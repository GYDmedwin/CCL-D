#!/usr/bin/env python3
"""Experimental three-round Ring wait scoring; an explicit slow gate is required."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def graph_wait_scores(states: dict[int, dict]) -> dict:
    """Compute the historical per-rank-volume score over actual Ring edges."""
    ranks = set(states)
    if ranks != set(range(len(ranks))) or len(ranks) < 3:
        raise ValueError('a complete communicator with at least three ranks is required')
    if any(states[r].get('wait_probe') is not True for r in ranks):
        raise ValueError('Ring wait probe is unavailable on at least one rank')
    maps = {r: {c.get('id'): c for c in states[r].get('channels', [])} for r in ranks}
    ids = set(maps[0])
    if (not ids or any(type(i) is not int for i in ids) or
            any(set(maps[r]) != ids or len(maps[r]) != len(states[r].get('channels', [])) for r in ranks)):
        raise ValueError('missing, duplicate, or inconsistent channel sets')
    delta: dict[int, int] = defaultdict(int)
    volume = 0
    for cid in sorted(ids):
        for r in sorted(ranks):
            c = maps[r][cid]
            nxt, prev = c.get('ring_next'), c.get('ring_prev')
            if (type(nxt) is not int or type(prev) is not int or nxt not in ranks or prev not in ranks or
                    nxt == r or prev == r or maps[nxt][cid].get('ring_prev') != r or
                    maps[prev][cid].get('ring_next') != r):
                raise ValueError(f'channel {cid}: invalid Ring links')
            credit, data = c.get('wait_credit_spins'), maps[nxt][cid].get('wait_data_spins')
            if any(type(v) is not int or v < 0 for v in (credit, data)):
                raise ValueError(f'channel {cid}: invalid wait values')
            delta[r] += data - credit
            delta[nxt] += credit - data
            volume += credit + data
        seen, node = set(), 0
        while node not in seen:
            seen.add(node)
            node = maps[node][cid]['ring_next']
        if node != 0 or seen != ranks:
            raise ValueError(f'channel {cid}: topology is not one Ring')
    if not volume:
        return {'status': 'unavailable', 'scores': None, 'forced_top_ranks': [], 'top_gap': None}
    scores = {r: delta[r] * len(ranks) / (2 * volume) for r in sorted(ranks)}
    values = sorted(scores.values(), reverse=True)
    top = values[0]
    return {'status': 'exploratory', 'scores': {str(r): v for r, v in scores.items()},
            'forced_top_ranks': [r for r, value in scores.items()
                                 if math.isclose(value, top, rel_tol=1e-12, abs_tol=1e-12)],
            'top_gap': values[0] - values[1]}


def completed_rounds(directory: Path, protocol: str, operations: tuple[int, ...] = (4, 5, 6)) -> tuple[dict, str]:
    """Read one complete 8-rank, one-host communicator, rejecting ambiguous traces."""
    states = {op: {} for op in operations}
    inits, signatures = {}, set()
    files = sorted(Path(directory).glob('*.jsonl'))
    if not files:
        raise ValueError('no decoded trace files')
    for path in files:
        with path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    continue  # live decoder has not finished this line
                event = json.loads(line)
                kind = event.get('event')
                if kind in ('dropped', 'coverage_gap'):
                    raise ValueError('trace contains a dropped record or coverage gap')
                rank = event.get('rank')
                if kind == 'comm_init':
                    if rank in inits:
                        raise ValueError('duplicate rank initialization or multiple communicators')
                    inits[rank] = event
                elif kind == 'complete' and event.get('op_count') in states:
                    op = event['op_count']
                    if rank in states[op]:
                        raise ValueError('duplicate rank completion')
                    if (event.get('algorithm'), event.get('protocol')) != ('RING', protocol.upper()):
                        raise ValueError('actual algorithm or protocol differs')
                    states[op][rank] = event
                    signatures.add((event.get('count'), event.get('datatype'), event.get('n_channels')))
    if (set(inits) != set(range(8)) or len({e.get('comm_hash') for e in inits.values()}) != 1 or
            any(e.get('world_size') != 8 or e.get('poll_us') != 200 for e in inits.values())):
        raise ValueError('requires one complete 8-rank communicator with a 200 us polling interval')
    hosts = {e.get('host') for e in inits.values()}
    if len(hosts) != 1 or not isinstance(next(iter(hosts)), str) or not next(iter(hosts)):
        raise ValueError('single-host communicator identity is unavailable')
    if any(set(group) != set(range(8)) for group in states.values()) or len(signatures) != 1:
        raise ValueError('incomplete ranks or inconsistent workload signature')
    comm = next(iter(inits.values())).get('comm_hash')
    if any(e.get('comm_hash') != comm or e.get('world_size') != 8 or e.get('op') != 'AllReduce'
           for group in states.values() for e in group.values()):
        raise ValueError('completed operation does not match the communicator or AllReduce')
    return states, comm


def decide(states: dict, threshold: float, slow_confirmed: bool) -> dict:
    """Apply the three-round median/majority rule only after a confirmed slow event."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be finite and positive')
    if not slow_confirmed:
        return {'status': 'indeterminate', 'detected': None, 'candidate_ranks': [],
                'reason': 'matching slow diagnosis has not been confirmed'}
    if len(states) != 3:
        raise ValueError('exactly three completed operations are required')
    metrics = [graph_wait_scores(group) for _, group in sorted(states.items())]
    gaps = [m['top_gap'] for m in metrics]
    if any(gap is None for gap in gaps):
        return {'status': 'indeterminate', 'detected': None, 'candidate_ranks': [],
                'reason': 'normalization is unavailable'}
    detected = statistics.median(gaps) >= threshold
    votes = Counter(m['forced_top_ranks'][0] for m in metrics
                    if m['top_gap'] >= threshold and len(m['forced_top_ranks']) == 1)
    return {'status': 'evaluated', 'detected': detected,
            'candidate_ranks': [r for r, n in sorted(votes.items()) if n >= 2] if detected else [],
            'trial_median_gap': statistics.median(gaps), 'operation_gaps': gaps,
            'qualifying_votes': dict(sorted(votes.items()))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace_dir', type=Path, help='decoded rank JSONL files for one communicator')
    p.add_argument('--protocol', required=True, choices=('SIMPLE', 'LL', 'LL128'))
    p.add_argument('--threshold', type=float, required=True, help='threshold calibrated for this specific build/workload')
    p.add_argument('--slow-confirmed', action='store_true', help='matching slow result already confirmed separately')
    args = p.parse_args()
    try:
        states, comm = completed_rounds(args.trace_dir, args.protocol)
        result = decide(states, args.threshold, args.slow_confirmed)
        result.update(comm_hash=comm, method='volume_per_rank')
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {'status': 'indeterminate', 'detected': None, 'candidate_ranks': [], 'reason': str(exc)}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
