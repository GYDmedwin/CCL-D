#!/usr/bin/env python3
"""Experimental three-round Ring wait scoring; an explicit slow gate is required."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
import statistics
import time


PHASE_WINDOW_NS = 2_500_000
PHASE_ENTRY_GAP_NS = 1_000_000
WAIT_FIELDS = ('wait_data_spins', 'wait_credit_spins')


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


def phase_boundaries(states, snapshots, hosts):
    """Bracket the last entry with observed samples on the same host clock."""
    if not snapshots or len(states) < 3:
        return None
    clock_hosts = {hosts.get(rank) for rank in states}
    if len(clock_hosts) != 1 or not all(isinstance(h, str) and h for h in clock_hosts):
        return None
    if any(type(event.get('start_mono_ns')) is not int for event in states.values()):
        return None
    ordered = sorted((e['start_mono_ns'], rank) for rank, e in states.items())
    cutoff, last = ordered[-1]
    if cutoff - ordered[-2][0] <= PHASE_ENTRY_GAP_NS or any(e['mono_ns'] < cutoff for e in states.values()):
        return None
    bounds = {}
    for rank, final in states.items():
        if rank == last:
            continue
        before, after = None, final if final['mono_ns'] <= cutoff + PHASE_WINDOW_NS else None
        for event in snapshots.get(rank, ()):
            if event.get('start_mono_ns') != final['start_mono_ns']:
                continue
            when = event['mono_ns']
            if cutoff - PHASE_WINDOW_NS <= when <= cutoff and (before is None or when > before['mono_ns']):
                before = event
            if cutoff <= when <= cutoff + PHASE_WINDOW_NS and (after is None or when < after['mono_ns']):
                after = event
        if before is None or after is None:
            return None
        bounds[rank] = before, after
    return last, bounds


def phase_vote(states, boundaries, gap):
    """Vote only when subtraction on both sides produces the same strong root."""
    last, bounds = boundaries
    roots = []
    for index in range(2):
        views = {}
        for rank, final in states.items():
            baseline = bounds[rank][index] if rank != last else None
            rows = {c.get('id'): c for c in baseline['channels']} if baseline is not None else {}
            if baseline is not None and (baseline.get('wait_probe') is not True or
                    set(rows) != {c.get('id') for c in final['channels']} or len(rows) != len(baseline['channels'])):
                return None
            channels = []
            for channel in final['channels']:
                old = rows.get(channel.get('id'))
                if old is not None and any(old.get(k) != channel.get(k) for k in ('ring_prev', 'ring_next')):
                    return None
                deltas = {}
                for field in WAIT_FIELDS:
                    end = channel.get(field)
                    start = old.get(field) if old is not None else 0
                    if type(end) is not int or type(start) is not int or not 0 <= start <= end:
                        return None
                    deltas[field] = end - start
                channels.append(dict(channel, **deltas))
            views[rank] = dict(final, channels=channels)
        score = graph_wait_scores(views)
        candidates = score['forced_top_ranks']
        if len(candidates) != 1 or score['top_gap'] < gap or candidates[0] == last:
            return None
        roots.append(candidates[0])
    return roots[0] if roots[0] == roots[1] else None


class RingWaitAttribution:
    """Require directionally consistent waits before attributing communication slowdown."""

    def __init__(self, gap, slow_threshold, use_phases=False, alpha=0.4):
        self.gap = gap
        self.slow_threshold = slow_threshold
        self.use_phases = use_phases
        self.alpha = alpha
        self.rounds = deque(maxlen=3)
        self.last_window = None
        self.last_phase = False

    def refine(self, item, states, operation, window, boundaries=None):
        if item is None or not all(state.get('wait_probe') is True for state in states.values()):
            self.rounds.clear()
            return item
        kind = item['kind']
        mixed = kind == 'S3' or (kind == 'S1' and
                item['t_min_ms'] > item['t_base_ms'] * (1 + self.slow_threshold))
        if self.use_phases and kind == 'undetermined' and item['p'] >= self.alpha:
            mixed = True
        verify_s1 = self.use_phases and kind == 'S1' and not mixed
        if not (mixed or verify_s1 or kind in ('S2', 'undetermined')):
            self.rounds.clear()
            return item
        phased = self.use_phases and (mixed or verify_s1)
        if phased != self.last_phase or (self.last_window is not None and window - self.last_window > 2) or (
                self.rounds and self.rounds[-1][0] >= operation):
            self.rounds.clear()
        self.last_window = window
        self.last_phase = phased
        started = time.perf_counter_ns()
        comp = item['compute_roots'] if mixed or verify_s1 else []
        if phased:
            vote = None
            if boundaries is not None and comp == [boundaries[0]]:
                cutoff = states[comp[0]]['start_mono_ns']
                tail_ns = statistics.median(e['mono_ns'] - cutoff for e in states.values())
                # The early finish of the late rank can hide a slow tail on its peers.
                if tail_ns > max(PHASE_ENTRY_GAP_NS, item['t_base_ms'] * 1e6):
                    try:
                        vote = phase_vote(states, boundaries, self.gap)
                    except ValueError:
                        pass
            self.rounds.append((operation, (tuple(comp), vote)))
        else:
            self.rounds.append((operation, states))
        if len(self.rounds) < 3:
            if verify_s1:
                return item
            return dict(item, kind='detected', roots=[], reason='waiting for three directional samples')

        if phased:
            votes = Counter(rank for _, (compute, rank) in self.rounds if rank is not None and compute == tuple(comp))
            suspects = [rank for rank, support in votes.items() if support >= 2 and (not verify_s1 or rank == vote)]
        else:
            try:
                suspects = decide(dict(self.rounds), self.gap, slow_confirmed=True)['candidate_ranks']
            except ValueError:
                suspects = []
        if len(suspects) != 1 or suspects[0] in comp:
            if verify_s1:
                return item
            return dict(item, kind='undetermined', roots=[], compute_roots=comp,
                        communication_roots=[], reason=('no consistent post-entry wait attribution' if phased
                                                        else 'no consistent directional wait attribution'))
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return dict(item, kind='S3' if mixed or phased else 'S2',
                    roots=sorted(set(comp + suspects)), communication_roots=suspects,
                    compute_roots=comp, method='post_entry_ring_wait' if phased else 'directional_ring_wait',
                    location_latency_ms=(item.get('location_latency_ms') or 0) + elapsed_ms)


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
