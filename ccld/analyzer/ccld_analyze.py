#!/usr/bin/env python3
"""CCL-D decision rules over per-rank JSONL. See docs/telemetry.md.

Duration and sequence numbers are local to a rank; no cross-host wall-clock
comparisons are made. An absent or incomplete observation is not a zero count.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .ring_wait import RingWaitAttribution
else:
    from ring_wait import RingWaitAttribution


@dataclass(frozen=True)
class Config:
    hang_s: float = 300.0
    window_s: float = 60.0
    baseline_ms: float = 1.0
    slow_threshold: float = 3.0
    repeats: int = 2
    alpha: float = 0.4
    beta: float = 0.6
    baseline_rounds: int = 100
    baseline_s: float = 120.0
    rate_metric: str = 'paper'
    flush_last_window: bool = False
    baseline_skip: int = 0
    ring_wait_gap: float = 0.2


def load_records(paths: list[Path], allow_partial: bool = False) -> tuple[list[dict], list[str]]:
    events: list[dict] = []
    problems: list[str] = []
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError as exc:
            problems.append(f'{path}: {exc}')
            continue
        lines = data.splitlines(keepends=True)
        for line_no, line in enumerate(lines, 1):
            if not line.endswith(b'\n'):
                if not allow_partial:
                    problems.append(f'{path}:{line_no}: incomplete line')
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError('event must be a JSON object')
                for field in ('event', 'comm_hash', 'rank', 'world_size', 'mono_ns'):
                    if field not in event:
                        raise ValueError(f'missing {field}')
                if type(event['rank']) is not int or type(event['world_size']) is not int:
                    raise ValueError('rank and world_size must be integers')
                if not (0 <= event['rank'] < event['world_size']):
                    raise ValueError('rank outside communicator')
                if type(event['mono_ns']) is not int or event['mono_ns'] < 0:
                    raise ValueError('mono_ns must be a nonnegative integer')
                if event['event'] in ('entered', 'snapshot', 'complete', 'heartbeat', 'dropped'):
                    if type(event.get('op_count')) is not int or event['op_count'] < 0:
                        raise ValueError('op_count must be a nonnegative integer')
                if event['event'] in ('snapshot', 'complete'):
                    if type(event.get('duration_ns')) is not int or event['duration_ns'] < 0:
                        raise ValueError('duration_ns must be a nonnegative integer')
                event['_source'] = str(path)
                event['_line'] = line_no
                events.append(event)
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                problems.append(f'{path}:{line_no}: {exc}')
    return events, problems


def barrier(rec: dict) -> bool:
    return str(rec.get('op', '')).lower().replace('_', '') == 'allreduce' and rec.get('bytes', math.inf) <= 4


def latest_by_rank(records: list[dict]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for event in records:
        r = event['rank']
        # Only compare timestamps from the same rank/process, never across hosts.
        if r not in result or (event['mono_ns'], event.get('seq', 0)) >= (result[r]['mono_ns'], result[r].get('seq', 0)):
            result[r] = event
    return result


def channel_map(event: dict) -> dict[int, dict]:
    result = {}
    for channel in event.get('channels', []):
        if isinstance(channel, dict) and type(channel.get('id')) is int:
            result[channel['id']] = channel
    return result


def channel_parents(event: dict) -> dict[int, int]:
    parents = getattr(event.get('channels'), 'parents', None)
    if parents is not None:
        return parents()
    return {cid: channel.get('tree_parent') for cid, channel in channel_map(event).items()}


def operation_signature_partition(states: dict[int, dict], world_size: int) -> dict:
    """Separate inconsistent all-hung signatures only under a strict-majority assumption."""
    if set(states) != set(range(world_size)):
        return {'status': 'incomplete', 'groups': []}
    groups: dict[tuple, list[int]] = defaultdict(list)
    for rank in range(world_size):
        event = states[rank]
        op, count, dtype, algorithm, protocol = (event.get(k) for k in
                                                  ('op', 'count', 'datatype', 'algorithm', 'protocol'))
        if (event.get('event') != 'snapshot' or type(count) is not int or count < 0 or
                any(not isinstance(x, str) or not x.strip() for x in (op, dtype, algorithm, protocol))):
            return {'status': 'incomplete', 'groups': []}
        key = (op.strip().lower(), count, dtype.strip().lower(), algorithm.strip().lower(), protocol.strip().lower())
        groups[key].append(rank)
    reported = [{'signature': list(key), 'ranks': ranks} for key, ranks in sorted(groups.items())]
    if len(groups) == 1:
        return {'status': 'consistent', 'groups': reported}
    majority = next((ranks for ranks in groups.values() if len(ranks) > world_size / 2), None)
    if majority is None:
        return {'status': 'ambiguous', 'groups': reported}
    roots = sorted(set(range(world_size)) - set(majority))
    return {'status': 'minority', 'groups': reported, 'majority_ranks': sorted(majority), 'minority_ranks': roots}


def comparable_groups(rank_events: dict[int, dict], world_size: int) -> tuple[list[tuple[str, list[int], int | None]], list[str]]:
    """Return (label, ranks, channel) groups. Tree groups need parent on all ranks."""
    algos = {str(event.get('algorithm', '')).lower() for event in rank_events.values()}
    algos.discard('')
    if len(algos) != 1:
        return [], ['algorithm missing or differs across observed ranks']
    algorithm = next(iter(algos))
    if algorithm == 'ring':
        return [('ring', sorted(rank_events), None)], []
    if algorithm != 'tree':
        return [], [f'unsupported comparison algorithm: {algorithm}']
    if len(rank_events) != world_size:
        return [], ['tree topology requires observations from every rank']
    channels = [channel_parents(rank_events[r]) for r in range(world_size)]
    ids = set(channels[0])
    if not ids or any(set(c) != ids for c in channels):
        return [], ['tree channel sets are missing or unequal']
    groups: list[tuple[str, list[int], int | None]] = []
    errors: list[str] = []
    for cid in sorted(ids):
        parent = {r: channels[r][cid] for r in range(world_size)}
        if any(type(p) is not int or p < -1 or p >= world_size or p == r for r, p in parent.items()):
            errors.append(f'channel {cid}: invalid or absent tree_parent')
            continue
        if sum(p == -1 for p in parent.values()) != 1:
            errors.append(f'channel {cid}: expected one root')
            continue
        levels: dict[int, list[int]] = defaultdict(list)
        try:
            for rank in range(world_size):
                depth, node, seen = 0, rank, set()
                while parent[node] != -1:
                    if node in seen:
                        raise ValueError('cycle')
                    seen.add(node)
                    node = parent[node]
                    depth += 1
                levels[depth].append(rank)
        except ValueError:
            errors.append(f'channel {cid}: cycle in tree topology')
            continue
        groups.extend((f'tree:channel={cid}:depth={depth}', members, cid)
                      for depth, members in sorted(levels.items()) if len(members) > 1)
    if not groups and not errors:
        errors.append('tree has no layer with at least two ranks')
    return groups, errors


def slow_groups(states, world):
    groups, warnings = comparable_groups(states, world)
    if warnings != ['tree has no layer with at least two ranks'] or world < 3:
        return groups, warnings
    hosts = {event.get('host') for event in states.values()}
    if len(hosts) == 1 and all(hosts):
        return [('tree-chain', sorted(states), None)], []
    return groups, warnings


def count_score(event: dict, cid: int | None) -> float | None:
    channels = channel_map(event)
    chosen = [channels[cid]] if cid is not None and cid in channels else list(channels.values()) if cid is None else []
    if not chosen:
        return None
    vals = []
    for c in chosen:
        a, b = c.get('send_count'), c.get('recv_count')
        if type(a) is not int or type(b) is not int or min(a, b) < 0:
            return None
        vals.append(a + b)
    return sum(vals)


def rate_score(event: dict, cid: int | None, metric: str) -> float | None:
    channels = channel_map(event)
    chosen = [channels[cid]] if cid is not None and cid in channels else list(channels.values()) if cid is None else []
    if not chosen:
        return None
    names = ('send_rate', 'recv_rate') if metric == 'paper' else ('send_delta_per_ms', 'recv_delta_per_ms')
    values = []
    for c in chosen:
        for name in names:
            val = c.get(name)
            # A direction with no samples is unknown, not an artificial zero.
            if type(val) in (int, float) and math.isfinite(val) and val >= 0:
                values.append(float(val))
    return sum(values) / len(values) if values else None


def comparable_rate_scores(states: dict[int, dict], members: list[int], cid: int | None,
                           metric: str) -> dict[int, float | None]:
    """Do not rank different channel sets or mismatched missing observations."""
    channels = {r: channel_map(states[r]) for r in members}
    ids = [{cid} if cid is not None and cid in channels[r] else set(channels[r]) if cid is None else set()
           for r in members]
    if not ids or not ids[0] or any(group != ids[0] for group in ids[1:]):
        return {r: None for r in members}
    names = ('send_rate','recv_rate') if metric == 'paper' else ('send_delta_per_ms','recv_delta_per_ms')
    for channel in ids[0]:
        for name in names:
            available = []
            for rank in members:
                value = channels[rank][channel].get(name)
                available.append(type(value) in (int,float) and math.isfinite(value) and value >= 0)
            if any(available) and not all(available):
                return {r: None for r in members}
    return {r: rate_score(states[r],cid,metric) for r in members}


def min_candidates(scores: dict[int, float | None]) -> list[int]:
    if not scores or any(v is None for v in scores.values()):
        return []
    m = min(scores.values())
    return sorted(k for k, v in scores.items() if math.isclose(v, m, rel_tol=1e-12, abs_tol=1e-12))


def result_common(comm: str, op_count: int | None, category: str) -> dict:
    return {'comm_hash': comm, 'op_count': op_count, 'category': category,
            'detection_latency_ms': None, 'location_latency_ms': None,
            'timing_note': 'offline input has no analyzer-side fault/arrival timestamps'}


def slow_decision(comm, op, states, label, window, members, cid, base, baseline, detections, cfg):
    durs = {r: states[r]['duration_ns'] / 1e6 for r in members}
    tmax, tmin = max(durs.values()), min(durs.values())
    ratio = (tmax-base)/base
    if ratio <= cfg.slow_threshold:
        return None
    p = (tmax-tmin)/(tmax-base)
    comp = min_candidates(durs)
    entry_gap_ms = None
    if label == 'tree-chain':
        if p <= cfg.beta or any(type(states[r].get('start_mono_ns')) is not int for r in members):
            return None
        starts = sorted((states[r]['start_mono_ns'], r) for r in members)
        latest, late_rank = starts[-1]
        entry_gap_ms = (latest - starts[-2][0]) / 1e6
        if (entry_gap_ms <= base or comp != [late_rank] or
                any(states[r]['mono_ns'] < latest for r in members)):
            return None
    detections[label] += 1
    item = result_common(comm, op, 'slow')
    item.update(group=label, window=int(window), t_base_ms=base, t_max_ms=tmax, t_min_ms=tmin,
                slowdown_ratio=ratio, detections=detections[label], baseline=baseline)
    if detections[label] < cfg.repeats:
        item.update(kind='detected', roots=[], reason='waiting for repetition threshold')
        return item
    locate_start = time.perf_counter_ns()
    rates = comparable_rate_scores(states, members, cid, cfg.rate_metric)
    comm_roots = min_candidates(rates)
    if p > cfg.beta:
        kind, roots = 'S1', comp
    elif p < cfg.alpha:
        kind, roots = ('S2', comm_roots) if comm_roots else ('undetermined', [])
    else:
        kind, roots = ('S3', sorted(set(comp+comm_roots))) if comm_roots else ('undetermined', [])
    item.update(kind=kind, roots=roots, p=p, compute_roots=comp,
                communication_roots=comm_roots, rate_metric=cfg.rate_metric, rate_scores=rates)
    if entry_gap_ms is not None:
        item.update(method='same_host_tree_entry_gap', entry_gap_ms=entry_gap_ms)
    if p < 0 or p > 1:
        item['baseline_warning'] = 'P is outside [0,1]; baseline is outside the observed duration range'
    item['location_latency_ms'] = (time.perf_counter_ns() - locate_start) / 1e6
    return item


def analyze_comm(comm: str, events: list[dict], cfg: Config) -> list[dict]:
    sizes = {e['world_size'] for e in events}
    if len(sizes) != 1:
        return [{'comm_hash': comm, 'category': 'invalid', 'reason': 'inconsistent world_size'}]
    world = sizes.pop()
    by_op: dict[int, list[dict]] = defaultdict(list)
    dropped: dict[int, list[dict]] = defaultdict(list)
    # Include init and heartbeat to distinguish an observed but lagging rank.
    history = defaultdict(list)
    for e in events:
        history[e['rank']].append(e)
        if e['event'] == 'dropped':
            dropped[e['op_count']].append(e)
        if e['event'] in ('entered', 'snapshot', 'complete'):
            by_op[e['op_count']].append(e)
    latest_progress = {}
    for rank, hist in history.items():
        progress = [e for e in hist if 'op_count' in e]
        latest_progress[rank] = max((e['op_count'] for e in progress), default=0)

    output: list[dict] = []
    gap_boundaries = []
    for event in events:
        if event['event'] == 'coverage_gap':
            boundary = event.get('op_count')
            gap_boundaries.append(boundary if type(boundary) is int and boundary >= 0 else 0)
            output.append({'comm_hash': comm, 'op_count': event.get('op_count'),
                           'rank': event['rank'], 'category': 'coverage_gap',
                           'kind': 'undetermined', 'roots': [],
                           'reason': event.get('reason', 'untraced activity')})
    completed: list[tuple[int, dict[int, dict], int]] = []
    earliest_incomplete = None
    for op, records in sorted(by_op.items()):
        if gap_boundaries and op > min(gap_boundaries):
            item = result_common(comm, op, 'invalid')
            item.update(kind='undetermined', roots=[],
                        reason='an earlier coverage gap may have broken cross-rank operation alignment',
                        coverage_gap_after_op_count=min(gap_boundaries))
            output.append(item)
            if earliest_incomplete is None:
                earliest_incomplete = op
            continue
        if op in dropped:
            item = result_common(comm, op, 'invalid')
            item.update(kind='undetermined', roots=[], reason='telemetry explicitly dropped',
                        dropped=[{'rank': e['rank'], 'reason': e.get('reason')} for e in dropped[op]])
            output.append(item)
            if earliest_incomplete is None:
                earliest_incomplete = op
            continue
        # Prefer complete over a stale snapshot regardless of logging order.
        states = {}
        for r in {e['rank'] for e in records}:
            rr = [e for e in records if e['rank'] == r]
            done = [e for e in rr if e['event'] == 'complete']
            states[r] = latest_by_rank(done or rr)[r]
        measured = [e for e in states.values() if e['event'] in ('snapshot', 'complete')]
        is_barrier = bool(measured) and all(barrier(e) for e in measured)
        all_complete = len(states) == world and all(e['event'] == 'complete' for e in states.values())
        blocked_by = earliest_incomplete
        if not all_complete and not is_barrier and earliest_incomplete is None:
            earliest_incomplete = op
        if any(e.get('op') == 'P2PAllPeers' for e in records):
            item = result_common(comm, op, 'unsupported')
            item.update(kind='undetermined', roots=[],
                        reason='all-peer P2P progress is not a validated Ring or Tree diagnosis')
            output.append(item)
            continue
        if not measured or is_barrier:
            continue
        if all_complete and blocked_by is None:
            # Rank 0's local completion clock defines window boundaries.
            completed.append((op, states, states[0]['mono_ns']))
        long_events = [e for e in measured if e['duration_ns'] >= cfg.hang_s * 1e9]
        if not long_events:
            continue
        locate_start = time.perf_counter_ns()
        hang = result_common(comm, op, 'hang')
        hang['threshold_s'] = cfg.hang_s
        hang['observed_elapsed_ms'] = max(e['duration_ns'] for e in long_events) / 1e6
        if blocked_by is not None:
            hang.update(kind='undetermined', roots=[], blocked_by_op_count=blocked_by,
                        reason='an earlier operation is incomplete; later rank differences may be consequences')
            hang['location_latency_ms'] = (time.perf_counter_ns() - locate_start) / 1e6
            output.append(hang)
            continue
        unknown = [r for r in range(world) if r not in history]
        missing = [r for r in range(world) if r not in states and r in history and latest_progress[r] < op]
        advanced = [r for r in range(world) if r not in states and latest_progress.get(r, 0) >= op]
        if unknown or advanced:
            hang.update(kind='undetermined', roots=[], unknown_ranks=unknown,
                        missing_trace_ranks=advanced, reason='missing telemetry cannot establish entry state')
        elif missing:
            hang.update(kind='H1', roots=missing,
                        reason='last observed operation count is behind; stale logs/process death can look identical')
        elif len(states) != world:
            hang.update(kind='undetermined', roots=[], reason='incomplete rank evidence')
        else:
            finished = sorted(r for r, e in states.items() if e['event'] == 'complete')
            # Preserve the per-rank operation metadata even when all ranks
            # hang and channel counts cannot be compared. This is evidence,
            # not an automatic classification or root-cause decision.
            signatures = {r: [e.get(x) for x in ('op', 'count', 'datatype', 'algorithm', 'protocol')]
                          for r, e in states.items()}
            hang['operation_signatures'] = signatures
            # A completion after the hang threshold is not evidence that this
            # rank was un-hung: an error/abort can release a pending callback.
            not_hung = [r for r in finished if states[r]['duration_ns'] < cfg.hang_s * 1e9]
            if len(finished) == world:
                hang.update(kind='resolved', roots=[], reason='all ranks completed in the input')
            elif not_hung:
                hang.update(kind='H2', roots=not_hung,
                            reason='entered ranks include completions before threshold and outstanding operations')
            elif finished:
                hang.update(kind='undetermined', roots=[],
                            reason='some completions occurred only after the hang threshold; success is unknown')
            else:
                partition = operation_signature_partition(states, world)
                hang['operation_signature_groups'] = partition['groups']
                if partition['status'] == 'minority':
                    hang.update(kind='H2', roots=partition['minority_ranks'],
                                signature_majority_ranks=partition['majority_ranks'],
                                signature_heuristic='strict_majority_assumed_correct',
                                reason='all ranks are hung and minority operation signatures differ from a strict majority')
                    hang['location_latency_ms'] = (time.perf_counter_ns() - locate_start) / 1e6
                    output.append(hang)
                    continue
                if partition['status'] in ('incomplete', 'ambiguous'):
                    hang.update(kind='undetermined', roots=[],
                                reason=('incomplete operation signatures' if partition['status'] == 'incomplete'
                                        else 'conflicting operation signatures have no strict majority'))
                    hang['location_latency_ms'] = (time.perf_counter_ns() - locate_start) / 1e6
                    output.append(hang)
                    continue
                groups, errors = comparable_groups(states, world)
                comparisons = []
                for label, members, cid in groups:
                    channel_sets = [set(channel_map(states[r])) for r in members]
                    same_channels = cid is not None or (channel_sets and all(x == channel_sets[0] for x in channel_sets))
                    scores = {r: count_score(states[r], cid) if same_channels else None for r in members}
                    comparisons.append({'group': label, 'roots': min_candidates(scores), 'scores': scores})
                comparable = bool(comparisons and any(c['roots'] for c in comparisons))
                hang.update(kind='H3' if comparable else 'undetermined',
                            roots=sorted({r for c in comparisons for r in c['roots']}),
                            comparisons=comparisons, warnings=errors,
                            reason=('minimum send+recv counts; tied ranks are all candidates' if comparable
                                    else 'no comparable per-rank channel counts'))
        hang['location_latency_ms'] = (time.perf_counter_ns() - locate_start) / 1e6
        output.append(hang)

    # Windows use rank 0's monotonically increasing local timestamps only.
    completed.sort(key=lambda x: x[2])
    if not completed:
        return output
    first_ns = completed[0][2]
    baseline_values = []
    baseline_history = completed[cfg.baseline_skip:]
    baseline_first_ns = baseline_history[0][2] if baseline_history else None
    for _, states, t in baseline_history:
        if len(baseline_values) >= cfg.baseline_rounds or t - baseline_first_ns > cfg.baseline_s * 1e9:
            break
        baseline_values.append(max(e['duration_ns'] for e in states.values()) / 1e6)
    baseline_ready_ns = None
    if len(baseline_values) == cfg.baseline_rounds:
        baseline_ready_ns = baseline_history[cfg.baseline_rounds-1][2]
    elif baseline_first_ns is not None and completed[-1][2] - baseline_first_ns >= cfg.baseline_s * 1e9 and baseline_values:
        baseline_ready_ns = baseline_first_ns + int(cfg.baseline_s * 1e9)
    dynamic_base = sum(baseline_values) / len(baseline_values) if baseline_values else None
    window_ns = max(1, int(cfg.window_s * 1e9))
    windows = defaultdict(list)
    for op, states, t in completed:
        groups, errors = slow_groups(states, world)
        if not groups:
            item = result_common(comm, op, 'coverage')
            item.update(kind='undetermined', roots=[],
                        reason='slow comparison unavailable', warnings=errors)
            output.append(item)
            continue
        for label, members, cid in groups:
            if len(members) < 2:
                continue
            times = [states[r]['duration_ns'] / 1e6 for r in members]
            windows[((t-first_ns)//window_ns, label)].append((max(times)-min(times), op, states, members, cid, t))
    detections = defaultdict(int)
    ring_wait = RingWaitAttribution(cfg.ring_wait_gap, cfg.slow_threshold)
    last_ns = completed[-1][2]
    for (window, label), entries in sorted(windows.items()):
        end_ns = first_ns + (window+1)*window_ns
        if end_ns > last_ns and not cfg.flush_last_window:
            continue
        _, op, states, members, cid, t = max(entries, key=lambda x: (x[0], -x[1]))
        base = dynamic_base if baseline_ready_ns is not None and t > baseline_ready_ns else cfg.baseline_ms
        baseline = 'dynamic' if base == dynamic_base and baseline_ready_ns is not None and t > baseline_ready_ns else 'initial'
        item = slow_decision(comm, op, states, label, window, members, cid, base, baseline, detections, cfg)
        if label == 'ring':
            item = ring_wait.refine(item, states, op, window)
        if item:
            output.append(item)
    return output


def analyze(events: list[dict], cfg: Config) -> list[dict]:
    grouped = defaultdict(list)
    for event in events:
        grouped[str(event['comm_hash'])].append(event)
    return [result for comm, records in sorted(grouped.items()) for result in analyze_comm(comm, records, cfg)]



def annotate_live_latency(item: dict, events: list[dict], faults: list[dict],
                          observer_host: str, detection_ns: int) -> dict:
    """Measure only when injection and observer share the Linux monotonic clock.

    For each marker, infer the target collective as one greater than the last
    entered collective on that rank before the marker. Refuse ambiguous comms.
    """
    matches = []
    for fault in faults:
        rank, host, when = fault.get('rank'), fault.get('host'), fault.get('mono_ns')
        if type(rank) is not int or type(when) is not int or host != observer_host or when > detection_ns:
            continue
        rank_events = [e for e in events if e['rank'] == rank and e['mono_ns'] <= when
                       and e['event'] in ('comm_init', 'entered')]
        comms = {str(e['comm_hash']) for e in rank_events}
        if len(comms) != 1 or item.get('comm_hash') not in comms:
            continue
        before = [e['op_count'] for e in rank_events if e['event'] == 'entered']
        target = max(before, default=0) + 1
        if item.get('op_count') == target:
            matches.append(fault)
    if matches:
        fault = max(matches, key=lambda f: f['mono_ns'])
        item = dict(item)
        item['detection_latency_ms'] = (detection_ns - fault['mono_ns']) / 1e6
        item['timing_note'] = ('same-host CLOCK_MONOTONIC injection to analyzer observation; '
                               'includes file polling and assumes one active communicator')
        item['fault_marker'] = {'kind': fault.get('kind'), 'rank': fault['rank'],
                                'host': fault['host'], 'iteration': fault.get('iteration')}
    return item


def load_fault_markers(paths: list[Path]) -> list[dict]:
    markers = []
    for path in paths:
        files = sorted(path.rglob('*.ndjson')) if path.is_dir() else [path]
        for f in files:
            try:
                lines = f.read_bytes().splitlines(keepends=True)
                for line in lines:
                    if not line.endswith(b'\n'):
                        continue
                    event = json.loads(line)
                    if isinstance(event, dict):
                        markers.append(event)
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                print(f'{f}: fault marker: {exc}', file=sys.stderr)
    return markers


def add_decision_arguments(p):
    p.add_argument('--hang-s', type=float, default=300)
    p.add_argument('--window-s', type=float, default=60)
    p.add_argument('--baseline-ms', type=float, required=True)
    p.add_argument('--slow-threshold', type=float, default=3)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--alpha', type=float, default=.4)
    p.add_argument('--beta', type=float, default=.6)
    p.add_argument('--baseline-rounds', type=int, default=100)
    p.add_argument('--baseline-skip', type=int, default=0, help='exclude initial warmup rounds from dynamic baseline')
    p.add_argument('--baseline-s', type=float, default=120)
    p.add_argument('--rate-metric', choices=('paper', 'delta'), default='paper')
    p.add_argument('--ring-wait-gap', type=float, default=0.2,
                   help='minimum normalized directional wait score gap')


def decision_config(p, args, flush_last_window=False):
    if (any(not math.isfinite(x) or x <= 0 for x in
            (args.hang_s, args.window_s, args.baseline_ms, args.baseline_s, args.ring_wait_gap)) or
            not math.isfinite(args.slow_threshold) or args.slow_threshold < 0 or
            args.repeats < 1 or args.baseline_rounds < 1 or args.baseline_skip < 0 or
            not (0 <= args.alpha < args.beta <= 1)):
        p.error('durations, baseline and counts must be positive; require 0 <= alpha < beta <= 1')
    return Config(args.hang_s, args.window_s, args.baseline_ms, args.slow_threshold,
                  args.repeats, args.alpha, args.beta, args.baseline_rounds, args.baseline_s,
                  args.rate_metric, flush_last_window, args.baseline_skip, args.ring_wait_gap)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths', nargs='+', type=Path, help='JSONL files or directories (searched recursively)')
    add_decision_arguments(p)
    p.add_argument('--flush-last-window', action='store_true', help='analyze incomplete last window (offline experiments)')
    p.add_argument('--follow', action='store_true', help='poll for complete new JSONL lines')
    p.add_argument('--fault-events', nargs='*', type=Path, default=[], help='injector .ndjson files/directories for same-host live latency')
    p.add_argument('--poll-s', type=float, default=1)
    args = p.parse_args()
    if not math.isfinite(args.poll_s) or args.poll_s <= 0:
        p.error('--poll-s must be positive')
    cfg = decision_config(p, args, args.flush_last_window)
    emitted = set()
    reported_problems = set()
    while True:
        files = sorted({f for path in args.paths for f in (path.rglob('*.jsonl') if path.is_dir() else [path])})
        events, problems = load_records(files, allow_partial=args.follow)
        for problem in problems:
            if not args.follow or problem not in reported_problems:
                print(problem, file=sys.stderr)
                reported_problems.add(problem)
        markers = load_fault_markers(args.fault_events) if args.follow else []
        for item in analyze(events, cfg):
            if args.follow and sys.platform.startswith('linux') and item.get('category') in ('hang', 'slow'):
                item = annotate_live_latency(item, events, markers, socket.gethostname(), time.monotonic_ns())
            key = (item.get('comm_hash'), item.get('op_count'), item.get('category'), item.get('kind'), item.get('group'), item.get('window'), item.get('rank'), tuple(item.get('roots', [])))
            if not args.follow or key not in emitted:
                print(json.dumps(item, ensure_ascii=False, sort_keys=True), flush=True)
                emitted.add(key)
        if not args.follow:
            return 1 if problems else 0
        time.sleep(args.poll_s)


if __name__ == '__main__':
    raise SystemExit(main())
