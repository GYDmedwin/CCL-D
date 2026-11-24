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
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

if __package__:
    from .admission import RankAdmission
    from .ring_wait import RankMixedWait, RingWaitAttribution, phase_boundaries
    from .tree_hang import tree_hang
    from .tree_wait import TreeWaitAttribution, tree_phase, uniform_slowdown
else:
    from admission import RankAdmission
    from ring_wait import RankMixedWait, RingWaitAttribution, phase_boundaries
    from tree_hang import tree_hang
    from tree_wait import TreeWaitAttribution, tree_phase, uniform_slowdown


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
    slow_warmup_ops: int = 0
    baseline_scope: str = 'communicator'
    ring_wait_phases: bool = False
    tree_wait_gap: float = 0.2
    tree_cross_host_phases: bool = False


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


SIGNATURE_FIELDS = ('op', 'count', 'datatype', 'algorithm', 'protocol')
ENTRY_FIELDS = SIGNATURE_FIELDS[:3]
GROUPED_PLAN_SKIP = 'multiple_collectives'
RECOVERABLE_DROP_REASONS = {GROUPED_PLAN_SKIP, 'frame_slot_busy'}


def signature_fields(states, fields=SIGNATURE_FIELDS):
    if states and states.get(0, {}).get('reduction_op') and all(
            isinstance(e.get('reduction_op'), str) and e['reduction_op'].strip()
            for e in states.values()):
        return fields + ('reduction_op',)
    return fields


def acknowledged_exclusion(entry, dropped):
    """Recognize a producer's explicit omission of one recorded operation."""
    reason = dropped.get('reason')
    if (reason not in RECOVERABLE_DROP_REASONS or entry is None or
            entry['event'] != 'entered' or dropped['mono_ns'] < entry['mono_ns']):
        return None
    op, count, dtype = (entry.get(field) for field in ENTRY_FIELDS)
    if (type(count) is not int or count < 0 or
            any(not isinstance(value, str) or not value for value in (op, dtype))):
        return None
    return dict(entry, event='skipped', reason=reason)


def aligned_exclusion(states, world):
    """Retire an incomplete operation only with consistent evidence from all ranks."""
    if set(states) != set(range(world)) or any(e['event'] not in ('complete', 'skipped') for e in states.values()):
        return None
    reasons = {e['reason'] for e in states.values() if e['event'] == 'skipped'}
    if len(reasons) != 1:
        return None
    reason = next(iter(reasons))
    if reason == GROUPED_PLAN_SKIP and any(e['event'] != 'skipped' for e in states.values()):
        return None
    first = states[0]
    fields = signature_fields(states, ENTRY_FIELDS)
    if all(all(e.get(k) == first.get(k) for k in fields) for e in states.values()):
        return reason
    return None


def exclusion_report(comm, op, reason):
    item = result_common(comm, op, 'unsupported' if reason == GROUPED_PLAN_SKIP else 'coverage')
    item.update(kind='skipped', roots=[], reason=reason)
    return item


def recorded_exclusion(records, dropped, world):
    drops = {e['rank']: e for e in dropped}
    if len(drops) != len(dropped) or not drops.keys() <= set(range(world)):
        return None
    by_rank = defaultdict(list)
    for event in records:
        by_rank[event['rank']].append(event)
    states = {}
    for rank in range(world):
        evidence = by_rank.get(rank, [])
        if rank in drops:
            skipped = acknowledged_exclusion(evidence[0], drops[rank]) if len(evidence) == 1 else None
            if skipped is None:
                return None
            states[rank] = skipped
        else:
            finished = [event for event in evidence if event['event'] == 'complete']
            if len(finished) != 1:
                return None
            states[rank] = finished[0]
    return aligned_exclusion(states, world)


def operation_signature_partition(states: dict[int, dict], world_size: int,
                                  event_type: str = 'snapshot') -> dict:
    """Attribute an inconsistent signature only when a strict majority agrees."""
    if set(states) != set(range(world_size)):
        return {'status': 'incomplete', 'groups': []}
    groups: dict[tuple, list[int]] = defaultdict(list)
    fields = signature_fields(states)
    for rank in range(world_size):
        event = states[rank]
        op, count, dtype, algorithm, protocol = (event.get(k) for k in SIGNATURE_FIELDS)
        if (event.get('event') != event_type or type(count) is not int or count < 0 or
                any(not isinstance(x, str) or not x.strip() for x in (op, dtype, algorithm, protocol))):
            return {'status': 'incomplete', 'groups': []}
        key = (op.strip().lower(), count, dtype.strip().lower(), algorithm.strip().lower(), protocol.strip().lower())
        if len(fields) > len(SIGNATURE_FIELDS):
            key += (event['reduction_op'].strip().lower(),)
        groups[key].append(rank)
    reported = [{'signature': list(key), 'ranks': ranks} for key, ranks in sorted(groups.items())]
    if len(groups) == 1:
        return {'status': 'consistent', 'groups': reported}
    majority = next((ranks for ranks in groups.values() if len(ranks) > world_size / 2), None)
    if majority is None:
        return {'status': 'ambiguous', 'groups': reported}
    roots = sorted(set(range(world_size)) - set(majority))
    return {'status': 'minority', 'groups': reported, 'majority_ranks': sorted(majority), 'minority_ranks': roots}


@lru_cache(maxsize=256)
def tree_peer_layers(parents: tuple[int, ...]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Compute and reuse peer layers for a channel's exact parent vector."""
    world = len(parents)
    if any(p < -1 or p >= world or p == r for r, p in enumerate(parents)):
        raise ValueError('invalid or absent tree_parent')
    if parents.count(-1) != 1:
        raise ValueError('expected one root')

    root = parents.index(-1)
    depths: dict[int, int] = {root: 0}
    layers: dict[int, list[int]] = defaultdict(list)
    for rank in range(world):
        node, path = rank, []
        while node not in depths:
            depths[node] = -1
            path.append(node)
            node = parents[node]
        if depths[node] == -1:
            raise ValueError('cycle in tree topology')
        depth = depths[node]
        for member in reversed(path):
            depth += 1
            depths[member] = depth
        layers[depths[rank]].append(rank)
    return tuple((depth, tuple(members)) for depth, members in sorted(layers.items()) if len(members) > 1)


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
    if not world_size or set(rank_events) != set(range(world_size)):
        return [], ['tree topology requires observations from every rank']
    channels = [channel_parents(rank_events[r]) for r in range(world_size)]
    ids = set(channels[0])
    if not ids or any(set(c) != ids for c in channels):
        return [], ['tree channel sets are missing or unequal']
    groups: list[tuple[str, list[int], int | None]] = []
    errors: list[str] = []
    for cid in sorted(ids):
        parents = tuple(channel[cid] for channel in channels)
        if any(type(p) is not int for p in parents):
            errors.append(f'channel {cid}: invalid or absent tree_parent')
            continue
        try:
            layers = tree_peer_layers(parents)
        except ValueError as error:
            errors.append(f'channel {cid}: {error}')
            continue
        groups.extend((f'tree:channel={cid}:depth={depth}', list(members), cid)
                      for depth, members in layers)
    if not groups and not errors:
        errors.append('tree has no layer with at least two ranks')
    return groups, errors


def slow_groups(states, world, rank_hosts=None):
    groups, warnings = comparable_groups(states, world)
    if warnings != ['tree has no layer with at least two ranks'] or world < 3:
        return groups, warnings
    rank_hosts = rank_hosts or {}
    hosts = {rank_hosts.get(rank, event.get('host')) for rank, event in states.items()}
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


def completed_mismatch(comm, op, states, world):
    """Check metadata disagreement even when every rank reports completion."""
    first = states[0]
    fields = signature_fields(states)
    if all(event is first or all(event.get(k) == first.get(k) for k in fields)
           for event in states.values()):
        return None
    started = time.perf_counter_ns()
    partition = operation_signature_partition(states, world, event_type='complete')
    if partition['status'] not in ('minority', 'ambiguous'):
        return None
    item = result_common(comm, op, 'collective_mismatch')
    item.update(kind='H2' if partition['status'] == 'minority' else 'undetermined',
                roots=partition.get('minority_ranks', []),
                operation_signature_groups=partition['groups'],
                reason='all ranks reported completion with inconsistent operation signatures',
                location_latency_ms=(time.perf_counter_ns() - started) / 1e6)
    if partition['status'] == 'minority':
        item['signature_majority_ranks'] = partition['majority_ranks']
        item['signature_heuristic'] = 'strict_majority_assumed_correct'
    return item


def slow_decision(comm, op, states, label, window, members, cid, base, baseline, detections, cfg):
    durs = {r: states[r]['duration_ns'] / 1e6 for r in members}
    tmax, tmin = max(durs.values()), min(durs.values())
    ratio = (tmax-base)/base
    if ratio <= cfg.slow_threshold:
        return None
    p = (tmax-tmin)/(tmax-base)
    comp = min_candidates(durs)
    entry_gap_ms = None
    uniform_tree_slow = False
    if label == 'tree-chain':
        if p > cfg.beta and all(type(states[r].get('start_mono_ns')) is int for r in members):
            starts = sorted((states[r]['start_mono_ns'], r) for r in members)
            latest, late_rank = starts[-1]
            gap_ms = (latest - starts[-2][0]) / 1e6
            if gap_ms > base and comp == [late_rank] and all(states[r]['mono_ns'] >= latest for r in members):
                entry_gap_ms = gap_ms
        uniform_tree_slow = (entry_gap_ms is None and cfg.baseline_scope == 'signature' and
                             baseline == 'dynamic' and tmin > base * (1 + cfg.slow_threshold))
        if entry_gap_ms is None and not uniform_tree_slow:
            return None
    detection_key = f'{label}:uniform' if uniform_tree_slow else label
    detections[detection_key] += 1
    item = result_common(comm, op, 'slow')
    item.update(group=label, window=int(window), t_base_ms=base, t_max_ms=tmax, t_min_ms=tmin,
                slowdown_ratio=ratio, detections=detections[detection_key], baseline=baseline)
    if detections[detection_key] < cfg.repeats:
        item.update(kind='detected', roots=[], reason='waiting for repetition threshold')
        return item
    if label in ('tree-wait', 'tree-mixed', 'ring-mixed', 'ring-wait'):
        item.update(kind='S2' if label == 'ring-wait' else 'undetermined', roots=[],
                    compute_roots=[], communication_roots=[], baseline_metric='fastest_rank')
        return item
    locate_start = time.perf_counter_ns()
    if uniform_tree_slow:
        item.update(kind='undetermined', roots=[], p=p, compute_roots=[], communication_roots=[],
                    reason='all Tree ranks are slow and no peer layer is available for attribution',
                    location_latency_ms=(time.perf_counter_ns() - locate_start) / 1e6)
        return item
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


PROFILE_FIELDS = ('op', 'count', 'datatype', 'algorithm', 'protocol', 'n_channels', 'reduction_op')


def operation_profile(event):
    count = event.get('count')
    channels = event.get('n_channels')
    if channels is None:
        channels = len(event.get('channels') or ())
    names = tuple(event.get(field) for field in ('op', 'datatype', 'algorithm', 'protocol'))
    if (type(count) is not int or count < 0 or type(channels) is not int or channels < 1 or
            any(not isinstance(name, str) or not name.strip() for name in names)):
        return None
    op, datatype, algorithm, protocol = (name.strip().lower() for name in names)
    reduction = str(event.get('reduction_op', '')).strip().lower()
    return op, count, datatype, algorithm, protocol, channels, reduction


class SlowProfile:
    def __init__(self, cfg):
        self.skip = cfg.baseline_skip
        self.first = self.ready = None
        self.count = 0
        self.total_ms = 0.
        self.fastest_total_ms = 0.
        self.winners = {}
        self.detections = defaultdict(int)
        self.ring_wait = RingWaitAttribution(cfg.ring_wait_gap, cfg.slow_threshold, cfg.ring_wait_phases, cfg.alpha)
        self.tree_wait = {}
        self.admission = None
        self.ring_mixed = None

    def observe(self, when, states, cfg, collect_fastest=False):
        if self.skip:
            self.skip -= 1
        elif self.ready is None:
            if self.first is None:
                self.first = when
            cutoff = self.first + int(cfg.baseline_s * 1e9)
            if when <= cutoff:
                self.count += 1
                self.total_ms += max(e['duration_ns'] for e in states.values()) / 1e6
                if collect_fastest:
                    self.fastest_total_ms += min(e['duration_ns'] for e in states.values()) / 1e6
            if self.count == cfg.baseline_rounds or when >= cutoff:
                self.ready = min(when, cutoff)
        if self.total_ms and self.ready is not None and when > self.ready:
            return self.total_ms / self.count, 'dynamic'
        return cfg.baseline_ms, 'initial'


class SlowWindows:
    """Use rank 0's clock to select a worst round for each comparable group."""

    MAX_PROFILES = 128

    def __init__(self, comm, cfg, emit):
        self.comm, self.cfg, self.emit = comm, cfg, emit
        self.first = None
        self.window = 0
        self.window_ns = max(1, int(cfg.window_s * 1e9))
        self.warmup_ops = cfg.slow_warmup_ops
        self.profiles = OrderedDict()
        self.previous_tree = None

    def add(self, op, states, world, rank_hosts=None, snapshots=None, clocks=None):
        previous_ends = None
        if states[0].get('tree_wait_probe') is True:
            if self.previous_tree is not None and self.previous_tree[0] == op - 1:
                previous_ends = self.previous_tree[1]
            self.previous_tree = op, {rank: event['mono_ns'] for rank, event in states.items()}
        if self.warmup_ops:
            self.warmup_ops -= 1
            return
        when = states[0]['mono_ns']
        if self.first is None:
            self.first = when
        window = (when - self.first) // self.window_ns
        if window < self.window:
            raise ValueError('a rank-0 completion arrived after its slow window closed')
        if window > self.window:
            self.flush()
            self.window = window
        key = operation_profile(states[0]) if self.cfg.baseline_scope == 'signature' else ()
        if key is None:
            return
        profile = self.profiles.get(key)
        if profile is None:
            if len(self.profiles) == self.MAX_PROFILES:
                self.profiles.popitem(last=False)
            profile = self.profiles[key] = SlowProfile(self.cfg)
        else:
            self.profiles.move_to_end(key)
        tree_probe = key and states[0].get('tree_wait_probe') is True
        ring_probe = key and self.cfg.ring_wait_phases and states[0].get('wait_probe') is True
        hosts = {rank_hosts.get(rank) for rank in states} if rank_hosts and (tree_probe or ring_probe) else set()
        cross_host = len(hosts) > 1 and all(isinstance(host, str) and host for host in hosts)
        track_admission = bool(tree_probe or (ring_probe and cross_host))
        base, source = profile.observe(when, states, self.cfg, track_admission)
        admission = None
        if track_admission:
            if profile.admission is None:
                profile.admission = RankAdmission()
            admission = profile.admission.observe(op, states, source == 'dynamic')
        if key and source == 'initial':
            return
        phased_tree = tree_probe and cross_host and self.cfg.tree_cross_host_phases
        groups, warnings = ([], []) if phased_tree else slow_groups(states, world, rank_hosts)
        tree_base = ring_base = None
        tree_bounds = None
        if tree_probe and source == 'dynamic' and profile.fastest_total_ms:
            fastest = profile.fastest_total_ms / profile.count
            if uniform_slowdown(states, rank_hosts or {}, fastest, self.cfg.slow_threshold,
                               previous_ends, admission):
                tree_base = fastest
                groups = [group for group in groups if group[0] != 'tree-chain']
                groups.append(('tree-wait', sorted(states), None))
            elif admission is not None and admission.confirmed_rank is not None and (snapshots or clocks is not None):
                bounds = tree_phase(states, snapshots, rank_hosts or {}, admission.confirmed_rank, clocks)
                if bounds is not None:
                    tree_bounds = bounds
                    tree_base = fastest
                    groups.append(('tree-mixed', sorted(states), None))
        if (ring_probe and cross_host and source == 'dynamic' and profile.fastest_total_ms
                and admission is not None):
            fastest = profile.fastest_total_ms / profile.count
            all_slow = min(event['duration_ns'] for event in states.values()) > fastest * (1 + self.cfg.slow_threshold) * 1e6
            if all_slow:
                if admission.confirmed_rank is not None:
                    ring_base = fastest
                    groups = [('ring-mixed', sorted(states), None)]
                elif admission.delayed_rank is None and admission.spread_ns <= max(3_000_000, fastest * 1e6):
                    ring_base = fastest
                    groups = [('ring-wait', sorted(states), None)]
        if not groups and not phased_tree:
            item = result_common(self.comm, op, 'coverage')
            item.update(kind='undetermined', roots=[], reason='slow comparison unavailable', warnings=warnings)
            self.emit(item)
        for label, members, channel in groups:
            if len(members) < 2:
                continue
            durations = [states[r]['duration_ns'] for r in members]
            if label == 'tree-mixed' and cross_host:
                score = (tree_bounds.limits is not None, tree_bounds.confirmed_admission,
                         tree_bounds.tail_ns, -op)
            else:
                score = (min(durations) if label in ('tree-wait', 'ring-mixed', 'ring-wait') else
                         max(durations) - min(durations)), -op
            if label not in profile.winners or score > profile.winners[label][0]:
                bounds = phase_boundaries(states, snapshots, rank_hosts or {}) if label == 'ring' and snapshots else None
                if label == 'tree-mixed':
                    bounds = tree_bounds
                group_base = tree_base if label in ('tree-wait', 'tree-mixed') else base
                if label in ('ring-mixed', 'ring-wait'):
                    group_base = ring_base
                profile.winners[label] = score, op, states, members, channel, group_base, source, bounds, admission

    def decisions(self):
        for key, profile in sorted(self.profiles.items()):
            for label, (_, op, states, members, channel, base, source, bounds, admission) in sorted(profile.winners.items()):
                item = slow_decision(self.comm, op, states, label, self.window, members, channel,
                                     base, source, profile.detections, self.cfg)
                if label in ('ring', 'ring-wait'):
                    item = profile.ring_wait.refine(item, states, op, self.window, bounds)
                elif label == 'ring-mixed' and item:
                    if profile.ring_mixed is None:
                        profile.ring_mixed = RankMixedWait(self.cfg.ring_wait_gap)
                    item = profile.ring_mixed.refine(item, states, op, self.window, admission.confirmed_rank)
                elif label in ('tree-wait', 'tree-mixed') and item:
                    if label not in profile.tree_wait:
                        profile.tree_wait[label] = TreeWaitAttribution(self.cfg.tree_wait_gap)
                    item = profile.tree_wait[label].refine(item, states, op, self.window, bounds)
                if item:
                    if key:
                        item['operation_profile'] = dict(zip(PROFILE_FIELDS, key))
                    yield key, label, states, item
            profile.winners.clear()

    def flush(self):
        for _, _, _, item in self.decisions():
            self.emit(item)


def analyze_comm(comm: str, events: list[dict], cfg: Config, *, tree_references=()) -> list[dict]:
    sizes = {e['world_size'] for e in events}
    if len(sizes) != 1:
        return [{'comm_hash': comm, 'category': 'invalid', 'reason': 'inconsistent world_size'}]
    world = sizes.pop()
    by_op: dict[int, list[dict]] = defaultdict(list)
    dropped: dict[int, list[dict]] = defaultdict(list)
    # Include init and heartbeat to distinguish an observed but lagging rank.
    history = defaultdict(list)
    rank_hosts = {}
    for e in events:
        history[e['rank']].append(e)
        if e['event'] == 'comm_init':
            rank_hosts[e['rank']] = e.get('host')
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
    tree_references = deque(tree_references, maxlen=4)
    for op in sorted(by_op.keys() | dropped.keys()):
        records = by_op.get(op, [])
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
            reason = recorded_exclusion(records, dropped[op], world)
            if reason:
                output.append(exclusion_report(comm, op, reason))
                continue
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
            mismatch = completed_mismatch(comm, op, states, world)
            if mismatch:
                output.append(mismatch)
                continue
            # Rank 0's local completion clock defines window boundaries.
            completed.append((op, states, states[0]['mono_ns']))
            if str(states[0].get('algorithm', '')).lower() == 'tree':
                tree_references.append(states)
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
            fields = signature_fields(states)
            signatures = {r: [e.get(x) for x in fields] for r, e in states.items()}
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
                if str(states[0].get('algorithm', '')).lower() == 'tree':
                    hang.update(tree_hang(states, tree_references, cfg.hang_s * 1e9))
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

    slow = SlowWindows(comm, cfg, output.append)
    for op, states, _ in sorted(completed, key=lambda x: x[2]):
        snapshots = None
        if cfg.ring_wait_phases and (states[0].get('wait_probe') is True or states[0].get('tree_wait_probe') is True):
            snapshots = defaultdict(list)
            for event in by_op[op]:
                if event['event'] == 'snapshot' and (event.get('wait_probe') is True or event.get('tree_wait_probe') is True):
                    snapshots[event['rank']].append(event)
        slow.add(op, states, world, rank_hosts, snapshots)
    if cfg.flush_last_window:
        slow.flush()
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
    p.add_argument('--slow-warmup-ops', type=int, default=0,
                   help='skip these initial completed operations per communicator for slow analysis')
    p.add_argument('--baseline-s', type=float, default=120)
    p.add_argument('--baseline-scope', choices=('communicator', 'signature'), default='communicator',
                   help='learn independently by operation signature and wait for each baseline')
    p.add_argument('--rate-metric', choices=('paper', 'delta'), default='paper')
    p.add_argument('--ring-wait-gap', type=float, default=0.2,
                   help='minimum normalized directional wait score gap')
    p.add_argument('--ring-wait-phases', action='store_true',
                   help='require same-host post-entry wait snapshots for mixed Ring slowdown')
    p.add_argument('--tree-wait-gap', type=float, default=0.2,
                   help='minimum relative difference between the top Tree wait scores')


def decision_config(p, args, flush_last_window=False):
    if (any(not math.isfinite(x) or x <= 0 for x in
            (args.hang_s, args.window_s, args.baseline_ms, args.baseline_s, args.ring_wait_gap, args.tree_wait_gap)) or
            not math.isfinite(args.slow_threshold) or args.slow_threshold < 0 or
            args.repeats < 1 or args.baseline_rounds < 1 or args.baseline_skip < 0 or args.slow_warmup_ops < 0 or
            not (0 <= args.alpha < args.beta <= 1)):
        p.error('durations, baseline and counts must be positive; require 0 <= alpha < beta <= 1')
    return Config(args.hang_s, args.window_s, args.baseline_ms, args.slow_threshold,
                  args.repeats, args.alpha, args.beta, args.baseline_rounds, args.baseline_s,
                  args.rate_metric, flush_last_window, args.baseline_skip, args.ring_wait_gap,
                  args.slow_warmup_ops, args.baseline_scope, args.ring_wait_phases, args.tree_wait_gap,
                  getattr(args, 'tree_cross_host_phases', False))


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
