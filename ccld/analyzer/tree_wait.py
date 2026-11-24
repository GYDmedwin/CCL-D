"""Locate repeated communication slowdown with directional Tree edge waits."""
from __future__ import annotations

from collections import defaultdict, deque
from statistics import median
from typing import NamedTuple
import time

if __package__:
    from .clock_bounds import bracket_peer_waits, rank_clock_shifts
    from .ring_wait import PHASE_ENTRY_GAP_NS, PHASE_WINDOW_NS, phase_boundaries
else:
    from clock_bounds import bracket_peer_waits, rank_clock_shifts
    from ring_wait import PHASE_ENTRY_GAP_NS, PHASE_WINDOW_NS, phase_boundaries


class TreePhase(NamedTuple):
    admission_rank: int
    limits: dict | None
    tail_ns: float
    uncertainty_ns: int | None = None
    confirmed_admission: bool = True


def tree_phase(states, snapshots, hosts, compute_rank, clocks=None):
    if len(set(hosts.values())) == 1:
        bounds = phase_boundaries(states, snapshots, hosts) if snapshots else None
        if bounds is None or bounds[0] != compute_rank:
            return None
        cutoff = states[compute_rank]['start_mono_ns']
        return TreePhase(compute_rank, bounds[1], median(e['mono_ns'] - cutoff for e in states.values()))
    if clocks is None:
        return None
    shifts = rank_clock_shifts(states, hosts, compute_rank, clocks)
    if shifts is None:
        return None
    cutoff = states[compute_rank]['start_mono_ns']
    entries = {r: (e['start_mono_ns'] + shifts[r][0], e['start_mono_ns'] + shifts[r][1])
               for r, e in states.items()}
    confirmed = all(hi <= cutoff - PHASE_ENTRY_GAP_NS for r, (_, hi) in entries.items() if r != compute_rank)
    ready = (max(lo for lo, _ in entries.values()), max(hi for _, hi in entries.values()))
    tails = [event['mono_ns'] - ready[1] + shifts[rank][0] for rank, event in states.items()]
    uncertainty = max(hi - lo for lo, hi in shifts.values())
    limits = None
    if snapshots and min(tails) >= 0:
        limits = bracket_peer_waits(states, snapshots, shifts, ready, PHASE_WINDOW_NS)
    return TreePhase(compute_rank, limits, median(tails), uncertainty, confirmed)


def uniform_slowdown(states, hosts, base_ms, threshold, previous_ends=None, admission=None):
    clock_hosts = {hosts.get(rank) for rank in states}
    if not all(isinstance(host, str) and host for host in clock_hosts):
        return False
    if len(clock_hosts) > 1 and admission is None:
        return False
    starts = {rank: event.get('start_mono_ns') for rank, event in states.items()}
    if (any(event.get('tree_wait_probe') is not True for event in states.values()) or
            any(type(when) is not int for when in starts.values())):
        return False
    timing = starts.values()
    if previous_ends is not None and previous_ends.keys() == starts.keys():
        # Different completion times can lead to different entries in the
        # next round even when every rank performs the same host work.
        intervals = [when - previous_ends[rank] for rank, when in starts.items()]
        if min(intervals) >= 0:
            timing = intervals
    spread = admission.spread_ns if admission is not None else max(timing) - min(timing)
    return (spread <= max(1_000_000, base_ms * 1e6) and
            min(event['duration_ns'] for event in states.values()) > base_ms * (1 + threshold) * 1e6)


def directional_scores(states):
    """Subtract inherited waits from independently observed Tree edges."""
    world = len(states)
    if world < 3 or set(states) != set(range(world)):
        raise ValueError('Tree wait attribution requires every rank')
    channels = {rank: {c.get('id'): c for c in event['channels']} for rank, event in states.items()}
    ids = set(channels[0])
    if (not ids or any(type(cid) is not int for cid in ids) or
            any(set(rows) != ids or len(rows) != len(states[r]['channels']) for r, rows in channels.items())):
        raise ValueError('Tree channel sets are missing or inconsistent')
    scores = dict.fromkeys(range(world), 0.0)
    for cid in sorted(ids):
        parents = {r: channels[r][cid].get('tree_parent') for r in range(world)}
        roots = [r for r, p in parents.items() if p == -1]
        if (len(roots) != 1 or any(type(p) is not int or p < -1 or p >= world or p == r
                                    for r, p in parents.items())):
            raise ValueError('invalid Tree parent topology')
        children = defaultdict(list)
        for rank, parent in parents.items():
            if parent >= 0:
                children[parent].append(rank)
        reachable = roots.copy()
        for rank in reachable:
            reachable.extend(children[rank])
        if len(reachable) != world:
            raise ValueError('disconnected or cyclic Tree edges')

        edges = {}
        for rank, parent in parents.items():
            expected = set(children[rank])
            if parent >= 0:
                expected.add(parent)
            for peer in channels[rank][cid].get('tree_wait_peers', ()):
                node, data, credit = (peer.get(key) for key in ('rank', 'data_spins', 'credit_spins'))
                if (type(node) is not int or node not in expected or
                        any(type(v) is not int or not 0 <= v < 2**32 - 1 for v in (data, credit))):
                    raise ValueError('missing, duplicate, or saturated Tree wait evidence')
                edges[rank, node] = data, credit
                expected.remove(node)
            if expected:
                raise ValueError('missing Tree peer wait evidence')

        remaining = {(r, p): max(0, edges[p, r][0] - credit) for (r, p), (_, credit) in edges.items()}
        for rank, parent in parents.items():
            incoming_up = max((remaining[child, rank] for child in children[rank]), default=0)
            if parent >= 0:
                scores[rank] += max(0, remaining[rank, parent] - incoming_up)
            if children[rank]:
                inherited_down = remaining[parent, rank] if parent >= 0 else incoming_up
                outgoing_down = sum(remaining[rank, child] for child in children[rank]) / len(children[rank])
                scores[rank] += max(0, outgoing_down - inherited_down)
    return scores


def preferred_rank(states, gap):
    ordered = sorted(((score, rank) for rank, score in directional_scores(states).items()), reverse=True)
    top, rank = ordered[0]
    if top <= 0:
        return None, None
    separation = (top - ordered[1][0]) / top
    return (rank if separation >= gap else None), separation


def wait_delta(final, earlier):
    channels = []
    prior = {channel['id']: channel for channel in earlier['channels']}
    for channel in final['channels']:
        old = prior.get(channel['id'])
        if old is None or old['tree_parent'] != channel['tree_parent']:
            raise ValueError('Tree topology changed between wait samples')
        peers = {row['rank']: row for row in old['tree_wait_peers']}
        rows = []
        for edge in channel['tree_wait_peers']:
            before = peers.pop(edge['rank'], None)
            if before is None:
                raise ValueError('Tree peer changed between wait samples')
            data, credit = (edge[key] - before[key] for key in ('data_spins', 'credit_spins'))
            if min(data, credit) < 0:
                raise ValueError('Tree wait counter regressed')
            rows.append(dict(rank=edge['rank'], data_spins=data, credit_spins=credit))
        if peers:
            raise ValueError('Tree peer coverage changed between wait samples')
        channels.append(dict(channel, tree_wait_peers=rows))
    return dict(final, channels=channels)


def phase_rank(states, boundaries, base_ms, gap):
    delayed, limits = boundaries.admission_rank, boundaries.limits
    if limits is None or boundaries.tail_ns <= max(2_000_000, 1.5 * base_ms * 1e6):
        return None, None
    votes = []
    for index in (0, 1):
        view = {}
        for rank, event in states.items():
            earlier = limits[rank][index] if rank in limits else None
            view[rank] = wait_delta(event, earlier) if earlier is not None else event
        votes.append(preferred_rank(view, gap))
    (early, a), (late_vote, b) = votes
    if early is None or early != late_vote or early == delayed:
        return None, None
    return early, min(a, b)


class TreeWaitAttribution:
    def __init__(self, gap):
        self.gap = gap
        self.rounds = deque(maxlen=5)
        self.last_window = None

    def refine(self, item, states, operation, window, boundaries=None):
        if item['kind'] == 'detected':
            return item
        if (self.last_window is not None and window - self.last_window > 2 or
                self.rounds and self.rounds[-1][1] >= operation):
            self.rounds.clear()
        self.last_window = window
        started = time.perf_counter_ns()
        compute = boundaries.admission_rank if boundaries is not None and boundaries.confirmed_admission else None
        candidate, separation = None, None
        try:
            if boundaries is None:
                candidate, separation = preferred_rank(states, self.gap)
            else:
                candidate, separation = phase_rank(states, boundaries, item['t_base_ms'], self.gap)
        except ValueError:
            pass
        self.rounds.append((window, operation, (compute, candidate)))
        item.update(roots=[], compute_roots=[], communication_roots=[], tree_wait_gap=separation)
        cross_host = boundaries is not None and boundaries.uncertainty_ns is not None
        if cross_host:
            item['clock_uncertainty_us'] = boundaries.uncertainty_ns / 1e3
            if compute is None:
                item['admission_candidates'] = [boundaries.admission_rank]
            recent = [vote[1] for when, _, vote in self.rounds if window - when <= 4 and vote[1] is not None]
            votes = recent.count(candidate)
            repeated = len(self.rounds) >= 3 and votes >= 2 and votes > len(recent) // 2
        else:
            recent = list(self.rounds)[-3:]
            repeated = len(recent) == 3 and sum(vote == (compute, candidate) for _, _, vote in recent) >= 2
        if candidate is not None and repeated:
            roots = [candidate] if compute is None else sorted([compute, candidate])
            item.update(kind='S2' if compute is None else 'S3', roots=roots,
                        compute_roots=[] if compute is None else [compute], communication_roots=[candidate],
                        method='tree_directional_wait' if boundaries is None else 'post_entry_tree_wait')
        elif cross_host and compute is not None:
            reason = ('no bounded wait samples at the last admission' if boundaries.limits is None else
                      'no repeated post-entry communication root')
            item.update(kind='S1', roots=[compute], compute_roots=[compute],
                        method='tree_rank_admission', reason=reason)
        elif len(self.rounds) < 3:
            item.update(kind='detected', reason='waiting for three directional Tree samples')
        else:
            item.update(kind='undetermined', reason='no repeated directional Tree root')
        if item['roots']:
            item['location_latency_ms'] = (item.get('location_latency_ms') or 0) + (
                time.perf_counter_ns() - started) / 1e6
        return item
