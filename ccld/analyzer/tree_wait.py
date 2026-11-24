"""Use Tree edge waits to locate a repeated, same-host communication slowdown."""
from collections import defaultdict, deque
import time


def uniform_slowdown(states, hosts, base_ms, threshold):
    clock_hosts = {hosts.get(rank) for rank in states}
    if len(clock_hosts) != 1 or not all(isinstance(host, str) and host for host in clock_hosts):
        return False
    starts = [event.get('start_mono_ns') for event in states.values()]
    return (all(event.get('tree_wait_probe') is True for event in states.values()) and
            all(type(when) is int for when in starts) and max(starts) - min(starts) <= max(1_000_000, base_ms * 1e6) and
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


class TreeWaitAttribution:
    def __init__(self, gap):
        self.gap = gap
        self.rounds = deque(maxlen=3)
        self.last_window = None

    def refine(self, item, states, operation, window):
        if item['kind'] == 'detected':
            return item
        if (self.last_window is not None and window - self.last_window > 2 or
                self.rounds and self.rounds[-1][0] >= operation):
            self.rounds.clear()
        self.last_window = window
        started = time.perf_counter_ns()
        candidate, separation = None, None
        try:
            ordered = sorted(((score, rank) for rank, score in directional_scores(states).items()), reverse=True)
            top, rank = ordered[0]
            if top > 0:
                separation = (top - ordered[1][0]) / top
                if separation >= self.gap:
                    candidate = rank
        except ValueError:
            pass
        self.rounds.append((operation, candidate))
        item.update(roots=[], compute_roots=[], communication_roots=[], tree_wait_gap=separation)
        if len(self.rounds) < 3:
            item.update(kind='detected', reason='waiting for three directional Tree samples')
        elif candidate is not None and sum(rank == candidate for _, rank in self.rounds) >= 2:
            item.update(kind='S2', roots=[candidate], communication_roots=[candidate], method='tree_directional_wait',
                        location_latency_ms=(item.get('location_latency_ms') or 0) +
                        (time.perf_counter_ns() - started) / 1e6)
        else:
            item.update(kind='undetermined', reason='no repeated directional Tree root')
        return item
