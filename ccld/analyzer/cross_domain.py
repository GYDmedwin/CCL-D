"""Raise cross-domain H1 explanations backed by observed wait scopes."""
from collections import defaultdict


def upstream_frontier(operation, predecessors):
    frontier, visited = set(), {operation}
    remaining = list(predecessors[operation])
    while remaining:
        other = remaining.pop()
        if other in visited:
            continue
        visited.add(other)
        following = predecessors.get(other)
        if following:
            remaining.extend(following)
        else:
            frontier.add(other)
    return sorted(frontier)


class CrossDomainWaits:
    def __init__(self):
        self.active = {}
        self.reported = {}

    def add(self, event):
        key = (event['comm_hash'], event['op_count'], event['rank'], event['wait_id'])
        if event['active']:
            self.active[key] = (event['producer_comm'], event['producer_op'], event['producer_rank'])
        else:
            self.active.pop(key, None)

    def evaluate(self, comms, emit):
        if not self.active and not self.reported:
            return
        covered = {key: comm for key, comm in comms.items()
                   if not comm.gap and not comm.ended and len(comm.observed) == comm.world}
        links = defaultdict(set)
        predecessors = defaultdict(set)
        for (comm, op, rank, _), (upstream, up_op, up_rank) in self.active.items():
            current, preceding = covered.get(comm), covered.get(upstream)
            if current is None or preceding is None or current.next_op != op or not current.last_hang:
                continue
            alert_op, kind, missing = current.last_hang
            if (alert_op != op or kind != 'H1' or rank not in missing
                    or current.observed.get(rank, op) >= op
                    or rank in current.pending.get(op, {})):
                continue
            state = preceding.pending.get(up_op, {}).get(up_rank)
            if state is None or state['event'] not in ('entered', 'snapshot'):
                continue
            links[(comm, op)].add((rank, upstream, up_op, up_rank))
            predecessors[(comm, op)].add((upstream, up_op))

        results = {}
        for (comm, op), edges in sorted(links.items()):
            missing = set(covered[comm].last_hang[2])
            unresolved = missing - {rank for rank, _, _, _ in edges}
            item = {
                'category': 'cross_domain', 'kind': 'witnessed_wait', 'comm_hash': comm, 'op_count': op,
                'links': [{'waiter_rank': rank, 'comm_hash': upstream, 'op_count': up_op, 'rank': up_rank}
                          for rank, upstream, up_op, up_rank in sorted(edges)],
                'upstream_candidates': [{'comm_hash': upstream, 'op_count': up_op}
                                        for upstream, up_op in upstream_frontier((comm, op), predecessors)],
                'unresolved_ranks': sorted(unresolved),
            }
            if self.reported.get((comm, op)) != item:
                emit(item)
            results[(comm, op)] = item
        for comm, op in sorted(self.reported.keys() - results.keys()):
            emit({'category': 'cross_domain', 'kind': 'attribution_cleared', 'comm_hash': comm, 'op_count': op})
        self.reported = results
