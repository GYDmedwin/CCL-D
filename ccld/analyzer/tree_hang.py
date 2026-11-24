"""Compare Tree progress with a completed round of the same workload."""
from fractions import Fraction


def _profile(event):
    return (tuple(str(event.get(k, '')).lower() for k in
                  ('algorithm', 'protocol', 'op', 'datatype', 'reduction_op')),
            event.get('count'))


def _progress(event):
    channels = {}
    for channel in event.get('channels', ()):
        cid, parent = channel.get('id'), channel.get('tree_parent')
        sent, received = channel.get('send_count'), channel.get('recv_count')
        if (type(cid) is not int or type(parent) is not int or cid in channels or
                type(sent) is not int or type(received) is not int or min(sent, received) < 0):
            return {}
        channels[cid] = parent, sent + received
    return channels


def tree_hang(states, references, threshold_ns):
    """Use rank-local normalized progress and retain ambiguity on ties."""
    result = dict(kind='undetermined', roots=[], method='tree_normalized_progress')
    if any(event.get('duration_ns', 0) < threshold_ns for event in states.values()):
        return dict(result, reason='waiting for a post-threshold snapshot on each Tree rank')

    current = {rank: _progress(event) for rank, event in states.items()}
    for reference in reversed(references):
        if reference.keys() != states.keys() or any(
                _profile(reference[r]) != _profile(event) for r, event in states.items()):
            continue
        scores = {}
        for rank, channels in current.items():
            prior = _progress(reference[rank])
            if (not prior or prior.keys() != channels.keys() or
                    any(prior[cid][0] != parent or amount > prior[cid][1]
                        for cid, (parent, amount) in channels.items())):
                break
            total = sum(amount for _, amount in prior.values())
            if total <= 0:
                break
            scores[rank] = Fraction(sum(amount for _, amount in channels.values()), total)
        if scores.keys() != states.keys():
            continue
        least = min(scores.values())
        candidates = sorted(r for r, score in scores.items() if score == least)
        result.update(reference_op_count=reference[0]['op_count'], progress_fraction={
            rank: float(fraction) for rank, fraction in scores.items()
        })
        if len(candidates) == 1:
            return dict(result, kind='H3', roots=candidates,
                        reason='unique minimum progress relative to a matching completed Tree round')
        return dict(result, candidate_ranks=candidates,
                    reason='multiple Tree ranks have the same minimum normalized progress')
    return dict(result, reason='no comparable completed Tree reference')
