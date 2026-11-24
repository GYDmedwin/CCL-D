"""Clock translation bounds for phase comparisons across training nodes."""
from collections import deque


class ClockBounds:
    MAX_AGE_NS = 2_000_000_000
    # Allow for relative monotonic-clock drift, including ordinary clock slewing.
    DRIFT_DIVISOR = 1000

    def __init__(self):
        self.samples = {}
        self.endpoints = {}

    def add(self, endpoint, sample):
        binding = self.endpoints.setdefault(sample.host, (endpoint, sample.epoch))
        if binding != (endpoint, sample.epoch):
            # Two agents claiming the same host, or an epoch change, cannot
            # provide an unambiguous clock for buffered records.
            self.samples.pop(sample.host, None)
            self.endpoints[sample.host] = None
            return
        self.samples.setdefault(sample.host, deque(maxlen=16)).append(sample)

    def offset(self, host, start, end):
        """Bound the clock offset throughout one round on a given host."""
        candidates = []
        for sample in self.samples.get(host, ()):
            age = max(abs(start - sample.agent_ns), abs(end - sample.agent_ns))
            if age <= self.MAX_AGE_NS:
                pad = (age + self.DRIFT_DIVISOR - 1) // self.DRIFT_DIVISOR
                candidates.append((sample.lower_ns - pad, sample.upper_ns + pad))
        if candidates:
            lower, upper = max(lo for lo, _ in candidates), min(hi for _, hi in candidates)
            if lower <= upper:
                return lower, upper
        return None


def rank_clock_shifts(states, hosts, anchor, clocks):
    """Bound each clock's offset relative to the anchor rank's clock."""
    if len(states) < 3 or anchor not in states:
        return None
    if any(type(e.get('start_mono_ns')) is not int or e['mono_ns'] < e['start_mono_ns']
           or not hosts.get(r) for r, e in states.items()):
        return None
    by_host = {}
    for rank, event in states.items():
        host = hosts[rank]
        start, end = event['start_mono_ns'], event['mono_ns']
        if host in by_host:
            old_start, old_end = by_host[host]
            start, end = min(start, old_start), max(end, old_end)
        by_host[host] = (start, end)
    offsets = {host: clocks.offset(host, start, end) for host, (start, end) in by_host.items()}
    if any(offset is None for offset in offsets.values()):
        return None

    anchor_host = hosts[anchor]
    anchor_lo, anchor_hi = offsets[anchor_host]
    shifts = {}
    for rank in states:
        if hosts[rank] == anchor_host:
            shifts[rank] = (0, 0)
        else:
            lo, hi = offsets[hosts[rank]]
            shifts[rank] = (lo - anchor_hi, hi - anchor_lo)
    return shifts


def bracket_peer_waits(states, snapshots, shifts, ready, window_ns):
    """Bracket the all-rank admission interval with sample times outside it."""
    ready_lo, ready_hi = ready
    samples = {}
    for rank, final in states.items():
        shift_lo, shift_hi = shifts[rank]
        start = final['start_mono_ns']
        if start + shift_lo == ready_hi:
            # This rank is definitely last and its counters start at zero.
            continue
        before, after = None, None
        for event in (*snapshots.get(rank, ()), final):
            if event.get('start_mono_ns') != start:
                continue
            when = event['mono_ns']
            lo, hi = when + shift_lo, when + shift_hi
            if ready_hi - window_ns <= lo and hi <= ready_lo and (before is None or when > before['mono_ns']):
                before = event
            if ready_hi <= lo and hi <= ready_lo + window_ns and (after is None or when < after['mono_ns']):
                after = event
        # Zero counters at admission are a valid earlier bound for a rank
        # whose entry might itself lie inside the common ready interval.
        if after is None or (before is None and start + shift_lo < ready_hi - window_ns):
            return None
        samples[rank] = before, after
    return samples
