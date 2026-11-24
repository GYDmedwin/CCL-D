"""Rank-local host intervals for overlapping communication diagnoses."""
from __future__ import annotations

from collections import deque
from statistics import median
from typing import NamedTuple


class AdmissionEvidence(NamedTuple):
    spread_ns: float
    delayed_rank: int | None
    confirmed_rank: int | None


class RankAdmission:
    """Compare consecutive admission intervals with their own rank baselines."""
    MIN_BASELINE = 3
    BASELINE_LIMIT = 8
    MIN_EXCESS_NS = 3_000_000
    MIN_SEPARATION_NS = 1_000_000

    def __init__(self):
        self.previous = None
        self.samples = []
        self.baseline = None
        self.votes = deque(maxlen=5)

    def observe(self, operation, states, baseline_ready):
        if len(states) < 2:
            return None
        previous = self.previous
        self.previous = operation, {rank: event['mono_ns'] for rank, event in states.items()}
        if previous is None or previous[0] != operation - 1 or previous[1].keys() != states.keys():
            self.votes.clear()
            return None
        gaps = {}
        for rank, event in states.items():
            start = event.get('start_mono_ns')
            if type(start) is not int or start < previous[1][rank]:
                self.votes.clear()
                return None
            gaps[rank] = start - previous[1][rank]
        if not baseline_ready:
            if len(self.samples) < self.BASELINE_LIMIT:
                self.samples.append(gaps)
            return None
        if self.baseline is None:
            if len(self.samples) < self.MIN_BASELINE:
                return None
            self.baseline = {rank: median(row[rank] for row in self.samples) for rank in gaps}
            self.samples.clear()
        excess = sorted(((gap - self.baseline[rank], rank) for rank, gap in gaps.items()), reverse=True)
        (largest, rank), (second, _) = excess[:2]
        delayed = rank if largest >= self.MIN_EXCESS_NS and largest - max(second, 0) >= self.MIN_SEPARATION_NS else None
        self.votes.append(delayed)
        confirmed = delayed if delayed is not None and self.votes.count(delayed) >= 3 else None
        return AdmissionEvidence(largest - excess[-1][0], delayed, confirmed)
