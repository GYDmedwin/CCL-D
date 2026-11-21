#!/usr/bin/env python3
"""Summarize dual-node S2 cycle scans using the NV Ring-wait score function.

The script reads decoded RCCL traces and never changes the NV analyzer rules.
It reports per-run winner votes, top-gap quantiles, and target-rank durations so
that a threshold can be selected from paired control/fault data.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/workspace/nv/CCL-D/ccld/analyzer")
from ring_wait import graph_wait_scores  # noqa: E402


def load(path: Path, first: int, rounds: int) -> dict[int, dict[int, dict]]:
    last = first + rounds - 1
    states: dict[int, dict[int, dict]] = {}
    for file in sorted(path.rglob("*.jsonl")):
        if "jsonl-final" in file.parts:
            continue
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "complete":
                continue
            op = event.get("op_count", -1)
            if first <= op <= last:
                states.setdefault(op, {})[event["rank"]] = event
    return states


def summarize(path: Path, target: int, first: int, rounds: int) -> dict:
    states = load(path, first, rounds)
    winners: list[int] = []
    gaps: list[float] = []
    target_durations: list[float] = []
    complete_ops = []
    for op in range(first, first + rounds):
        group = states.get(op, {})
        if set(group) != set(range(8)):
            continue
        metric = graph_wait_scores(group)
        complete_ops.append(op)
        winners.extend(metric["forced_top_ranks"])
        gaps.append(float(metric["top_gap"]))
        target_durations.append(group[target]["duration_ns"] / 1e6)
    if not gaps:
        raise ValueError(f"{path}: no complete 8-rank operations in requested range")
    manifests = sorted(path.rglob("fault-config.json"))
    manifest = json.loads(manifests[0].read_text()) if manifests else {}
    return {
        "run": str(path),
        "fault_cycles": manifest.get("fault_cycles"),
        "target_rank": target,
        "first_op": first,
        "rounds_requested": rounds,
        "complete_ops": complete_ops,
        "winner_votes": dict(sorted(Counter(winners).items())),
        "target_votes": winners.count(target),
        "gap_min": min(gaps),
        "gap_median": statistics.median(gaps),
        "gap_max": max(gaps),
        "target_duration_ms_median": statistics.median(target_durations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--first-op", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    try:
        for run in args.runs:
            print(json.dumps(summarize(run, args.target_rank, args.first_op, args.rounds), sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"S2 scan analysis failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
