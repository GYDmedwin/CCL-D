#!/usr/bin/env python3
"""Detect rank signature mismatches when RCCL completes every rank's operation.

The NV base analyzer reports an operation as resolved when all ranks complete.
This companion check preserves that outcome and emits a separate H2 mismatch
finding with ``hang_observed=false`` when completed ranks have inconsistent
operation signatures. It never changes the NV analyzer's hang classification.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

SIGNATURE_FIELDS = ("op", "count", "datatype", "algorithm", "protocol")


def load_complete(paths: list[Path]) -> dict[tuple[str, int], dict[int, dict]]:
    grouped: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for path in paths:
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            for line in file.read_text().splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event") != "complete":
                    continue
                comm = event.get("comm_hash")
                op_count = event.get("op_count")
                rank = event.get("rank")
                if isinstance(comm, str) and type(op_count) is int and type(rank) is int:
                    key = (comm, op_count)
                    if rank in grouped[key]:
                        raise ValueError(f"duplicate completion for comm={comm} op={op_count} rank={rank}")
                    grouped[key][rank] = event
    return grouped


def findings(paths: list[Path], world_size: int) -> list[dict]:
    output = []
    for (comm, op_count), ranks in sorted(load_complete(paths).items()):
        if set(ranks) != set(range(world_size)):
            continue
        signatures = {
            rank: tuple(event.get(field) for field in SIGNATURE_FIELDS)
            for rank, event in ranks.items()
        }
        groups: dict[tuple, list[int]] = defaultdict(list)
        for rank, signature in signatures.items():
            groups[signature].append(rank)
        if len(groups) <= 1:
            continue
        ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[1]))
        majority_signature, majority_ranks = ordered[0]
        minority_ranks = sorted(rank for _, members in ordered[1:] for rank in members)
        unique_majority = len(majority_ranks) > world_size / 2
        roots = minority_ranks if unique_majority else []
        mismatch_events = []
        reference = ranks[majority_ranks[0]]
        for rank in minority_ranks:
            event = ranks[rank]
            mismatch_events.append({
                "rank": rank,
                "signature": dict(zip(SIGNATURE_FIELDS, signatures[rank])),
                "count_delta_bytes_from_majority": (
                    event.get("bytes") - reference.get("bytes")
                    if type(event.get("bytes")) is int and type(reference.get("bytes")) is int
                    else None
                ),
            })
        output.append({
            "category": "collective_mismatch",
            "kind": "H2",
            "status": "detected" if roots else "ambiguous",
            "detected": True,
            "hang_observed": False,
            "comm_hash": comm,
            "op_count": op_count,
            "world_size": world_size,
            "roots": roots,
            "majority_ranks": sorted(majority_ranks),
            "majority_signature": dict(zip(SIGNATURE_FIELDS, majority_signature)),
            "operation_signatures": {str(rank): dict(zip(SIGNATURE_FIELDS, signature))
                                      for rank, signature in sorted(signatures.items())},
            "mismatches": mismatch_events,
            "reason": "completed rank operation signatures differ",
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="decoded JSONL files or directories")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output", type=Path, help="write findings to this JSONL file")
    parser.add_argument("--expect-rank", type=int, help="require a finding for this rank")
    parser.add_argument("--expect-op-count", type=int, help="check the expected rank at this operation")
    args = parser.parse_args()
    if args.world_size < 2:
        parser.error("world-size must be at least 2")
    if args.expect_rank is not None and not 0 <= args.expect_rank < args.world_size:
        parser.error("expected rank must be within the communicator")
    if args.expect_op_count is not None and (args.expect_rank is None or args.expect_op_count < 1):
        parser.error("expected operation requires an expected rank and a positive count")
    try:
        rows = findings(args.paths, args.world_size)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"completed mismatch scan failed: {exc}", file=sys.stderr)
        return 2
    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    print(json.dumps({"complete_operations_checked": "all full rank sets",
                      "mismatches_detected": len(rows),
                      "output": str(args.output) if args.output else "stdout"}), file=sys.stderr)
    if args.expect_rank is not None and not any(
        row["roots"] == [args.expect_rank]
        and (args.expect_op_count is None or row["op_count"] == args.expect_op_count)
        for row in rows
    ):
        print("missing expected completed-operation mismatch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
