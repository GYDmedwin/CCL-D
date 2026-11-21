#!/usr/bin/env python3
"""Compare source-defined H3/S2 target rounds against an NV-workload control."""
import argparse
import json
from pathlib import Path
import statistics


def load(path, world):
    rows = []
    for file in sorted((path / "jsonl").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in file.read_text().splitlines() if line)
    completes = [row for row in rows if row["event"] == "complete"]
    assert len(completes) == world * 9, (path, len(completes), world * 9)
    by_op = {}
    for row in completes:
        by_op.setdefault(row["op_count"], {})[row["rank"]] = row["duration_ns"]
    assert set(by_op) == set(range(1, 10)), (path, by_op.keys())
    assert all(set(ranks) == set(range(world)) for ranks in by_op.values()), path
    return by_op


def rank_max_median(by_op, op):
    return statistics.median(by_op[op].values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("control", type=Path)
    p.add_argument("fault", type=Path)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--fault-op", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--fault-rank", type=int, default=0)
    p.add_argument("--kind", choices=("H3", "S2"), required=True)
    a = p.parse_args()
    control, fault = load(a.control, a.world_size), load(a.fault, a.world_size)
    target_ops = set(range(a.fault_op, a.fault_op + a.repeats))
    assert target_ops <= set(range(1, 10)), target_ops
    results = {}
    for op in sorted(target_ops):
        base = rank_max_median(control, op)
        injected = rank_max_median(fault, op)
        target = fault[op][a.fault_rank]
        assert injected > base, {"kind": a.kind, "op": op, "control_median_ns": base,
                                 "fault_median_ns": injected, "target_rank_ns": target}
        results[op] = {"control_rank_median_ns": base, "fault_rank_median_ns": injected,
                       "target_rank_duration_ns": target, "increase_ratio": injected / base}
    # Confirm complete numerical validation by the shared CCL-D workload logs.
    for path in (a.control, a.fault):
        logs = sorted((path / "workload").glob("rank-*.ndjson"))
        assert len(logs) == a.world_size, (path, len(logs))
        for log in logs:
            entries = [json.loads(line) for line in log.read_text().splitlines() if line]
            checks = [row for row in entries if "iteration" in row]
            assert len(checks) == 9 and all(row["verified"] for row in checks), log
    print(json.dumps({"kind": a.kind, "world_size": a.world_size, "target_rank": a.fault_rank,
                      "target_ops": results, "result": "bounded injected delay observed; all outputs verified"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
