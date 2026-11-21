#!/usr/bin/env python3
"""Validate a CCL-D torch_collectives case using the NV decoder/analyzer."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--op", default="allreduce")
    p.add_argument("--algorithm", default="Ring")
    p.add_argument("--protocol", default="Simple")
    a = p.parse_args()
    traces = sorted((a.run / "bin").glob("*.ccldbin"))
    if len(traces) != a.world_size:
        raise AssertionError(f"expected {a.world_size} traces, found {len(traces)}")
    decoded = a.run / "jsonl"
    decoded.mkdir(parents=True, exist_ok=True)
    if len(list(decoded.glob("*.jsonl"))) != a.world_size:
        raise AssertionError(f"expected {a.world_size} decoded rank files")
    workload = sorted((a.run / "workload").glob("rank-*.ndjson"))
    if len(workload) != a.world_size:
        raise AssertionError(f"expected {a.world_size} workload logs, found {len(workload)}")
    seen = set()
    expected_events = a.warmup + a.iters
    for path in workload:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        start = next(row for row in rows if "runtime_nccl_version" in row)
        assert start["runtime_nccl_version"] == 22203, start
        assert any("librccl.so" in item for item in start["mapped_nccl"]), start
        rank = start["rank"]
        seen.add(rank)
        checks = [row for row in rows if "iteration" in row]
        assert len(checks) == expected_events, (path, len(checks), expected_events)
        assert all(row["verified"] for row in checks), path
    assert seen == set(range(a.world_size)), seen
    all_events = []
    for path in sorted(decoded.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        all_events.extend(rows)
        init = [row for row in rows if row["event"] == "comm_init"]
        entered = [row for row in rows if row["event"] == "entered"]
        complete = [row for row in rows if row["event"] == "complete"]
        assert len(init) == 1 and init[0]["world_size"] == a.world_size, (path, init)
        assert not [row for row in rows if row["event"] in ("dropped", "coverage_gap")], path
        # The NV results explicitly record AllToAll as numerical-only: the
        # grouped P2P path emitted no AllToAll operation records.
        if a.op == "alltoall":
            assert not entered and not complete, (path, entered, complete)
            continue
        assert len(entered) == expected_events, (path, len(entered))
        assert len(complete) == expected_events, (path, len(complete))
        for event in complete:
            assert event["op"] == {"allreduce": "AllReduce", "allgather": "AllGather",
                                    "reducescatter": "ReduceScatter", "alltoall": "AllToAll"}[a.op], event
            assert event["algorithm"] == a.algorithm and event["protocol"] == a.protocol, event
            channels = event["channels"][:event["n_channels"]]
            assert channels and all(c["send_count"] > 0 and c["recv_count"] > 0 for c in channels), event
            if a.algorithm == "Ring":
                if a.op == "allreduce":
                    steps = 2 * (a.world_size - 1)
                    assert all(c["send_count"] == steps and c["recv_count"] == steps for c in channels), event
                else:
                    assert all(c["send_count"] == c["recv_count"] for c in channels), event

    if a.algorithm == "Tree" and a.op != "alltoall":
        completed = [row for row in all_events if row["event"] == "complete"]
        by_op = {}
        for event in completed:
            by_op.setdefault(event["op_count"], {}).setdefault(event["rank"], event)
        for op_count, ranks in by_op.items():
            assert set(ranks) == set(range(a.world_size)), (op_count, ranks.keys())
            channel_ids = {c["id"] for c in ranks[0]["channels"][:ranks[0]["n_channels"]]}
            for channel_id in channel_ids:
                parents = {}
                for rank, event in ranks.items():
                    metrics = {c["id"]: c for c in event["channels"][:event["n_channels"]]}
                    assert set(metrics) == channel_ids, (op_count, rank, metrics.keys(), channel_ids)
                    parent = metrics[channel_id]["tree_parent"]
                    assert -1 <= parent < a.world_size and parent != rank, (op_count, channel_id, rank, parent)
                    parents[rank] = parent
                roots = [rank for rank, parent in parents.items() if parent == -1]
                assert len(roots) == 1, (op_count, channel_id, roots)
                for rank in range(a.world_size):
                    seen, current = set(), rank
                    while current != -1:
                        assert current not in seen, (op_count, channel_id, rank, parents)
                        seen.add(current)
                        current = parents[current]
    print(f"CCL-D trace/numerical case validated: world={a.world_size} op={a.op} algorithm={a.algorithm} protocol={a.protocol}")


if __name__ == "__main__":
    main()
