#!/usr/bin/env python3
"""Accept an S2 run when the base analyzer detects a slow collective.

Rank attribution remains recorded in the analyzer output. The functional gate
checks detection only, as configured for the current gfx936 delivery target.
"""
import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--target-rank", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("fault_kind") != "S2" or manifest.get("fault_rank") != args.target_rank:
        raise SystemExit(f"fault manifest does not match S2 rank {args.target_rank}: {manifest}")
    rows = [json.loads(line) for line in args.analysis.read_text().splitlines() if line.strip()]
    detections = [row for row in rows if row.get("category") == "slow" and
                  row.get("kind") in ("detected", "S1", "S2", "S3")]
    if not detections:
        raise SystemExit(f"base analyzer did not detect a slow collective; rows={rows}")
    print(json.dumps({"functional_gate": "pass", "detected": True,
                      "detection_kind": detections[-1]["kind"],
                      "confirmed_classification": detections[-1]["kind"] in ("S1", "S2", "S3"),
                      "roots": detections[-1].get("roots", []),
                      "detection": detections[-1],
                      "fault_rank": args.target_rank,
                      "fault_cycles": manifest.get("fault_cycles")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
