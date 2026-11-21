#!/usr/bin/env python3
"""Check a post-collective non-collective stall without inventing a root rank."""
import argparse
import json
from pathlib import Path
import time


def load_rows(path: Path):
    rows = []
    for f in sorted(path.rglob('*.jsonl')) if path.is_dir() else [path]:
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jsonl', type=Path)
    p.add_argument('--fault-events', type=Path, required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--threshold-s', type=float, default=1.0)
    p.add_argument('--now-ns', type=int, default=0, help='offline observation time; default uses CLOCK_MONOTONIC')
    a = p.parse_args()
    markers = []
    for f in sorted(a.fault_events.rglob('*.ndjson')) if a.fault_events.is_dir() else [a.fault_events]:
        for line in f.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                if x.get('rank') == a.rank and x.get('kind') == 'POST':
                    markers.append(x)
    rows = load_rows(a.jsonl)
    marker = max(markers, key=lambda x: x['mono_ns']) if markers else None
    if marker is None:
        raise SystemExit('POST marker not found')
    rank_rows = [x for x in rows if x.get('rank') == a.rank and x.get('event') in ('comm_init', 'entered', 'complete')]
    before = [x for x in rank_rows if x.get('event') == 'entered' and x.get('mono_ns', 0) <= marker['mono_ns']]
    after = [x for x in rank_rows if x.get('mono_ns', 0) > marker['mono_ns']]
    observation = a.now_ns or time.monotonic_ns()
    elapsed_ms = max(0.0, observation - marker['mono_ns']) / 1e6
    result = {
        'category': 'hang', 'kind': 'undetermined', 'roots': [],
        'reason': 'non_collective boundary stall; collective evidence cannot identify a root rank',
        'fault_marker': marker, 'last_entered_op': max((x.get('op_count', 0) for x in before), default=0),
        'post_marker_collective_events': len(after),
        'observed_elapsed_ms': elapsed_ms, 'threshold_s': a.threshold_s,
        'detected': elapsed_ms >= a.threshold_s * 1000 and not after,
    }
    print(json.dumps(result, sort_keys=True))
    if not result['detected']:
        raise SystemExit('POST boundary has not exceeded threshold or has later collective evidence')


if __name__ == '__main__':
    main()
