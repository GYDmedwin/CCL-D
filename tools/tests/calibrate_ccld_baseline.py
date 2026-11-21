#!/usr/bin/env python3
"""Calibrate an NV CCL-D baseline from a same-workload control trace."""
import argparse
import json
import math
from pathlib import Path


def durations(root: Path):
    paths = sorted(root.rglob('*.jsonl')) if root.is_dir() else [root]
    values = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('event') != 'complete':
                continue
            value = row.get('duration_ns')
            if type(value) is int and value > 0:
                values.append(value / 1e6)
    return sorted(values)


def quantile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError('duration sample or quantile is invalid')
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--quantile', type=float, default=0.05)
    parser.add_argument('--margin', type=float, default=0.95)
    args = parser.parse_args()
    if not math.isfinite(args.quantile) or not 0 <= args.quantile <= 1:
        parser.error('--quantile must be in [0,1]')
    if not math.isfinite(args.margin) or not 0 < args.margin <= 1:
        parser.error('--margin must be in (0,1]')
    values = durations(args.trace)
    if not values:
        parser.error('control trace has no complete duration samples')
    baseline = quantile(values, args.quantile) * args.margin
    print(json.dumps({'baseline_ms': baseline, 'quantile': args.quantile,
                      'margin': args.margin, 'samples': len(values),
                      'min_ms': values[0], 'max_ms': values[-1]}, sort_keys=True))


if __name__ == '__main__':
    main()
