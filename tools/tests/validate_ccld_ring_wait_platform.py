#!/usr/bin/env python3
"""Run NV CCL-D ring_wait scoring on platform traces without changing its rules."""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

NV_ANALYZER = Path(os.environ.get('CCLD_NV_ROOT', str(Path(__file__).resolve().parents[2]))) / 'ccld/analyzer'
sys.path.insert(0, str(NV_ANALYZER))
from ring_wait import decide, graph_wait_scores  # noqa: E402


def states(path):
    if (path / 'jsonl').is_dir():
        source = path / 'jsonl'
        files = sorted(source.glob('*.jsonl'))
    else:
        files = sorted(path.rglob('jsonl/*.jsonl'))
        if not files:
            files = sorted(path.glob('*.jsonl'))
    result = {}
    for file in files:
        for line in file.read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'complete' and event.get('op_count') in (4, 5, 6):
                result.setdefault(event['op_count'], {})[event['rank']] = event
    return result


def scores(path):
    data = states(path)
    return data, {op: graph_wait_scores(ranks) for op, ranks in sorted(data.items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', action='append', required=True, type=Path)
    p.add_argument('--holdout-control', action='append', required=True, type=Path)
    p.add_argument('--holdout-analysis', action='append', required=True, type=Path)
    p.add_argument('--fault', required=True, type=Path)
    p.add_argument('--fault-analysis', required=True, type=Path)
    p.add_argument('--world-size', required=True, type=int)
    p.add_argument('--target-rank', required=True, type=int)
    p.add_argument('--margin', default=1.0, type=float,
                   help='threshold multiplier over the largest training-control median')
    args = p.parse_args()
    if len(args.holdout_control) != len(args.holdout_analysis):
        p.error('each holdout control requires its NV analyzer output')
    train = []
    for path in args.control:
        data, metric = scores(path)
        assert all(set(ranks) == set(range(args.world_size)) for ranks in data.values()), path
        train.append(statistics.median(metric[op]['top_gap'] for op in (4, 5, 6)))
    threshold = max(train) * args.margin
    controls = []
    accepted_slow = {'S1', 'S2', 'S3', 'undetermined'}
    for path, analysis_path in zip(args.holdout_control, args.holdout_analysis):
        data, metric = scores(path)
        assert all(set(ranks) == set(range(args.world_size)) for ranks in data.values()), path
        analysis = [json.loads(line) for line in analysis_path.read_text().splitlines() if line]
        confirmed = [row for row in analysis if row.get('category') == 'slow' and
                     row.get('kind') in accepted_slow and row.get('op_count') in (4, 5, 6)]
        assert not confirmed, (path, confirmed)
        controls.append(decide(data, threshold, False))
    fault_data, metric = scores(args.fault)
    assert all(set(ranks) == set(range(args.world_size)) for ranks in fault_data.values()), args.fault
    fault_analysis = [json.loads(line) for line in args.fault_analysis.read_text().splitlines() if line]
    confirmed = [row for row in fault_analysis if row.get('category') == 'slow' and
                 row.get('kind') in accepted_slow and row.get('op_count') in (4, 5, 6)]
    assert confirmed, (args.fault, 'NV base analyzer did not confirm slow in the wait-scoring operations')
    result = decide(fault_data, threshold, True)
    assert result['candidate_ranks'] == [args.target_rank], result
    print(json.dumps({'threshold': threshold, 'training_control_medians': train,
                      'holdout_controls_gated': controls, 'confirmed_base_slow': confirmed, 'fault': result,
                      'fault_candidate_matches_target': True}, sort_keys=True))


if __name__ == '__main__':
    main()
