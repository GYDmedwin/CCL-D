#!/usr/bin/env python3
"""Tree S1 candidate from complete per-channel group decisions.

This optional second stage is applied only after the base slow analyzer has
confirmed and classified a complete group window. It does not diagnose S2,
S3, or a hang, and it does not infer that absent evidence is a no-fault run.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

if __package__:
    from .ccld_analyze import comparable_groups, load_records
else:
    from ccld_analyze import comparable_groups, load_records

CLASSIFIED = {'S1', 'S2', 'S3', 'undetermined'}


def vote(states, decisions):
    """Vote only with S1 groups; require every eligible channel to agree."""
    if not states:
        raise ValueError('no rank states')
    world = states[0].get('world_size') if 0 in states else None
    if type(world) is not int or set(states) != set(range(world)):
        raise ValueError('all communicator ranks are required')
    if any(not isinstance(state.get('comm_hash'),str) or not state['comm_hash'] or
           type(state.get('op_count')) is not int or state['op_count'] < 0
           for state in states.values()):
        raise ValueError('invalid communicator or operation identifier')
    comms = {state['comm_hash'] for state in states.values()}
    ops = {state['op_count'] for state in states.values()}
    if (len(comms) != 1 or len(ops) != 1 or
            any(state.get('world_size') != world or state.get('rank') != rank or
                state.get('event') != 'complete' or str(state.get('algorithm','')).upper() != 'TREE'
                for rank,state in states.items())):
        raise ValueError('rank states do not describe one completed Tree operation')
    signatures = set()
    for state in states.values():
        op_name, count, dtype, algorithm, protocol = (state.get(k) for k in
                                                        ('op','count','datatype','algorithm','protocol'))
        if (type(count) is not int or count < 0 or
                any(not isinstance(value,str) or not value for value in (op_name,dtype,algorithm,protocol))):
            raise ValueError('operation signature is incomplete')
        signatures.add((op_name,count,dtype,algorithm,protocol))
    if len(signatures) != 1:
        raise ValueError('rank states do not describe one completed Tree operation')
    groups, errors = comparable_groups(states, world)
    if errors or not groups:
        raise ValueError('Tree topology is unavailable or incomplete')
    expected = {label:(members,cid) for label,members,cid in groups}
    if len(expected) != len(groups):
        raise ValueError('duplicate topology group')
    seen = {}
    windows = set()
    for decision in decisions:
        label = decision.get('group')
        if (decision.get('category') != 'slow' or decision.get('kind') not in CLASSIFIED or
                decision.get('comm_hash') not in comms or decision.get('op_count') not in ops or
                type(decision.get('window')) is not int or label not in expected or label in seen):
            raise ValueError('inconsistent, duplicate or unconfirmed group decision')
        members, _ = expected[label]
        roots = decision.get('roots')
        if (not isinstance(roots,list) or any(type(rank) is not int or rank not in members for rank in roots) or
                len(roots) != len(set(roots))):
            raise ValueError('group candidate is outside the comparable layer')
        windows.add(decision['window'])
        seen[label] = decision
    if len(windows) != 1 or set(seen) != set(expected):
        raise ValueError('incomplete or mixed decision window')
    eligible, votes = defaultdict(set), defaultdict(set)
    kinds = Counter()
    for label,(members,cid) in expected.items():
        decision = seen[label]
        kinds[decision['kind']] += 1
        for rank in members:
            eligible[rank].add(cid)
        if decision['kind'] == 'S1':
            for rank in decision['roots']:
                votes[rank].add(cid)
    scores = {str(rank):{'s1_votes':len(votes[rank]), 'eligible_channels':len(eligible[rank])}
              for rank in range(world)}
    if not kinds['S1']:
        return {'status':'not_applicable', 'roots':[], 'scores':scores, 'group_kinds':dict(kinds)}
    roots = [rank for rank in range(world) if len(eligible[rank]) >= 2 and votes[rank] == eligible[rank]]
    return {'status':'candidate' if len(roots) == 1 else 'undetermined',
            'roots':roots if len(roots) == 1 else [], 'scores':scores, 'group_kinds':dict(kinds)}


def analyze_files(trace_dir, decision_file):
    paths = sorted(Path(trace_dir).glob('ccld-*.jsonl'))
    if not paths:
        raise ValueError('no decoded trace files')
    records, problems = load_records(paths)
    if problems:
        raise ValueError('invalid or incomplete trace: '+problems[0])
    raw = Path(decision_file).read_bytes().splitlines(keepends=True)
    if any(not line.endswith(b'\n') for line in raw):
        raise ValueError('incomplete decision file')
    decisions = [json.loads(line) for line in raw if line.strip()]
    if any(not isinstance(item,dict) for item in decisions):
        raise ValueError('invalid decision record')
    by_comm = defaultdict(list)
    for record in records:
        by_comm[str(record['comm_hash'])].append(record)
    result = []
    for comm,events in sorted(by_comm.items()):
        base = {'comm_hash':comm}
        inits = [e for e in events if e['event'] == 'comm_init']
        sizes = {e['world_size'] for e in inits}
        if (len(sizes) != 1 or len(inits) != next(iter(sizes),0) or
                {e['rank'] for e in inits} != set(range(next(iter(sizes),0))) or
                any(e['event'] in ('dropped','coverage_gap') for e in events)):
            result.append(dict(base,status='incomplete',roots=[],reason='missing rank initialization or telemetry gap'))
            continue
        world = sizes.pop()
        cd = [d for d in decisions if str(d.get('comm_hash')) == comm]
        if any(d.get('category') in ('invalid','coverage_gap') for d in cd):
            result.append(dict(base,status='incomplete',roots=[],reason='base analyzer reported invalid coverage'))
            continue
        confirmed = [d for d in cd if d.get('category') == 'slow' and d.get('kind') in CLASSIFIED and
                     str(d.get('group','')).startswith('tree:')]
        if not confirmed:
            result.append(dict(base,status='no_confirmed_tree_slow',roots=[]))
            continue
        if any(type(d.get('window')) is not int for d in confirmed):
            result.append(dict(base,status='incomplete',roots=[],reason='invalid decision window'))
            continue
        window = min(d['window'] for d in confirmed)
        batch = [d for d in confirmed if d['window'] == window]
        ops = {d.get('op_count') for d in batch}
        if len(ops) != 1 or type(next(iter(ops))) is not int:
            result.append(dict(base,status='incomplete',roots=[],window=window,reason='window mixes operations'))
            continue
        op = next(iter(ops))
        complete = [e for e in events if e['event'] == 'complete' and e.get('op_count') == op]
        states = {e['rank']:e for e in complete}
        if len(complete) != world or len(states) != world:
            result.append(dict(base,status='incomplete',roots=[],window=window,op_count=op,
                               reason='missing or duplicate completed rank records'))
            continue
        try:
            outcome = vote(states,batch)
            result.append(dict(base,window=window,op_count=op,**outcome))
        except ValueError as error:
            result.append(dict(base,status='incomplete',roots=[],window=window,op_count=op,reason=str(error)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces',type=Path,help='directory of decoded ccld-*.jsonl files')
    parser.add_argument('decisions',type=Path,help='completed base-analyzer JSONL output')
    args = parser.parse_args()
    for result in analyze_files(args.traces,args.decisions):
        print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
