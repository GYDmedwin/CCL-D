#!/usr/bin/env python3
"""Validate a strict per-send S2 fault with NV Ring-wait scores over 10 rounds."""
import argparse, json, statistics, sys
from pathlib import Path
sys.path.insert(0, '/workspace/nv/CCL-D/ccld/analyzer')
from ring_wait import graph_wait_scores

def load(path, first, last):
    out={}
    for f in sorted(path.rglob('*.jsonl')) if path.is_dir() else [path]:
        for line in f.read_text().splitlines():
            if not line.strip(): continue
            e=json.loads(line)
            if e.get('event')=='complete' and first <= e.get('op_count',-1) <= last:
                out.setdefault(e['op_count'],{})[e['rank']]=e
    return out

def score(data, first, last, world):
    rows=[]
    for op in range(first,last+1):
        if op not in data or set(data[op]) != set(range(world)):
            raise SystemExit(f'incomplete op {op}')
        m=graph_wait_scores(data[op]); rows.append((op,m))
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--control',action='append',required=True,type=Path); p.add_argument('--fault',required=True,type=Path); p.add_argument('--world-size',type=int,required=True); p.add_argument('--target-rank',type=int,required=True); p.add_argument('--first-op',type=int,default=4); p.add_argument('--rounds',type=int,default=10); p.add_argument('--threshold',type=float,default=0.1); p.add_argument('--fault-events',type=Path)
    a=p.parse_args(); last=a.first_op+a.rounds-1
    controls=[]; control_winners=[]
    for c in a.control:
        rows=score(load(c,a.first_op,last),a.first_op,last,a.world_size); controls.extend(x[1]['top_gap'] for x in rows); control_winners.extend(m['forced_top_ranks'][0] for _,m in rows if len(m['forced_top_ranks'])==1)
    fault=score(load(a.fault,a.first_op,last),a.first_op,last,a.world_size)
    winners=[m['forced_top_ranks'][0] for _,m in fault if len(m['forced_top_ranks'])==1]
    target_votes=sum(r==a.target_rank for r in winners)
    control_median=statistics.median(controls)
    control_target_votes=sum(r==a.target_rank for r in control_winners); fault_median=statistics.median(m['top_gap'] for _,m in fault)
    result={'world_size':a.world_size,'target_rank':a.target_rank,'rounds':a.rounds,'target_votes':target_votes,'winner_count':len(winners),'fault_gaps':[m['top_gap'] for _,m in fault],'fault_gap_median':fault_median,'control_gap_median':control_median,'control_target_votes':control_target_votes,'control_winners':control_winners,'threshold':a.threshold,'result':'pass' if target_votes >= (a.rounds*7)//10 and fault_median >= a.threshold and target_votes > control_target_votes else 'fail'}
    if a.fault_events:
        marks=[]
        for f in sorted(a.fault_events.rglob('*.ndjson')):
            for line in f.read_text().splitlines():
                if line.strip(): marks.append(json.loads(line))
        if marks: result['fault_marker_ns']=min(x['mono_ns'] for x in marks); result['fault_marker_kind']=marks[0].get('kind')
    print(json.dumps(result,sort_keys=True))
    if result['result']!='pass': raise SystemExit(1)
if __name__=='__main__': main()
