#!/usr/bin/env python3
"""Plot the recorded 16-GPU FSDP CPU-affinity comparison (requires matplotlib)."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    data=json.loads(args.evidence.read_text())['cpu_affinity_rotations']['two_hardware_threads_per_core']
    modes=(('stock','Upstream','#3a6689'),('disabled','Modified, disabled','#88a7b8'),
           ('base','Base probe enabled','#de8c41'))
    cores=(8,24)
    fig,ax=plt.subplots(figsize=(8.4,5.0))
    fig.subplots_adjust(left=.10,right=.98,top=.88,bottom=.23)
    width=0.23
    for idx,(mode,label,color) in enumerate(modes):
        for x,core in enumerate(cores):
            values=[c['mean_max_rank_step_ms'] for c in data['cases']
                    if c['mode']==mode and c['physical_cores_per_node']==core]
            if len(values)!=2:
                raise ValueError(f'expected two passes for {mode}/{core}')
            position=x+(idx-1)*width
            mean=statistics.mean(values)
            ax.bar(position,mean,width=width*.84,color=color,label=label if x==0 else None,zorder=2)
            ax.scatter([position-width*.12,position+width*.12],values,
                       s=28,facecolors='white',edgecolors='#263238',linewidths=.8,zorder=3)
            ax.text(position,max(mean,*values)+4,f'{mean:.1f}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(range(2),['8 cores / 16 threads','24 cores / 48 threads'])
    ax.set_ylabel('Maximum-rank mean step time (ms)')
    ax.set_xlabel('CPU affinity per node')
    ax.set_ylim(0,165)
    ax.set_title('16-GPU FSDP: sensitivity to CPU affinity',loc='left',fontweight='bold',pad=16)
    ax.grid(axis='y',color='#dbe2e7',linewidth=.7,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,ncol=3,loc='upper right',fontsize=8)
    fig.text(.10,.05,'117M parameters, synthetic tokens; 30 steps/pass, first 5 excluded. Dots: individual passes.',
             fontsize=8,color='#45535d')
    fig.savefig(args.output,metadata={'Date':None})
    if args.output.suffix.lower() == '.svg':
        args.output.write_text('\n'.join(line.rstrip() for line in args.output.read_text().splitlines())+'\n')


if __name__=='__main__':
    main()
