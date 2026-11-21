#!/usr/bin/env python3
"""Persist the CCL-D fault settings used by a gfx936 test run."""
import json
import os
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: write_ccld_fault_config.py OUTPUT.json')
    config = {
        'algorithm': os.environ.get('NCCL_ALGO', ''),
        'protocol': os.environ.get('NCCL_PROTO', ''),
        'world_size': int(os.environ.get('CCLD_NNODES', '1')) * int(os.environ.get('CCLD_NPROC_PER_NODE', '1')),
        'fault_kind': os.environ.get('NCCL_CCLD_FAULT_KIND', ''),
        'fault_rank': int(os.environ.get('NCCL_CCLD_FAULT_RANK', '-1')),
        'fault_first_op': int(os.environ.get('NCCL_CCLD_FAULT_OP', '-1')),
        'fault_repeats': int(os.environ.get('NCCL_CCLD_FAULT_REPEATS', '-1')),
        'fault_cycles': int(os.environ.get('NCCL_CCLD_FAULT_CYCLES', '-1')),
        'rccl_library': os.environ.get('RCCL_LIB', ''),
        'rccl_build_dir': os.environ.get('RCCL_BUILD_DIR', ''),
        'workload_fault_kind': os.environ.get('CCLD_WORKLOAD_FAULT', 'none'),
        'workload_fault_rank': int(os.environ.get('CCLD_WORKLOAD_FAULT_RANK', '-1')),
        'workload_fault_iteration': int(os.environ.get('CCLD_WORKLOAD_FAULT_ITERATION', '-1')),
        'workload_numel': int(os.environ.get('CCLD_NUMEL', '262144')),
        'workload_h2_delta_bytes': int(os.environ.get('CCLD_WORKLOAD_H2_DELTA_BYTES', '1024')),
        'workload_h2_count_divisor': int(os.environ.get('CCLD_WORKLOAD_H2_COUNT_DIVISOR', '0')),
        'workload_h2_mismatch': os.environ.get('CCLD_WORKLOAD_H2_MISMATCH', 'count'),
        'workload_h2_operation': os.environ.get('CCLD_WORKLOAD_H2_OPERATION', 'reducescatter'),
    }
    ccld_fault = bool(config['fault_kind'])
    workload_fault = config['workload_fault_kind'] != 'none'
    config['workload_h2_injected_count'] = (config['workload_numel'] - config['workload_h2_delta_bytes'] // 4 if config['workload_h2_count_divisor'] == 0 else max(1, config['workload_numel'] // config['workload_h2_count_divisor']))
    if not ccld_fault and not workload_fault:
        raise SystemExit('at least one fault mode must be configured')
    if ccld_fault and (config['fault_rank'] < 0 or config['fault_first_op'] < 1 or config['fault_repeats'] < 1 or config['fault_cycles'] < 1):
        raise SystemExit('incomplete CCL-D device fault configuration')
    if workload_fault and (config['workload_fault_rank'] < 0 or config['workload_fault_iteration'] < 0):
        raise SystemExit('incomplete workload fault configuration')
    if not config['algorithm'] or not config['protocol'] or not config['rccl_library']:
        raise SystemExit('algorithm, protocol and RCCL library are required in fault manifest')
    target = Path(sys.argv[1])
    target.write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
