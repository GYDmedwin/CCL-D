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
    }
    if not config['fault_kind'] or config['fault_rank'] < 0 or config['fault_first_op'] < 1 or \
            config['fault_repeats'] < 1 or config['fault_cycles'] < 1:
        raise SystemExit('incomplete CCL-D fault configuration')
    if not config['algorithm'] or not config['protocol'] or not config['rccl_library']:
        raise SystemExit('algorithm, protocol and RCCL library are required in fault manifest')
    target = Path(sys.argv[1])
    target.write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
