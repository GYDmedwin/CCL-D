#!/usr/bin/env python3
"""RCCL topology adapter for the NV CCL-D online Ring-wait decision."""
import argparse
import json
import os
import math
from pathlib import Path
import selectors
import signal
import subprocess
import sys

NV_ANALYZER = Path(os.environ.get('CCLD_NV_ROOT', str(Path(__file__).resolve().parents[2]))) / 'ccld/analyzer'
sys.path.insert(0, str(NV_ANALYZER))
from ring_wait import decide  # noqa: E402


def completed_rounds(trace_dir: Path, protocol: str, world_size: int):
    if world_size not in (4, 8):
        raise ValueError('gfx936 Ring-wait accepts 4 ranks on one node or 8 ranks on two nodes')
    states = {op: {} for op in (4, 5, 6)}
    inits, signatures = {}, set()
    files = sorted(trace_dir.rglob('*.jsonl'))
    if not files:
        raise ValueError('no decoded trace files')
    for path in files:
        with path.open('rb') as stream:
            for raw in stream:
                if not raw.endswith(b'\n'):
                    continue
                event = json.loads(raw)
                kind = event.get('event')
                if kind in ('dropped', 'coverage_gap'):
                    raise ValueError('trace contains a dropped record or coverage gap')
                rank = event.get('rank')
                if kind == 'comm_init':
                    if rank in inits:
                        raise ValueError('duplicate rank initialization or multiple communicators')
                    inits[rank] = event
                elif kind == 'complete' and event.get('op_count') in states:
                    op = event['op_count']
                    if rank in states[op]:
                        raise ValueError('duplicate rank completion')
                    if (str(event.get('algorithm', '')).upper(), str(event.get('protocol', '')).upper()) != ('RING', protocol.upper()):
                        raise ValueError('actual algorithm or protocol differs')
                    states[op][rank] = event
                    signatures.add((event.get('count'), event.get('datatype'), event.get('n_channels')))

    ranks = set(range(world_size))
    if set(inits) != ranks:
        raise ValueError('communicator rank set is incomplete or differs from requested world size')
    if len({e.get('comm_hash') for e in inits.values()}) != 1:
        raise ValueError('trace contains multiple communicators')
    if any(e.get('world_size') != world_size or e.get('poll_us') != 200 for e in inits.values()):
        raise ValueError('world size or polling interval differs from the tested RCCL configuration')
    hosts = {}
    for rank, event in inits.items():
        host = event.get('host')
        if not isinstance(host, str) or not host:
            raise ValueError('host identity is missing')
        hosts.setdefault(host, set()).add(rank)
    host_sizes = sorted(map(len, hosts.values()))
    expected_host_sizes = [4] if world_size == 4 else [4, 4]
    if host_sizes != expected_host_sizes:
        raise ValueError('rank-to-node layout differs from gfx936 4-rank single-node or 4+4 dual-node setup')
    if any(set(group) != ranks for group in states.values()) or len(signatures) != 1:
        raise ValueError('one or more operations are incomplete or workload signatures differ')
    comm = next(iter(inits.values())).get('comm_hash')
    if any(e.get('comm_hash') != comm or e.get('world_size') != world_size or e.get('op') != 'AllReduce'
           for group in states.values() for e in group.values()):
        raise ValueError('completed operation does not match the initialized AllReduce communicator')
    return states, comm


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace_dir', type=Path)
    p.add_argument('--protocol', required=True, choices=('SIMPLE', 'LL'))
    p.add_argument('--world-size', required=True, type=int, choices=(4, 8))
    p.add_argument('--threshold', required=True, type=float,
                   help='E5 threshold calibrated for this RCCL build/workload')
    p.add_argument('--interval-s', type=float, default=0.1)
    args, base_args = p.parse_known_args()
    if (not math.isfinite(args.threshold) or args.threshold <= 0 or
            not math.isfinite(args.interval_s) or args.interval_s <= 0):
        p.error('threshold and interval must be finite and positive')
    if '--follow' in base_args:
        p.error('follow mode is enabled by this adapter')

    base_script = NV_ANALYZER / 'ccld_analyze.py'
    cmd = [sys.executable, '-u', str(base_script), str(args.trace_dir), '--follow', *base_args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    pending, confirmed, stopped = b'', {}, False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    reason = 'waiting for a confirmed base slow result and complete operations 4, 5, 6'
    try:
        while not stopped:
            for key, _ in selector.select(timeout=args.interval_s):
                data = key.fileobj.read(65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                pending += data
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    item = json.loads(line)
                    print(line.decode('utf-8'), flush=True)
                    if (item.get('category') == 'slow' and item.get('group') == 'ring' and
                            item.get('op_count') in (4, 5, 6) and
                            item.get('kind') in ('S1', 'S2', 'S3', 'undetermined')):
                        confirmed[item.get('comm_hash')] = item
            if proc.poll() is not None:
                raise RuntimeError(f'NV base analyzer exited with status {proc.returncode}')
            if confirmed:
                try:
                    states, comm = completed_rounds(args.trace_dir, args.protocol, args.world_size)
                    if comm in confirmed:
                        result = decide(states, args.threshold, slow_confirmed=True)
                        result.update(category='online_wait', kind='three_round_decision',
                                      comm_hash=comm, method='volume_per_rank',
                                      base_slow_kind=confirmed[comm]['kind'], world_size=args.world_size)
                        print(json.dumps(result, sort_keys=True), flush=True)
                        return 0
                    reason = 'complete communicator does not match the confirmed slow event'
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    reason = str(exc)
        print(json.dumps({'category': 'online_wait', 'status': 'indeterminate',
                          'detected': None, 'candidate_ranks': [], 'reason': reason}, sort_keys=True), flush=True)
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        selector.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
