#!/usr/bin/env python3
"""One-shot live Ring-wait decision after a matching base slow diagnosis."""
import argparse
import json
import math
from pathlib import Path
import selectors
import signal
import subprocess
import sys

from ring_wait import completed_rounds, decide


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace_dir', type=Path, help='directory of decoded JSONL files')
    p.add_argument('--protocol', required=True, choices=('SIMPLE', 'LL', 'LL128'))
    p.add_argument('--threshold', type=float, required=True, help='threshold calibrated for this build and workload')
    p.add_argument('--interval-s', type=float, default=0.1)
    args, base_args = p.parse_known_args()
    if not math.isfinite(args.threshold) or args.threshold <= 0 or not math.isfinite(args.interval_s) or args.interval_s <= 0:
        p.error('threshold and interval must be finite and positive')
    if '--follow' in base_args:
        p.error('follow mode is already enabled')
    cmd = [sys.executable, '-u', str(Path(__file__).with_name('ccld_analyze.py')),
           str(args.trace_dir), '--follow', *base_args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    pending, confirmed, stopped = b'', {}, False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_error = 'waiting for a confirmed slow event and three complete operations'
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
                raise RuntimeError(f'base analyzer exited with status {proc.returncode}')
            if confirmed:
                try:
                    states, comm = completed_rounds(args.trace_dir, args.protocol)
                    if comm in confirmed:
                        result = decide(states, args.threshold, slow_confirmed=True)
                        result.update(category='online_wait', kind='three_round_decision',
                                      comm_hash=comm, method='volume_per_rank',
                                      base_slow_kind=confirmed[comm]['kind'])
                        print(json.dumps(result, sort_keys=True), flush=True)
                        return 0
                    last_error = 'completed communicator does not match the confirmed slow event'
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    last_error = str(exc)
        print(json.dumps({'category': 'online_wait', 'status': 'indeterminate',
                          'detected': None, 'candidate_ranks': [], 'reason': last_error}, sort_keys=True), flush=True)
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
