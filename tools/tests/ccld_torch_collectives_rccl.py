#!/usr/bin/env python3
"""Small torchrun collective workload; see RESULTS.md for validated cases."""
import argparse
import ctypes
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist


def mapped_nccl():
    paths = set()
    try:
        for line in Path('/proc/self/maps').read_text().splitlines():
            # RCCL is NCCL-API compatible but is mapped as librccl.so.
            if 'libnccl.so' in line or 'librccl.so' in line:
                paths.add(line.split()[-1])
    except OSError:
        pass
    return sorted(paths)


def runtime_nccl_version(paths):
    if len(paths) != 1:
        raise RuntimeError(f'expected exactly one mapped NCCL library, got {paths}')
    lib = ctypes.CDLL(paths[0])
    fn = lib.ncclGetVersion
    fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.c_int
    version = ctypes.c_int()
    code = fn(ctypes.byref(version))
    if code != 0:
        raise RuntimeError(f'ncclGetVersion failed: {code}')
    return version.value


def log_workload(args, rank, event):
    line = json.dumps(event, sort_keys=True)
    if args.workload_log_dir:
        root = Path(args.workload_log_dir)
        root.mkdir(parents=True, exist_ok=True)
        with (root / f'rank-{rank}.ndjson').open('a', encoding='utf-8') as out:
            out.write(line + '\n')
            out.flush()
    print(line, flush=True)


def mark_fault(args, rank, iteration):
    root = Path(args.fault_log_dir)
    root.mkdir(parents=True, exist_ok=True)
    event = {'kind': args.fault, 'rank': rank, 'iteration': iteration,
             'host': socket.gethostname(), 'mono_ns': time.monotonic_ns()}
    with (root / f'rank-{rank}.ndjson').open('a', encoding='utf-8') as out:
        out.write(json.dumps(event, sort_keys=True) + '\n')
        out.flush()
    print('INJECT ' + json.dumps(event, sort_keys=True), flush=True)


def collective(op, n, rank, world, count_override=None):
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    if op == 'allreduce':
        # A one-element mismatch can still fit the same NCCL transfer chunks;
        # use a substantial count mismatch for the H2 experiment.
        count = n if count_override is None else count_override
        x = torch.full((count,), float(rank + 1), device=device)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x, float(world * (world + 1) // 2)
    if op == 'allgather':
        x = torch.full((n,), float(rank + 1), device=device)
        y = torch.empty((n * world,), device=device)
        dist.all_gather_into_tensor(y, x)
        return y, None
    if op == 'reducescatter':
        x = torch.full((n * world,), float(rank + 1), device=device)
        y = torch.empty((n,), device=device)
        dist.reduce_scatter_tensor(y, x, op=dist.ReduceOp.SUM)
        return y, float(world * (world + 1) // 2)
    if op == 'alltoall':
        x = torch.arange(world, device=device, dtype=torch.float32).repeat_interleave(n) + rank * world
        y = torch.empty_like(x)
        dist.all_to_all_single(y, x)
        return y, None
    raise ValueError(op)


def verify(op, out, expected, n, rank, world):
    if expected is not None:
        return bool(torch.all(out == expected).item())
    if op == 'allgather':
        return all(bool(torch.all(out[i*n:(i+1)*n] == i+1).item()) for i in range(world))
    return all(bool(torch.all(out[i*n:(i+1)*n] == i*world+rank).item()) for i in range(world))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--op', choices=('allreduce', 'allgather', 'reducescatter', 'alltoall'), default='allreduce')
    p.add_argument('--numel', type=int, default=1 << 18, help='elements per rank/chunk')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--fault', choices=('none', 'H1', 'H2', 'S1'), default='none')
    p.add_argument('--fault-rank', type=int, default=0)
    p.add_argument('--fault-iteration', type=int, default=3)
    p.add_argument('--h2-delta-bytes', type=int, default=1024, help='bytes removed from the injected H2 rank count')
    p.add_argument('--sleep-ms', type=int, default=1000)
    p.add_argument('--gap-ms', type=int, default=0, help='delay between collective rounds')
    p.add_argument('--hang-seconds', type=int, default=20)
    p.add_argument('--timeout-seconds', type=int, default=90)
    p.add_argument('--fault-log-dir', default='fault-events')
    p.add_argument('--workload-log-dir', default='')
    p.add_argument('--expect-nccl-prefix', default='')
    p.add_argument('--expect-nccl-version', type=int, default=0)
    args = p.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if args.numel < 2 or args.iters < 1 or args.warmup < 0 or args.gap_ms < 0 or args.h2_delta_bytes < 4 or args.h2_delta_bytes % 4 != 0 or args.h2_delta_bytes // 4 >= args.numel or not 0 <= args.fault_rank < world:
        p.error('invalid size, iteration count, round delay, or fault rank')
    if args.fault == 'H2' and args.op != 'allreduce':
        p.error('H2 count mismatch is implemented only for AllReduce')
    profile = os.environ.get('CCLD_RESTRICTED_BUILD', '')
    supported = {'1': {'allreduce'}, 'expanded': {'allreduce', 'allgather', 'reducescatter'},
                 'p2p': {'allreduce', 'alltoall'}}
    if profile and (profile not in supported or args.op not in supported[profile]):
        p.error(f'operation {args.op} is not allowed for restricted NCCL profile {profile!r}')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=args.timeout_seconds))
    mapped = mapped_nccl()
    if args.expect_nccl_prefix and not any(str(Path(p).resolve()).startswith(args.expect_nccl_prefix) for p in mapped):
        raise RuntimeError(f'loaded NCCL path {mapped} does not match {args.expect_nccl_prefix!r}')
    runtime_version = runtime_nccl_version(mapped)
    if args.expect_nccl_version and runtime_version != args.expect_nccl_version:
        raise RuntimeError(f'loaded NCCL version {runtime_version} != expected {args.expect_nccl_version}')
    log_workload(args, rank, {'rank': rank, 'local_rank': local_rank, 'host': socket.gethostname(),
                              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch': torch.__version__,
                              'torch_cuda': torch.version.cuda, 'torch_build_nccl': torch.cuda.nccl.version(),
                              'runtime_nccl_version': runtime_version, 'mapped_nccl': mapped, 'op': args.op})
    for iteration in range(args.warmup + args.iters):
        injected = rank == args.fault_rank and iteration >= args.fault_iteration
        if injected and args.fault == 'S1':
            mark_fault(args, rank, iteration)
            time.sleep(args.sleep_ms / 1000)
        elif injected and args.fault == 'H1':
            mark_fault(args, rank, iteration)
            # Do not enter this round. Keep the probe thread alive for snapshots;
            # the launcher must terminate the job after the configured timeout.
            time.sleep(args.hang_seconds)
            os._exit(77)
        elif injected and args.fault == 'H2':
            mark_fault(args, rank, iteration)
        collective_start_ns = time.monotonic_ns()
        h2_count = args.numel - args.h2_delta_bytes // 4 if injected and args.fault == 'H2' and rank == args.fault_rank else None
        x, expected = collective(args.op, args.numel, rank, world, count_override=h2_count)
        torch.cuda.synchronize()
        collective_end_ns = time.monotonic_ns()
        mismatched_round = args.fault == 'H2' and iteration >= args.fault_iteration
        if not mismatched_round and not verify(args.op, x, expected, args.numel, rank, world):
            raise RuntimeError(f'numerical verification failed rank={rank} iteration={iteration}')
        log_workload(args, rank, {'rank': rank, 'iteration': iteration, 'verified': not mismatched_round,
                                  'collective_start_ns': collective_start_ns,
                                  'collective_end_ns': collective_end_ns,
                                  'collective_duration_ns': collective_end_ns - collective_start_ns,
                                  'count_override': h2_count, 'count_delta_bytes': args.h2_delta_bytes if h2_count is not None else 0})
        if args.gap_ms and iteration + 1 < args.warmup + args.iters:
            time.sleep(args.gap_ms / 1000)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
