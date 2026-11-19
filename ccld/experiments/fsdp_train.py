#!/usr/bin/env python3
"""Train a small causal decoder with FSDP and fixed synthetic token data."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

from torch_collectives import mapped_nccl, runtime_nccl_version


class DecoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.width, self.heads = width, heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width, bias=False)
        self.fc2 = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        b, s, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, s, 3, self.heads, d // self.heads)
        q, k, v = qkv.unbind(2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, s, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Decoder(nn.Module):
    def __init__(self, vocab, width, heads, layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(DecoderBlock(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=('smoke', 'medium', 'train'), required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--expect-nccl-prefix', required=True)
    a = p.parse_args()
    if not 3 <= a.steps <= 300:
        p.error('steps must be 3–300')
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world < 1 or not 0 <= local < torch.cuda.device_count():
        raise RuntimeError('local rank does not map to a visible GPU')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(seconds=480))
    libs = mapped_nccl()
    if (len(libs) != 1 or not str(Path(libs[0]).resolve()).startswith(a.expect_nccl_prefix) or
            runtime_nccl_version(libs) != 22403):
        raise RuntimeError(f'unexpected NCCL mapping {libs}')
    configs = {
        'smoke': {'width': 256, 'heads': 4, 'layers': 2, 'vocab': 1024, 'sequence': 32},
        'medium': {'width': 1024, 'heads': 16, 'layers': 8, 'vocab': 8192, 'sequence': 128},
        'train': {'width': 2048, 'heads': 16, 'layers': 24, 'vocab': 16384, 'sequence': 64},
    }
    config = configs[a.profile]
    torch.set_num_threads(1)
    torch.manual_seed(24016)
    model = Decoder(config['vocab'], config['width'], config['heads'], config['layers'])
    nparams = sum(p.numel() for p in model.parameters())
    init_sample = model.embedding.weight.detach().reshape(-1)[:1024].numpy().tobytes()
    init_sha = hashlib.sha256(init_sample).hexdigest()
    generator = torch.Generator(device='cpu').manual_seed(90000 + rank)
    tokens = torch.randint(config['vocab'], (1, config['sequence'] + 1), generator=generator)
    data_sha = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    x, target = tokens[:, :-1].to(local), tokens[:, 1:].to(local)
    precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                               buffer_dtype=torch.bfloat16)
    model.blocks = nn.ModuleList(FSDP(block, device_id=local, mixed_precision=precision,
                                      use_orig_params=True, limit_all_gathers=True)
                                 for block in model.blocks)
    model = FSDP(model, device_id=local, mixed_precision=precision,
                 use_orig_params=True, limit_all_gathers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    dist.barrier(device_ids=[local])
    durations_ms, losses, first_grad_norm = [], [], None
    training_started_mono_ns = time.monotonic_ns()
    for step in range(a.steps):
        torch.cuda.synchronize(local)
        begin = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, config['vocab']), target.reshape(-1))
        loss.backward()
        if step == 0:
            norms = [p.grad.detach().float().norm() for p in model.parameters()
                     if p.grad is not None and p.grad.numel()]
            first_grad_norm = float(torch.stack(norms).sum().item()) if norms else 0.0
        optimizer.step()
        torch.cuda.synchronize(local)
        durations_ms.append((time.perf_counter_ns() - begin) / 1e6)
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise RuntimeError(f'rank {rank} non-finite loss at step {step}')
        losses.append(value)
        if rank == 0 and (step == 0 or (step + 1) % 20 == 0 or step + 1 == a.steps):
            print(f'FSDP_STEP {step + 1}/{a.steps} loss={value:.6f} ms={durations_ms[-1]:.3f}', flush=True)
    training_finished_mono_ns = time.monotonic_ns()
    result = {'rank': rank, 'local_rank': local, 'world_size': world, 'host': socket.gethostname(), 'profile': a.profile,
              'steps': a.steps, 'config': config, 'parameters': nparams,
              'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
              'torch_cpu_threads': torch.get_num_threads(),
              'initial_embedding_sample_sha256': init_sha, 'tokens_sha256': data_sha,
              'mapped_nccl': libs, 'runtime_nccl_version': 22403,
              'loss': losses, 'step_ms': durations_ms, 'first_gradient_norm': first_grad_norm,
              'training_started_mono_ns': training_started_mono_ns,
              'training_finished_mono_ns': training_finished_mono_ns,
              'scope': 'fixed synthetic tokens, causal decoder, FSDP bf16, AdamW; not model-quality evaluation'}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    tmp = a.output_dir / f'.rank-{rank}-{os.getpid()}.tmp'
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, a.output_dir / f'rank-{rank}.json')
    dist.barrier(device_ids=[local])
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
