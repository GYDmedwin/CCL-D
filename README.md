# CCL-D-style diagnostics for NCCL (experimental)

This repository contains an experimental NCCL implementation of diagnostic
ideas from [CCL-D](https://doi.org/10.1145/3774934.3786429), together with
trace decoders and analyzers. It is a separate NCCL-based implementation; it
does not contain the original AntCCL implementation. The starting point is
NVIDIA NCCL **v2.24.3-1**, commit
`6aae37927840b2bd7b7d42d2f0050e75f88ee97f`.

**Research prototype:** the Ring wait rule has only narrow synthetic
single-node validation. Its scores and thresholds are not proven to transfer
to other models, sizes, hardware, builds, protocols, or topologies. This is
not a production hang detector. See [RESULTS.md](RESULTS.md) before use.

## What is included

- NCCL source with opt-in mapped-host progress counters, per-communicator host
  probing, asynchronous binary trace writing, and periodic hang snapshots.
- An optional Ring-only wait counter and a three-completed-operation
  volume-normalized ranking rule, gated on a confirmed slow diagnosis.
- A binary trace decoder, a paper-inspired hang/slow analyzer, a one-shot live
  Ring observer, a small PyTorch workload, unit tests, and a Slurm example.
- The ordered patches in [ccld/patches](ccld/patches/README.md), so the
  modifications can be reviewed against the original NCCL release.

There is no claim that the Tree wait rule, grouped point-to-point/AlltoAll
coverage, graph capture/replay, multi-node Ring wait localization, or FSDP
training is validated by these experiments. Progress snapshots may be
unavailable or dropped; missing evidence is not treated as a zero counter.
The current store-only probe supports **mode 1 only**. Telemetry is disabled by
default; this does not establish zero overhead versus unmodified NCCL.

## Build

Build with Linux, a compatible CUDA toolkit and C++ toolchain. The historical
measurements used CUDA 12.8.2 and NVIDIA H100; this example limits the build
to H100 (`sm_90`). Set `CUDA_HOME` for your installation and omit or change
`NVCC_GENCODE` for a different target supported by your toolkit.

```bash
make -j8 src.build CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
python3 -m unittest discover -s ccld/tests -v
```

The default output is `build/lib/libnccl.so.*`. This full build has fault
injection compiled out unless explicitly enabled at build time. A successful
build does not guarantee that a framework loads this library; verify the
actual mapped library before interpreting traces or performance. See the
upstream [NCCL README](README.NVIDIA.md) for other build options.

## Example: one-node Ring experiment

The following is an example for an isolated eight-GPU Linux allocation and an
installed CUDA-enabled PyTorch. It forces Ring/Simple for this experiment,
which may change normal algorithm selection. Choose a fresh output directory.
`--expect-nccl-prefix` makes the workload fail if the modified NCCL is not the
library that PyTorch actually mapped.

```bash
mkdir -p runs/example/bin runs/example/jsonl
export LD_LIBRARY_PATH="$PWD/build/lib:${LD_LIBRARY_PATH:-}"
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1 NCCL_CCLD_WAIT_ENABLE=1
export NCCL_CCLD_POLL_US=200 NCCL_CCLD_FRAME_BLOCKS=512
export NCCL_CCLD_QUEUE_SLOTS=2048 NCCL_CCLD_COMPLETION=event
export NCCL_CCLD_OUT_DIR="$PWD/runs/example/bin"
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  ccld/experiments/torch_collectives.py --op allreduce \
  --expect-nccl-prefix "$PWD/build" --expect-nccl-version 22403
python3 ccld/tools/decode_async_wait.py runs/example/bin runs/example/jsonl
python3 ccld/analyzer/ccld_analyze.py runs/example/jsonl \
  --baseline-ms 1 --flush-last-window
```

`--baseline-ms 1` is an example, not a generally valid baseline. Inspect the
decoded records and calibrate the detector for the workload. The Slurm example
is in [ccld/slurm/single_node_smoke.sh](ccld/slurm/single_node_smoke.sh); set
your site's account and partition at submission.

For live use, start the decoder with `--follow` while the program is running,
and analyze its JSONL output with `--follow`. The optional one-shot Ring
observer below forwards base analyzer events, waits for a confirmed slow
diagnosis for the same communicator, then evaluates completed operations
4–6. `CCLD_THRESHOLD` must be calibrated for the exact build and workload.

```bash
python3 ccld/tools/decode_async_wait.py runs/example/bin runs/live-jsonl --follow &
python3 ccld/analyzer/online_ring_wait.py runs/live-jsonl \
  --protocol SIMPLE --threshold "$CCLD_THRESHOLD" \
  --baseline-ms 1 --window-s 1 --repeats 2
```

The observer is intentionally restricted to one eight-rank, single-host
AllReduce communicator, 200 µs polling, and completed operations 4–6. It
rejects missing ranks, inconsistent signatures, invalid Ring links and
reported dropped records. These choices mirror the narrow historical
experiment; they do not make its threshold universally applicable. The
standalone `ring_wait.py` accepts `--slow-confirmed` only when the matching
base slow result has already been checked. A lack of confirmation or data is
indeterminate, not a healthy verdict.

The portable online observer has been exercised on H100 with five synthetic
S2 faults and five controls per protocol, using the historical restricted
fault-injection library and workload. All faults were localized and no control
raised an online alert. LL and LL128 faults were classified as S3 by the base
analyzer. An offline experiment that bypassed the slow gate crossed the old
threshold in one Simple control, without a majority rank candidate. See
[RESULTS.md](RESULTS.md) for the limits and timing of this validation.

## Fault simulation

The small workload supports `--fault H1`, `H2`, and `S1` for isolated,
time-limited tests. H1/H2 can intentionally stall communication; use a job
timeout and an isolated allocation. The source also contains bounded H3/S2
GPU fault simulation guarded by `CCLD_FAULT_INJECTION`; the default build does
not enable it. Such delays are simulations, not hardware failures. The
historical fault-validation library was a restricted-function build and must
not be substituted for a normal framework library.

For a separate, explicitly fault-enabled test build on H100:

```bash
CXXFLAGS='-DCCLD_FAULT_INJECTION=1' make -j8 src.build \
  CUDA_HOME=/usr/local/cuda BUILDDIR="$PWD/build-fault" \
  NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90 -DCCLD_FAULT_INJECTION=1'
```

Use `build-fault/lib` only in the isolated test environment. Alongside the
ordinary probe environment, set `NCCL_CCLD_FAULT_KIND=S2` (send delay) or
`H3` (bounded stall), `NCCL_CCLD_FAULT_RANK`, `NCCL_CCLD_FAULT_OP`, and
`NCCL_CCLD_FAULT_CYCLES`; optionally set `NCCL_CCLD_FAULT_REPEATS`.
The operation number is the recorded communicator operation counter, and GPU
cycles are not microseconds. Apply a job timeout and confirm the actual
injection and operation in the collected traces.

## License and citation

NCCL's license is retained in [LICENSE.txt](LICENSE.txt), NVTX's in
[LICENSE-NVTX.txt](LICENSE-NVTX.txt), and original reproduction additions in
[LICENSE-CCLD.txt](LICENSE-CCLD.txt). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The paper and experiment
logs are not distributed in this repository.

Yida Gu, Fakang Wang, et al., “CCL-D: A High-Precision Diagnostic System for
Slow and Hang Anomalies in Large-Scale Model Training,” PPoPP 2026.
DOI: [10.1145/3774934.3786429](https://doi.org/10.1145/3774934.3786429).
