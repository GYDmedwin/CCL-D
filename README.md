# CCL-D for NCCL

This repository provides an NCCL-based research implementation of the
diagnostic ideas in [CCL-D](https://doi.org/10.1145/3774934.3786429), with
GPU-level communication probes, host-side monitoring, and analysis tools for
slow and hanging collective operations. It is based on NVIDIA NCCL
**v2.24.3-1**, commit
`6aae37927840b2bd7b7d42d2f0050e75f88ee97f`.

## What is included

- NCCL source with opt-in mapped-host progress counters, per-communicator host
  probing, asynchronous binary trace writing, and periodic hang snapshots.
- An optional Ring-only wait counter and a three-completed-operation
  volume-normalized ranking rule, gated on a confirmed slow diagnosis.
- A binary trace decoder, a paper-inspired hang/slow analyzer, a one-shot live
  Ring observer, PyTorch collective and FSDP training examples, unit tests,
  and a Slurm example.
- The ordered patches in [ccld/patches](ccld/patches/README.md), so the
  modifications can be reviewed against the original NCCL release.

Telemetry is enabled through environment variables and uses mode 1 for the
store-only probe. The Ring wait observer combines a slow-detection gate with
per-rank wait information to identify candidate ranks.

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

The default output is `build/lib/libnccl.so.*`. Fault simulation can be
enabled in a separate build as described below. See the upstream
[NCCL README](README.NVIDIA.md) for other build options.

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

Set `--baseline-ms` to suit the workload. The Slurm example is in
[ccld/slurm/single_node_smoke.sh](ccld/slurm/single_node_smoke.sh); set your
site's account and partition at submission.

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

The current observer targets a single-host, eight-rank AllReduce communicator,
200 µs polling, and completed operations 4–6. The standalone
`ring_wait.py` accepts `--slow-confirmed` after a matching base slow diagnosis
has been confirmed.

For a recorded Tree AllReduce run, the optional second stage combines the
base analyzer's completed group decisions to identify an S1 candidate across
channels:

```bash
python3 ccld/analyzer/tree_s1_vote.py runs/tree/jsonl runs/tree/diagnosis.jsonl
```

It consumes decoded traces and base-analyzer JSONL, and reports a candidate
only when all eligible channels agree. Other outcomes remain available as
`not_applicable`, `undetermined`, or `incomplete`.

## Example: FSDP model training

The following example trains a synthetic causal decoder with FSDP, bfloat16,
and AdamW on eight GPUs. It runs the decoder and base analyzer alongside
training. Start in the repository root with CUDA-enabled PyTorch installed and
use a fresh output directory:

```bash
export CCLD_RUN="$PWD/runs/fsdp-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$CCLD_RUN/bin" "$CCLD_RUN/jsonl" "$CCLD_RUN/workload"
export LD_LIBRARY_PATH="$PWD/build/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$PWD/build/lib/libnccl.so.2"
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1 NCCL_CCLD_WAIT_ENABLE=1
export NCCL_CCLD_POLL_US=200 NCCL_CCLD_FRAME_BLOCKS=32
export NCCL_CCLD_QUEUE_SLOTS=2048 NCCL_CCLD_COMPLETION=event
export NCCL_CCLD_OUT_DIR="$CCLD_RUN/bin"

python3 ccld/tools/decode_async_wait.py "$CCLD_RUN/bin" "$CCLD_RUN/jsonl" --follow &
decoder=$!
python3 ccld/analyzer/ccld_analyze.py "$CCLD_RUN/jsonl" --follow \
  --baseline-ms 1 --baseline-rounds 20 --baseline-skip 2 \
  --window-s 1 --repeats 2 --poll-s 0.2 > "$CCLD_RUN/diagnosis.jsonl" &
analyzer=$!
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  ccld/experiments/fsdp_train.py --profile medium --steps 20 \
  --output-dir "$CCLD_RUN/workload" --expect-nccl-prefix "$PWD/build/lib"
kill "$decoder"; wait "$decoder"
sleep 1
kill "$analyzer"; wait "$analyzer" 2>/dev/null || true
```

The example checks the NCCL library mapped by every rank and writes per-rank
losses, step times and CPU affinity to `workload/`. The `medium` profile has
approximately 117 million parameters; `smoke` is a smaller setup check. The
tokens are synthetic, so the loss is useful for checking run consistency
rather than model quality. Choose analysis windows and baseline values for
your workload.

To monitor an existing PyTorch training job, enable the same NCCL environment
on every rank, launch the training command as usual, and run the decoder and
base analyzer on its trace directory. The covered operations are Ring
AllReduce, AllGather and ReduceScatter, and Tree AllReduce. For multiple nodes,
use a directory visible to the observer and every rank. On Linux, check
`/proc/<training-pid>/maps` for the mapped `libnccl.so` when using your own
workload. The one-shot Ring observer above is a separate example for its
specified eight-rank AllReduce workload.

For a two-node Slurm launch, activate the PyTorch environment, set
`CCLD_ROOT` to the shared repository path, and submit
[ccld/slurm/fsdp_train.sh](ccld/slurm/fsdp_train.sh) with your site's
account, partition and GPU resource options. The script uses eight GPUs per
node by default; `CCLD_GPUS_PER_NODE`, `CCLD_PROFILE`, `CCLD_STEPS`, and
`CCLD_RUN` can be set before submission. The default 32 probe frames can be
changed through `NCCL_CCLD_FRAME_BLOCKS`. Adjust the CPU request to the site
and workload so the training workers and NCCL/monitoring threads have room to
run; [RESULTS.md](RESULTS.md) includes an example CPU allocation comparison.

## Validation

The online observer has been evaluated with single-node, eight-H100
AllReduce workloads using Ring/Simple, Ring/LL, and Ring/LL128. Each protocol
was exercised with five simulated S2 send delays and five no-fault controls.
Collective correctness and FSDP training have also been exercised across two
eight-H100 nodes.
See [RESULTS.md](RESULTS.md) for the test configuration and results.

## Fault simulation

The small workload supports `--fault H1`, `H2`, and `S1`. The source also
provides bounded H3 and S2 GPU fault simulations, enabled with
`CCLD_FAULT_INJECTION`. Run fault simulations in an isolated, time-limited
allocation.

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
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Yida Gu, Fakang Wang, et al., “CCL-D: A High-Precision Diagnostic System for
Slow and Hang Anomalies in Large-Scale Model Training,” PPoPP 2026.
DOI: [10.1145/3774934.3786429](https://doi.org/10.1145/3774934.3786429).
