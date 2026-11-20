# Replay and fault experiments

These examples use the optional file-based workflow and the NCCL build from
the [README](../../README.md#build). For memory-pull diagnosis during training,
see [online deployment](live.md).

## Example: one-node Ring experiment

Run from the repository root in an isolated eight-GPU Linux allocation with
CUDA-enabled PyTorch. This experiment forces Ring/Simple and uses the optional
Ring wait probe. The workload checks that PyTorch loaded the modified NCCL.

```bash
mkdir -p "$PWD/runs"
run=$(mktemp -d "$PWD/runs/ring.XXXXXXXX")
mkdir "$run/bin" "$run/jsonl"
export CCLD_NCCL_LIB="$(pwd -P)/build/lib"
export LD_LIBRARY_PATH="$CCLD_NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1 NCCL_CCLD_WAIT_ENABLE=1
export NCCL_CCLD_POLL_US=200 NCCL_CCLD_FRAME_BLOCKS=512
export NCCL_CCLD_QUEUE_SLOTS=2048 NCCL_CCLD_COMPLETION=event
unset NCCL_CCLD_NODE_SOCKET
export NCCL_CCLD_OUT_DIR="$run/bin"
LD_PRELOAD="$CCLD_NCCL_LIB/libnccl.so.2" \
  torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  ccld/experiments/torch_collectives.py --op allreduce \
  --expect-nccl-prefix "$CCLD_NCCL_LIB" --expect-nccl-version 22403
python3 ccld/tools/decode_async_wait.py "$run/bin" "$run/jsonl"
python3 ccld/analyzer/ccld_analyze.py "$run/jsonl" \
  --baseline-ms 1 --flush-last-window
```

Set `--baseline-ms` to suit the workload. To run the capture and decode steps
under Slurm, submit [single_node_smoke.sh](../slurm/single_node_smoke.sh) with
your site account and partition.

## Follow a Ring recording

For a live file-based Ring experiment, run the decoder and observer in a
second shell while the workload is running. Set `run` to the capture's
directory and calibrate `CCLD_THRESHOLD` for the exact build and workload.

```bash
: "${CCLD_THRESHOLD:?set the calibrated Ring wait threshold}"
mkdir -p "$run/live-jsonl"
python3 ccld/tools/decode_async_wait.py "$run/bin" "$run/live-jsonl" --follow &
python3 ccld/analyzer/online_ring_wait.py "$run/live-jsonl" \
  --protocol SIMPLE --threshold "$CCLD_THRESHOLD" \
  --baseline-ms 1 --window-s 1 --repeats 2
```

The one-shot observer forwards base analyzer events. After a confirmed slow
diagnosis for the same communicator, it compares per-rank wait information
normalized by operation volume. The tested configuration uses a single-host,
eight-rank AllReduce communicator, 200 µs polling, and completed operations
4–6. For replay, `ring_wait.py` accepts `--slow-confirmed` after a matching
base slow diagnosis has been confirmed.

## Vote across Tree channels

For recorded Tree AllReduce, the offline S1 stage reads decoded traces and
base analyzer decisions from the same run.

```bash
python3 ccld/analyzer/tree_s1_vote.py runs/tree/jsonl runs/tree/diagnosis.jsonl
```

It reports a candidate when all eligible channels support the same rank.
The other statuses are `not_applicable`, `undetermined`, and `incomplete`.
See [how to interpret online Tree decisions](live.md#interpret-tree-slow-decisions).

## Fault simulation

The [collective workload](../experiments/torch_collectives.py) supports
`--fault H1`, `H2`, and `S1`. The NCCL source provides bounded GPU-side H3 and
S2 simulations behind `CCLD_FAULT_INJECTION`. Use an isolated, time-limited
allocation for injected faults.

For a separate fault-enabled H100 build:

```bash
CXXFLAGS='-DCCLD_FAULT_INJECTION=1' make -j8 src.build \
  CUDA_HOME=/usr/local/cuda BUILDDIR="$PWD/build-fault" \
  NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90 -DCCLD_FAULT_INJECTION=1'
```

Point the test to `build-fault/lib`. Alongside the ordinary probe
environment, set `NCCL_CCLD_FAULT_KIND=S2` for a send delay or `H3` for a
bounded stall, plus `NCCL_CCLD_FAULT_RANK`, `NCCL_CCLD_FAULT_OP`, and
`NCCL_CCLD_FAULT_CYCLES`. Set `NCCL_CCLD_FAULT_REPEATS` for repeated
injections. The operation number is the recorded communicator operation
counter. Delay is expressed in GPU cycles. Confirm the actual injection
and operation in the collected traces.

See the [validation results](../../RESULTS.md) for the tested configurations.
