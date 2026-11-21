#!/usr/bin/env bash
set -euo pipefail

# Docker replacement for the NV CCL-D torchrun/Slurm entry point. Invoke this
# once on each node with the same RUN directory and different NODE_RANK values.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python3}
TORCHRUN=${TORCHRUN:-torchrun}
RCCL_LIB=${RCCL_LIB:-$ROOT/build-ccld-gfx936-only/librccl.so.1.0}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-only}
WORKLOAD=${CCLD_WORKLOAD:-$ROOT/tools/tests/ccld_torch_collectives_rccl.py}
RUN=${CCLD_RUN:?set CCLD_RUN to a shared output directory}
NODE_RANK=${CCLD_NODE_RANK:?set CCLD_NODE_RANK to 0 or 1}
MASTER_ADDR=${CCLD_MASTER_ADDR:?set CCLD_MASTER_ADDR}
MASTER_PORT=${CCLD_MASTER_PORT:-29501}
NNODES=${CCLD_NNODES:-2}
NPROC_PER_NODE=${CCLD_NPROC_PER_NODE:-4}
OP=${CCLD_OP:-allreduce}
NUMEL=${CCLD_NUMEL:-262144}
WARMUP=${CCLD_WARMUP:-3}
ITERS=${CCLD_ITERS:-3}
GAP_MS=${CCLD_GAP_MS:-0}
ALGO=${CCLD_ALGO:-Ring}
PROTO=${CCLD_PROTO:-Simple}
FAULT=${CCLD_WORKLOAD_FAULT:-none}
FAULT_RANK=${CCLD_WORKLOAD_FAULT_RANK:-0}
FAULT_ITERATION=${CCLD_WORKLOAD_FAULT_ITERATION:-3}
SLEEP_MS=${CCLD_WORKLOAD_SLEEP_MS:-1000}
HANG_SECONDS=${CCLD_WORKLOAD_HANG_SECONDS:-20}
TIMEOUT_SECONDS=${CCLD_WORKLOAD_TIMEOUT_SECONDS:-90}

[[ -f "$RCCL_LIB" ]] || { echo "RCCL_LIB not found: $RCCL_LIB" >&2; exit 2; }
[[ -f "$WORKLOAD" ]] || { echo "workload not found: $WORKLOAD" >&2; exit 2; }
[[ -f "$RCCL_BUILD_DIR/include/rccl.h" ]] || { echo "RCCL headers not found" >&2; exit 2; }
[[ "$NODE_RANK" =~ ^[0-9]+$ && "$NODE_RANK" -lt "$NNODES" ]] || { echo "invalid CCLD_NODE_RANK" >&2; exit 2; }
mkdir -p "$RUN/bin" "$RUN/jsonl" "$RUN/workload" "$RUN/fault-events"
LIBDIR=$(dirname "$RCCL_LIB")
RUNTIME_LIBS="$LIBDIR:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/hyhal/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then RUNTIME_LIBS="$RUNTIME_LIBS:$LD_LIBRARY_PATH"; fi
export LD_LIBRARY_PATH="$RUNTIME_LIBS"
export LD_PRELOAD="$RCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
export NCCL_ALGO="$ALGO" NCCL_PROTO="$PROTO"
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1 NCCL_CCLD_WAIT_ENABLE=1
export NCCL_CCLD_POLL_US=200 NCCL_CCLD_FRAME_BLOCKS=32
export NCCL_CCLD_QUEUE_SLOTS=2048 NCCL_CCLD_COMPLETION=event
if [[ -n "${NCCL_CCLD_NODE_SOCKET:-}" ]]; then
  unset NCCL_CCLD_OUT_DIR
else
  export NCCL_CCLD_OUT_DIR="$RUN/bin"
fi
if [[ -n "${NCCL_CCLD_FAULT_KIND:-}" ]]; then export NCCL_CCLD_FAULT_KIND; fi
if [[ -n "${NCCL_CCLD_FAULT_RANK:-}" ]]; then export NCCL_CCLD_FAULT_RANK; fi
if [[ -n "${NCCL_CCLD_FAULT_OP:-}" ]]; then export NCCL_CCLD_FAULT_OP; fi
if [[ -n "${NCCL_CCLD_FAULT_REPEATS:-}" ]]; then export NCCL_CCLD_FAULT_REPEATS; fi
if [[ -n "${NCCL_CCLD_FAULT_CYCLES:-}" ]]; then export NCCL_CCLD_FAULT_CYCLES; fi
if [[ "$NODE_RANK" == 0 && -n "${NCCL_CCLD_FAULT_KIND:-}" ]]; then
  python3 "$ROOT/tools/tests/write_ccld_fault_config.py" "$RUN/fault-config.json"
fi

exec "$TORCHRUN" \
  --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$WORKLOAD" --op "$OP" --numel "$NUMEL" --warmup "$WARMUP" --iters "$ITERS" \
  --gap-ms "$GAP_MS" --fault "$FAULT" --fault-rank "$FAULT_RANK" \
  --fault-iteration "$FAULT_ITERATION" --sleep-ms "$SLEEP_MS" --hang-seconds "$HANG_SECONDS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --workload-log-dir "$RUN/workload" --fault-log-dir "$RUN/fault-events" \
  --expect-nccl-prefix "$RCCL_BUILD_DIR" --expect-nccl-version 22203
