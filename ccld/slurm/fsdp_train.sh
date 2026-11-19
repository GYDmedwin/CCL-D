#!/usr/bin/env bash
#SBATCH --job-name=ccld-fsdp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=00:10:00
set -euo pipefail

# Set CCLD_ROOT to an absolute path visible on all nodes; activate a PyTorch
# environment and select the site-specific account/partition and CPU request
# at submission. Eight GPU workers and NCCL/probe threads share the CPUs.
: "${CCLD_ROOT:?set CCLD_ROOT to the repository root}"
CCLD_ROOT=$(cd "$CCLD_ROOT" && pwd -P)
export CCLD_ROOT
export CCLD_NCCL_LIB="${CCLD_NCCL_LIB:-$CCLD_ROOT/build/lib}"
export CCLD_GPUS_PER_NODE="${CCLD_GPUS_PER_NODE:-8}"
export CCLD_PROFILE="${CCLD_PROFILE:-medium}"
export CCLD_STEPS="${CCLD_STEPS:-20}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CCLD_RUN="${CCLD_RUN:-$CCLD_ROOT/runs/fsdp-$SLURM_JOB_ID}"
export CCLD_TORCHRUN="$(command -v torchrun)"
mkdir -p "$(dirname "$CCLD_RUN")"
mkdir "$CCLD_RUN"
mkdir "$CCLD_RUN/bin" "$CCLD_RUN/jsonl" "$CCLD_RUN/workload"

export LD_LIBRARY_PATH="$CCLD_NCCL_LIB:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CCLD_NCCL_LIB/libnccl.so.2"
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1 NCCL_CCLD_WAIT_ENABLE=1
export NCCL_CCLD_POLL_US=200 NCCL_CCLD_FRAME_BLOCKS=512
export NCCL_CCLD_QUEUE_SLOTS=2048 NCCL_CCLD_COMPLETION=event
export NCCL_CCLD_OUT_DIR="$CCLD_RUN/bin"

decoder=''; analyzer=''
cleanup() {
  for pid in "$decoder" "$analyzer"; do [ -z "$pid" ] || kill "$pid" 2>/dev/null || true; done
  for pid in "$decoder" "$analyzer"; do [ -z "$pid" ] || wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
python3 -u "$CCLD_ROOT/ccld/tools/decode_async_wait.py" "$CCLD_RUN/bin" "$CCLD_RUN/jsonl" --follow &
decoder=$!
python3 -u "$CCLD_ROOT/ccld/analyzer/ccld_analyze.py" "$CCLD_RUN/jsonl" --follow \
  --baseline-ms 1 --baseline-rounds 20 --baseline-skip 2 --window-s 1 --repeats 2 --poll-s 0.2 \
  > "$CCLD_RUN/diagnosis.jsonl" &
analyzer=$!

export CCLD_MASTER_ADDR
CCLD_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export CCLD_MASTER_PORT
CCLD_MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpu-bind=none --gpu-bind=none \
  bash -c 'exec "$CCLD_TORCHRUN" --nnodes="$SLURM_NNODES" --nproc_per_node="$CCLD_GPUS_PER_NODE" \
    --node_rank="$SLURM_PROCID" --master_addr="$CCLD_MASTER_ADDR" --master_port="$CCLD_MASTER_PORT" \
    "$CCLD_ROOT/ccld/experiments/fsdp_train.py" --profile "$CCLD_PROFILE" --steps "$CCLD_STEPS" \
    --output-dir "$CCLD_RUN/workload" --expect-nccl-prefix "$CCLD_NCCL_LIB"'
kill "$decoder"; wait "$decoder"; decoder=''
sleep 1
kill "$analyzer"; wait "$analyzer" 2>/dev/null || true; analyzer=''
printf 'Training outputs: %s\n' "$CCLD_RUN"
