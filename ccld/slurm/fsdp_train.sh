#!/usr/bin/env bash
#SBATCH --job-name=ccld-fsdp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=00:10:00
set -euo pipefail

# Set CCLD_ROOT to an absolute path visible on all training nodes; activate a PyTorch
# environment and select the site-specific account/partition and CPU request
# at submission. Eight GPU workers and NCCL/probe threads share the CPUs.
: "${CCLD_ROOT:?set CCLD_ROOT to the repository root}"
CCLD_ROOT=$(cd "$CCLD_ROOT" && pwd -P)
export CCLD_ROOT
cd "$CCLD_ROOT"
export CCLD_NCCL_LIB="${CCLD_NCCL_LIB:-$CCLD_ROOT/build/lib}"
export CCLD_GPUS_PER_NODE="${CCLD_GPUS_PER_NODE:-8}"
export CCLD_PROFILE="${CCLD_PROFILE:-medium}"
export CCLD_STEPS="${CCLD_STEPS:-20}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CCLD_RUN="${CCLD_RUN:-$CCLD_ROOT/runs/fsdp-$SLURM_JOB_ID}"
export CCLD_TORCHRUN="$(command -v torchrun)"
export CCLD_PULL_RECORDS="${CCLD_PULL_RECORDS:-65536}"
mkdir -p "$(dirname "$CCLD_RUN")"
mkdir "$CCLD_RUN"
mkdir "$CCLD_RUN/workload"

source "$CCLD_ROOT/ccld/slurm/live_job.sh"
ccld_live_start --baseline-ms 1 --baseline-rounds 20 --baseline-skip 2 \
  --window-s 1 --repeats 2 --poll-s "${CCLD_PULL_POLL_S:-0.2}" \
  --decode-workers "${CCLD_DECODE_WORKERS:-1}"

export LD_LIBRARY_PATH="$CCLD_NCCL_LIB:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CCLD_NCCL_LIB/libnccl.so.2"
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1 NCCL_CCLD_WAIT_ENABLE=1
export NCCL_CCLD_POLL_US=200 NCCL_CCLD_FRAME_BLOCKS="${NCCL_CCLD_FRAME_BLOCKS:-32}"
export NCCL_CCLD_QUEUE_SLOTS=2048 NCCL_CCLD_COMPLETION=event

export CCLD_MASTER_PORT
CCLD_MASTER_PORT=$("$CCLD_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpu-bind=none --gpu-bind=none \
  bash -c 'exec "$CCLD_TORCHRUN" --nnodes="$SLURM_NNODES" --nproc_per_node="$CCLD_GPUS_PER_NODE" \
    --node_rank="$SLURM_PROCID" --master_addr="$CCLD_MASTER_ADDR" --master_port="$CCLD_MASTER_PORT" \
    "$CCLD_ROOT/ccld/experiments/fsdp_train.py" --profile "$CCLD_PROFILE" --steps "$CCLD_STEPS" \
    --output-dir "$CCLD_RUN/workload" --expect-nccl-prefix "$CCLD_NCCL_LIB"'
ccld_live_stop
printf 'Training outputs: %s\n' "$CCLD_RUN"
