#!/usr/bin/env bash
#SBATCH --job-name=ccld-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:10:00
set -euo pipefail

# Submit from the repository root or set CCLD_ROOT explicitly. Set the
# partition/account and GPU resource syntax required by your Slurm site.
root="${CCLD_ROOT:-${SLURM_SUBMIT_DIR:?}}"
build="${CCLD_NCCL_BUILD:-$root/build}"
run="${CCLD_RUN_DIR:-$root/runs/$SLURM_JOB_ID}"
test -d "$build/lib"
mkdir -p "$run/bin" "$run/jsonl"
export LD_LIBRARY_PATH="$build/lib:${LD_LIBRARY_PATH:-}"
export NCCL_ALGO=Ring NCCL_PROTO="${NCCL_PROTO:-Simple}"
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1 NCCL_CCLD_WAIT_ENABLE=1
export NCCL_CCLD_POLL_US=200 NCCL_CCLD_FRAME_BLOCKS=512 NCCL_CCLD_QUEUE_SLOTS=2048
export NCCL_CCLD_COMPLETION=event NCCL_CCLD_OUT_DIR="$run/bin"
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  "$root/ccld/experiments/torch_collectives.py" --op allreduce \
  --expect-nccl-prefix "$build" --expect-nccl-version 22403
python3 "$root/ccld/tools/decode_async_wait.py" "$run/bin" "$run/jsonl"
printf 'Decoded JSONL traces: %s\n' "$run/jsonl"
