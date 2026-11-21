#!/usr/bin/env bash
set -euo pipefail

# Run CCL-D's torch_collectives workload on two Docker hosts without Slurm.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NODE_SCRIPT="$ROOT/tools/tests/run_ccld_gfx936_docker_node.sh"
VALIDATOR="$ROOT/tools/tests/validate_ccld_torch_case.py"
DECODER=${CCLD_DECODER:-/workspace/nv/CCL-D/ccld/tools/decode_async_wait.py}
ANALYZER=${CCLD_ANALYZER:-/workspace/nv/CCL-D/ccld/analyzer/ccld_analyze.py}
RCCL_LIB=${RCCL_LIB:-$ROOT/build-ccld-gfx936-only/librccl.so.1.0}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-only}
MASTER_ADDR=${CCLD_MASTER_ADDR:-HPDPS-6}
MASTER_PORT=${CCLD_MASTER_PORT:-29571}
HOST1=${CCLD_NODE1:-HPDPS-6}
HOST2=${CCLD_NODE2:-HPDPS-7}
RUN=${CCLD_RUN:-$ROOT/../results/ccld-gfx936-2node-$(date +%Y%m%d-%H%M%S)}
WARMUP=${CCLD_WARMUP:-3}
ITERS=${CCLD_ITERS:-3}
NUMEL=${CCLD_NUMEL:-262144}
GAP_MS=${CCLD_GAP_MS:-0}
FAULT=${CCLD_WORKLOAD_FAULT:-none}
FAULT_RANK=${CCLD_WORKLOAD_FAULT_RANK:-0}
FAULT_ITERATION=${CCLD_WORKLOAD_FAULT_ITERATION:-3}
SLEEP_MS=${CCLD_WORKLOAD_SLEEP_MS:-1000}
HANG_SECONDS=${CCLD_WORKLOAD_HANG_SECONDS:-20}
TIMEOUT_SECONDS=${CCLD_WORKLOAD_TIMEOUT_SECONDS:-90}
CASE_FILTER=${CCLD_CASE_FILTER:-all}

[[ -f "$RCCL_LIB" ]] || { echo "RCCL_LIB not found: $RCCL_LIB" >&2; exit 2; }
[[ -f "$DECODER" && -f "$ANALYZER" ]] || { echo "NV CCL-D decoder/analyzer not found" >&2; exit 2; }
if [[ -n "${NCCL_CCLD_FAULT_KIND:-}" ]]; then
  grep -q '^CCLD_FAULT_INJECTION:BOOL=ON$' "$RCCL_BUILD_DIR/CMakeCache.txt" || { echo "fault-enabled RCCL build required" >&2; exit 2; }
fi
mkdir -p "$RUN"
trap '[[ -z "${remote_pid:-}" ]] || kill "$remote_pid" 2>/dev/null || true' EXIT

run_case() {
  local algorithm=$1 protocol=$2 op=$3 case_name="$1-$2-$3"
  if [[ "$CASE_FILTER" != all && ",$CASE_FILTER," != *",$case_name,"* ]]; then return 0; fi
  local case_run="$RUN/$case_name"
  local fault_env=''
  if [[ -n "${NCCL_CCLD_FAULT_KIND:-}" ]]; then
    printf -v fault_env ' NCCL_CCLD_FAULT_KIND=%q NCCL_CCLD_FAULT_RANK=%q NCCL_CCLD_FAULT_OP=%q NCCL_CCLD_FAULT_REPEATS=%q NCCL_CCLD_FAULT_CYCLES=%q' \
      "$NCCL_CCLD_FAULT_KIND" "${NCCL_CCLD_FAULT_RANK:-0}" "${NCCL_CCLD_FAULT_OP:-1}" \
      "${NCCL_CCLD_FAULT_REPEATS:-1}" "${NCCL_CCLD_FAULT_CYCLES:-50000000}"
  fi
  mkdir -p "$case_run/bin" "$case_run/jsonl" "$case_run/workload" "$case_run/fault-events"
  local remote_command
  printf -v remote_command 'env RCCL_LIB=%q RCCL_BUILD_DIR=%q CCLD_RUN=%q CCLD_NODE_RANK=1 CCLD_NNODES=2 CCLD_NPROC_PER_NODE=4 CCLD_MASTER_ADDR=%q CCLD_MASTER_PORT=%q CCLD_ALGO=%q CCLD_PROTO=%q CCLD_OP=%q CCLD_NUMEL=%q CCLD_WARMUP=%q CCLD_ITERS=%q CCLD_GAP_MS=%q NCCL_CCLD_ENABLE=%q NCCL_CCLD_WAIT_ENABLE=%q NCCL_CCLD_POLL_US=%q NCCL_CCLD_FRAME_BLOCKS=%q CCLD_WORKLOAD_FAULT=%q CCLD_WORKLOAD_FAULT_RANK=%q CCLD_WORKLOAD_FAULT_ITERATION=%q CCLD_WORKLOAD_SLEEP_MS=%q CCLD_WORKLOAD_HANG_SECONDS=%q CCLD_WORKLOAD_TIMEOUT_SECONDS=%q%s bash %q' \
    "$RCCL_LIB" "$RCCL_BUILD_DIR" "$case_run" "$MASTER_ADDR" "$MASTER_PORT" \
    "$algorithm" "$protocol" "$op" "$NUMEL" "$WARMUP" "$ITERS" "$GAP_MS" \
    "${NCCL_CCLD_ENABLE:-1}" "${NCCL_CCLD_WAIT_ENABLE:-1}" "${NCCL_CCLD_POLL_US:-200}" "${NCCL_CCLD_FRAME_BLOCKS:-32}" \
    "$FAULT" "$FAULT_RANK" "$FAULT_ITERATION" "$SLEEP_MS" "$HANG_SECONDS" "$TIMEOUT_SECONDS" "$fault_env" "$NODE_SCRIPT"
  ssh -o BatchMode=yes "$HOST2" "$remote_command" >"$case_run/node1.log" 2>&1 &
  remote_pid=$!
  RCCL_LIB="$RCCL_LIB" RCCL_BUILD_DIR="$RCCL_BUILD_DIR" \
    CCLD_RUN="$case_run" CCLD_NODE_RANK=0 CCLD_NNODES=2 CCLD_NPROC_PER_NODE=4 \
    CCLD_MASTER_ADDR="$MASTER_ADDR" CCLD_MASTER_PORT="$MASTER_PORT" \
    CCLD_ALGO="$algorithm" CCLD_PROTO="$protocol" CCLD_OP="$op" \
    CCLD_NUMEL="$NUMEL" CCLD_WARMUP="$WARMUP" CCLD_ITERS="$ITERS" CCLD_GAP_MS="$GAP_MS" \
    NCCL_CCLD_ENABLE="${NCCL_CCLD_ENABLE:-1}" NCCL_CCLD_WAIT_ENABLE="${NCCL_CCLD_WAIT_ENABLE:-1}" \
    NCCL_CCLD_POLL_US="${NCCL_CCLD_POLL_US:-200}" NCCL_CCLD_FRAME_BLOCKS="${NCCL_CCLD_FRAME_BLOCKS:-32}" \
    CCLD_WORKLOAD_FAULT="$FAULT" CCLD_WORKLOAD_FAULT_RANK="$FAULT_RANK" \
    CCLD_WORKLOAD_FAULT_ITERATION="$FAULT_ITERATION" CCLD_WORKLOAD_SLEEP_MS="$SLEEP_MS" \
    CCLD_WORKLOAD_HANG_SECONDS="$HANG_SECONDS" CCLD_WORKLOAD_TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
    "$NODE_SCRIPT" >"$case_run/node0.log" 2>&1
  wait "$remote_pid"
  remote_pid=''
  if ! rg -q 'NET/IB[ :]' "$case_run/node0.log" || ! rg -q 'NET/IB[ :]' "$case_run/node1.log"; then
    echo "RCCL logs did not confirm IB on both nodes for $case_name" >&2
    exit 1
  fi
  if [[ "${NCCL_CCLD_ENABLE:-1}" == 1 ]]; then
    python3 "$DECODER" "$case_run/bin" "$case_run/jsonl"
    python3 "$VALIDATOR" "$case_run" --world-size 8 --warmup "$WARMUP" --iters "$ITERS" \
      --op "$op" --algorithm "$algorithm" --protocol "$protocol"
  fi
  if [[ -n "${CCLD_BASELINE_MS:-}" ]]; then
    python3 "$ANALYZER" "$case_run/jsonl" --baseline-ms "$CCLD_BASELINE_MS" \
      --flush-last-window >"$case_run/analysis.jsonl"
  fi
  if [[ -f "$case_run/analysis.jsonl" ]]; then python3 - "$case_run/analysis.jsonl" <<'PY'
import collections, json, pathlib, sys
rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line]
counts = collections.Counter((row.get("category"), row.get("kind")) for row in rows)
print("NV CCL-D analyzer output:", dict(sorted(counts.items())))
PY
  fi
  printf '%s\n' "case=$case_name ranks=8 hosts=$HOST1,$HOST2 RCCL_IB=confirmed artifacts=$case_run"
}

# Matrix mirrors RESULTS.md's two-node Ring/Tree collective checks. gfx936
# does not support LL128, so only Simple and LL are scheduled.
for protocol in Simple LL; do
  for op in allreduce allgather reducescatter; do run_case Ring "$protocol" "$op"; done
  run_case Tree "$protocol" allreduce
done
# NV records AllToAll as a numerical-only case: its grouped P2P implementation
# produced no AllToAll entry/completion telemetry in the reference setup.
run_case Ring Simple alltoall
printf 'Two-node CCL-D test artifacts: %s\n' "$RUN"
