#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NODE_SCRIPT="$ROOT/tools/tests/run_ccld_gfx936_docker_node.sh"
VALIDATOR="$ROOT/tools/tests/validate_ccld_torch_case.py"
DECODER=${CCLD_DECODER:-/workspace/nv/CCL-D/ccld/tools/decode_async_wait.py}
ANALYZER=${CCLD_ANALYZER:-/workspace/nv/CCL-D/ccld/analyzer/ccld_analyze.py}
RCCL_LIB=${RCCL_LIB:-$ROOT/build-ccld-gfx936-only/librccl.so.1.0}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-only}
RUN=${CCLD_RUN:-$ROOT/../results/ccld-gfx936-single-$(date +%Y%m%d-%H%M%S)}
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

[[ -f "$RCCL_LIB" && -f "$DECODER" && -f "$ANALYZER" ]] || { echo "RCCL/NV CCL-D inputs are missing" >&2; exit 2; }
mkdir -p "$RUN"

run_case() {
  local algorithm=$1 protocol=$2 op=$3 case_name="$1-$2-$3"
  if [[ "$CASE_FILTER" != all && ",$CASE_FILTER," != *",$case_name,"* ]]; then return 0; fi
  local case_run="$RUN/$case_name"
  mkdir -p "$case_run/bin" "$case_run/jsonl" "$case_run/workload" "$case_run/fault-events"
  RCCL_LIB="$RCCL_LIB" RCCL_BUILD_DIR="$RCCL_BUILD_DIR" \
    CCLD_RUN="$case_run" CCLD_NODE_RANK=0 CCLD_NNODES=1 CCLD_NPROC_PER_NODE=4 \
    CCLD_MASTER_ADDR=127.0.0.1 CCLD_MASTER_PORT=${CCLD_MASTER_PORT:-29571} \
    CCLD_ALGO="$algorithm" CCLD_PROTO="$protocol" CCLD_OP="$op" \
    CCLD_NUMEL="$NUMEL" CCLD_WARMUP="$WARMUP" CCLD_ITERS="$ITERS" CCLD_GAP_MS="$GAP_MS" \
    CCLD_WORKLOAD_FAULT="$FAULT" CCLD_WORKLOAD_FAULT_RANK="$FAULT_RANK" \
    CCLD_WORKLOAD_FAULT_ITERATION="$FAULT_ITERATION" CCLD_WORKLOAD_SLEEP_MS="$SLEEP_MS" \
    CCLD_WORKLOAD_HANG_SECONDS="$HANG_SECONDS" CCLD_WORKLOAD_TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
    "$NODE_SCRIPT" >"$case_run/torchrun.log" 2>&1
  python3 "$DECODER" "$case_run/bin" "$case_run/jsonl"
  python3 "$VALIDATOR" "$case_run" --world-size 4 --warmup "$WARMUP" --iters "$ITERS" \
    --op "$op" --algorithm "$algorithm" --protocol "$protocol"
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
  printf '%s\n' "case=$case_name ranks=4 host=$(hostname -s) artifacts=$case_run"
}

# Same NV collective cases as the two-node matrix; gfx936 excludes LL128.
for protocol in Simple LL; do
  for op in allreduce allgather reducescatter; do run_case Ring "$protocol" "$op"; done
  run_case Tree "$protocol" allreduce
done
run_case Ring Simple alltoall
printf 'Single-node CCL-D test artifacts: %s\n' "$RUN"
