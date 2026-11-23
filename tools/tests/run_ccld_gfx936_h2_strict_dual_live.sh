#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DECODER=${CCLD_DECODER:-${CCLD_NV_ROOT:-$ROOT}/ccld/tools/decode_async_wait.py}
ANALYZER=${CCLD_ANALYZER:-${CCLD_NV_ROOT:-$ROOT}/ccld/analyzer/ccld_analyze.py}
LAUNCHER="$ROOT/tools/tests/run_ccld_gfx936_docker_2node.sh"
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-fault}
RCCL_LIB=${RCCL_LIB:-$RCCL_BUILD_DIR/librccl.so.1.0}
RUN=${CCLD_RUN:-$ROOT/../results/ccld-h2-strict-dual-live-$(date +%Y%m%d-%H%M%S)}
CASE=Ring-Simple-allreduce
mkdir -p "$RUN/$CASE/bin" "$RUN/$CASE/jsonl" "$RUN/$CASE/workload" "$RUN/$CASE/fault-events"
python3 "$DECODER" "$RUN/$CASE/bin" "$RUN/$CASE/jsonl" --follow --interval-s 0.05 >"$RUN/$CASE/decoder-live.log" 2>&1 & decoder_pid=$!
python3 -u "$ANALYZER" "$RUN/$CASE/jsonl" --follow --poll-s 0.05 --fault-events "$RUN/$CASE/fault-events" --baseline-ms 300 --hang-s 1 --window-s 0.2 --repeats 1 >"$RUN/$CASE/analysis-live.jsonl" 2>"$RUN/$CASE/analyzer-live.err" & analyzer_pid=$!
cleanup() { kill "$decoder_pid" "$analyzer_pid" 2>/dev/null || true; wait "$decoder_pid" "$analyzer_pid" 2>/dev/null || true; }
trap cleanup EXIT
set +e
CCLD_SKIP_POST_DECODE=1 CCLD_RUN="$RUN" CCLD_CASE_FILTER="$CASE" CCLD_WORKLOAD_FAULT=H2_SIM CCLD_WORKLOAD_FAULT_RANK=2 CCLD_WORKLOAD_FAULT_ITERATION=3 CCLD_WORKLOAD_H2_DELTA_BYTES=1024 CCLD_WORKLOAD_H2_COUNT_DIVISOR=0 CCLD_H2_SIM_STALL_RANKS=0,1,3,4,5,6,7 CCLD_H2_SIM_CYCLES=${CCLD_H2_SIM_CYCLES:-30000000000} CCLD_WARMUP=3 CCLD_ITERS=3 CCLD_TIMEOUT_SECONDS=30 RCCL_BUILD_DIR="$RCCL_BUILD_DIR" RCCL_LIB="$RCCL_LIB" NCCL_CCLD_FAULT_KIND=H3 NCCL_CCLD_FAULT_RANK=0 NCCL_CCLD_FAULT_OP=4 NCCL_CCLD_FAULT_REPEATS=1 NCCL_CCLD_FAULT_CYCLES=${CCLD_H3_CYCLES:-30000000000} "$LAUNCHER" >"$RUN/launcher.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$RUN/workload-exit-status"
sleep 1
cleanup
trap - EXIT
python3 - "$RUN/$CASE/analysis-live.jsonl" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
h=[x for x in rows if x.get('category')=='hang' and x.get('kind')=='H2' and 2 in x.get('roots',[])]
print(json.dumps({'live_h2_rank2':h[-1] if h else None},sort_keys=True))
if not h: raise SystemExit('dual live analyzer did not identify H2 rank 2')
PY
printf 'Dual H2 strict live artifacts: %s\n' "$RUN"
