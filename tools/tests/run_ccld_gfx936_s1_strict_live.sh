#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NODE="$ROOT/tools/tests/run_ccld_gfx936_docker_node.sh"
DECODER=${CCLD_DECODER:-${CCLD_NV_ROOT:-$ROOT}/ccld/tools/decode_async_wait.py}
ANALYZER=${CCLD_ANALYZER:-${CCLD_NV_ROOT:-$ROOT}/ccld/analyzer/ccld_analyze.py}
CALIBRATOR="$ROOT/tools/tests/calibrate_ccld_baseline.py"
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-only}
RCCL_LIB=${RCCL_LIB:-$RCCL_BUILD_DIR/librccl.so.1.0}
RUN=${CCLD_RUN:-$ROOT/../results/ccld-eval51-s1-strict-$(date +%Y%m%d-%H%M%S)}
RANK=${CCLD_S1_RANK:-2}
WARMUP=${CCLD_WARMUP:-5}
ITERS=${CCLD_ITERS:-25}
SLEEP_MS=${CCLD_S1_SLEEP_MS:-3000}
HANG_S=${CCLD_HANG_S:-10}
WINDOW_S=${CCLD_WINDOW_S:-5}
REPEATS=${CCLD_ANALYZER_REPEATS:-2}
mkdir -p "$RUN/control/bin" "$RUN/control/jsonl" "$RUN/control/workload" "$RUN/control/fault-events" \
         "$RUN/fault/bin" "$RUN/fault/jsonl-live" "$RUN/fault/jsonl-final" "$RUN/fault/workload" "$RUN/fault/fault-events"
run_node() {
  local dir=$1 kind=$2
  RCCL_LIB="$RCCL_LIB" RCCL_BUILD_DIR="$RCCL_BUILD_DIR" CCLD_RUN="$dir" \
  CCLD_NODE_RANK=0 CCLD_NNODES=1 CCLD_NPROC_PER_NODE=4 CCLD_MASTER_ADDR=127.0.0.1 \
  CCLD_MASTER_PORT=${CCLD_MASTER_PORT:-29631} CCLD_ALGO=Ring CCLD_PROTO=Simple CCLD_OP=allreduce \
  CCLD_NUMEL=262144 CCLD_WARMUP="$WARMUP" CCLD_ITERS="$ITERS" \
  CCLD_WORKLOAD_FAULT="$kind" CCLD_WORKLOAD_FAULT_RANK="$RANK" \
  CCLD_WORKLOAD_FAULT_ITERATION="$WARMUP" CCLD_WORKLOAD_SLEEP_MS="$SLEEP_MS" \
  CCLD_WORKLOAD_TIMEOUT_SECONDS=120 "$NODE" >"$dir/workload.log" 2>&1
}
run_node "$RUN/control" none
python3 "$DECODER" "$RUN/control/bin" "$RUN/control/jsonl"
python3 "$CALIBRATOR" "$RUN/control/jsonl" --quantile "${CCLD_S1_BASELINE_QUANTILE:-0.95}" --margin "${CCLD_S1_BASELINE_MARGIN:-1.00}" > "$RUN/control/baseline.json"
BASE=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["baseline_ms"])' "$RUN/control/baseline.json")
python3 "$ANALYZER" "$RUN/control/jsonl" --baseline-ms "$BASE" --hang-s "$HANG_S" --window-s "$WINDOW_S" --repeats "$REPEATS" --flush-last-window > "$RUN/control/analysis.jsonl"
python3 - "$RUN/control/analysis.jsonl" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
confirmed=[x for x in rows if x.get('category')=='slow' and x.get('kind') in ('S1','S2','S3')]
print(json.dumps({'control_confirmed_slow':confirmed},sort_keys=True))
if confirmed: raise SystemExit('control produced confirmed slow diagnosis; baseline calibration rejected')
PY
python3 "$DECODER" "$RUN/fault/bin" "$RUN/fault/jsonl-live" --follow --interval-s 0.05 >"$RUN/fault/decoder-live.log" 2>&1 & decoder_pid=$!
python3 -u "$ANALYZER" "$RUN/fault/jsonl-live" --follow --poll-s 0.05 --fault-events "$RUN/fault/fault-events" \
  --baseline-ms "$BASE" --hang-s "$HANG_S" --window-s "$WINDOW_S" --repeats "$REPEATS" \
  >"$RUN/fault/analysis-live.jsonl" 2>"$RUN/fault/analyzer-live.err" & analyzer_pid=$!
cleanup(){ kill "$decoder_pid" "$analyzer_pid" 2>/dev/null || true; wait "$decoder_pid" "$analyzer_pid" 2>/dev/null || true; }
trap cleanup EXIT
set +e
RCCL_LIB="$RCCL_LIB" RCCL_BUILD_DIR="$RCCL_BUILD_DIR" CCLD_RUN="$RUN/fault" \
  CCLD_NODE_RANK=0 CCLD_NNODES=1 CCLD_NPROC_PER_NODE=4 CCLD_MASTER_ADDR=127.0.0.1 \
  CCLD_MASTER_PORT=${CCLD_MASTER_PORT:-29632} CCLD_ALGO=Ring CCLD_PROTO=Simple CCLD_OP=allreduce \
  CCLD_NUMEL=262144 CCLD_WARMUP="$WARMUP" CCLD_ITERS="$ITERS" \
  CCLD_WORKLOAD_FAULT=S1 CCLD_WORKLOAD_FAULT_RANK="$RANK" CCLD_WORKLOAD_FAULT_ITERATION="$WARMUP" \
  CCLD_WORKLOAD_SLEEP_MS="$SLEEP_MS" CCLD_WORKLOAD_TIMEOUT_SECONDS=120 \
  "$NODE" >"$RUN/fault/workload.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$RUN/fault/workload-exit-status"
sleep 1
cleanup
trap - EXIT
python3 "$DECODER" "$RUN/fault/bin" "$RUN/fault/jsonl-final"
python3 "$ANALYZER" "$RUN/fault/jsonl-final" --baseline-ms "$BASE" --hang-s "$HANG_S" \
  --window-s "$WINDOW_S" --repeats "$REPEATS" --flush-last-window > "$RUN/fault/analysis-final.jsonl"
python3 - "$RUN" "$RANK" "$BASE" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rank=int(sys.argv[2]); base=float(sys.argv[3])
rows=[json.loads(x) for x in (root/'fault/analysis-live.jsonl').read_text().splitlines() if x]
hits=[x for x in rows if x.get('category')=='slow' and x.get('kind')=='S1' and x.get('roots')==[rank]]
print(json.dumps({'baseline_ms':base,'live_s1_rank':rank,'matching_live_decisions':len(hits),'last':hits[-1] if hits else None,'run':str(root)},sort_keys=True))
if len(hits)<2: raise SystemExit('live analyzer did not repeatedly identify S1 rank')
PY
printf 'Strict S1 live artifacts: %s\n' "$RUN"
