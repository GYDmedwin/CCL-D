#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SINGLE="$ROOT/tools/tests/run_ccld_gfx936_docker_single_node_matrix.sh"
DUAL="$ROOT/tools/tests/run_ccld_gfx936_docker_2node.sh"
NODE_DECODER=${CCLD_DECODER:-${CCLD_NV_ROOT:-$ROOT}/ccld/tools/decode_async_wait.py}
ANALYZER=${CCLD_ANALYZER:-${CCLD_NV_ROOT:-$ROOT}/ccld/analyzer/ccld_analyze.py}
ADAPTER="$ROOT/tools/tests/ccld_gfx936_online_ring_wait.py"
S2_GATE="$ROOT/tools/tests/validate_ccld_s2_detected.py"
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-fault}
RCCL_LIB=${RCCL_LIB:-$RCCL_BUILD_DIR/librccl.so.1.0}
TOPOLOGY=${CCLD_TOPOLOGY:-single}
PROTO=${CCLD_PROTO:-Simple}
FAULT_RANK=${CCLD_S2_RANK:-1}
FAULT_OP=${CCLD_S2_OP:-4}
FAULT_REPEATS=${CCLD_S2_REPEATS:-3}
FAULT_CYCLES=${CCLD_S2_CYCLES:-50000000}
WARMUP=${CCLD_WARMUP:-3}
ITERS=${CCLD_ITERS:-13}
NUMEL=${CCLD_NUMEL:-67108864}
THRESHOLD=${CCLD_WAIT_THRESHOLD:?set calibrated CCLD_WAIT_THRESHOLD for topology/protocol}
BASELINE_MS=${CCLD_BASELINE_MS:-1}
WINDOW_S=${CCLD_WINDOW_S:-0.5}
RUN=${CCLD_RUN:-$ROOT/../results/ccld-p9-live-${TOPOLOGY}-${PROTO,,}-rank${FAULT_RANK}-$(date +%Y%m%d-%H%M%S)}
CASE="Ring-${PROTO}-allreduce"

[[ -f "$RCCL_LIB" && -f "$NODE_DECODER" && -f "$ANALYZER" && -f "$ADAPTER" ]] || { echo "RCCL/NV CCL-D inputs are missing" >&2; exit 2; }
grep -q '^CCLD_FAULT_INJECTION:BOOL=ON$' "$RCCL_BUILD_DIR/CMakeCache.txt" || { echo "fault-enabled RCCL build required" >&2; exit 2; }
[[ "$TOPOLOGY" == single || "$TOPOLOGY" == dual ]] || { echo "CCLD_TOPOLOGY must be single or dual" >&2; exit 2; }
WORLD_SIZE=4
[[ "$TOPOLOGY" == dual ]] && WORLD_SIZE=8
[[ "$FAULT_RANK" =~ ^[0-9]+$ && "$FAULT_RANK" -lt "$WORLD_SIZE" ]] || {
  echo "CCLD_S2_RANK must be in [0,$((WORLD_SIZE - 1))] for $TOPOLOGY topology" >&2; exit 2;
}
mkdir -p "$RUN/$CASE/bin" "$RUN/$CASE/jsonl" "$RUN/$CASE/jsonl-final" "$RUN/$CASE/workload" "$RUN/$CASE/fault-events"
TRACE="$RUN/$CASE/jsonl"

python3 "$NODE_DECODER" "$RUN/$CASE/bin" "$TRACE" --follow --interval-s 0.05 \
  >"$RUN/$CASE/decoder-live.log" 2>&1 &
decoder_pid=$!
python3 -u "$ADAPTER" "$TRACE" --protocol "${PROTO^^}" \
  --world-size "$WORLD_SIZE" \
  --threshold "$THRESHOLD" --interval-s 0.1 --baseline-ms "$BASELINE_MS" \
  --window-s "$WINDOW_S" --repeats 2 --hang-s 10 \
  >"$RUN/$CASE/adapter-live.jsonl" 2>"$RUN/$CASE/adapter-live.err" &
adapter_pid=$!
cleanup() {
  kill "$decoder_pid" "$adapter_pid" 2>/dev/null || true
  wait "$decoder_pid" "$adapter_pid" 2>/dev/null || true
}
trap cleanup EXIT

export CCLD_RUN="$RUN" CCLD_CASE_FILTER="$CASE" CCLD_PROTO="$PROTO" CCLD_GAP_MS="${CCLD_GAP_MS:-1200}"
export CCLD_WARMUP="$WARMUP" CCLD_ITERS="$ITERS" CCLD_NUMEL="$NUMEL"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-262144}"
export CCLD_SKIP_POST_DECODE=1 CCLD_BASELINE_MS="$BASELINE_MS" CCLD_HANG_S=10
export NCCL_CCLD_FAULT_KIND=S2 NCCL_CCLD_FAULT_RANK="$FAULT_RANK"
export NCCL_CCLD_FAULT_OP="$FAULT_OP" NCCL_CCLD_FAULT_REPEATS="$FAULT_REPEATS" NCCL_CCLD_FAULT_CYCLES="$FAULT_CYCLES"
export RCCL_LIB RCCL_BUILD_DIR
if [[ "$TOPOLOGY" == single ]]; then
  "$SINGLE" >"$RUN/launcher.log" 2>&1
else
  "$DUAL" >"$RUN/launcher.log" 2>&1
fi
for _ in $(seq 1 80); do
  if [[ -s "$RUN/$CASE/adapter-live.jsonl" ]] && grep -q '"kind": "three_round_decision"' "$RUN/$CASE/adapter-live.jsonl"; then break; fi
  sleep 0.25
done
sleep 0.5
cleanup
trap - EXIT
python3 "$NODE_DECODER" "$RUN/$CASE/bin" "$RUN/$CASE/jsonl-final"
python3 "$ANALYZER" "$RUN/$CASE/jsonl-final" --baseline-ms "${CCLD_FINAL_BASELINE_MS:-1}" \
  --hang-s 10 --window-s "$WINDOW_S" --repeats 2 --flush-last-window \
  >"$RUN/$CASE/analysis-final.jsonl"
python3 "$S2_GATE" "$RUN/$CASE/analysis-final.jsonl" "$RUN/$CASE/fault-config.json" --target-rank "$FAULT_RANK"
python3 - "$RUN/$CASE/adapter-live.jsonl" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
decisions=[x for x in rows if x.get('category')=='online_wait' and x.get('kind')=='three_round_decision']
print(json.dumps({'online_wait_decision':decisions[-1] if decisions else None}, sort_keys=True))
PY
printf 'Ring-wait live artifacts: %s\n' "$RUN/$CASE"
