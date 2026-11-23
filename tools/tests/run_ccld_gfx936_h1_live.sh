#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DECODER=${CCLD_DECODER:-${CCLD_NV_ROOT:-/workspace/nv/CCL-D}/ccld/tools/decode_async_wait.py}
ANALYZER=${CCLD_ANALYZER:-${CCLD_NV_ROOT:-/workspace/nv/CCL-D}/ccld/analyzer/ccld_analyze.py}
SINGLE="$ROOT/tools/tests/run_ccld_gfx936_docker_single_node.sh"
DUAL="$ROOT/tools/tests/run_ccld_gfx936_docker_2node.sh"
TOPOLOGY=${CCLD_TOPOLOGY:-single}
RANK=${CCLD_H1_RANK:-2}
RUN=${CCLD_RUN:-$ROOT/../results/ccld-h1-live-${TOPOLOGY}-r${RANK}-$(date +%Y%m%d-%H%M%S)}
CASE=Ring-Simple-allreduce

[[ "$TOPOLOGY" == single || "$TOPOLOGY" == dual ]] || { echo 'CCLD_TOPOLOGY must be single or dual' >&2; exit 2; }
WORLD=4; LAUNCHER="$SINGLE"
if [[ "$TOPOLOGY" == dual ]]; then WORLD=8; LAUNCHER="$DUAL"; fi
[[ "$RANK" =~ ^[0-9]+$ && "$RANK" -lt "$WORLD" ]] || { echo 'invalid H1 rank' >&2; exit 2; }
TRACE_ROOT="$RUN"
[[ "$TOPOLOGY" == dual ]] && TRACE_ROOT="$RUN/$CASE"
mkdir -p "$TRACE_ROOT/bin" "$TRACE_ROOT/jsonl" "$TRACE_ROOT/workload" "$TRACE_ROOT/fault-events"

python3 "$DECODER" "$TRACE_ROOT/bin" "$TRACE_ROOT/jsonl" --follow --interval-s 0.05 >"$TRACE_ROOT/decoder-live.log" 2>&1 & decoder_pid=$!
python3 -u "$ANALYZER" "$TRACE_ROOT/jsonl" --follow --poll-s 0.05 --fault-events "$TRACE_ROOT/fault-events" --baseline-ms 300 --hang-s 1 --window-s 0.2 --repeats 1 >"$TRACE_ROOT/analysis-live.jsonl" 2>"$TRACE_ROOT/analyzer-live.err" & analyzer_pid=$!
cleanup() { kill "$decoder_pid" "$analyzer_pid" 2>/dev/null || true; wait "$decoder_pid" "$analyzer_pid" 2>/dev/null || true; }
trap cleanup EXIT

export CCLD_RUN="$RUN" CCLD_CASE_FILTER="$CASE" CCLD_WORKLOAD_FAULT=H1 CCLD_WORKLOAD_FAULT_RANK="$RANK" CCLD_WORKLOAD_FAULT_ITERATION=3
export CCLD_WARMUP=3 CCLD_ITERS=3 CCLD_NUMEL=262144 CCLD_HANG_SECONDS=5 CCLD_TIMEOUT_SECONDS=15 CCLD_SKIP_POST_DECODE=1
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_WAIT_ENABLE=1
RCCL_LIB=${RCCL_LIB:-$ROOT/build-ccld-gfx936-only/librccl.so.1.0}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-only}
export RCCL_LIB RCCL_BUILD_DIR
"$LAUNCHER" >"$RUN/launcher.log" 2>&1 || true
sleep 0.5
cleanup
trap - EXIT
python3 "$DECODER" "$TRACE_ROOT/bin" "$TRACE_ROOT/jsonl-final" 2>/dev/null || true
python3 - "$TRACE_ROOT/analysis-live.jsonl" "$RANK" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
rank = int(sys.argv[2])
h = [x for x in rows if x.get('category') == 'hang' and x.get('kind') == 'H1' and x.get('roots') == [rank]]
if not h:
    raise SystemExit('live analyzer did not identify H1 target rank')
print(json.dumps({'h1': h[-1], 'count': len(h)}, sort_keys=True))
PY
printf 'H1 live artifacts: %s\n' "$RUN"
