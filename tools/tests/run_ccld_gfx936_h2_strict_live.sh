#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NODE_SCRIPT="$ROOT/tools/tests/run_ccld_gfx936_docker_node.sh"
DECODER=${CCLD_DECODER:-/workspace/nv/CCL-D/ccld/tools/decode_async_wait.py}
ANALYZER=${CCLD_ANALYZER:-/workspace/nv/CCL-D/ccld/analyzer/ccld_analyze.py}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-fault}
RCCL_LIB=${RCCL_LIB:-$RCCL_BUILD_DIR/librccl.so.1.0}
RUN=${CCLD_RUN:-$ROOT/../results/ccld-h2-strict-live-$(date +%Y%m%d-%H%M%S)}
HANG_S=${CCLD_HANG_S:-1}
DUMP_MS=${NCCL_CCLD_DUMP_MS:-10}
mkdir -p "$RUN/bin" "$RUN/jsonl-live" "$RUN/jsonl-final" "$RUN/workload" "$RUN/fault-events"
grep -q '^CCLD_FAULT_INJECTION:BOOL=ON$' "$RCCL_BUILD_DIR/CMakeCache.txt"
python3 "$DECODER" "$RUN/bin" "$RUN/jsonl-live" --follow --interval-s 0.05 >"$RUN/decoder-live.log" 2>&1 & decoder_pid=$!
python3 -u "$ANALYZER" "$RUN/jsonl-live" --follow --poll-s 0.05 --fault-events "$RUN/fault-events" --baseline-ms 300 --hang-s "$HANG_S" --window-s 0.2 --repeats 1 >"$RUN/analysis-live.jsonl" 2>"$RUN/analyzer-live.err" & analyzer_pid=$!
cleanup() { kill "$decoder_pid" "$analyzer_pid" 2>/dev/null || true; wait "$decoder_pid" "$analyzer_pid" 2>/dev/null || true; }
trap cleanup EXIT
set +e
RCCL_LIB="$RCCL_LIB" RCCL_BUILD_DIR="$RCCL_BUILD_DIR" CCLD_RUN="$RUN" CCLD_NODE_RANK=0 CCLD_NNODES=1 CCLD_NPROC_PER_NODE=4 CCLD_MASTER_ADDR=127.0.0.1 CCLD_MASTER_PORT=${CCLD_MASTER_PORT:-29611} CCLD_ALGO=Ring CCLD_PROTO=Simple CCLD_OP=allreduce CCLD_NUMEL=262144 CCLD_WARMUP=3 CCLD_ITERS=3 CCLD_WORKLOAD_FAULT=H2_SIM CCLD_WORKLOAD_FAULT_RANK=2 CCLD_WORKLOAD_FAULT_ITERATION=3 CCLD_WORKLOAD_H2_DELTA_BYTES=1024 CCLD_WORKLOAD_H2_COUNT_DIVISOR=0 CCLD_WORKLOAD_H2_MISMATCH=count CCLD_H2_SIM_STALL_RANKS=0,1,3 CCLD_H2_SIM_CYCLES=${CCLD_H2_SIM_CYCLES:-30000000000} CCLD_WORKLOAD_TIMEOUT_SECONDS=30 NCCL_CCLD_FAULT_KIND=H3 NCCL_CCLD_FAULT_RANK=0 NCCL_CCLD_FAULT_OP=4 NCCL_CCLD_FAULT_REPEATS=1 NCCL_CCLD_FAULT_CYCLES=${CCLD_H3_CYCLES:-30000000000} NCCL_CCLD_DUMP_MS="$DUMP_MS" "$NODE_SCRIPT" >"$RUN/workload.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$RUN/workload-exit-status"
sleep 1
cleanup
trap - EXIT
python3 "$DECODER" "$RUN/bin" "$RUN/jsonl-final"
python3 "$ANALYZER" "$RUN/jsonl-final" --baseline-ms 300 --hang-s "$HANG_S" --window-s 0.2 --repeats 1 --flush-last-window >"$RUN/analysis-final.jsonl"
python3 - "$RUN" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); rows=[json.loads(x) for x in (p/'analysis-live.jsonl').read_text().splitlines() if x]
h2=[x for x in rows if x.get('category')=='hang' and x.get('kind')=='H2' and 2 in x.get('roots',[])]
print(json.dumps({'live_h2_rank2':h2[-1] if h2 else None,'workload_status':(p/'workload-exit-status').read_text().strip(),'run':str(p)},sort_keys=True))
if not h2: raise SystemExit('live analyzer did not identify H2 rank 2')
PY
