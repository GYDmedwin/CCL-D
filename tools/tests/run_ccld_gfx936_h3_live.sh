#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NODE_SCRIPT="$ROOT/tools/tests/run_ccld_gfx936_docker_node.sh"
DECODER=${CCLD_DECODER:-/workspace/nv/CCL-D/ccld/tools/decode_async_wait.py}
ANALYZER=${CCLD_ANALYZER:-/workspace/nv/CCL-D/ccld/analyzer/ccld_analyze.py}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-fault}
RCCL_LIB=${RCCL_LIB:-$RCCL_BUILD_DIR/librccl.so.1.0}
RUN=${CCLD_RUN:-$ROOT/../results/ccld-h3-live-$(date +%Y%m%d-%H%M%S)}
FAULT_RANK=${CCLD_H3_RANK:-1}
FAULT_OP=${CCLD_H3_OP:-4}
FAULT_REPEATS=${CCLD_H3_REPEATS:-1}
FAULT_CYCLES=${CCLD_H3_CYCLES:-30000000000}
HANG_S=${CCLD_HANG_S:-1}
BASELINE_MS=${CCLD_BASELINE_MS:-1000}
WINDOW_S=${CCLD_WINDOW_S:-0.2}
REPEATS=${CCLD_ANALYZER_REPEATS:-1}

[[ -f "$RCCL_LIB" && -f "$DECODER" && -f "$ANALYZER" ]] || { echo "RCCL or NV CCL-D tool is missing" >&2; exit 2; }
grep -q '^CCLD_FAULT_INJECTION:BOOL=ON$' "$RCCL_BUILD_DIR/CMakeCache.txt" || {
  echo "RCCL_BUILD_DIR must use CCLD_FAULT_INJECTION=ON" >&2; exit 2;
}
mkdir -p "$RUN/bin" "$RUN/jsonl-live" "$RUN/jsonl-final" "$RUN/workload" "$RUN/fault-events"

python3 "$DECODER" "$RUN/bin" "$RUN/jsonl-live" --follow --interval-s 0.05 \
  >"$RUN/decoder-live.log" 2>&1 &
decoder_pid=$!
python3 -u "$ANALYZER" "$RUN/jsonl-live" --follow --poll-s 0.05 \
  --fault-events "$RUN/fault-events" --baseline-ms "$BASELINE_MS" \
  --hang-s "$HANG_S" --window-s "$WINDOW_S" --repeats "$REPEATS" \
  >"$RUN/analysis-live.jsonl" 2>"$RUN/analyzer-live.err" &
analyzer_pid=$!
cleanup() {
  kill "$decoder_pid" "$analyzer_pid" 2>/dev/null || true
  wait "$decoder_pid" "$analyzer_pid" 2>/dev/null || true
}
trap cleanup EXIT

set +e
RCCL_LIB="$RCCL_LIB" RCCL_BUILD_DIR="$RCCL_BUILD_DIR" CCLD_RUN="$RUN" \
  CCLD_NODE_RANK=0 CCLD_NNODES=1 CCLD_NPROC_PER_NODE=4 \
  CCLD_MASTER_ADDR=127.0.0.1 CCLD_MASTER_PORT="${CCLD_MASTER_PORT:-29601}" \
  CCLD_ALGO=Ring CCLD_PROTO=Simple CCLD_OP=allreduce \
  CCLD_NUMEL="${CCLD_NUMEL:-262144}" CCLD_WARMUP=3 CCLD_ITERS=3 \
  CCLD_WORKLOAD_FAULT=none CCLD_WORKLOAD_TIMEOUT_SECONDS="${CCLD_WORKLOAD_TIMEOUT_SECONDS:-90}" \
  NCCL_CCLD_FAULT_KIND=H3 NCCL_CCLD_FAULT_RANK="$FAULT_RANK" \
  NCCL_CCLD_FAULT_OP="$FAULT_OP" NCCL_CCLD_FAULT_REPEATS="$FAULT_REPEATS" \
  NCCL_CCLD_FAULT_CYCLES="$FAULT_CYCLES" \
  "$NODE_SCRIPT" >"$RUN/workload.log" 2>&1
workload_status=$?
set -e
printf '%s\n' "$workload_status" > "$RUN/workload-exit-status"
sleep 1
cleanup
trap - EXIT

python3 "$DECODER" "$RUN/bin" "$RUN/jsonl-final"
python3 "$ANALYZER" "$RUN/jsonl-final" --baseline-ms "$BASELINE_MS" \
  --hang-s "$HANG_S" --window-s "$WINDOW_S" --repeats "$REPEATS" \
  --flush-last-window >"$RUN/analysis-final.jsonl"
python3 - "$RUN" "$FAULT_RANK" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
target = int(sys.argv[2])
live = [json.loads(line) for line in (run / 'analysis-live.jsonl').read_text().splitlines() if line]
h3 = [row for row in live if row.get('category') == 'hang' and row.get('kind') == 'H3' and target in row.get('roots', [])]
if not h3:
    raise SystemExit(f'live NV analyzer did not identify H3 rank {target}; see {run / "analysis-live.jsonl"}')
manifest = json.loads((run / 'fault-config.json').read_text())
assert manifest['fault_kind'] == 'H3' and manifest['fault_rank'] == target, manifest
assert manifest['fault_first_op'] > 0 and manifest['fault_cycles'] > 0, manifest
print(json.dumps({'live_h3': h3[-1], 'fault_manifest': str(run / 'fault-config.json'),
                  'live_analysis': str(run / 'analysis-live.jsonl'),
                  'final_analysis': str(run / 'analysis-final.jsonl'),
                  'workload_exit_status': (run / 'workload-exit-status').read_text().strip()}, sort_keys=True))
PY
printf 'H3 live diagnostic artifacts: %s\n' "$RUN"
