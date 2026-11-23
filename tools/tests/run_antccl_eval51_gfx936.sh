#!/usr/bin/env bash
set -euo pipefail

# Single public entry point for the AntCCL Evaluation 5.1 fault experiments.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TESTS="$ROOT/tools/tests"
DECODER=${CCLD_DECODER:-${CCLD_NV_ROOT:-$ROOT}/ccld/tools/decode_async_wait.py}
TOPOLOGY=${CCLD_TOPOLOGY:-single}
CASE=${CCLD_EVAL51_CASE:-all}
RUN=${CCLD_RUN:-$ROOT/../results/antccl-eval51-${TOPOLOGY}-$(date +%Y%m%d-%H%M%S)}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-fault}
RCCL_LIB=${RCCL_LIB:-$RCCL_BUILD_DIR/librccl.so.1.0}
RANK=${CCLD_FAULT_RANK:-2}
PROTO=${CCLD_PROTO:-Simple}
SINGLE_MATRIX="$ROOT/tools/tests/run_ccld_gfx936_docker_single_node_matrix.sh"
DUAL_MATRIX="$ROOT/tools/tests/run_ccld_gfx936_docker_2node.sh"

usage() {
  cat <<'EOF'
Usage: run_antccl_eval51_gfx936.sh [--topology single|dual] [--case all|H1|H2|H2_SIM|H3|POST|S1|S2]

Environment overrides: CCLD_RUN RCCL_BUILD_DIR RCCL_LIB CCLD_NV_ROOT CCLD_FAULT_RANK
CCLD_S2_CYCLES CCLD_BASELINE_MS CCLD_PROTO
The S2 and all cases require CCLD_WAIT_THRESHOLD for the selected setup.
EOF
}

while (($#)); do
  case "$1" in
    --topology) TOPOLOGY=${2:?}; shift 2 ;;
    --case) CASE=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$TOPOLOGY" == single || "$TOPOLOGY" == dual ]] || { echo "invalid topology: $TOPOLOGY" >&2; exit 2; }
[[ "$CASE" == all || "$CASE" =~ ^(H1|H2|H2_SIM|H3|POST|S1|S2)$ ]] || { echo "invalid case: $CASE" >&2; exit 2; }
[[ -f "$RCCL_LIB" ]] || { echo "RCCL library not found: $RCCL_LIB" >&2; exit 2; }
if [[ "$CASE" == all || "$CASE" == S2 ]]; then
  : "${CCLD_WAIT_THRESHOLD:?set CCLD_WAIT_THRESHOLD calibrated for topology/protocol}"
fi
MATRIX_RUNNER=$SINGLE_MATRIX
WORLD_SIZE=4
if [[ "$TOPOLOGY" == dual ]]; then
  MATRIX_RUNNER=$DUAL_MATRIX
  WORLD_SIZE=8
fi
mkdir -p "$RUN"
export RCCL_BUILD_DIR RCCL_LIB CCLD_TOPOLOGY="$TOPOLOGY" CCLD_H1_RANK="$RANK"

run_one() {
  local kind=$1 case_run="$RUN/$1"
  export CCLD_RUN="$case_run"
  printf '\n=== AntCCL Evaluation 5.1: %s (%s) ===\n' "$kind" "$TOPOLOGY"
  case "$kind" in
    H1)
      "$TESTS/run_ccld_gfx936_h1_live.sh"
      ;;
    H2)
      local launch_log="${case_run}-launcher.log"
      CCLD_CASE_FILTER=Ring-Simple-allreduce CCLD_WORKLOAD_FAULT=H2 \
        CCLD_WORKLOAD_FAULT_RANK="$RANK" CCLD_WORKLOAD_FAULT_ITERATION=3 \
        CCLD_WORKLOAD_H2_DELTA_BYTES=1024 CCLD_WORKLOAD_H2_COUNT_DIVISOR=0 \
        CCLD_WARMUP=3 CCLD_ITERS=3 CCLD_NUMEL=262144 CCLD_SKIP_POST_DECODE=1 \
        "$MATRIX_RUNNER" \
        >"$launch_log" 2>&1
      local case_dir="$case_run/Ring-Simple-allreduce"
      python3 "$DECODER" "$case_dir/bin" "$case_dir/jsonl"
      python3 "$TESTS/ccld_completed_mismatch.py" "$case_dir/jsonl" \
        --world-size "$WORLD_SIZE" \
        --expect-rank "$RANK" --expect-op-count 4 \
        --output "$case_dir/completed-mismatches.jsonl"
      cat "$case_dir/completed-mismatches.jsonl"
      ;;
    H2_SIM)
      if [[ "$TOPOLOGY" == single ]]; then
        "$TESTS/run_ccld_gfx936_h2_strict_live.sh"
      else
        "$TESTS/run_ccld_gfx936_h2_strict_dual_live.sh"
      fi
      ;;
    H3)
      [[ "$TOPOLOGY" == single ]] || { echo 'H3 live runner currently uses the single-node 4-rank case' >&2; return 2; }
      CCLD_H3_RANK="$RANK" "$TESTS/run_ccld_gfx936_h3_live.sh"
      ;;
    POST)
      local launch_log="${case_run}-launcher.log"
      CCLD_CASE_FILTER=Ring-Simple-allreduce CCLD_WORKLOAD_FAULT=POST \
        CCLD_WORKLOAD_FAULT_RANK="$RANK" CCLD_WORKLOAD_FAULT_ITERATION=3 \
        CCLD_WORKLOAD_HANG_SECONDS=5 CCLD_WORKLOAD_TIMEOUT_SECONDS=15 \
        CCLD_WARMUP=3 CCLD_ITERS=3 CCLD_NUMEL=262144 CCLD_SKIP_POST_DECODE=1 \
        "$MATRIX_RUNNER" \
        >"$launch_log" 2>&1
      local case_dir="$case_run/Ring-Simple-allreduce"
      python3 "$DECODER" "$case_dir/bin" "$case_dir/jsonl"
      python3 "$TESTS/ccld_post_boundary_check.py" "$case_dir/jsonl" \
        --fault-events "$case_dir/fault-events" --rank "$RANK" --threshold-s 1.0
      ;;
    S1)
      [[ "$TOPOLOGY" == single ]] || { echo 'S1 strict calibrated runner currently uses the single-node 4-rank case' >&2; return 2; }
      CCLD_S1_RANK="$RANK" "$TESTS/run_ccld_gfx936_s1_strict_live.sh"
      ;;
    S2)
      local cycles=${CCLD_S2_CYCLES:-2300}
      [[ "$TOPOLOGY" == dual && -z "${CCLD_S2_CYCLES:-}" ]] && cycles=115000
      CCLD_PROTO="$PROTO" CCLD_S2_RANK="$RANK" \
        CCLD_S2_CYCLES="$cycles" "$TESTS/run_ccld_gfx936_ring_wait_live.sh"
      ;;
  esac
}

if [[ "$CASE" == all ]]; then
  if [[ "$TOPOLOGY" == single ]]; then
    for kind in H1 H2 H2_SIM H3 POST S1 S2; do run_one "$kind"; done
  else
    for kind in H1 H2 H2_SIM POST S2; do run_one "$kind"; done
    printf 'H3 and S1 use the single-node 4-rank cases in this Evaluation 5.1 matrix.\n'
  fi
else
  run_one "$CASE"
fi
printf '\nEvaluation 5.1 artifacts: %s\n' "$RUN"
