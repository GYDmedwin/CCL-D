#!/usr/bin/env bash
# Run with --agent on a training node, or source for an in-allocation demo.
if [ "${1:-}" = --agent ]; then
  set -euo pipefail
  umask 077
  cd "$CCLD_ROOT"
  dir=$(dirname "$NCCL_CCLD_NODE_SOCKET")
  mkdir -m 700 -- "$dir"
  agent=''
  cleanup_agent() {
    if [ -n "$agent" ]; then
      kill "$agent" 2>/dev/null || true
      wait "$agent" 2>/dev/null || true
    fi
    rmdir -- "$dir" 2>/dev/null || true
  }
  trap cleanup_agent EXIT
  trap 'exit 0' TERM INT
  env -u LD_PRELOAD "$CCLD_PYTHON" -m ccld.tools.live_transport \
    --socket "$NCCL_CCLD_NODE_SOCKET" --bind "$(hostname -s)" --port "$CCLD_PULL_PORT" \
    --records "${CCLD_PULL_RECORDS:-16384}" &
  agent=$!
  wait "$agent"
  exit
fi

ccld_live_cleanup() {
  local pid
  for pid in "$ccld_analyzer" "$ccld_agents"; do
    [ -z "$pid" ] || kill "$pid" 2>/dev/null || true
  done
  for pid in "$ccld_analyzer" "$ccld_agents"; do
    [ -z "$pid" ] || wait "$pid" 2>/dev/null || true
  done
}

ccld_live_start() {
  local node attempt
  local -a endpoints=() nodes=()
  mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  [ "${#nodes[@]}" -eq "$SLURM_NNODES" ]
  export CCLD_PYTHON="$(command -v python3)"
  export CCLD_PULL_TOKEN="$("$CCLD_PYTHON" -c 'import secrets; print(secrets.token_hex(32))')"
  export CCLD_PULL_PORT="${CCLD_PULL_PORT:-$("$CCLD_PYTHON" -c 'import secrets; print(20000+secrets.randbelow(20000))')}"
  export NCCL_CCLD_NODE_SOCKET="/dev/shm/ccld-$SLURM_JOB_ID-${BASHPID}/node.sock"
  export CCLD_MASTER_ADDR="${nodes[0]}"
  for node in "${nodes[@]}"; do endpoints+=(--node "$node:$CCLD_PULL_PORT"); done

  ccld_agents=''; ccld_analyzer=''
  trap ccld_live_cleanup EXIT
  srun --overlap --gres=none --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --cpus-per-task=1 --cpu-bind=none bash "$CCLD_ROOT/ccld/slurm/live_job.sh" --agent \
    > "$CCLD_RUN/agents.log" 2>&1 &
  ccld_agents=$!
  env -u LD_PRELOAD "$CCLD_PYTHON" -m ccld.analyzer.live "${endpoints[@]}" "$@" \
    > "$CCLD_RUN/diagnosis.log" 2> "$CCLD_RUN/analyzer.log" &
  ccld_analyzer=$!
  unset CCLD_PULL_TOKEN
  for attempt in {1..75}; do
    if grep -qx CCLD_ANALYZER_READY "$CCLD_RUN/analyzer.log"; then return 0; fi
    kill -0 "$ccld_analyzer" "$ccld_agents" 2>/dev/null || break
    sleep 0.2
  done
  echo "CCL-D analyzer could not reach all node agents. See $CCLD_RUN/agents.log and diagnosis.log." >&2
  return 1
}

ccld_live_stop() {
  # Keep the agents available while the analyzer drains the final records.
  kill "$ccld_analyzer" 2>/dev/null || true
  wait "$ccld_analyzer"
  ccld_analyzer=''
}
