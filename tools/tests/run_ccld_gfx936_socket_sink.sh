#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HIPCC=${HIPCC:-hipcc}
DECODER=${CCLD_DECODER:-${CCLD_NV_ROOT:-$ROOT}/ccld/tools/decode_ccld_binary.py}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936}
RCCL_LIB=${RCCL_LIB:-$RCCL_BUILD_DIR/librccl.so.1.0}
WORK=${CCLD_TEST_WORK:-$(mktemp -d /tmp/ccld-gfx936-socket.XXXXXX)}
KEEP_WORK=${CCLD_KEEP_TEST_WORK:-0}
mkdir -p "$WORK/streams" "$WORK/decoded"
cleanup() {
  if [[ "$KEEP_WORK" != 1 ]]; then rm -rf "$WORK"; else echo "CCL-D socket artifacts: $WORK"; fi
}
trap cleanup EXIT

LIBDIR=$(dirname "$RCCL_LIB")
"$HIPCC" --offload-arch=gfx936 -std=c++14 -I"$RCCL_BUILD_DIR/include" \
  "$ROOT/tools/tests/ccld_simple_allreduce_test.cpp" -L"$LIBDIR" -Wl,-rpath,"$LIBDIR" \
  -l:$(basename "$RCCL_LIB") -o "$WORK/ccld_simple_allreduce_test"

SOCKET="$WORK/agent.sock"
python3 "$ROOT/tools/tests/ccld_socket_capture.py" "$SOCKET" "$WORK/streams" 4 &
CAPTURE_PID=$!
for _ in $(seq 1 100); do [[ -S "$SOCKET" ]] && break; sleep 0.05; done
[[ -S "$SOCKET" ]]
LD_LIBRARY_PATH="$LIBDIR:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/hyhal/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
NCCL_CCLD_ENABLE=1 NCCL_CCLD_NODE_SOCKET="$SOCKET" NCCL_CCLD_DUMP_MS=10 NCCL_CCLD_POLL_US=20 \
NCCL_ALGO=Ring NCCL_PROTO=Simple "$WORK/ccld_simple_allreduce_test"
wait "$CAPTURE_PID"

mapfile -t traces < <(find "$WORK/streams" -maxdepth 1 -type f -name '*.ccldbin' | sort)
[[ "${#traces[@]}" -eq 4 ]]
for trace in "${traces[@]}"; do
  python3 "$DECODER" "$trace" "$WORK/decoded/$(basename "$trace" .ccldbin).jsonl"
done
python3 - "$WORK/decoded" <<'PY'
import collections, json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob('*.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    counts = collections.Counter(row['event'] for row in rows)
    assert counts['comm_init'] == 1 and counts['comm_end'] == 1, (path, counts)
    assert counts['entered'] == 3 and counts['complete'] == 3, (path, counts)
    for event in (r for r in rows if r['event'] == 'complete'):
        channels = event['channels'][:event['n_channels']]
        assert event['n_channels'] == 16, event
        assert {c['send_count'] for c in channels} == {6}, event
        assert {c['recv_count'] for c in channels} == {6}, event
print('gfx936 CCL-D node socket sink validation passed')
PY
