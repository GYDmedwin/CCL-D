#!/usr/bin/env bash
set -euo pipefail

# This test deliberately builds and runs one target: gfx936.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HIPCC=${HIPCC:-/opt/dtk/bin/hipcc}
DECODER=${CCLD_DECODER:-/workspace/nv/CCL-D/ccld/tools/decode_ccld_binary.py}
RCCL_LIB=${RCCL_LIB:-/workspace/dcu/results/gfx936-20261121/librccl.so.1.0}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-only}
EXPECTED_RANKS=${CCLD_EXPECTED_RANKS:-4}
PROTO=${CCLD_TEST_PROTO:-Simple}
ALGO=${CCLD_TEST_ALGO:-Ring}
EXPECTED_CHANNELS=${CCLD_EXPECTED_CHANNELS:-$([[ "$PROTO" == "LL" ]] && echo 32 || echo 16)}
WORK=${CCLD_TEST_WORK:-$(mktemp -d /tmp/ccld-gfx936-ring.XXXXXX)}
KEEP_WORK=${CCLD_KEEP_TEST_WORK:-0}

cleanup() {
  if [[ "$KEEP_WORK" != 1 ]]; then
    rm -rf "$WORK"
  else
    echo "CCL-D test artifacts: $WORK"
  fi
}
trap cleanup EXIT

if [[ ! -x "$HIPCC" ]]; then
  echo "HIPCC not found: $HIPCC" >&2
  exit 2
fi
if [[ ! -f "$RCCL_LIB" ]]; then
  echo "RCCL_LIB not found: $RCCL_LIB" >&2
  exit 2
fi
if [[ ! -f "$DECODER" ]]; then
  echo "CCL-D decoder not found: $DECODER" >&2
  exit 2
fi
if [[ ! -f "$RCCL_BUILD_DIR/include/rccl.h" ]]; then
  echo "RCCL headers not found: $RCCL_BUILD_DIR/include/rccl.h" >&2
  exit 2
fi

mkdir -p "$WORK/traces" "$WORK/decoded"
LIBDIR=$(dirname "$RCCL_LIB")
"$HIPCC" --offload-arch=gfx936 -std=c++14 \
  -I"$RCCL_BUILD_DIR/include" \
  "$ROOT/tools/tests/ccld_simple_allreduce_test.cpp" \
  -L"$LIBDIR" -Wl,-rpath,"$LIBDIR" -l:$(basename "$RCCL_LIB") \
  -o "$WORK/ccld_simple_allreduce_test"

RUNTIME_LIBS="$LIBDIR:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/hyhal/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  RUNTIME_LIBS="$RUNTIME_LIBS:$LD_LIBRARY_PATH"
fi
NCCL_CCLD_ENABLE=1 \
NCCL_CCLD_OUT_DIR="$WORK/traces" \
NCCL_CCLD_DUMP_MS=10 \
NCCL_CCLD_POLL_US=20 \
NCCL_ALGO="$ALGO" \
NCCL_PROTO="$PROTO" \
LD_LIBRARY_PATH="$RUNTIME_LIBS" \
  "$WORK/ccld_simple_allreduce_test"

mapfile -t traces < <(find "$WORK/traces" -maxdepth 1 -type f -name '*.ccldbin' | sort)
if [[ "${#traces[@]}" -ne "$EXPECTED_RANKS" ]]; then
  echo "expected $EXPECTED_RANKS rank traces, found ${#traces[@]}" >&2
  exit 1
fi
for trace in "${traces[@]}"; do
  stem=$(basename "$trace" .ccldbin)
  python3 "$DECODER" "$trace" "$WORK/decoded/$stem.jsonl"
done

EXPECTED_SEND_RECV=$((2 * (EXPECTED_RANKS - 1)))
python3 - "$WORK/decoded" "$EXPECTED_RANKS" "$EXPECTED_SEND_RECV" "$PROTO" "$ALGO" "$EXPECTED_CHANNELS" <<'PY'
import collections
import json
import pathlib
import sys

decoded = pathlib.Path(sys.argv[1])
expected_ranks = int(sys.argv[2])
expected_count = int(sys.argv[3])
files = sorted(decoded.glob("*.jsonl"))
assert len(files) == expected_ranks, (len(files), expected_ranks)
for path in files:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    events = collections.Counter(row["event"] for row in rows)
    assert events["comm_init"] == 1, (path, events)
    assert events["comm_end"] == 1, (path, events)
    assert events["entered"] == 3, (path, events)
    completes = [row for row in rows if row["event"] == "complete"]
    assert len(completes) == 3, (path, events)
    for event in completes:
        assert event["algorithm"] == sys.argv[5], event
        assert event["protocol"] == sys.argv[4], event
        channels = event["channels"][:event["n_channels"]]
        assert event["n_channels"] > 0, event
        assert len(channels) == int(sys.argv[6]), event
        if sys.argv[5] == "Ring":
            assert {channel["send_count"] for channel in channels} == {expected_count}, event
            assert {channel["recv_count"] for channel in channels} == {expected_count}, event
        else:
            assert any(channel["tree_parent"] != -1 for channel in channels), event
            assert all(channel["send_count"] > 0 and channel["recv_count"] > 0 for channel in channels), event
print(f"gfx936 CCL-D {sys.argv[5]} validation passed: ranks={expected_ranks}, "
      f"complete_events={len(completes)} per rank, send_recv={expected_count}")
PY
