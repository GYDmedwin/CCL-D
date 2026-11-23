#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HIPCC=${HIPCC:-/opt/dtk/bin/hipcc}
DECODER=${CCLD_DECODER:-${CCLD_NV_ROOT:-$ROOT}/ccld/tools/decode_ccld_binary.py}
RCCL_LIB=${RCCL_LIB:-/workspace/dcu/results/gfx936-20261121/librccl.so.1.0}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-only}
WORK=${CCLD_TEST_WORK:-$(mktemp -d /tmp/ccld-gfx936-collectives.XXXXXX)}
trap '[[ "${CCLD_KEEP_TEST_WORK:-0}" == 1 ]] || rm -rf "$WORK"' EXIT

mkdir -p "$WORK/traces" "$WORK/decoded"
LIBDIR=$(dirname "$RCCL_LIB")
"$HIPCC" --offload-arch=gfx936 -std=c++14 -I"$RCCL_BUILD_DIR/include" \
  "$ROOT/tools/tests/ccld_ring_collectives_test.cpp" -L"$LIBDIR" -Wl,-rpath,"$LIBDIR" \
  -l:$(basename "$RCCL_LIB") -o "$WORK/ccld_ring_collectives_test"
LD_LIBRARY_PATH="$LIBDIR:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/hyhal/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
NCCL_CCLD_ENABLE=1 NCCL_CCLD_OUT_DIR="$WORK/traces" NCCL_CCLD_DUMP_MS=10 NCCL_CCLD_POLL_US=20 \
NCCL_ALGO=Ring NCCL_PROTO=Simple "$WORK/ccld_ring_collectives_test"

mapfile -t traces < <(find "$WORK/traces" -maxdepth 1 -type f -name '*.ccldbin' | sort)
[[ "${#traces[@]}" -eq 4 ]]
for trace in "${traces[@]}"; do
  python3 "$DECODER" "$trace" "$WORK/decoded/$(basename "$trace" .ccldbin).jsonl"
done
python3 - "$WORK/decoded" <<'PY'
import collections, json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob('*.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    complete = [r for r in rows if r['event'] == 'complete']
    assert len(complete) == 2, (path, collections.Counter(r['event'] for r in rows))
    assert [r['op'] for r in complete] == ['AllGather', 'ReduceScatter'], complete
    for event in complete:
        assert event['algorithm'] == 'Ring' and event['protocol'] == 'Simple', event
        channels = event['channels'][:event['n_channels']]
        assert event['n_channels'] > 0, event
        assert len(channels) == event['n_channels'], event
        assert all(c['send_count'] == c['recv_count'] for c in channels), event
        assert all(c['send_count'] > 0 for c in channels), event
print('gfx936 Ring AllGather/ReduceScatter CCL-D validation passed')
PY
