#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HIPCC=${HIPCC:-/opt/dtk/bin/hipcc}
DECODER=${CCLD_WAIT_DECODER:-${CCLD_NV_ROOT:-/workspace/nv/CCL-D}/ccld/tools/decode_async_wait.py}
RCCL_LIB=${RCCL_LIB:-/workspace/dcu/results/gfx936-20261121/librccl.so.1.0}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-only}
WORK=${CCLD_TEST_WORK:-$(mktemp -d /tmp/ccld-gfx936-wait.XXXXXX)}
trap '[[ "${CCLD_KEEP_TEST_WORK:-0}" == 1 ]] || rm -rf "$WORK"' EXIT
mkdir -p "$WORK/traces" "$WORK/decoded"
LIBDIR=$(dirname "$RCCL_LIB")
"$HIPCC" --offload-arch=gfx936 -std=c++14 -I"$RCCL_BUILD_DIR/include" \
  "$ROOT/tools/tests/ccld_simple_allreduce_test.cpp" -L"$LIBDIR" -Wl,-rpath,"$LIBDIR" \
  -l:$(basename "$RCCL_LIB") -o "$WORK/ccld_simple_allreduce_test"
LD_LIBRARY_PATH="$LIBDIR:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/hyhal/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
NCCL_CCLD_ENABLE=1 NCCL_CCLD_OUT_DIR="$WORK/traces" NCCL_CCLD_WAIT_ENABLE=1 \
NCCL_CCLD_DUMP_MS=10 NCCL_CCLD_POLL_US=20 NCCL_ALGO=Ring NCCL_PROTO=Simple \
  "$WORK/ccld_simple_allreduce_test"

mapfile -t traces < <(find "$WORK/traces" -maxdepth 1 -type f -name '*.ccldbin' | sort)
[[ "${#traces[@]}" -eq 4 ]]
for trace in "${traces[@]}"; do
  python3 "$DECODER" "$trace" "$WORK/decoded/$(basename "$trace" .ccldbin).jsonl"
done
python3 - "$WORK/decoded" <<'PY'
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob('*.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    init = next(row for row in rows if row['event'] == 'comm_init')
    assert init['wait_requested'] and init['wait_frame_bytes'] > 0, (path, init)
    completes = [row for row in rows if row['event'] == 'complete']
    assert len(completes) == 3, (path, rows)
    for event in completes:
        assert event['wait_probe'] is True, event
        channels = event['channels'][:event['n_channels']]
        assert len(channels) == 16, event
        for channel in channels:
            assert channel['ring_prev'] >= 0 and channel['ring_next'] >= 0, channel
            assert channel['wait_data_spins'] >= 0 and channel['wait_credit_spins'] >= 0, channel
print('gfx936 Ring wait probe binary-pair validation passed')
PY
