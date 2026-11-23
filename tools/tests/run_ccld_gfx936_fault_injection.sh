#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HIPCC=${HIPCC:-/opt/dtk/bin/hipcc}
DECODER=${CCLD_DECODER:-${CCLD_NV_ROOT:-$ROOT}/ccld/tools/decode_ccld_binary.py}
RCCL_LIB=${RCCL_LIB:-$ROOT/build-ccld-gfx936-only/lib/librccl.so.1.0}
RCCL_BUILD_DIR=${RCCL_BUILD_DIR:-$ROOT/build-ccld-gfx936-only}
WORK=${CCLD_TEST_WORK:-$(mktemp -d /tmp/ccld-gfx936-fault.XXXXXX)}
trap '[[ "${CCLD_KEEP_TEST_WORK:-0}" == 1 ]] || rm -rf "$WORK"' EXIT

if ! grep -q '^CCLD_FAULT_INJECTION:BOOL=ON$' "$RCCL_BUILD_DIR/CMakeCache.txt"; then
  echo "RCCL_BUILD_DIR must be configured with CCLD_FAULT_INJECTION=ON" >&2
  exit 2
fi
mkdir -p "$WORK/traces" "$WORK/decoded"
LIBDIR=$(dirname "$RCCL_LIB")
"$HIPCC" --offload-arch=gfx936 -std=c++14 -I"$RCCL_BUILD_DIR/include" \
  "$ROOT/tools/tests/ccld_simple_allreduce_test.cpp" -L"$LIBDIR" -Wl,-rpath,"$LIBDIR" \
  -l:$(basename "$RCCL_LIB") -o "$WORK/ccld_simple_allreduce_test"
LD_LIBRARY_PATH="$LIBDIR:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/hyhal/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
NCCL_CCLD_ENABLE=1 NCCL_CCLD_OUT_DIR="$WORK/traces" NCCL_CCLD_DUMP_MS=10 NCCL_CCLD_POLL_US=20 \
NCCL_CCLD_FAULT_KIND=S2 NCCL_CCLD_FAULT_RANK=0 NCCL_CCLD_FAULT_OP=1 \
NCCL_CCLD_FAULT_REPEATS=1 NCCL_CCLD_FAULT_CYCLES=10000000 \
NCCL_ALGO=Ring NCCL_PROTO=Simple "$WORK/ccld_simple_allreduce_test"

mapfile -t traces < <(find "$WORK/traces" -maxdepth 1 -type f -name '*.ccldbin' | sort)
[[ "${#traces[@]}" -eq 4 ]]
for trace in "${traces[@]}"; do
  python3 "$DECODER" "$trace" "$WORK/decoded/$(basename "$trace" .ccldbin).jsonl"
done
python3 - "$WORK/decoded" <<'PY'
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob('*.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    complete = [row for row in rows if row['event'] == 'complete']
    assert len(complete) == 3, (path, rows)
    assert all(row['algorithm'] == 'Ring' and row['protocol'] == 'Simple' for row in complete)
    for row in complete:
        channels = row['channels'][:row['n_channels']]
        assert len(channels) == 16 and all(c['send_count'] > 0 and c['recv_count'] > 0 for c in channels), row
    if 'rank0-' in path.name:
        selected = next(row for row in complete if row['op_count'] == 1)
        assert selected['duration_ns'] >= 1_000_000, selected
print('gfx936 isolated CCL-D S2 fault injection validation passed')
PY
