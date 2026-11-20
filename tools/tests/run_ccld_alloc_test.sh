#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HIPCC=${HIPCC:-/opt/dtk/bin/hipcc}
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

"$HIPCC" --offload-arch=gfx936 -std=c++14 -Wno-return-type -Wno-cpp \
  -I"$ROOT/src/include" -I/opt/hyhal/hsa/include \
  "$ROOT/tools/tests/ccld_alloc_test.cpp" -o "$TMP_DIR/ccld_alloc_test" -lamdhip64
"$TMP_DIR/ccld_alloc_test"
