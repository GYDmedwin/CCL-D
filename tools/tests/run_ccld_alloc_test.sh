#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HIPCC=${HIPCC:-/opt/dtk/bin/hipcc}

"$HIPCC" --offload-arch=gfx936 -std=c++14 -Wno-return-type -Wno-cpp -fsyntax-only \
  -I"$ROOT/src/include" -I/opt/hyhal/hsa/include \
  "$ROOT/tools/tests/ccld_alloc_test.cpp"
