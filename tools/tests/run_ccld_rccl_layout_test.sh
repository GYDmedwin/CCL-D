#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HIPCC=${HIPCC:-/opt/dtk/bin/hipcc}
BUILD_DIR=${RCCL_BUILD_DIR:-/tmp/rccl-ccld-gfx936-local-20261121a}
if ! command -v ninja >/dev/null || [ ! -f "$BUILD_DIR/build.ninja" ]; then
  echo "RCCL_BUILD_DIR must point to a configured RCCL Ninja build" >&2
  exit 2
fi
ninja -C "$BUILD_DIR" hipify/src/include/device.h hipify/src/include/comm.h >/dev/null

"$HIPCC" --offload-arch=gfx936 -std=c++14 -Wno-return-type -Wno-cpp -fsyntax-only \
  -I"$BUILD_DIR/hipify/src/include" -I"$BUILD_DIR/include" -I"$ROOT/src/include" \
  -I/opt/dtk-26.04/cuda/cuda-12/targets/x86_64-linux/include -I/opt/hyhal/hsa/include \
  "$ROOT/tools/tests/ccld_rccl_layout_test.cpp"
