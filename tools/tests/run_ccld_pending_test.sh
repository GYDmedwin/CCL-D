#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

${CXX:-c++} -std=c++14 -Wall -Wextra -Werror -pthread \
  -I"$ROOT/src/include" "$ROOT/tools/tests/ccld_pending_test.cpp" \
  -o "$TMP_DIR/ccld_pending_test"
"$TMP_DIR/ccld_pending_test"
