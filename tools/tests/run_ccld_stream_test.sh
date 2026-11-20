#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TMP=$(mktemp -d /tmp/ccld-stream-test.XXXXXX)
trap 'rm -rf "$TMP"' EXIT
${CXX:-c++} -std=c++14 -Wall -Wextra -Werror -pthread -I"$ROOT/src/include" \
  "$ROOT/tools/tests/ccld_stream_test.cpp" -o "$TMP/ccld_stream_test"
"$TMP/ccld_stream_test"
