#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CCLD_RUN=${CCLD_RUN:-$ROOT/../results/ccld-gfx936-single-$(date +%Y%m%d-%H%M%S)}
export CCLD_NODE_RANK=0 CCLD_NNODES=1 CCLD_NPROC_PER_NODE=4
export CCLD_MASTER_ADDR=127.0.0.1 CCLD_MASTER_PORT=${CCLD_MASTER_PORT:-29501}
exec "$ROOT/tools/tests/run_ccld_gfx936_docker_node.sh"
