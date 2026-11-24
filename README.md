# CCL-D for RCCL

CCL-D detects and locates slow or stalled collective operations in distributed
training. This branch instruments RCCL **2.22.3-1**. DCU kernels write
progress counters to mapped host memory. Rank-local monitors sample the
counters, node agents buffer the records in memory, and a separate CPU
analyzer pulls those records over the job network.

[Paper](https://doi.org/10.1145/3774934.3786429) · [NCCL version](https://github.com/GYDmedwin/CCL-D/tree/main) · [Validation results](RESULTS.md)

## Build

Load your HIP toolchain and use Python 3.10 or later. This example builds
for BW1000 (`gfx936`) and writes `build-ccld-gfx936/librccl.so.1.0`.

```bash
git clone --branch rccl https://github.com/GYDmedwin/CCL-D.git
cd CCL-D
cmake -S . -B build-ccld-gfx936 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx936 -DCOMPILING_TARGETS=gfx936 \
  -DCOLLTRACE=ON -DCCLD_FAULT_INJECTION=OFF -DBUILD_TESTS=OFF
cmake --build build-ccld-gfx936 --parallel 12
python3 -m unittest discover -s ccld/tests
```

The BW1000 validation used DTK 26.04 and PyTorch 2.7.1. Use the full build
for training that selects collectives and datatypes at runtime. See the
[targeted-build examples](RESULTS.md).

## Run with your training job

Use the same CCL-D revision for the agents and analyzer. Generate one token
with `openssl rand -hex 32` and export it as `CCLD_PULL_TOKEN` wherever the
agents or analyzer run.

1. **Start one agent per training node.** Set `TRAIN_NODE_ADDR` to the node
   address reachable from the analyzer. Keep the local socket path for
   that node's training ranks.

   ```bash
   socket_dir="$(mktemp -d)"
   export NCCL_CCLD_NODE_SOCKET="$socket_dir/ccld.sock"
   python3 -m ccld.tools.live_transport \
     --socket "$NCCL_CCLD_NODE_SOCKET" --bind "$TRAIN_NODE_ADDR" --port 0
   ```

2. **Start the analyzer on a CPU host.** Use the `HOST:PORT` shown in each
   agent's `CCLD_NODE_READY` line. The analyzer prints JSON decisions to
   standard output.

   ```bash
   python3 -m ccld.analyzer.live --node "$AGENT_A" --node "$AGENT_B" \
     --baseline-ms 1 --baseline-scope signature --poll-s 0.2
   ```

3. **Launch training with the instrumented library.** Export the agent
   socket for each node into that node's training ranks. Set
   `RCCL_BUILD_DIR` to the build directory and use the job's existing
   PyTorch or MPI launcher.

   ```bash
   export NCCL_CCLD_NODE_SOCKET=/path/to/local/ccld.sock
   export RCCL_BUILD_DIR=/path/to/CCL-D/build-ccld-gfx936
   export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1
   export LD_LIBRARY_PATH="$RCCL_BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   export LD_PRELOAD="$RCCL_BUILD_DIR/librccl.so.1.0${LD_PRELOAD:+:$LD_PRELOAD}"
   ```

Keep the agents running until the analyzer drains their buffers. The
analyzer's `--poll-s` controls how often it pulls records. Set
`--baseline-scope signature` to learn separate slow baselines by collective
and `--slow-warmup-ops` to exclude initial calls from slow analysis.

## Diagnose collectives

H1 identifies a rank that has yet to enter a collective. H2 uses mismatched
signatures or partial completion. H3 uses stalled progress. A node agent
can also report a verified rank process exit. Slow diagnoses can identify
delayed entry, communication slowdown, or both. The analyzer reports the
evidence and any root ranks it can confirm.

### Directional waits

- **Ring/Simple or Ring/LL.** Set `NCCL_ALGO=Ring` and
  `NCCL_CCLD_WAIT_ENABLE=1` on the training ranks. For mixed slowdown across hosts,
  add `--ring-wait-phases` and `--baseline-scope signature` to the analyzer.
- **Tree/Simple AllReduce.** Set `NCCL_ALGO=Tree NCCL_PROTO=Simple` and
  `NCCL_CCLD_TREE_WAIT_ENABLE=1`. For mixed slowdown across hosts, set
  `NCCL_CCLD_DUMP_MS=1` and add `--tree-cross-host-phases` and
  `--baseline-scope signature` to the analyzer. The analyzer bounds the
  host clock offsets and compares Tree edge waits after the ranks enter.

### Replay and validate

For offline replay, set `NCCL_CCLD_OUT_DIR` to an existing directory on each
training node. Decode the collected traces and run the analyzer.

```bash
python3 -m ccld.tools.decode_async_wait "$TRACE_DIR" "$JSONL_DIR"
python3 -m ccld.analyzer.ccld_analyze "$JSONL_DIR" \
  --baseline-ms 1 --baseline-scope signature
```

For controlled fault validation, build a separate library with
`-DCCLD_FAULT_INJECTION=ON` and use the evaluation runner.

```bash
RCCL_BUILD_DIR="$PWD/build-ccld-gfx936-fault" \
  tools/tests/run_antccl_eval51_gfx936.sh --case H1
```

The runner's `--help` lists its fault and topology options. See
[validation results](RESULTS.md) for the tested collective and training
configurations.

## Licenses and attribution

The branch preserves the history of
[niheartent's RCCL port](https://github.com/niheartent/CCL-D-DCU/tree/486a4ced7fce37b41b6b60689d6c2143b34f0a28).
See the [RCCL license](LICENSE.txt) and the
[CCL-D license](ccld/LICENSE.txt).
