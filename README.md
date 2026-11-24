# CCL-D for RCCL

CCL-D diagnoses slow and hanging collectives in RCCL workloads. GPU probes update mapped host-memory frames. A host thread samples the frames, and a node agent buffers the resulting binary records. An independent analyzer pulls the records over TCP and reports the affected collective and any supported root ranks.

This branch targets RCCL 2.22.3-1 on gfx936 BW1000 accelerators. The validation environment uses DTK 26.04, HIP, and PyTorch 2.7.1. It includes single-node collective tests and 16-rank InfiniBand and FSDP runs. [Validation results](RESULTS.md)

## Build

Load the ROCm or DTK environment on the build host and run these commands from the repository root.

```bash
cmake -S . -B build-ccld-gfx936 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx936 -DCOMPILING_TARGETS=gfx936 \
  -DCOLLTRACE=ON -DCCLD_FAULT_INJECTION=OFF -DBUILD_TESTS=OFF
cmake --build build-ccld-gfx936 --parallel 12
```

For quick compile and test cycles, set `ONLY_FUNCS` to the RCCL device functions exercised by the workload. Use the full build for workloads with dynamic collective or datatype selection. [Targeted build examples](RESULTS.md)

## Run alongside training

Use the same code revision on the training and analysis hosts. Run the Python commands from the repository root with Python 3.10 or later. Create one token for the job with `openssl rand -hex 32` and export it as `CCLD_PULL_TOKEN` to every agent and to the analyzer.

### Start a node agent

On each training node, choose a job network address reachable by the analyzer and set `TRAIN_NODE_ADDR` to that address. Start one agent per node and keep the socket path for the training launcher on that node.

```bash
socket_dir="$(mktemp -d)"
export NCCL_CCLD_NODE_SOCKET="$socket_dir/ccld.sock"
python3 -m ccld.tools.live_transport \
  --socket "$NCCL_CCLD_NODE_SOCKET" --bind "$TRAIN_NODE_ADDR" --port 0
```

### Start the analyzer

On the analysis host, use the `HOST:PORT` printed in each agent's `CCLD_NODE_READY` line. Pass one `--node` per training node. The analyzer reads the binary buffers in memory and writes diagnosis events to standard output.

```bash
python3 -m ccld.analyzer.live --node "$NODE_AGENT_ADDR" \
  --poll-s 0.025 --hang-s 300 --window-s 60 --baseline-ms 1 \
  --baseline-scope signature --decode-workers 4 --coalesce-completed
```

### Start the training workload

Pass the node-local socket path from the agent step to the training processes and load the instrumented library. The example uses four local ranks. For distributed training, use the existing launcher and pass the corresponding socket path on each node.

```bash
export RCCL_BUILD_DIR="$PWD/build-ccld-gfx936"
export LD_LIBRARY_PATH="$RCCL_BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$RCCL_BUILD_DIR/librccl.so.1.0"
unset NCCL_CCLD_OUT_DIR
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1 NCCL_CCLD_POLL_US=200
export NCCL_CCLD_FRAME_BLOCKS=64
torchrun --standalone --nproc-per-node=4 train.py
```

## Configure diagnosis

| Setting | Purpose |
| --- | --- |
| `NCCL_CCLD_POLL_US` | Time between samples of mapped progress counters on the training host. |
| `NCCL_CCLD_DUMP_MS` | Time between progress snapshots for in-flight collectives. Completions emit their own records. |
| `--poll-s` | Time between analyzer pulls from node agents. |
| `--baseline-scope signature` | Learn separate slow baselines by operation, size, datatype, algorithm, protocol, channel count, and known reduction operator. |
| `--slow-warmup-ops` | Skip the first configured number of eligible operations in each communicator when learning and evaluating slow baselines. Hang and signature checks continue. |
| `--baseline-rounds`, `--baseline-s` | Set the baseline sample budget and learning time. Calibrate on a healthy run before reducing the slow window or the hang threshold. |

### Directional waits

For Ring with Simple or LL, set `NCCL_ALGO=Ring NCCL_CCLD_WAIT_ENABLE=1`
on the training processes. Enable `--ring-wait-phases` in the analyzer
for mixed slowdown. Within one host, set `NCCL_CCLD_DUMP_MS=1` to sample
waits around the last rank's entry. Across hosts, use
`--baseline-scope signature` and collect at least three intervals between
completed and subsequent collective calls. The analyzer compares these
intervals with each rank's own baseline and checks them against repeated
directional waits. The host timestamps stay on their own clocks.

For Tree/Simple AllReduce, set
`NCCL_ALGO=Tree NCCL_PROTO=Simple NCCL_CCLD_TREE_WAIT_ENABLE=1` and use
`--baseline-scope signature`. The analyzer scores parent and child waits
separately. Same-host mixed attribution also uses
`NCCL_CCLD_DUMP_MS=1` and `--ring-wait-phases` to compare waits before
and after the delayed rank enters. `--tree-wait-gap` sets the required
separation between the top communication scores. [Validation](RESULTS.md)

### Hang and process exit

Tree H3 compares each rank's progress against a matching completed round.
Tied progress can leave several candidates. A live node agent reports
`rank_exit` with the communicator rank and its last observed operation
counter when it verifies local process exit. If it can establish only
the socket closure, it reports a coverage gap. Upgrade the node agent and
analyzer together.

### Record a trace for replay

Set `NCCL_CCLD_OUT_DIR` to an existing directory on each training node to write binary files. Decode the directory from the same revision.

```bash
python3 -m ccld.tools.decode_async_wait "$TRACE_DIR" "$JSONL_DIR"
python3 -m ccld.analyzer.ccld_analyze "$JSONL_DIR" --baseline-ms 1 --baseline-scope signature
```

## Run validation

Build a separate library with `CCLD_FAULT_INJECTION=ON` to run the controlled fault experiments. The Evaluation 5.1 runner accepts `RCCL_BUILD_DIR` and `RCCL_LIB` to select that build.

```bash
RCCL_BUILD_DIR="$PWD/build-ccld-gfx936-fault" \
  tools/tests/run_antccl_eval51_gfx936.sh --case H1
```

Run the script with `--help` for available cases and topology options. The S2 suite uses `CCLD_WAIT_THRESHOLD` calibrated for that protocol, message size, and topology. [Workloads, accuracy, and overhead](RESULTS.md)

## Attribution and licenses

This branch builds on niheartent's RCCL port and preserves its commit history. [Original RCCL port](https://github.com/niheartent/CCL-D-DCU/tree/486a4ced7fce37b41b6b60689d6c2143b34f0a28)

See the [RCCL license](LICENSE.txt) and the [CCL-D license](ccld/LICENSE.txt).
