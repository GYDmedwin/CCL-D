# CCL-D for NCCL

CCL-D detects and locates slow or stalled collective operations in distributed
training. This branch instruments NVIDIA NCCL **v2.24.3-1**. GPU kernels write
progress counters to mapped host memory. Rank-local monitors sample the
counters, node agents buffer the records in memory, and a separate CPU
analyzer pulls those records over the job network.

[Paper](https://doi.org/10.1145/3774934.3786429) · [RCCL version](https://github.com/GYDmedwin/CCL-D/tree/rccl) · [Validation results](RESULTS.md)

## Build

Use Linux, a CUDA and C++ toolchain, and Python 3.9 or later. This example
builds for H100. Set `CUDA_HOME` to your toolkit and adjust `NVCC_GENCODE` for
your GPU. The build writes `build/lib/libnccl.so.2`.

```bash
git clone https://github.com/GYDmedwin/CCL-D.git
cd CCL-D
make -j8 src.build CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
python3 -m unittest discover -s ccld/tests
```

The H100 validation used CUDA 12.8.2. See the
[NCCL build options](README.NVIDIA.md#build).

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
   `CCLD_NCCL_LIB` to the build's library directory and use the job's
   existing PyTorch or MPI launcher.

   ```bash
   export NCCL_CCLD_NODE_SOCKET=/path/to/local/ccld.sock
   export CCLD_NCCL_LIB=/path/to/CCL-D/build/lib
   export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1
   export LD_LIBRARY_PATH="$CCLD_NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   export LD_PRELOAD="$CCLD_NCCL_LIB/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
   ```

Keep the agents running until the analyzer drains their buffers. See the
[deployment guide](ccld/docs/live.md) for retention, pull tuning, and observed
waits between communicators.

### Slurm FSDP example

Run from a shared checkout with your CUDA PyTorch environment active. The
example starts node agents and runs the analyzer on the batch host. It
checks the NCCL library loaded by every rank and saves the training timings,
losses, and diagnostic output under `runs/fsdp-$SLURM_JOB_ID`.

```bash
export CCLD_ROOT="$PWD"
CCLD_PROFILE=smoke sbatch --account YOUR_ACCOUNT --partition YOUR_GPU_PARTITION \
  ccld/slurm/fsdp_train.sh
```

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
  `NCCL_CCLD_TREE_SNAPSHOT_MS=1` and add `--tree-cross-host-phases` and
  `--baseline-scope signature` to the analyzer. The analyzer bounds the
  host clock offsets and compares Tree edge waits after the ranks enter.
  The faster snapshot interval applies to active Tree operations.

See the [telemetry reference](ccld/docs/telemetry.md) and the
[replay and fault-injection examples](ccld/docs/experiments.md).

## License and citation

See the [NCCL license](LICENSE.txt), [NVTX license](LICENSE-NVTX.txt),
[CCL-D license](LICENSE-CCLD.txt), and
[third-party notices](THIRD_PARTY_NOTICES.md).

Yida Gu, Fakang Wang, et al., *CCL-D: A High-Precision Diagnostic System for
Slow and Hang Anomalies in Large-Scale Model Training*, PPoPP 2026.
[DOI](https://doi.org/10.1145/3774934.3786429).
