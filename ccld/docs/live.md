# Online deployment

The GPU updates the existing UVA-mapped pinned probing frame, and the
rank-local diagnostic thread samples it. The rank writer sends structured
events over a Unix socket to an agent on the same training node. One agent
keeps a bounded buffer in host memory for that node's ranks. The analyzer
pulls compressed binary batches over TCP from its last cursor. It groups
the records by communicator and operation count and evaluates hang and slow
decisions in memory.

## Probe and metric computation

The store-only build uses one writer per active primitive subgroup in a
channel. Its local thread zero increments the raw send and receive totals
in GPU shared memory and publishes them to the UVA-mapped pinned frame
with volatile stores. A forced Ring run uses one subgroup slot per
channel. The host diagnostic thread sums the active subgroup totals and
records counter changes and increments over the sampling window. CPU
decoding computes rates and bandwidth. The CPU analyzer compares ranks,
selects slow candidates and diagnoses hangs.

The optional Ring wait probe (`NCCL_CCLD_WAIT_ENABLE=1`) also counts
waiting loop iterations on the corresponding GPU send and receive roles.
The CPU reads those cumulative counts from the separate mapped wait frame
and uses them for wait-based attribution.

## Training nodes

Use Python 3.9 or later and the modified NCCL library. Generate one
32-byte hexadecimal `CCLD_PULL_TOKEN` per training run and provide it
through the environment of all node agents and the analyzer. Select an
address on the job network for each agent. Start the following commands
from the repository root on **each** training node:

```bash
node_dir=$(mktemp -d /dev/shm/ccld.XXXXXXXX)
export NCCL_CCLD_NODE_SOCKET="$node_dir/agent.sock"
python3 -m ccld.tools.live_transport --socket "$NCCL_CCLD_NODE_SOCKET" \
  --bind "$TRAINING_NODE_ADDRESS" --port 38801 --records 65536
```

When the agent reports `CCLD_NODE_READY`, launch your usual training command
with `NCCL_CCLD_ENABLE=1`, the node's `NCCL_CCLD_NODE_SOCKET`, and your
modified NCCL library on every rank. Each agent should stay alive until the
analyzer has drained the final events. Remove the empty socket directories
when the agents stop. The [Slurm launcher](../slurm/fsdp_train.sh) manages
these steps automatically for its FSDP example.

## Analyzer host

Copy the Python source to a reachable CPU host and provide the same job
token. Start the analyzer from that checkout:

```bash
python3 -m ccld.analyzer.live --node node-a:38801 --node node-b:38801 \
  --baseline-ms 1 --poll-s 0.2 --decode-workers 4
```

The process uses the Python standard library and TCP connections to the
agents. CUDA, PyTorch and shared storage are unnecessary on the analyzer
host. Allocate its CPUs separately from training when possible. Each
connection uses the run token on the trusted job network.

Agent and analyzer must run the same wire protocol version. Set
`--decode-workers` to decode communicator/rank streams on separate CPU
processes while retaining the cross-rank decision state in the coordinator.
Worker processes start when their streams receive data. Leave CPU capacity
for the coordinator as well as the workers. Four workers gave the lowest
observed lag in the dedicated eight-GPU benchmark with ample CPUs.
See [results](../../RESULTS.md#pull-rate-and-separate-analysis-cpu) for the
CPU-time tradeoff and the effect of worker count.

Use `--coalesce-completed` with multiple decode workers to test batch
aggregation. For each rank and operation, the worker folds entered and
snapshot states superseded by a completion in the same batch. The
coordinator retains the logical event counts and the batch ordering
needed for coverage checks. Progress for unfinished operations remains
available for hang detection.

## Set the pull interval and retention

The analyzer sleeps for `--poll-s` after it catches up. If an agent reports
unread records beyond the returned batch, the analyzer requests another batch
immediately. Fully observed operations advance the slow-window state with
each batch. A hang decision that depends on a missing rank waits until the
analyzer has caught up with every node.

The analyzer reads operation identifiers, progress and timing on every
round and retains compact channel data for the slow-window candidates.
Tree grouping reads channel identities and parent links. Detailed channel
counters and rates are expanded when a candidate reaches root localization.

Size the node buffer using records per second from the training workload.
Dividing its capacity by the producer rate estimates how long the reader can
fall behind. Allow room for the poll interval, transfer and analysis time,
and scheduling jitter. Measure unread records as the producer cursor minus
the reader cursor. A full retention buffer can still be healthy when it
contains records the analyzer already read. Sustained input faster than
the analyzer can drain requires higher throughput or fewer emitted records.

The probe sampling interval (`NCCL_CCLD_POLL_US`), node pull interval
(`--poll-s`), and diagnosis thresholds (`--hang-s` and `--window-s`) control
different parts of the pipeline. The probe samples progress between
diagnostic snapshots. The pull interval controls delivery, while the
thresholds control when the available evidence warrants a diagnosis.

Diagnostic decisions go to stdout as they are made. SIGTERM after training
asks the analyzer to drain the streams and print a summary. The bounded
buffers are volatile. A reader that falls behind retention, a restarted
agent, or a truncated stream produces an explicit coverage gap. Adjust the
agent's `--records` and the analyzer poll interval for the event rate. The
Slurm FSDP example starts with 65,536 records and a 200 ms pull interval.
That is about 130.5 MiB of binary payload per node before Python and socket
overhead. Set `CCLD_PULL_RECORDS`, `CCLD_PULL_POLL_S` or
`CCLD_DECODE_WORKERS` to change these settings. The pinned frame size and
the node buffer size are configured independently.

For an offline recording instead, set `NCCL_CCLD_OUT_DIR` and leave
`NCCL_CCLD_NODE_SOCKET` unset. The binary decoder and the JSONL analyzer
support recorded traces.

## Observe cross-communicator waits

The live analyzer evaluates observed waits automatically when an H1 alarm
identifies a rank that has yet to enter. An observed wait links a missing
participant to an outstanding operation in another communicator using
exact trace IDs and rank memberships from the same training process.

An integration supplies the producer and each consumer operation that the
current thread will issue after the wait. For example, around host
synchronization on a CUDA event for a known collective:

```python
from ccld.tools.wait_observation import Operation, observe_wait

producer = Operation(producer_comm_hash, producer_op_count, producer_rank)
consumer = Operation(consumer_comm_hash, consumer_op_count, consumer_rank)
with observe_wait(producer, [consumer]):
    ready_event.synchronize()
```

`local_communicators()` lists the hashes, local ranks and sizes registered
by the calling process. The framework maps these registrations to its
groups and tracks the correct operation IDs at both endpoints. The Linux
node agent verifies the caller using Unix peer credentials and its
registered rank streams. It stores active scopes in RAM via a companion
socket with the suffix `.wait`. Each analyzer pull exports changes in the
active scopes. Waits that open and close between pulls can elapse without
adding a telemetry record.

The analyzer emits each domain's H1 alarm and adds `cross_domain` records
with observed `links`, `upstream_candidates` and `unresolved_ranks`.
Multiple concurrent symptoms and sources can appear in the same run.
A producer that completes or loses coverage clears the corresponding
attribution. The upstream candidates are the ends of the observed
dependency paths and identify where to continue domain-local diagnosis.
Use the same agent and analyzer checkout for observed waits.

## Interpret Tree slow decisions

An online `group_candidate` carries `group_kind` and `candidate_ranks` for
one comparable Tree layer. For a communicator-level result, read the
`tree_s1_channel_vote` record. A `candidate` status requires one uniquely
supported S1 rank across its eligible channels, using classified groups
from the same operation and window. Conflicting operations or missing votes
produce an unresolved result. Root attribution for Tree S2 needs its own
directional evidence.
