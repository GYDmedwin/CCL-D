# Telemetry contract and prototype limitations

The online path reads binary telemetry directly. The offline decoder can export
one JSON object per line. A rank may have more than one communicator.
`comm_hash` identifies the communicator and `op_count` is its rank-local
collective sequence. These IDs do not need cross-host clock synchronization.
Trace files are per rank, process, and communicator. Records use communicator
ranks.

Required fields for all records: `event` (`comm_init`, `heartbeat`, `entered`, `snapshot`, `complete`, or `fault`), `comm_hash` (stable string), `rank` (zero-based), `world_size`, `mono_ns` (integer local monotonic nanoseconds). `entered`, `snapshot`, `complete`, and `heartbeat` additionally require a nonnegative integer `op_count`; `snapshot` and `complete` require nonnegative integer `duration_ns` measured on that same rank. `entered` records the host-side invocation, even if the device never begins. `heartbeat` has the highest local invocation counter, including while idle, so not-entered detection has evidence. A `snapshot` is emitted periodically even while a kernel is hung. A `complete` must be emitted only after the stream completion signal, never when a call merely returns.

Operation records carry `op`, `algorithm` (`Ring`/`Tree`), `protocol`, `count`, `datatype`, `bytes`, `n_channels`, and `channels`. Each channel is `{"id": 0, "tree_parent": -1, "send_count": 4, "recv_count": 4, "send_rate": 0.25, "recv_rate": 0.25, "send_delta_per_ms": 0.4, "recv_delta_per_ms": 0.4}`. For Tree every rank must report its **actual parent for each channel**, using `-1` for the root; a global tree-depth field or an assumed rank-to-tree mapping is insufficient. Rates with no change are JSON `null`, not infinity or zero. The `bytes` value is the collective's logical size; for all-gather/reduce-scatter the busBW size is total data across ranks. Preserve the original count and datatype separately.

A producer must identify and disclose dropped/overwritten frames, unsupported algorithms, capture/replay, grouped operations, and allocation errors rather than silently returning fabricated metrics. Eight reusable pinned blocks are not safe to overwrite when more than eight operations are in flight. `cudaLaunchHostFunc` callbacks cannot use CUDA API calls and do not execute normally when previous device work never completes; periodic snapshots must be independent of the callback. GPU atomic behavior on mapped host memory depends on platform support, and GPU atomics are not necessarily atomic from the host's perspective ([NVIDIA CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)). This store-only version supports mode 1; requesting mode 2 disables telemetry.

## Analyzer semantics and limitations

With the Tree/Simple AllReduce wait probe, each channel snapshot and completion
carries `tree_wait_peers` entries for its parent and children. Each entry
contains a peer `rank`, `data_spins`, and `credit_spins`. The
`NCCL_CCLD_TREE_SNAPSHOT_MS` setting controls the snapshot interval for
active Tree probes. `NCCL_CCLD_DUMP_MS` still controls the heartbeat interval
and snapshots for other operations. The online analyzer can request clock
samples from each node agent to bound the all-rank entry time.

The analyzer groups records by communicator and operation counter. Ring
compares progress across ranks with matching channels. For Tree H3 it
compares each rank's progress with that rank's completed progress on the
same operation signature and channel topology. Tied minima retain their
candidate ranks and leave the root undetermined. A missing reference leaves
the diagnosis undetermined. Equal-depth comparisons remain available for
Tree slow analysis.

A completed round with inconsistent operation signatures emits
`collective_mismatch`. The analyzer locates the minority ranks when one
signature has a strict majority. It keeps the result undetermined on a tie
and excludes the round from slow baseline learning.

The live node agent records a premature rank stream closure with
`comm_hash`, rank and the last observed operation counter. It can verify
process exit using the local process identity captured at connection time.
A verified exit can produce a `rank_exit` diagnosis. A socket closure
without process evidence remains a rank-scoped coverage gap. The anomaly
output preserves earlier H2 or H3 diagnoses if rank exits follow them.

An incomplete earlier operation makes subsequent operations indeterminate, since ranks advancing from a mismatch can otherwise make blocked peers look like a new H1 cause. This conservative ordering can delay a diagnosis when independent streams overlap. For H2, a completion only counts as an un-hung rank if its duration is below the hang threshold; callbacks released after an error or abort can otherwise make victims appear to be causes. Completions are host-callback observations, not proof of successful CUDA execution.

Slow detection uses the requested initial baseline, then the mean of the first `min(100 rounds, 2 minutes)` rank-maximum durations. A complete window chooses its maximum duration spread. By default no rounds are skipped; `--baseline-skip` is an explicit warmup-related experimental deviation. P is not clamped; a value outside [0,1] emits a baseline warning. The static baseline remains in effect until the history limit has actually been reached. Repeated detections are counted per Ring or Tree-layer group; S1/S2/S3 use the requested ratio and boundaries. When a run ends before a full window, `--flush-last-window` opts into an explicitly partial window for testing. Rank 0's **local** monotonic completion timestamps delimit the windows; other ranks' local timestamps are not compared across hosts. This does not establish a true global event ordering.

For S2 the default score is the mean of available send/recv values over compared channels; `--rate-metric delta` uses measured count increments per millisecond instead. Both apply the specified minimum-score rule. Values without valid samples are unavailable and never silently scored as zero. The paper and supplied document describe a reciprocal of change counts yet call smaller values slower. If the count is literally the number of changes in one *fixed* window, the reciprocal decreases when there are more changes, i.e. points in the opposite direction. The paper's 1/2-versus-1/7 example appears to count observations needed to achieve a fixed amount of progress. This ambiguity must be assessed empirically and both quantities must be retained; neither is a validated root-cause metric here.

Detection latency is null for offline input or cross-node clocks. With `--follow --fault-events DIR`, it can be measured from an injection marker to analyzer observation only when the injector and analyzer run on the same Linux host and the marker maps unambiguously to one communicator; that duration includes file polling. The inferred target is the next collective after the marker on that rank. An ambiguous communicator or another host leaves latency null. Location latency is local Python decision processing time, not the paper's distributed pipeline end-to-end latency. `--follow` polls complete lines, but currently rereads all files and is a prototype, not a scalable online service. Incomplete final lines are withheld; malformed records are reported. Controlled NCCL tests and their limitations are recorded in `RESULTS.md`.

For count-based Ring localization the analyzer requires matching channel sets. For rate comparisons, a channel/direction that has samples on only some ranks invalidates the comparison rather than averaging different subsets. A direction missing on every rank may be omitted, but if no comparable samples remain, localization is indeterminate. These checks do not resolve the observed S2 propagation ambiguity.

When every rank is still hung, the analyzer first compares its complete
`(op, count, datatype, algorithm, protocol)` signatures. If one signature
has a strict majority and other ranks differ, it classifies H2 and returns all
minority ranks as candidates, explicitly recording the assumption that the
majority is correct. A tie or missing signature is `undetermined`; consistent
signatures proceed to the H3 count rule. The paper introduces Operation Type
Set for inconsistent operations, but its stated decision tree identifies H2
from ranks that have already completed. Applying a majority-signature rule
when all ranks hang is this reproduction's additional heuristic; a minority
is not proof of causation. A completed round is also checked for inconsistent signatures.

For a completed operation with no comparable rank group, the analyzer emits a
`category: coverage`, `kind: undetermined` record with a reason and topology
warnings. This signals that slow comparison is unavailable; it does not
assert that a slow anomaly occurred. One record is emitted per affected
operation, including on an ordinary single-node chain-shaped Tree.
