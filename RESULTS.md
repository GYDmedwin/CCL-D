# Historical results and validation boundaries

These are controlled research observations, not a claim that the release is
validated for arbitrary workloads. Raw traces are not included. The export
preserves the selected asynchronous Ring-wait NCCL patch chain; its portable
observer has now been exercised on the limited GPU matrix described below.

## Online memory transport

A full NCCL 2.24.3 build with CUDA 12.8.2 was tested on H100s with the
rank-to-agent Unix socket and independent TCP pulls. The online analyzer
decoded records directly into its decision state. The test directories
contained zero raw binary trace or per-rank JSONL telemetry files.

| Setup | Check | Observed result |
|---|---|---|
| Two training nodes, 16 H100s, analyzer on the batch host | 30 FSDP steps | All 12,512 expected rank completions arrived during or by the end of the run, with no reported gaps. The first completion was received during training. All 480 rank-step losses matched the earlier file-mode reference for the same model and inputs. |
| Same two-node allocation | Rank 12 enters the last AllReduce 12 s late | H1 named rank 12 while the operation was outstanding. All 112 rank operations completed and passed the numerical checks. |
| One 8-H100 training node and a different two-CPU analyzer host | Rank 5 enters the last AllReduce 12 s late | The remote analyzer named rank 5 while the operation was outstanding. It subsequently reported recovery, with all 56 rank operations completed and verified. |

Both hang checks used a 5 s threshold, a 250 ms snapshot interval and a
200 ms analyzer poll. The separate analyzer allocation requested zero GPUs
and ran the Python code from a private local copy. The FSDP case used Ring
wait counters and 32 probing frames. These runs verify transport and
decisions under the tested settings. Long-running CPU and network cost
remain unmeasured. See [deployment](ccld/docs/live.md) for the online
commands.

## Pull rate and separate analysis CPU

A separate CPU allocation received events from an eight-H100 FSDP training
node with 48 logical CPUs, Ring/Simple, wait counters and 32 probing frames.
The model used the same synthetic tokens in every pass. After warmup, the
file writer produced about 26,000 raw records/s on the training node. The
p95 rate across 100 ms bins was about 28,000 records/s.

The first tests found three issues under sustained traffic. The file decoder
could stop before it revisited the earlier rank files, the online analyzer
deferred retiring ready operations until its pull cursor caught up, and the
16,384-record node buffer overwrote unread records in several runs. After
fixing the two software issues, a 65,536-record buffer sustained the 300-step
runs, but the uncompressed 1,500-step test still reported a coverage gap.
The analyzer kept training separate from the diagnosis and reported the
gap rather than using incomplete telemetry for rank attribution.

The next version compressed the TCP payload and increased the maximum
batch from 1,024 to 4,096 raw records. The 1,500-step, 200 ms serial run
received all 312,016 rank completions. Its binary payload was about 1.82 GiB
before encoding and 131 MiB on the wire, excluding TCP/IP headers. Three
300-step runs on a separate four-CPU analyzer with the same 65,536-record
buffer also delivered all 62,416 completions each.

| Idle poll interval | Requests over process lifetime | Maximum unread records at a node pull |
|---|---:|---:|
| 50 ms | 455 | 16,969 |
| 200 ms | 149 | 8,613 |
| 1 s | 68 | 29,284 |

These short cases ran once per interval. The 200 ms setting gives a starting
point for this workload. The highest observed backlog includes scheduling
and processing variability as well as the poll interval. A full retention
buffer can still contain previously consumed records.

With compression and the same 200 ms interval, the following three
1,500-step cases were run in one allocation. Each used a 65,536-record
buffer. CPU masks applied to the entire analysis process tree.

| Analysis CPU mask | Decode workers | Rank completions received | Maximum unread records | Analyzer CPU seconds including workers |
|---:|---:|---:|---:|---:|
| 2 logical CPUs | 1 | 234,291 / 312,016, overrun reported | 65,662 | 22.4 before overrun |
| 4 logical CPUs | 1 | 312,016 / 312,016 | 44,261 | 30.2 |
| 4 logical CPUs | 2 | 312,016 / 312,016 | 9,926 | 64.7 |

The two complete cases measured 25.04 and 25.11 ms per step on the same
training configuration. The decoded events were processed in two worker
processes and combined in order by the coordinator when parallel decoding
was enabled. Up to two batches may already be in the analyzer while worker
processing proceeds. The maximum unread metric measures the node-side
cursor lag and excludes those in-flight batches.

The site's per-job limit allowed a separate 96-logical-CPU analysis
allocation with zero GPUs. The training allocation and the 200 ms poll,
compression and 65,536-record buffer were the same for all three cases.
Each 1,500-step case received 312,016 completions and matched the 12,000
rank-step loss values in the first case.

| Decode workers with 96 CPUs available | Maximum unread records | Analyzer CPU seconds including workers | Mean max-rank step time |
|---:|---:|---:|---:|
| 2 | 11,409 | 52.3 | 24.09 ms |
| 4 | 8,776 | 55.9 | 24.09 ms |
| 8 | 11,537 | 61.5 | 24.65 ms |

Four workers had the smallest observed lag in this rotation. The eight-worker
case did not improve the lag and consumed more CPU. Each setting ran once.
The coordinator still joins rank evidence, and larger jobs with more
communicators can distribute decode work across more active shards. This
rotation does not establish an optimal worker count for other workloads.

The complete 300-step file-mode reference used 36.0 and 37.3 CPU seconds
for remote decoding plus analysis. The separate 300-step compressed
serial run at 200 ms used 6.2 CPU seconds on the analyzer. The node-side
probe and publisher plus agent used approximately 0.79 logical CPUs
during timed steps in that compressed run. These are job-level observations,
with one run for each long comparison setting. The experiments used a
single training node and describe functional coverage and observed CPU
and transport costs at that scale.

## Deferred channel decoding and Tree root candidates

Two eager and two deferred passes ran in eager, deferred, deferred, eager
order. The unchanged compressed transport pulled from a separate
eight-H100 FSDP node. Analysis used four decode workers, 200 ms polling
and a 65,536-record buffer. Every 1,500-step pass received exactly 312,016
rank completions with no coverage gaps. All 12,000 rank-step loss values
matched the first pass.

| Online decode path | Analyzer CPU seconds per pass | Maximum unread records per pass |
|---|---|---|
| Eager | 56.20 / 56.58 | 12,859 / 24,307 |
| Deferred | 31.52 / 31.87 | 7,822 / 40,760 |

Deferring the detailed metrics reduced mean analyzer CPU time by 44%
in this comparison. All four passes had full event coverage and transferred
130.93–131.18 MiB of compressed payload. Peak unread backlog varied between
7,822 and 40,760 records. The deferred path saw both endpoints of this range.

Tree layer results are now reported as candidates. The online analyzer
uses cross-channel S1 votes for communicator-level attribution, using
complete classified group evidence from the same operation and slow
window. Evidence from different operations, missing groups and
non-unique votes leave the communicator root undetermined.

A replay of the previous two-node Tree/Simple traces ran all six held-out
S1 injections and three controls through the online state machine. In
each injection the rank with the longest duration in the selected round
was a different rank from the injected one. The aggregated S1 candidate
matched the injected rank in all six. The three controls produced no
slow candidates. The separate strong S2 trial produced local candidate
groups and no communicator-level root. These are replays of recorded
traces with the frozen configuration and two injected S1 ranks.

Slurm job 16396 exercised the full NCCL 2.24.3 build through the live
binary transport on two eight-H100 nodes. It ran 64 MiB Tree/Simple
AllReduce with three warmup and three measured rounds per case. All
96 rank completions per case arrived and passed numerical verification.
The control emitted zero slow results. With a 3 s delay before the last
three collectives at rank 3, all 14 eligible channels gave that rank the
S1 vote for the confirmed candidate. Both cases finished without coverage
gaps. A separate analysis process ran on one of the two training nodes.
This functional test used 200 ms pulls, a 1 s slow window and two detections
for confirmation.

## Batch aggregation of completed progress

Jobs 16397 and 16398 ran two ordinary and two coalesced analyzer passes in
ABBA order. An eight-H100 FSDP trainer used the same Ring/Simple NCCL
build for each 1,500-step pass. The separate analyzer used four decode
workers, a 200 ms idle poll and 65,536 records of retention.
Every pass received 312,016 rank completions with complete coverage,
and all 12,000 rank-step loss checks matched the reference.

| Analyzer | Batch apply CPU seconds per pass | Total analyzer CPU seconds per pass | Peak unread records | Receive-to-apply P99 (ms) |
|---|---|---|---|---|
| Ordinary | 1.79 / 1.72 | 29.30 / 27.62 | 7,804 / 12,617 | 293.2 / 288.0 |
| Coalesced | 1.31 / 1.25 | 29.34 / 28.68 | 7,134 / 8,335 | 288.7 / 285.6 |

The workers folded about 307,000 superseded entry states per coalesced
pass, keeping logical event counts and the completed channel evidence.
The coordinator used 27% less CPU for batch application and evaluation.
Worker CPU increased and total analyzer CPU stayed near 28–29 seconds.
Peak unread counts were lower in both pairs, with training step duration
varying across runs. Receive-to-apply timing covers time after the
analyzer accepts a batch, including its wait for the next pull iteration.

CPU-only replay job 16400 read one archived 300-step FSDP capture.
Each pass applied the same 187,326 binary records in the same batches
and received 62,416 rank completions. The logical event counts and the
one preliminary slow finding agreed across all four passes. The file
sink terminates at EOF. After validating expected completions and
rank coverage, the replay supplies eight end markers to release the
remaining window state. Coalescing reduced coordinator batch-apply
CPU from 1.16–1.17 to 0.97–0.98 s on this fixed input. Either path
can be selected for future runtime tests.

## Online release validation

Slurm job 16401 ran the full NCCL 2.24.3 build on two eight-H100 nodes
with four decode workers and `--coalesce-completed`. Each case used
64 MiB AllReduce and the Simple protocol. Its analysis process ran on
one of the training nodes. The preflight observed idle compute processes
on one node. The cases check diagnostic and numerical behavior.

| Case | Verified rank completions | Diagnostic result |
|---|---:|---|
| Tree control | 96 | No alerts |
| Tree, 3 s delay at rank 3 on three rounds | 96 | S1 rank 3 with all 14 comparable channels voting for it |
| Ring, 12 s delay at rank 3 on the final round | 64 | H1 rank 3 during the delay, followed by a recovery report |

All three cases had complete event coverage. In the Ring case the H1
report appeared 5.33 s after the injected-delay marker at a 5 s threshold.
The injector and analyzer shared the same host clock for that observation.

## Cross-domain H1 over memory pull

Slurm job 16448 ran a PyTorch TP8×DP2 projection and gradient workload on
two eight-H100 nodes using instrumented NCCL 2.24.3. The workload supplied
exact collective IDs at CUDA completion waits. The node agents sampled
active waits on analyzer pulls, and the H1 threshold was 2 s. Each case
passed 80 numerical rank-step checks and delivered all 160 expected rank
completions with complete telemetry coverage.

| Case | Online diagnostic result |
|---|---|
| Control | No H1 alarms or wait records |
| TP entry delay | Seven DP H1 findings linked to their observed waits on TP0 |
| Simultaneous TP and independent DP entry delays | Both TP0 and DP0 raised H1 at operation 4. Six DP groups linked to TP0, while DP0's independent H1 had no observed link to TP0. |
| TP delay with an independent DP0 input | DP0 operation 5 linked to TP0 operation 4. TP1 operation 5 linked to seven earlier DP operations. |

The control in job 16447, which emitted wait events at every scope
transition, carried 220 wait records. Sampling at pull time reduced that
count to zero in job 16448. These runs exercise H1 attribution with
explicit training-runtime wait evidence.

### Shared-memory wait registration

The training process now attaches to a shared page during setup and
updates active wait scopes locally. A Linux CPU-only test with eight
concurrent rank callers at 100 registrations per second each measured
the complete scope with an empty body. Median time fell from 2.91 ms
with per-wait socket acknowledgements to 0.21 ms with shared memory.
P95 fell from 3.47 ms to 0.23 ms. These are the means of the respective
percentiles over two repetitions per implementation.

Job 16465 ran both implementations sequentially on the same two
eight-H100 nodes. Each run alternated eight timing blocks with the
same NCCL probes and online analyzer. The table shows the distribution
of the longest rank time at each measured step. Each cell uses
236 measured steps. Each timed rank-step had two wait registrations
when enabled.

| Wait registration path | Off median | On median | Added median time | Off P95 | On P95 |
|---|---:|---:|---:|---:|---:|
| Per-wait socket | 2.43 ms | 5.63 ms | 3.20 ms | 3.23 ms | 10.13 ms |
| Shared memory | 2.45 ms | 2.63 ms | 0.17 ms | 3.41 ms | 4.12 ms |

Each run passed 8,384 numerical rank-step checks and delivered 16,768
rank completions. The shared-memory build also passed the four
diagnostic cases above. The independent TP0 and DP0 H1 alarms stayed
separate, and DP0 operation 5 still linked to TP0 operation 4. These
timing results describe the short-step TP8×DP2 workload and its wait
frequency.

## Two-node collective and training checks

On 2026-11-19, the full, non-fault-injection NCCL build was exercised on two
eight-H100 nodes with CUDA 12.8.2 and PyTorch 2.4.1. Every process verified
its mapped NCCL library and runtime version. With Ring/Simple, Ring/LL and
Ring/LL128, AllReduce, AllGather and ReduceScatter each passed six numerical
checks per rank and produced matching completed probe records. Tree AllReduce
passed the same checks with each of the three protocols, including the
topology needed for same-level comparisons. Across these 12 protocol/operation
cases there were 1,152 numerical checks and 1,152 completed rank-operation
records. Each case was run once without injected faults; these are collective
and telemetry checks, not fault-localization accuracy measurements.

A separate two-node AlltoAll run passed 96 numerical checks with the modified
library loaded. PyTorch's `all_to_all_single` used grouped point-to-point
operations in this setup: the traces contained communicator initialization
and heartbeats but no AlltoAll entry, completion, or explicit coverage-gap
record. Its numerical success therefore does not establish AlltoAll probe or
diagnostic coverage; an absent record must not be interpreted as a healthy
AlltoAll operation.

The FSDP example trained a 117,475,328-parameter causal decoder with synthetic
tokens, bfloat16, and AdamW for 20 steps on all 16 GPUs. Each rank had a
nonzero first gradient; the first and last rank-0 losses were 9.190777 and
0.687858. All 320 rank/step loss values matched exactly between upstream NCCL
and the enabled CCL-D build in this run. The enabled run completed 5,440
AllGather, 2,880 ReduceScatter and 32 AllReduce rank-operations, with no
recorded gaps; the live decoder observed a completion during the training
loop. A smaller two-million-parameter, eight-GPU smoke run also had matching
losses and complete traces. These fixed-token tests check execution and run
consistency, not convergence on a real dataset or model quality.

Two additional 16-GPU, 30-step live-analyzer training runs used a 1 ms initial
baseline, a one-second window, two repetitions, and a dynamic baseline after
20 operations. One run produced an S1 decision with no fault deliberately
injected; the other did not reach the repetition threshold. The analyzer's
baseline aggregates heterogeneous FSDP collectives, so these short-window
settings require calibration for a training workload. We do not treat this
no-injection result as a measured general false-positive rate or as proof that
a hardware fault occurred. The separately gated one-shot Ring-wait observer
was not used for the multi-node or mixed-collective training workload.

The table shows a small paired rotation on the 117M model. Each cell is the
average of two runs, where each run averages the maximum rank step time over
the last 25 of 30 steps. The order was upstream, base probe, base plus wait,
base plus wait and live analyzer, then the reverse order. `Base` and `Wait`
write binary traces without a concurrent decoder; `Live` adds the decoder and
base analyzer. These are short exploratory measurements, not an overhead
guarantee or an isolation of the wait-counter cost.

| Placement | Upstream | Base | Wait | Live |
|---|---:|---:|---:|---:|
| 1 node, 8 ranks | 34.51 ms | 32.61 ms | 33.21 ms | 34.31 ms |
| 2 nodes, 4 ranks each | 30.84 ms | 32.98 ms | 34.57 ms | 34.50 ms |
| 2 nodes, 16 ranks | 80.03 ms | 133.28 ms | 123.16 ms | 143.37 ms |

Across these runs all rank/step losses matched the corresponding upstream
run exactly, and trace coverage checks passed. The two eight-rank placements
used the same initial model and per-rank synthetic tokens; their 240
corresponding loss values differed by at most 0.00250. They used separate
allocations, and the two-node eight-rank upstream pass means ranged from 26.56
to 35.12 ms. Thus the observed timing differences are suggestive but do not
isolate a causal mechanism. The 16-rank case also changes the rank count and
global batch. The detailed, sanitized aggregates are in
[fsdp-multinode-validation-20261119.json](ccld/evidence/fsdp-multinode-validation-20261119.json).

To investigate the 16-rank timing, we repeated the same 117M model in a
mirrored rotation with fixed CPU affinity, 30 steps per pass, two passes per
configuration and two hardware threads available per physical core. `Off`
loads the modified library with CCL-D disabled; `Base` enables the base probe
with 512 frames and without the Ring wait counters or live analysis. Each cell
averages the maximum-rank step time after excluding five warmup steps.

| Physical cores per node | Upstream | Modified, off | Base probe |
|---|---:|---:|---:|
| 8 cores, 16 hardware threads | 84.69 ms | 70.39 ms | 135.68 ms |
| 24 cores, 48 hardware threads | 50.41 ms | 48.43 ms | 46.96 ms |

![FSDP step time across CPU affinity settings](ccld/evidence/fsdp-cpu-affinity-20261119.svg)

The higher-affinity allocation substantially reduced the observed training
times and the gap between the upstream and base-probe runs. Sampled GPU SM
clocks during active periods were 1,980 MHz in these passes. The results show
sensitivity to available CPU resources and scheduling; they do not isolate
which NCCL, training or monitoring threads are responsible, establish a
general overhead, or rule out other hardware and network effects. With only
two passes per setting, the lower `Base` value in the larger allocation is
not evidence that instrumentation accelerates training. An 8-versus-24-core
rotation with one hardware thread per core showed still stronger sensitivity;
both rotations are retained in the aggregate evidence.
The figure can be regenerated with `ccld/tools/plot_fsdp_cpu.py` and the
aggregate JSON; it requires Matplotlib.

The example Slurm launcher was subsequently rerun on two nodes using 48
allocated logical CPUs per node. All 16 ranks recorded a 48-CPU affinity,
completed ten FSDP steps and produced complete AllGather, ReduceScatter and
AllReduce traces.

We also compared the base probe with 32 versus 512 frame blocks, using the
same model, 16 ranks, a 200 µs poll and two paired passes per setting. The
Ring wait counter and live analyzer were off for this comparison.

| Physical cores per node | 32 frames | 512 frames |
|---|---:|---:|
| 8 cores, 16 hardware threads | 130.71 ms | 151.04 ms |
| 24 cores, 48 hardware threads | 48.87 ms | 53.76 ms |

All eight passes produced complete traces without reported drops and the
training loss checks passed. Linux scheduler counters sampled for threads
present at both ends of the measured interval show the CCL-D threads waiting
to run for an average of 20.88 versus 24.17 ms per rank per step in the
limited-core 32/512-frame cases; at 24 cores the corresponding values were
0.009 versus 0.020 ms. These are sums across monitored threads, not elapsed
communication time, and exclude threads that appeared or exited between
snapshots. The small sample supports CPU scheduling pressure as a contributor;
it does not prove that frame scanning is the only cause or that 32 frames is
sufficient for arbitrary training workloads.

For the same frame comparison, the sampled threads' approximate CPU runtime
per node was as follows. One CPU equivalent corresponds to one logical CPU
busy for a time equal to the sum of timed steps; values average two passes and
two nodes.

| CPUs available per node | Frames | CCL-D probe and writer | Other matched training-process threads |
|---|---:|---:|---:|
| 8 physical cores, 16 logical CPUs | 32 | 0.49 CPU | 13.91 CPUs |
| 8 physical cores, 16 logical CPUs | 512 | 0.52 CPU | 13.92 CPUs |
| 24 physical cores, 48 logical CPUs | 32 | 1.60 CPUs | 14.87 CPUs |
| 24 physical cores, 48 logical CPUs | 512 | 1.63 CPUs | 14.98 CPUs |

For example, 0.49 CPU is approximately 0.26% of the node's 192 logical CPUs,
but approximately 3.1% of the 16 logical CPUs available to the process group.
The other group includes all other matched threads in the training processes,
which can include PyTorch, NCCL and CUDA threads; the separate live analyzer
was disabled and is not included. CPU
runtime and runnable waiting are different measurements. CPU equivalents here
divide sampled thread runtime by the sum of each rank's timed step durations;
because that denominator excludes work outside the timed steps, the values
are approximate, not full-interval utilization measurements. Threads created
or exited between snapshots are also excluded. The apparent increase in probe
CPU equivalents with more available CPUs occurred alongside faster training;
it is not a controlled per-operation CPU cost.

The published FSDP Slurm example was then rerun with its new 32-frame
default, 48 allocated logical CPUs per node and Ring wait counting enabled.
All 16 ranks completed ten steps with matching expected NCCL mappings and
complete traces; the separate AlltoAll numerical smoke also passed.

### Online decoder and analyzer CPU

An additional short two-node, 16-rank FSDP comparison used 32 frames, Ring
wait counting, a 200 µs probe poll, a decoder following every 50 ms and the
base analyzer following every 200 ms. There were two passes per setting, each
with 30 steps. The decoder and analyzer ran as separate processes on the master
node and shared the same logical-CPU affinity as its training processes.
Their CPU times were sampled over the rank-0 training interval; the probe
thread estimate uses timed steps after five warmup steps.

| CPUs available per node | Probe and writer, average per node | Decoder, master node | Analyzer, master node |
|---|---:|---:|---:|
| 8 physical cores, 16 logical CPUs | 0.45 CPU | 0.35 CPU | 0.55 CPU |
| 24 physical cores, 48 logical CPUs | 1.22 CPUs | 0.65 CPU | 0.41 CPU |

Each value is an approximate logical-CPU equivalent. In the 16-logical-CPU
setting, the two sidecars together used approximately 0.90 CPU on the master
node in addition to the probe threads. The two CPU measurement windows are
not identical, and these values are not a universal full-system maximum.
The online analyzer currently rereads accumulated JSONL on each poll; CPU
cost and analysis latency may therefore change with trace size and poll rate.

All eight passes had complete expected traces and matching loss checks. An
external GPU process appeared on the second node during the Slurm steps for
the four 16-logical-CPU cases; its utilization and exact overlap with timed
training steps are unknown. No process was detected during the four larger
CPU-allocation cases, but the periodic monitor can miss brief activity. The
short, potentially shared runs are not a clean performance comparison or a
long-training CPU characterization. The compact per-pass measurements are in
[fsdp-online-cpu-16371.json](ccld/evidence/fsdp-online-cpu-16371.json).

## Two-node, 16-rank bounded fault checks

The full non-injection NCCL build was also tested with the PyTorch AllReduce
workload and host-side software faults. Each line below is one independent
trial on two nodes, using Simple protocol, a five-second hang threshold, a
0.5-second slow window, and two slow detections before classification.
Faulty S1 ranks slept for one second before each of four operations. H1 ranks
skipped the operation and exited after a bounded wait; these are software
simulations, not hardware failures.

| Algorithm | Scenario and injected rank | Live result |
|---|---|---|
| Ring | no fault | No confirmed alert |
| Ring | S1, rank 2 | S1 with rank 2 as the only candidate |
| Ring | S1, rank 10 | S1 with rank 10 as the only candidate |
| Ring | H1, rank 2 | H1 with rank 2 as the only candidate |
| Tree | H1, rank 10 | H1 with rank 10 as the only candidate |
| Tree | S1, rank 10 | Rank 10 appeared as S1 in some groups, but other groups produced conflicting S2 candidates; no unique overall localization |

For the last case, 42 group decisions reported S1/rank 10, 221 reported S2
and 73 were undetermined; the union of all reported candidates covered all 16
ranks. Those are group-level decisions over multiple operations, not independent
trials. Detection on the observer's own host took 2.12 s in the Ring/S1 case
and 5.15 s in Ring/H1, including the specified thresholds and file polling;
cross-host detection latency was not calculated. The no-fault and S1 trials
each passed 112 rank-operation numerical checks, and both H1 trials passed
all 48 pre-injection checks. These few cases do not replace the planned
five-repeat, multi-protocol and multi-fault matrix.

An optional offline S1-only cross-channel vote was then tested on the Tree
rank-10 trial above and on a separate, restricted Tree-wait build. In the
rank-10 trial it selected the injected rank with 14 votes out of 14 eligible
channels, while all other ranks received zero S1 votes. Three matched
no-injection controls on the Tree-wait build had no confirmed slow alert; six
held-out S1 trials, targeting a leaf and an internal node on a reference
channel three times each, all selected the injected rank 14-to-0. A stronger
simulated S2 trial classified all 112 groups as S2, and the S1-only vote
correctly returned `not_applicable`. These are limited offline observations,
not a validated online or general Tree S2 detector. The stronger S2 test used
a different restricted library from the public training build. Detailed
extension results remain in the research workspace.

Five separate weaker, 5-million-cycle Tree S2 trials on that restricted build
did not trigger the base slow gate, so no S2 localization was evaluated in
those trials. Their host markers record configured injection but do not
directly prove that the device spin executed.

## Released online observer: H100 revalidation

On 2026-11-19, we ran 30 separate single-node, eight-H100 jobs using the
released decoder and online observer. For each Ring protocol, five jobs used a
bounded S2 send-delay injection and five were no-fault controls. The injected
ranks were 0 twice, 1 twice, and 7 once per protocol. Each job ran six
AllReduce operations of 262,144 float32 elements; all 1,440 rank-operation
numerical checks and telemetry validation passed.

| Protocol | S2 faults with correct wait rank | Controls with online alarm | Base slow classification on injected S2 | Controls over threshold when gate forcibly bypassed |
|---|---:|---:|---|---:|
| Simple | 5/5 | 0/5 | S2: 5/5 | 1/5 |
| LL | 5/5 | 0/5 | S3: 5/5 | 0/5 |
| LL128 | 5/5 | 0/5 | S3: 5/5 | 0/5 |

The last column is a separate **offline forced-score** analysis, not an
online false alarm: the normal Simple trial did not pass the confirmed-slow
gate, and the forced score did not produce a majority rank candidate. It shows
that the wait score can cross the old threshold on normal data and should not
be used as an ungated detector. The base analyzer's S3 classifications on LL
and LL128 are incorrect for these pure S2 injections, even though the separate
wait-based rank candidate was correct.

Across the 15 fault jobs, injection-to-first-alert was 1.98–2.09 s and
injection-to-wait-decision was 3.54–3.66 s. These same-host measurements
include trace polling and the configured 1.2-second gaps between operations;
they are not isolated analyzer execution time.

This test used the same restricted-function, fault-enabled library and the
previously fitted `0.6469719382` threshold, a strong 50-million-cycle software
send delay, and the historical 22-channel workload signature. Trials ran on
two H100 hosts, one host per job; this was not a multi-node communicator test.
No Tree, H1/H2/H3, S1/S3 mixed fault, FSDP, weaker fault, or full non-injection
library fault-localization claim follows from this matrix. The aggregate data
are in [online-ring-s2-validation-20261119.json](ccld/evidence/online-ring-s2-validation-20261119.json).

## Synthetic Ring/S2 faults

On one node with eight H100 GPUs, a restricted fault-injection build was
evaluated with a bounded 50-million-GPU-cycle send delay. The threshold was
fit on separate training data and then held fixed; ranks 0, 1 and 7 were held
out for fault validation. Each protocol had one new no-fault control.

| Ring protocol | Held-out S2 faults uniquely located | Control alarms | Injection to live decision |
|---|---:|---:|---:|
| Simple | 3/3 | 0/1 | 3.946–4.102 s |
| LL | 3/3 | 0/1 | 4.125–4.409 s |
| LL128 | 3/3 | 0/1 | 4.190–4.467 s |

All 12 jobs passed their recorded numerical checks. The measurement includes
same-host observation, file polling and configured 1.2-second gaps between
operations. The historical observer required a confirmed base slow diagnosis
and completed operations 4–6. These small, strong synthetic-fault samples
do **not** establish a general accuracy or false-positive rate. The observed
base paper-inspired analyzer's own attribution and classification were not
always correct; the wait result was reported separately.

The historic fitted threshold was about `0.64697`, bound to a specific
restricted fault-enabled library, 8-rank workload, 200 µs polling, data
signature, code hashes and host setup. It is **not** a validated default for
this export or a full non-injection build. Fit and independently validate a
threshold for the intended environment.

## Example overhead observations

These compare the full wait-enabled, no-injection library with upstream NCCL
on one 8-H100 node, Ring/Simple, using five ABBA pairs and 300 nonblocking
iterations per pass. They include the base instrumentation and wait counters.

| Operation and size | In-place elapsed time | Out-of-place elapsed time |
|---|---:|---:|
| AllReduce, 64 MiB | +1.73% | +1.77% |
| AllGather, 64 MiB | +1.61% | +2.12% |
| ReduceScatter, 64 MiB | +1.78% | +1.87% |
| AllReduce, 256 MiB | +1.30% | +1.27% |

This is a limited single-node measurement, not an aggregate or a direct
comparison with the paper's H20 multi-node measurements. The fault-validation
and full performance binaries were different. Waiting-counter iterations are
not elapsed time. Tree wait attribution and multi-node wait localization
accuracy remain unvalidated for this export.

At the 24-channel, 512-frame, 2,048-queue-slot setting used in the full-build
smoke, the base mapped frame was 204,832 bytes per communicator, the optional
mapped wait frame was 196,608 bytes, and queue cells used 4,292,608 bytes of
host memory, excluding other metadata and CUDA event resources. This is a
different configuration from the paper's approximately 1,184-byte frame and
should not be presented as the same per-rank footprint. Diagnostic CPU usage
and general training-step overhead have not been established for this export.

## Differences from the paper

- Earlier tests used one 8-H100 node and CUDA 12.8.2, rather than the
  paper's H20 multi-node setup and CUDA 12.2. The published NCCL base here is
  v2.24.3-1. The Ring-wait rule is an experimental addition, not a stated
  paper algorithm.
- The default completion path polls CUDA events; host-function completion is
  optional. Traces are written as binary records and decoded to JSONL. The
  wait ranking consumes completed operations and therefore cannot itself
  diagnose a kernel that never completes; separate periodic snapshots feed the
  hang analyzer.
- The implementation retains both reciprocal change-count rates and count
  deltas per millisecond. Their interpretation is ambiguous for fixed-window
  samples; neither the base rate heuristic nor the wait rule is a proven
  physical-link root-cause detector.
- H2/H3 GPU faults are bounded simulations or operator-level mismatches, not
  actual hardware failures. Coverage for arbitrary models, overlapping
  operations, capture/replay, and distributed deployments needs further work.

## Export checks

The reconstructed NCCL base tree matches the official v2.24.3-1 tree, and the
ordered patch series applies cleanly. The complete no-injection NCCL library
was built successfully for `sm_90` with CUDA 12.8.2 using eight CPU cores; the
source manifest matched this export. The Python test suite covers trace
decoding, dropped-record handling, base diagnosis and a synthetic live Ring
gate. The refactored normalized Ring score was also compared with the prior
implementation on 1,000 randomly generated valid topologies. These tests are
not a substitute for repeating the GPU validation matrix on this export.
