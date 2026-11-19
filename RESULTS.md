# Historical results and validation boundaries

These are observations from the preceding research runs, not a claim that this
clean export has already reproduced every result. Raw traces are not included.
The export preserves the selected asynchronous Ring-wait NCCL patch chain;
the portable Ring observer is a refactoring and requires fresh GPU validation.

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
not elapsed time. FSDP training, Tree wait attribution and multi-node wait
accuracy remain unvalidated for this export.

At the 24-channel, 512-frame, 2,048-queue-slot setting used in the full-build
smoke, the base mapped frame was 204,832 bytes per communicator, the optional
mapped wait frame was 196,608 bytes, and queue cells used 4,292,608 bytes of
host memory, excluding other metadata and CUDA event resources. This is a
different configuration from the paper's approximately 1,184-byte frame and
should not be presented as the same per-rank footprint. Diagnostic CPU usage
and training-step overhead have not been established for this export.

## Differences from the paper

- Historical tests used one 8-H100 node and CUDA 12.8.2, rather than the
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
