# NCCL patch series

The checked-in NCCL source already contains these changes. For a reviewable
reconstruction, start from the clean official `v2.24.3-1` commit
`6aae37927840b2bd7b7d42d2f0050e75f88ee97f` and apply the numbered patches in
lexicographic order. Gaps in numbering reflect omitted experiments.

The series adds base probes, compile-time-gated bounded fault simulations,
ReduceScatter Ring probing, configurable frames and queues, event completion,
binary asynchronous writing, optional Ring wait counters, and a node-local
socket sink for online analysis. The fault simulations are compiled out of the
ordinary build.

From this repository's root, with `UPSTREAM_NCCL` pointing to a clean checkout:

```bash
test "$(git -C "$UPSTREAM_NCCL" rev-parse HEAD)" = 6aae37927840b2bd7b7d42d2f0050e75f88ee97f
for patch in "$PWD"/ccld/patches/[0-9]*.patch; do
  git -C "$UPSTREAM_NCCL" apply --check "$patch"
  git -C "$UPSTREAM_NCCL" apply "$patch"
done
```
