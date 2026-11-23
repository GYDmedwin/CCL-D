# CCL-D-DCU

CCL-D-DCU 是 RCCL 上的集合通信诊断实现。它在通信过程中记录诊断事件，并复用 NCCL 版 CCL-D 的 decoder 和 analyzer 分析结果。

## 当前目标平台

- GPU：BW1000，`gfx936`
- RCCL：2.22.3-1
- 参考实现：NCCL 2.24.3-1 上的 CCL-D
- 现有验收入口：AntCCL Evaluation 5.1
- `gfx936` 不使用 LL128 协议

Evaluation 5.1 场景包括 collective 未进入、count 不一致、kernel hang、collective 后 hang、计算变慢和通信变慢。当前统一入口默认运行单节点 4 rank。双节点 8 rank 入口也已保留，节点是 HPDPS-6 和 HPDPS-7；运行前需要两台主机名均可解析并可通过 SSH 访问。

## 构建 RCCL

在已加载 ROCm/DTK 环境的机器上，从仓库根目录配置并构建 gfx936 版本：

```bash
cmake -S . -B build-ccld-gfx936-fault \
  -DGPU_TARGETS=gfx936 \
  -DCOLLTRACE=ON \
  -DCCLD_FAULT_INJECTION=ON
cmake --build build-ccld-gfx936-fault -j
```

`CCLD_FAULT_INJECTION` 用于构建 H3/S2 实验所需的隔离故障注入支持。普通诊断不需要另编译一套 RCCL；验收脚本默认读取上述目录中的 `librccl.so.1.0`。若库位于其他位置，可设置 `RCCL_BUILD_DIR` 和 `RCCL_LIB`。

## 运行 Evaluation 5.1

运行全部单节点场景：

```bash
tools/tests/run_antccl_eval51_gfx936.sh --case all
```

先运行一个例程，例如 H1（rank 2 暂停进入 collective）：

```bash
tools/tests/run_antccl_eval51_gfx936.sh --case H1
```

可选场景为 `H1`、`H2`、`H2_SIM`、`H3`、`POST`、`S1`、`S2`。每次运行会在 `../results/antccl-eval51-<拓扑>-<时间>/` 下保存 workload、诊断日志和分析结果。可以指定输出目录：

```bash
CCLD_RUN=/workspace/dcu/results/my-eval51 \
  tools/tests/run_antccl_eval51_gfx936.sh --case H1
```

双节点 8 rank/IB 验收命令：

```bash
CCLD_TOPOLOGY=dual \
  tools/tests/run_antccl_eval51_gfx936.sh --case all
```

双节点 runner 通过 SSH 启动 HPDPS-7，并检查两端 RCCL 日志中的 IB 记录。节点名解析或 SSH 未就绪时，该实验无法启动。

## 依赖与结果

验收入口使用 `/workspace/nv/CCL-D` 中的 decoder 和 analyzer。若参考源码在其他位置，可设置：

```bash
CCLD_DECODER=/path/to/CCL-D/ccld/tools/decode_async_wait.py \
CCLD_ANALYZER=/path/to/CCL-D/ccld/analyzer/ccld_analyze.py \
  tools/tests/run_antccl_eval51_gfx936.sh --case H1
```

运行脚本会将结果写入仓库同级的 `results/` 目录。项目设计、移植索引和逐项验收记录保存在工作区 `/workspace/dcu/doc/`，其中 `当前进度与有效文档.md` 是文档索引。
