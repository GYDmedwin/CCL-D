# CCL-D-DCU

CCL-D-DCU 是 RCCL 上的集合通信诊断实现。训练 rank 通过 HIP 映射的主机内存采集通信进度，节点代理在内存中缓冲二进制记录，独立分析器通过 TCP 拉取并诊断。

此分支延续 niheartent 的 RCCL 移植和提交历史。 [RCCL 移植来源](https://github.com/niheartent/CCL-D-DCU/tree/486a4ced7fce37b41b6b60689d6c2143b34f0a28)

## 当前目标平台

- GPU：BW1000，`gfx936`
- RCCL：2.22.3-1
- 参考实现：NCCL 2.24.3-1 上的 CCL-D
- 现有验收入口：AntCCL Evaluation 5.1
- 已验证 Simple 和 LL。该构建的 gfx936 路径未启用 LL128。

真机覆盖 AllReduce、AllGather、ReduceScatter、FSDP 和跨节点 IB。诊断的准确率、开销和适用范围见 [RESULTS.md](RESULTS.md)。

## 构建 RCCL

在已加载 ROCm/DTK 环境的机器上，从仓库根目录配置并构建 gfx936 版本：

```bash
cmake -S . -B build-ccld-gfx936 \
  -DGPU_TARGETS=gfx936 \
  -DCOLLTRACE=ON \
  -DCCLD_FAULT_INJECTION=OFF -DBUILD_TESTS=OFF
cmake --build build-ccld-gfx936 -j
```

`CCLD_FAULT_INJECTION=ON` 启用 H3/S2 实验的故障注入。在独立的 `build-ccld-gfx936-fault` 目录构建该版本。验收脚本支持用 `RCCL_BUILD_DIR` 和 `RCCL_LIB` 指定库。

## 随训练在线运行

在所有节点代理和分析器的环境中设置相同的 `CCLD_PULL_TOKEN`，可用 `openssl rand -hex 32` 生成。各训练节点先启动一个代理，`TRAIN_NODE_ADDR` 是分析主机可访问的作业网地址。

```bash
socket_dir="$(mktemp -d)"
export NCCL_CCLD_NODE_SOCKET="$socket_dir/ccld.sock"
python3 -m ccld.tools.live_transport --socket "$NCCL_CCLD_NODE_SOCKET" \
  --bind "$TRAIN_NODE_ADDR" --port 0
```

在分析主机从同一分支的仓库根目录运行。对每个代理传一个 `--node`，地址取该代理的 `CCLD_NODE_READY` 输出。

```bash
python3 -m ccld.analyzer.live --node "$NODE_AGENT_ADDR" \
  --poll-s 0.025 --hang-s 300 --window-s 60 --baseline-ms 1 \
  --decode-workers 4 --coalesce-completed
```

启动训练时继承本机代理的 socket 路径，并确保进程加载本分支的 `librccl`。以下是单节点示例，多节点用训练任务自己的分布式启动参数。

```bash
export LD_LIBRARY_PATH="$PWD/build-ccld-gfx936${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$PWD/build-ccld-gfx936/librccl.so.1.0"
unset NCCL_CCLD_OUT_DIR
export NCCL_CCLD_ENABLE=1 NCCL_CCLD_MODE=1 NCCL_CCLD_POLL_US=200
export NCCL_CCLD_FRAME_BLOCKS=64
torchrun --standalone --nproc-per-node=4 train.py
```

在 Ring 上额外启用方向等待计数时，设置 `NCCL_ALGO=Ring NCCL_CCLD_WAIT_ENABLE=1`。用正常训练标定 `--baseline-ms` 和 slow 窗口。用 `--slow-warmup-ops` 按 communicator 跳过启动期的慢诊断和基线学习，hang 诊断和签名检查仍实时运行。设置 `NCCL_CCLD_OUT_DIR` 可改用落盘的二进制记录供回放。

## 运行 Evaluation 5.1

运行全部单节点场景前，先按拓扑、协议和消息大小标定等待阈值，再通过 `CCLD_WAIT_THRESHOLD` 传入。

```bash
CCLD_WAIT_THRESHOLD="$CALIBRATED_WAIT_THRESHOLD" \
  tools/tests/run_antccl_eval51_gfx936.sh --case all
```

先运行一个例程，例如 H1（rank 2 暂停进入 collective）：

```bash
tools/tests/run_antccl_eval51_gfx936.sh --case H1
```

可选场景为 `H1`、`H2`、`H2_SIM`、`H3`、`POST`、`S1`、`S2`。每次运行会在 `../results/antccl-eval51-<拓扑>-<时间>/` 下保存 workload、诊断日志和分析结果。可以指定输出目录：

```bash
CCLD_RUN="$PWD/results/my-eval51" \
  tools/tests/run_antccl_eval51_gfx936.sh --case H1
```

双节点 8 rank/IB 验收命令：

```bash
CCLD_NODE1="$FIRST_HOST" CCLD_NODE2="$SECOND_HOST" CCLD_MASTER_ADDR="$FIRST_HOST" \
  CCLD_WAIT_THRESHOLD="$CALIBRATED_WAIT_THRESHOLD" \
  tools/tests/run_antccl_eval51_gfx936.sh --topology dual --case all
```

双节点 Docker runner 从第一台节点通过 SSH 启动第二台节点，并检查两端 RCCL 日志中的 IB 记录。使用该脚本时两端应能访问同一仓库和结果目录。

`H2` 检查已经完成的 collective 是否在预期轮次出现 count 或 operation 不一致。`H2_SIM` 验证模拟 hang 的分类。`S2` 的统一入口验证 slow 检测，并输出候选 rank。需要核对根因时，可使用 `tools/tests/validate_ccld_s2_strict.py` 检查多轮等待得分。

## 依赖与结果

本分支自带二进制 decoder、节点代理和分析器，来源是 NCCL 版 CCL-D 提交 `38e21d1`。验收入口默认使用当前检出的工具。用 `CCLD_NV_ROOT` 可以指定另一个分析器版本，也可用它为源码对照单测提供 NCCL 检出。

```bash
export CCLD_NV_ROOT=/path/to/nccl-CCL-D
tools/tests/run_antccl_eval51_gfx936.sh --case H1
```

验收脚本默认将结果写入仓库同级的 `results/` 目录。
