# RCCL / BW1000 验证结果

实测使用 RCCL 2.22.3-1、DTK 26.04（HIP 6.3.26045）、gfx936 BW1000、CMake 3.22.1 和 PyTorch 2.7.1。RCCL 移植来源为 `486a4ce`，原版开销对照使用同一 RCCL 版本的基线 `2939c54`。故障矩阵与通信基准使用定向编译的设备函数，训练使用覆盖 BF16 AllReduce、AllGather 和 ReduceScatter 等操作的定向构建。诊断实验用本分支自带的节点代理和分析器。 [原始移植](https://github.com/niheartent/CCL-D-DCU/tree/486a4ced7fce37b41b6b60689d6c2143b34f0a28)

默认禁用故障注入的 gfx936 完整库在 20 CPU 的构建作业中成功生成，`ONLY_FUNCS` 留空，耗时 27 分钟。全库的四卡 Ring/Tree × Simple/LL AllReduce 和 Ring/Simple、Ring/LL 的 AllGather/ReduceScatter 数值及解码检查通过，合计 64 条完成记录，未观察到采集缺口。

## 构建耗时

构建发生在主机 CPU 上。完整编译中大部分时间耗在 gfx936 设备代码链接。用 12 CPU 在节点本地目录单独测量了定向构建。

| 构建动作 | 实测墙钟时间 |
|---|---:|
| 完整 gfx936 库，20 CPU | 27 分 18 秒 |
| 定向构建，配置加编译和链接 | 54.21 秒 |
| 定向构建，修改一个主机端探针源文件后重编和链接 | 12.69 秒 |
| 定向构建，未变更源码后再次执行构建 | 0.31 秒 |

定向构建的时间适用于下文的函数集。改动设备公共头文件会影响更多编译单元。日常 C++ 验证用固定的定向构建目录，发布验证再构建完整库。纯分析器修改可直接复用已验证的库，并从二进制记录回放。

## 诊断准确度

单节点四卡，每种配置五次，覆盖四个注入 rank。主矩阵使用 FP32 AllReduce 和 512 KiB 数据。H1 让一个 rank 延迟 1.2 s 入场，H2 让一个 rank 少调用 256 个 FP32 元素，H3 用 `clock64` 限时停顿模拟设备卡死。S1 单独延迟入场 18 ms，S2 在 send 路径注入 120 万周期的限时延迟。S3 同时让一个 rank 延迟入场 6 ms，并在另一个 rank 注入 S2。

| 场景 | 类型和全部根因均正确 | 注入至首条正确结果中位数 (ms) | 首次正确定位计算中位数 (ms) |
|---|---:|---:|---:|
| H1 · Ring/Simple | 5/5 | 368.2 | 0.010 |
| H2 · Ring/Simple，四个 rank 均返回 | 5/5 | 16.6 | 0.040 |
| H3 · Ring/Simple，设备端限时停顿 | 5/5 | 376.0 | 0.258 |
| S1 · Ring/Simple | 5/5 | 213.5 | 0.280 |
| S1 · Ring/LL | 5/5 | 209.3 | 0.472 |
| S1 · Tree/Simple | 5/5 | 292.2 | 0.207 |
| S1 · Tree/LL | 5/5 | 299.5 | 0.349 |
| S2 · Ring/Simple | 5/5 | 319.7 | 0.745 |
| S2 · Ring/LL | 5/5 | 322.2 | 1.320 |
| S3 · Ring/Simple，两个不同 rank | 0/10，计算侧候选正确 10/10 | — | — |

同机用 Linux 单调时钟测注入至结果时间，包含探针轮询、拉取间隔、slow 窗和重复确认。主要设置为主机采样 200 µs、快照间隔 40 ms、TCP 拉取 20 ms、hang 阈值 350 ms、slow 窗 70 ms、初始基线 1 ms、跳过第一轮基线样本后学习 3 轮、slow 相对阈值 2、重复阈值 2。方向等待使用 3 个窗口的投票和 0.2 的归一化分差门槛。实际工作负载应以正常训练标定。表中定位计算时间仅计规则执行。H3 表中的命中结果对应人为限时停顿。

较早的 256 KiB 实验实测了另外一种 H2 表现，五次不同 count 都使所有 rank 挂起。超时与应用退出前，分析器五次均定位到少数派 rank。512 KiB 实验中各 rank 返回完成记录，新版在线分析器检查同一 opCount 的操作签名，结果归为 `collective_mismatch`。签名分组没有严格多数时保留未定。较早的分析器在 512 KiB 同类测试中漏报了 5/5。

S2 原规则在 256 KiB 检出 5/5，定位正确 0/5。对 Ring 的实际前驱和后继比较数据、credit 两类等待，再做多轮确认，512 KiB 的 Simple 和 LL 共 10/10 定位正确。S3 早期 256 KiB 回放 5/5 可复原两个根因，512 KiB 后续两种相对 rank 位置的通信等待得分接近，保留计算侧候选并输出 `undetermined`。单机 Tree 在链形拓扑下同层没有可比较同伴。来自 communicator 初始化的主机标识一致时，分析器用同一单调时钟下的入场差和最短通信时长交叉验证 S1。初版 Tree/Simple S1 为 0/5。

20 次健康及 POST 对照均没有确定的 H/S 告警。10 次健康 Ring 请求出现一次待确认 slow，未达到重复门槛。POST 模拟最后一次集合通信结束之后的进程停顿。

## 数值、采集和跨节点

- 四卡数值和记录验证覆盖 Ring/Simple、Ring/LL、Tree/Simple、Tree/LL 的 AllReduce，以及 Ring/Simple 和 Ring/LL 的 AllGather、ReduceScatter。
- 2+2 BW1000 的实际跨节点日志确认使用 InfiniBand。在 4 rank 的 Ring 和 Tree 健康实验中，聚合各收齐 48 个完成记录。Ring H1 在第 4 轮定位到 rank 2，故障恢复后收齐 48 条完成记录。分析器运行于其中一个训练节点。
- 8+8 BW1000 复测同样走 InfiniBand。Ring/Tree × Simple/LL 的四组健康 AllReduce 各收齐 208 条完成记录，所有 rank 通过末轮数值校验。Ring/Simple 的 H1 将延迟 1.2 s 的 rank 12 定位为根因，并在恢复后收齐 128 条记录。Ring/Simple 和 Tree/Simple 各一次 S1 注入，均定位到 rank 12，两个配置各报告八个有效窗口。短窗口的 Ring/Simple 健康对照在第二轮报了一次 S1，该轮仍使用初始基线。
- 同一 communicator 每次提交两个 collective 的压力场景曾导致后续单 collective 缺失。回收所有未跟踪任务的挂起元数据后，132 组成组调用之后的三个单 collective 都得到四个 rank 的完成记录。组内 collective 仍显式标为未覆盖。
- 32 帧、80 轮的 AllReduce 对照，旧库在循环复用后的 3,072 个 channel 增量全部与本轮计数不符。槽位准备时清零主机采样历史后，6 种有探针的配置中 86,016 个复用增量全部正确。
- `ncclAllToAll` 走 P2P 的数值校验通过。当前仅出现 communicator 生命周期记录。该路径的诊断覆盖仍有缺口。

## 集合通信开销

四卡单节点，原版库、移植库关闭诊断、移植库开启诊断。各配置按循环轮换顺序启动四次，使用 20 轮 warmup 和 60 轮计时，按每轮四个 GPU 同步完成时计时。输入输出较大一端的每 rank 容量为表中的数据量。88 个启动都通过最后一轮的数值校验。开启组使用 32 帧、200 µs 采样，Ring 同时启用方向等待计数。二进制 sink 在 tmpfs，表不含在线分析器。同一节点上另外四张卡曾同期运行独立故障矩阵。

| 操作与配置 | 原版 P50 / P95 (ms) | 移植关闭 P50 / P95 (ms) | 移植开启 P50 / P95 (ms) | 开启比原版 P50 配对变化 | 进程 CPU 时间配对变化 |
|---|---:|---:|---:|---:|---:|
| AllReduce · Ring/Simple · 256 KiB | 0.132 / 0.135 | 0.137 / 0.147 | 0.141 / 0.145 | +8.2% | +53.2% |
| AllReduce · Ring/Simple · 64 MiB | 1.008 / 1.027 | 1.007 / 1.018 | 1.001 / 1.015 | -1.0% | +36.1% |
| AllReduce · Ring/Simple · 256 MiB | 3.550 / 3.564 | 3.539 / 3.556 | 3.545 / 3.562 | -0.1% | +83.2% |
| AllReduce · Tree/LL · 256 KiB | 0.135 / 0.141 | 0.136 / 0.141 | 0.128 / 0.133 | -5.1% | +41.4% |
| AllGather · Ring/Simple · 64 MiB | 0.602 / 0.612 | 0.601 / 0.608 | 0.597 / 0.614 | -0.5% | +29.5% |
| ReduceScatter · Ring/Simple · 64 MiB | 0.613 / 0.619 | 0.622 / 0.628 | 0.618 / 0.626 | +1.9% | +39.8% |
| AllToAll · P2P · 64 MiB | 0.641 / 0.654 | 0.642 / 0.656 | 0.642 / 0.656 | +0.4% | +11.3% |

时延取四次启动各自分位数的中位数。配对变化对相同重复编号先取百分比，再取四个百分比的中位数。小消息 Ring AllReduce 中，开启比编入探针后关闭的 P50 高 1.9%，进程 CPU 时间高 45.5%。其对原版的变化还包含库代码路径的差别。负向变化在短时基准中存在，暂不据此推断加速。AllToAll 一行衡量装载探针库的运行代价。

32 个 Ring 帧的锁页计数区为 16,928 B/rank，方向等待区另 16,384 B/rank。256 帧对应 135,200 B 和 131,072 B。固定宽度 2,088 B 的日志记录进入独立主机队列。计时配置为 1,024 槽，约 2.05 MiB/rank，每个槽含序号。每个 communicator 独立分配。32 帧与 256 帧在 256 KiB 组的 P50 接近，32 帧少使用约 87.5% 锁页内存。并发流较深时应监控 `frame_slot_busy` 和队列溢出后再选择容量。

另以 32 帧、四组轮换测试 200/500/1,000 µs。64 MiB Ring AllReduce 的采样周期从 200 调到 1,000 µs，进程 CPU 时间配对中位数降 10.3%，P50 变化小于 0.1%，两组各收齐 1,280 条记录。相同调整在 256 KiB 组出现槽位忙，200 和 1,000 µs 分别缺了 10 条和 9 条记录。随后 256 KiB 上各以 32 和 64 帧搭配 200 和 1,000 µs，每档六次的 24 个启动均收齐。示例训练使用 64 帧和 200 µs。64 MiB 上的调参验证了数值、完成覆盖和开销，方向判定需要重新标定。

### 复跑通信基准

在同一节点的四张可见 BW1000 上、已加载 ROCm/DTK 的 shell 中编译。对原版库与移植库分别链接并循环轮换执行后比较。可以用 `ldd` 核对加载路径。

上述基准和故障矩阵的定向构建使用以下 CMake 选项。

```bash
only_funcs='AllReduce RING/TREE SIMPLE/LL Sum f32|AllGather RING SIMPLE/LL Sum i8|ReduceScatter RING SIMPLE/LL Sum f32|SendRecv RING SIMPLE Sum i8'
cmake -S . -B build-ccld-gfx936 -DGPU_TARGETS=gfx936 -DCOMPILING_TARGETS=gfx936 \
  -DCOLLTRACE=ON -DBUILD_TESTS=OFF -DCCLD_FAULT_INJECTION=OFF "-DONLY_FUNCS=$only_funcs"
cmake --build build-ccld-gfx936 --parallel 12
```

```bash
build="$PWD/build-ccld-gfx936"
hipcc --offload-arch=gfx936 -std=c++14 -I"$build/include" \
  tools/tests/ccld_rccl_benchmark.cpp -L"$build" -Wl,-rpath,"$build" \
  -l:librccl.so.1.0 -o ccld_rccl_benchmark

trace_dir="$(mktemp -d)"
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_CCLD_ENABLE=1 \
  NCCL_CCLD_WAIT_ENABLE=1 NCCL_CCLD_FRAME_BLOCKS=32 NCCL_CCLD_POLL_US=200 \
  NCCL_CCLD_QUEUE_SLOTS=1024 NCCL_CCLD_OUT_DIR="$trace_dir" \
  ./ccld_rccl_benchmark 262144 ar
```

`ar`、`ag`、`rs`、`a2a` 对应四种操作。将开关改为 `NCCL_CCLD_ENABLE=0` 得到本库的关闭组。记录可通过 `python3 -m ccld.tools.decode_async_wait TRACE_DIRECTORY OUTPUT_DIRECTORY` 解码。

## 端到端 FSDP

四卡、1.17 亿参数 Decoder、固定合成 token、BF16、AdamW，每次 96 步。同一驱动的两组都强制使用 PyTorch 数学 attention 后端，适配本环境的 fused attention 依赖。第一轮六组交错开关对照的每 rank loss 曲线逐步一致。训练稳态步时采用每步最慢 rank，剔除前八步。开启后的成对变化中位数为 P50 +3.06%、P95 +6.10%、全训练循环 +3.03%、四个 rank 进程 CPU 总时间 +4.61%。每次开启组收齐 9,992 条完成记录。该轮用了移植库的定向构建和初版在线分析器。六次无注入对照都在训练启动期给出了确定的 S1 结果，启动期慢诊断随后加入了可配置预热窗口。

新版用四组开关对照和三次训练中 S1 注入复测。分析器使用四个解码进程、合并同批中被完成记录覆盖的进度事件，拉取周期 25 ms。跳过前 416 个符合比较条件的完成操作，动态基线为其后 60 轮，slow 窗 0.5 s。四次健康开启组的确定告警为 0，三次出现待确认 slow。三个注入分别在 rank 2、0、3 的第 48–73 步入场前延迟 60 ms，三次全部正确定位，注入至首条正确结果中位数 799 ms。七次开启组都收齐每次 9,992 条记录，所有 11 次的各 rank loss 曲线保持一致。

新版对照的开启 P50 配对中位数 +4.83%，四对范围 −0.45% 至 +6.67%。全训练循环 +2.92%，四 rank CPU 累计 +3.53%。分析器协调进程与四个解码进程共用约 4.04 CPU 秒/次的中位数，节点代理的 CPU 未计入该数字。两轮对照在不同节点，采样和分析参数均应与工作负载一并记录。

gfx936 完整库另完成一次四卡、30 步开关对照。每 rank 的 loss 逐步一致，开启组收齐 3,128 条完成记录、无覆盖缺口，出现一条待确认 slow。

## 与 NCCL / H100 的差异和可采取的措施

| 能力 | RCCL / BW1000 当前行为 | 处理方式 |
|---|---|---|
| LL128 | H100 上已覆盖 LL128，测试用 gfx936 RCCL 构建关闭 LL128，移植探针只接受 Simple 和 LL | 在 gfx936 使用 Simple/LL。如目标 ROCm/RCCL 版本提供 LL128 实现，再单独移植计数位置并做数值、时延和准确度验证 |
| 跨 communicator H1 | H100 已验证训练运行时注册的等待关系。RCCL 分支含等待事件协议和分析代码，BW1000 训练尚未接入和验证 | 通过训练框架记录真实的上游与下游 collective ID，复用内存等待表，再在跨机训练中拿到完整依赖证据 |
| Tree S1/S2/S3 | 同机线形 Tree S1 已验证。8+8 Tree/Simple 的 S1 经同层和多通道投票定位。当前方向探针覆盖 Ring | 同节点以共同单调时钟验证晚入场。多机保留同层和多通道证据，补 Tree 分方向等待再做 S2/S3 验证 |
| AllToAll 和普通 P2P | 此处验证 P2P AllToAll 的数值。H100 参考发布也只验证了该路径的数值 | 给 P2P 分组增加显式覆盖记录，再按 peer 和方向进行探测及根因建模。Pivot AllToAll 应单独验证 |
| 同一计划多 collective、图重放 | 当前两套实现均以单 collective 计划为主要覆盖范围。RCCL 会排除多任务或 persistent 计划 | 对成组调用与图重放建立每个 work 的独立元数据及完成标记。诊断复核可先使用无图、单 collective 工作负载 |
| 完成时刻 | 两版默认轮询 event，也保留流回调。RCCL 使用 HIP event 和 HIP 映射内存 | HIP event 查询确认完成后读取探针。短操作需要同时报告采样间隔与应用端基准 |
| 双机规模 | 8+8 BW1000 已完成跨机数值、H1 和 S1 检查 | 扩大异常注入的消息大小和故障位置，复测跨节点 S2/S3 和训练负载下的采集容量 |

H100 参考数据与本次使用的消息、阈值和 GPU 数不同。论文中的低开销数字需要在各平台、各工作负载独立测量。 [NCCL 参考结果](https://github.com/GYDmedwin/CCL-D/blob/38e21d19b89abaae3d28fed935425534ce150090/RESULTS.md)
