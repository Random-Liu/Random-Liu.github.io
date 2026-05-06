# TPU 深入笔记：从单芯片到生产集群

> **目的**：给自己留一份从硬件原理一路到推理 / 集群层适配的端到端参考，重点在**为什么这么设计**而非术语解释。
> **对照对象**：每章末尾会有一个 `↔ GPU` 小节做就近对比。
> **关于"补充"**：正文中所有 `> **[补充 — Claude 加]** ...` 标记的内容都是原对话里没有、我补的，请你过一眼决定要不要保留。

---

## Part I — 硬件层：芯片、互联、封装

### 1. 单芯片：MXU、VPU、SPU 与脉动阵列

**一句话**：TPU 把矩阵乘法做成专用电路（MXU），用脉动阵列在芯片内部「传递数据」而不是「传递地址」，省掉了缓存层级。

#### 1.1 脉动阵列：绕开冯·诺依曼瓶颈

通用架构里每次 ALU 算加法都要去寄存器或 L1 取数、写回。LLM 推理动辄要几百 GB/s 的矩阵乘吞吐，这种「算一下查一次内存」的模式直接被带宽锁死。

TPU 的破局思路是把计算单元排成二维网格（v4 的 MXU 是 128×128 个 MAC），让数据像心脏泵血一样流过：

- **权重驻留**：算之前先把权重锁进每个 MAC 单元里
- **数据脉动**：激活值从左侧、上方流入，逐 cycle 向右、向下推进
- **邻居硬连线**：每个单元算完乘加，结果**直接通过物理走线**传给下一个邻居，不写寄存器

数据穿过整个阵列的过程中被复用了几百到上千次，期间一次都不碰 SRAM。这就是为什么同样的硅面积，TPU 能塞下比 GPU 多得多的纯算力单元。

#### 1.2 一个 2×2 的具体例子

设 $Y = X \times W$，激活和权重都是 2×2，MXU 是 2×2 网格，**权重驻留**模式：

```
权重位置：    PE(1,1)=W11   PE(1,2)=W12
              PE(2,1)=W21   PE(2,2)=W22

激活流入：    X 的行从左侧进入，第二行比第一行晚一个 cycle（"倾斜"进入）
部分和方向：  从上向下传递
```

逐 cycle 看：

- **Cycle 1**：$X_{11}$ 进入 PE(1,1)，算出 $X_{11} \times W_{11}$，向下传部分和，向右传 $X_{11}$。
- **Cycle 2**：$X_{11}$ 流到 PE(1,2)；$X_{21}$ 进入 PE(2,1)，把上方传来的 $X_{11} \times W_{11}$ 加上自己的 $X_{21} \times W_{21}$ —— 这一刻，PE(2,1) 向下输出的就是 $Y_{11}$ 的最终值。
- **Cycle 3**：右下角 PE(2,2) 收齐两路输入，输出 $Y_{12}$。
- **Cycle 4+**：剩余结果从阵列底部依次「滴落」，整个过程没有任何寄存器查表。

#### 1.3 宏观架构：MXU 之外还需要谁

只有 MXU 跑不了完整网络。一个 TPU 核心里：

| 组件 | 职责 |
|---|---|
| **MXU** | 稠密矩阵乘加，算力主力 |
| **VPU** | 向量运算：激活函数（GeLU/ReLU/Softmax）、LayerNorm 等没法压成矩阵乘的部分 |
| **SPU** | 标量与控制流：循环、分支、地址计算。算力很弱，但负责"指挥" |
| **Unified Buffer** | 几十 MB 的片上 SRAM，用来暂存从 HBM 搬进来的激活值和中间结果 |

指令集走 **CISC** 风格：一条 `MatrixMultiply` 就能触发 MXU 做几千次 MAC，把指令解码和调度的开销压到极低。

#### 1.4 SPU 与 VLIW：硬件不动，控制流谁管

MXU 自己不知道该算什么、数据从哪来。这套调度由 SPU + **VLIW（超长指令字）** 驱动：

XLA 编译器把多个无冲突的操作打包成一条长指令。一条指令同时包含四件事：

```
[ DMA ] | [ MXU ] | [ VPU ] | [ SPU 控制流 ]
```

举例：`[DMA 把下一块激活搬进 UB] | [MXU 算当前块] | [VPU 对上一块结果做 ReLU] | [SPU 更新地址]`

SPU 的工作模式：

- **不等结果就发指令**：把控制信号塞进 MXU 的 FIFO 队列，只要队列没满就一直往前跑
- **同步靠栅栏**：需要等 MXU 算完才能继续时，插一条 `WAIT_MXU_DONE`，硬件级阻塞
- **不参与计算**：SPU 算力极弱，但因为只做循环、地址、栅栏，开销极低

整个时序在程序运行前就 100% 确定。

#### 1.5 ↔ GPU

| 维度 | TPU | GPU |
|---|---|---|
| 矩阵单元 | MXU 128×128，单个巨型脉动阵列 | Tensor Core 16×8×16，散布在每个 SM 里 |
| 标量/向量单元 | SPU 弱、VPU 中等 | CUDA Core 多、强分支预测 |
| 数据缓冲 | Unified Buffer（软件显式 DMA 管理） | L1/L2 Cache（硬件自动预取） |
| 指令模型 | VLIW，编译期排好 | SIMT，运行时 Warp Scheduler 切换 |
| 隐藏访存延迟 | 静态流水线 | 海量并发线程切换 |

**Trade-off**：TPU 把灵活性交给编译器，硬件做傻瓜执行；GPU 把灵活性留在硬件，软件相对简单但硅面积花在调度上。

---

### 2. 多芯片互联：ICI 与 3D Torus

**一句话**：单芯片很强，但真正让 TPU 成为 TPU 的是片间网络——ICI 是物理层，3D Torus 是逻辑拓扑。

#### 2.1 ICI：网络做进硅片边缘

每颗 TPU 的硅片边缘集成了一组高速接口叫 **ICI（Inter-Core Interconnect）**，特性：

- **没有外部交换机**：Pod 内传数据不经过任何传统网络设备，从一颗芯片的光模块直接跳到另一颗的光模块
- **延迟极低且确定**：避开了交换机缓冲、路由表、拥塞控制等软件开销，XLA 能精确算出数据从 Pod 一端到另一端的纳秒数

> **[补充 — Claude 加]** 原对话没明说 ICI 链路的具体带宽数字，公开资料显示 v4 单链路是 4.5 TB/s 量级（每芯片所有 6 个方向加起来），不确定要不要写进笔记，请你定。

#### 2.2 3D Torus：六邻居 + 首尾相连

每颗芯片在 (X, Y, Z) 三维空间里有一个坐标，通过 ICI 连到 ±X, ±Y, ±Z 六个方向的邻居。**关键在于环回**：走到某个维度的边缘时，连线会绕回该维度的起点形成一个环，所以 Pod 里没有"边缘节点"。

为什么选 Torus 而不是 fat-tree：

- **冗余路径多**：一条线断了可以绕另一维度过去
- **集合通信友好**：All-Reduce / All-Gather 这种环形通信在物理上就是直跑环，不需要路由计算

代价：

- **网络直径线性**：随着规模 $N$ 增长，最长跳数是 $O(N^{1/3})$，而 fat-tree 是 $O(\log N)$
- **不规则点对点会拥塞**：MoE 的 All-to-All 这种乱序通信在 Torus 上会出现链路热点，第 19 章会展开

#### 2.3 ↔ GPU

| 维度 | TPU | GPU |
|---|---|---|
| 节点内 | ICI（直集成在芯片边） | NVLink / NVSwitch（独立交换芯片） |
| 节点间 | ICI + OCS（同一套，纯光） | InfiniBand + NIC + 交换机 |
| 拓扑 | 3D Torus | Fat-tree / 多层 Spine-Leaf |
| 延迟特性 | 静态可预测 | 动态有抖动 |

**Trade-off**：Torus 在规整集合通信上是降维打击，但面对动态稀疏通信（MoE）就吃亏；fat-tree 反过来。

---

### 3. OCS：光路交换让拓扑可重配

**一句话**：TPU Pod 在物理层是 star（每机架到 OCS 的若干光纤），逻辑层却被 OCS 接成 torus；MEMS 小镜子在微秒级切换光路。

#### 3.1 MEMS 镜子：纯反射、不读包

普通以太网/InfiniBand 交换机走的是 **O-E-O**（光-电-光）：光纤进来 → 转电信号 → 芯片读包头 → 查路由表 → 缓存排队 → 转回光信号出去。这套流程带来延迟、抖动，且交换芯片自己耗电极大。

OCS（Google 内部代号 Palomar）从 v4 开始引入，走纯 **O-O-O**：

- 内部是封闭充惰性气体的空腔
- 输入光纤的激光打在第一面 MEMS 微镜上（头发丝大小，靠静电引力偏转）
- 反射到对面第二面 MEMS 镜
- 第二面镜把光"矫正"成水平角度，射进通往目标 TPU 的光纤
- 全程**没有数字芯片，没有缓存，不解析数据包，没有带宽上限**

镜子在传输时是**完全静止**的——只在切换"道岔"时短暂调一下角度。所以 OCS 对数据是**透明**的（Data Agnostic）：不管激光以 100 Gbps 还是 800 Gbps 闪烁，它只管反射。光纤升级了，OCS 不用换。

#### 3.2 切分粒度：机架级，不是芯片级

**OCS 不能任意挑单颗芯片建环**。一个机架内的 64 颗 TPU（4×4×4 基础块）之间是用粗壮的 DAC 铜缆焊死的，便宜且距离短。机架对外暴露的接口才插光模块、拉光纤进 OCS。

所以 OCS 的"乐高积木"最小单位是一个 4×4×4 机架。要 256 芯就把 4 个机架的光纤拼起来，要 1024 芯就拼 16 个，以此类推。

#### 3.3 96 根线：4×4×4 机架的几何

这是个挺优雅的几何题。表面 56 颗芯片向外暴露的接口数怎么算？按位置分：

| 位置 | 数量 | 每颗出几根线 | 小计 |
|---|---|---|---|
| 8 个角 | 8 | 3（暴露 X、Y、Z 三方向） | 24 |
| 12 条棱（每条减去两端的角，剩 2 颗） | 24 | 2 | 48 |
| 6 个面（每面减去边角，剩 2×2=4 颗） | 24 | 1 | 24 |

合计 **96 根光纤**。也可以按面算验证：每个面是 4×4=16 个对外接口，6 个面 ×16 = **96**，吻合。

机房里这 96 根线不是一根根独立拉，而是用 MPO/MTP 高密度并行光缆（一根粗线含 16 或 32 芯光纤），从机架顶部"瀑布"式汇聚到中央的 OCS 网络机架。

#### 3.4 OCS 与 3D Torus 的关系

**3D Torus 是逻辑拓扑形态，OCS 是变形金刚的关节**：

- v2/v3 时代：拓扑是物理硬连线，机架 A→B→C→A 焊死，谁坏一颗整个区域瘫痪
- v4/v5 时代：物理走线变成 **star**（所有机架光纤都汇聚到 OCS），由 OCS 内部的镜子动态"折叠"出一个 3D Torus 闭环

切片场景：

| 你要 | OCS 怎么做 |
|---|---|
| 单机架 64 芯闭环 | 把这 96 根线在自己内部互折（左 16↔右 16，前↔后，上↔下） |
| 4×4×8（128 芯，跨 2 个机架） | X、Y 轴各自机架内闭环；Z 轴上把机架 A 顶面（16 根）接给机架 B 底面，反之亦然 |
| 绕过坏芯片 | 借隔壁机架一根线过来，强行维持 3D 环的逻辑完整性 |

#### 3.5 物理边界：Pod 内

激光在光纤和自由空间反射会衰减，OCS 网络的物理覆盖上限就是一个 Pod。v5p Pod 是 8960 颗芯片。**Pod 是光互联的绝对边界**。

跨 Pod 的"通信"必须走传统的数据中心以太网（DCN），延迟和抖动都没法跟 ICI 比，会破坏 XLA 的静态时钟。所以现实中：

- **同一次计算（Model Parallelism）**：绝不跨 Pod
- **服务调度（Load Balancing）**：跨 Pod 没问题——用户 A 完整路由给 Pod 1，用户 B 给 Pod 2，Pod 之间只通过 HTTP/gRPC 负载均衡器协调

#### 3.6 ↔ GPU

GPU 集群里**没有这个东西**。NVIDIA 的 NVSwitch 和 InfiniBand 都是 packet-switched 电交换，OCS 是 circuit-switched 光交换，理念完全不同。这条算是 TPU 体系最独特的设计。

> **[补充 — Claude 加]** 微软 Azure 在部分 AI 集群里也开始试用 Lumen 提供的 OCS，但量级和 Google TPU 不在一个层次。这条不写进正文，仅供你参考。

---

### 4. 集合通信：Ring All-Reduce 在 3D Torus 上的降维

**一句话**：3D Torus 上做 All-Reduce 的诀窍是把它拆成 X/Y/Z 三个 1D 圈分别做，再合起来——这是几何换算法。

#### 4.1 1D Ring All-Reduce：Reduce-Scatter + All-Gather

设 4 颗 TPU 首尾成环，每颗算出一个 400 MB 张量，要把 4 份逐元素相加然后让每颗都拿到完整结果。

直接发给 TPU 0 算会瞬间挤爆它的网络。XLA 的做法是**切碎 + 流水线接力**。把 400 MB 切成 4 个 100 MB（块 A、B、C、D）。

**阶段一：Reduce-Scatter（每颗最终持有一个块的完整总和）**

| Cycle | 动作 |
|---|---|
| 1 | 每颗芯片把自己的某个块往右传：TPU0→A→TPU1，TPU1→B→TPU2，TPU2→C→TPU3，TPU3→D→TPU0。**4 条线（含环回）全部满载** |
| 2 | TPU 1 把收到的 A 与本地 A 用 VPU 加起来变成"部分和 A"，继续往右传 |
| 3 | 再传一次。TPU 3 收到流转一圈的部分和 A，加上自己的 A，得到全网完整 A |

阶段结束时，TPU 0 持有完整 B，TPU 1 持有完整 C，TPU 2 持有完整 D，TPU 3 持有完整 A。

**阶段二：All-Gather（把各自的 1/4 答案分享出去）**

3 步接力广播，环上四个完整块并行飞奔。每颗芯片的 Unified Buffer 拼齐 A/B/C/D 后，DMA 才把最终 400 MB 刷回 HBM。

#### 4.2 关键硬件细节：网络是缓存的延伸

TPU 把网络变成了 SRAM 的直接延伸：

- 数据从发送方 Unified Buffer 抽出 → 光信号穿过 ICI → 接收方 Unified Buffer
- VPU 直接从 Unified Buffer 取数做加法，**全程不碰 HBM**
- 发送时同时附带一个极小的 **Sync Token**

GPU 的对照流程：HBM → PCIe → NIC → 交换机 → NIC → PCIe → HBM → 计算核心。每一跳都涉及内存读写，HBM 带宽被通信吃掉一大块。

#### 4.3 不靠全局时钟，靠硬件信号量

数据中心规模上维持几百颗芯片在纳秒级共享物理时钟，物理学上做不到（光速延迟、时钟漂移）。TPU 用的是 **XLA 静态时间表 + 硬件信号量异步握手**。

接收端的链式反应：

1. ICI 物理收完数据后**自动**把对应的硬件信号量 +1（SPU 完全不参与）
2. SPU 读 XLA 的指令，看到 `WAIT Semaphore_X >= 1` 就阻塞 VPU
3. 信号量翻 1 的瞬间，WAIT 放行，VPU 像弹簧一样起跳算加法
4. 算完触发下一条 DMA 把结果发出去，附新的 Sync Token，把自己的信号量清零

宏观看就像几千颗芯片长在同一个齿轮上，但实际上每颗芯片只盯着自己面前的几个硬件信号灯。XLA 提前把每一步算力匹配得严丝合缝，所以没有死锁、也没人空转太久。

#### 4.4 3D 降维：拆成 X、Y、Z 三个 1D 圈

如果在 4×4×4 机架里把 64 颗串成一个长 64 跳的单圈，灾难：

- 数据要跑 63 跳才转一圈
- 每颗芯片有 6 根线，只用了 1 根接收 + 1 根发送，**剩下 4 根全闲**

正确做法是 **多维正交环形同步**：把 3D 任务拆成三次并行的 1D 任务。

| 阶段 | 干什么 | 并行度 |
|---|---|---|
| X 轴同步 | 16 条平行 X 轴小环（长 4）同时跑 Reduce-Scatter + All-Gather | 16 |
| Y 轴同步 | 同步好的数据再切，16 条 Y 轴小环跑一次 | 16 |
| Z 轴同步 | 16 条 Z 轴小环跑一次 | 16 |

总跳数：4 + 4 + 4 = **12 跳**，远少于 64 跳。

光看每个阶段似乎只有 1/3 的线在跑，但 XLA 用 **流水线（Pipelining）**：当矩阵切片 1 在走 Y 轴时，切片 2 已经在走 X 轴。宏观上 6 根线全部满负荷发光。

#### 4.5 大圈是必然：跨机架的 Z 轴怎么缝合

4×4×8（128 芯，跨 2 机架）切片里，X/Y 还是长 4 小圈，Z 轴变成长 **8 的大圈**。物理形态：

```
[ 机架 A 的 Z 轴 ]                              [ 机架 B 的 Z 轴 ]
TPU(Z=0) — TPU(Z=1) — TPU(Z=2) — TPU(Z=3)      TPU(Z=4) — TPU(Z=5) — TPU(Z=6) — TPU(Z=7)
   ^                                  |                                            |
   |          ← OCS 跨机架光路缝合 ←   +— OCS — TPU(Z=4)                            |
   +———————————————————————————————————————— OCS 环回 ———————————————————————————————+
```

满规模 v5p Pod（8960 芯，比例可能是 16×16×35），最长边的 Z 轴会形成 **35 跳的大圈**。光物理传输的延迟叠加就足以让上层 MXU 等到饥饿。

#### 4.6 NUCA：铜缆和光路的延迟异构

8 跳 Z 轴大圈里，2 跳是光路（跨机架），6 跳是铜缆（机架内）。两种介质：

- **带宽必须严格一致**：流水线吞吐取决于最细那截管子。所以 TPU 设计时把光模块的调制速率和铜线 SerDes 速率对齐了
- **延迟必然不一致**：铜缆纳秒级（个位数到十位数），光路要走 E-O 转换 → 几十米光纤 → OCS 反射 → O-E 转换，到几百纳秒级别

带宽同构 + 延迟异构 → 圈在物理上不是完美对称的。这种现象叫 **NUCA（Non-Uniform Communication Architecture）**。

业务层有两种化解办法：

1. **流水线稳态掩盖**：刚启动时光路那两跳的延迟会让流水线出现微小气泡，但稳态后吞吐由带宽决定，几百纳秒的启动延迟在持续高吞吐下被淹没
2. **XLA 的拓扑感知映射**：见下节 9.1

#### 4.7 ↔ GPU

| 维度 | TPU 3D Torus | GPU NCCL on NVLink+IB |
|---|---|---|
| 算法 | Multi-dim Ring All-Reduce | Ring 或 Tree（可选） |
| 网络直径 | $O(N^{1/3})$ | $O(\log N)$ |
| 同步机制 | 硬件信号量 + 静态调度 | 软件 spin-wait + flag in HBM |
| 数据路径 | UB → ICI → UB → VPU | HBM → PCIe → NIC → IB → NIC → PCIe → HBM → SM |
| 计算资源占用 | VPU 顺手做加法，MXU 不停 | NCCL kernel 抢 SM 资源 |

**Trade-off**：环形 + 静态调度对规则集合通信极致优化，但牺牲了任意点对点的灵活性。

---

### 5. Host ↔ TPU：PCIe、NUMA 与 multi-host slice

**一句话**：TPU 不是独立机器，是挂在 CPU host 上的 PCIe 设备；slice 一旦跨 host，就是天然的分布式系统。

#### 5.1 物理形态：CPU 是包工头，TPU 是工人

每个机架上跑的是标准 x86 服务器主板（Intel/AMD），普通 DDR 内存。TPU 通过 PCIe 总线连接到 CPU 旁边。**典型配比是 1 个 CPU host 管 4 颗或 8 颗 TPU**，这是物理上焊死的比例。

分工：

- **CPU 干**：Linux 系统、Kubelet、HTTP/gRPC 接客、Python/PyTorch、vLLM 调度器（Radix Tree、PagedAttention 页表）、XLA 编译（CPU 密集）
- **TPU 干**：纯执行 CPU 编译好的机器码做矩阵乘加

一次 Decode Step 的全过程：

1. CPU 在系统内存里拼好 metadata（页表指针等）
2. PCIe DMA 把数据 + 新 token embedding 拷贝到 TPU HBM
3. CPU 给 TPU 发指令"按第 5 号编译图开干"
4. TPU 闭关算
5. PCIe 把 logits 拉回 CPU 内存
6. CPU 做采样（Argmax、Top-P 等）

#### 5.2 双网隔离：DCN 与 ICI

Pod 内部有两套**完全独立**的物理网络：

| 网络 | 谁用 | 介质 | 用途 |
|---|---|---|---|
| **DCN（数据中心网络）** | Host CPUs | 普通以太网交换机 | CPU 之间聊天，控制面同步 |
| **ICI（芯片互联）** | TPUs | OCS + 3D Torus 专用光纤 | TPU 之间狂飙数据，数据面 |

控制面（K8s 协调、Pod 启停）走 DCN，CPU 间用 gRPC。数据面（All-Reduce、KV 同步）完全不经 CPU。

#### 5.3 Multi-host slice：N:N 映射

很多人以为申请 64 芯 v4-64 切片会拿到一台带超大 CPU 的巨无霸 VM。**不是这样**。

物理上是 **16 台 Host VM**（每台 4 颗 TPU），16×4=64：

- 16 台 VM 通过 DCN 相连
- 64 颗 TPU 通过 OCS 组成 3D Torus
- 每台 VM 上跑一个 Kubelet
- K8s 调度 16 个 Pod，分别落在 16 台 VM 上
- 推理代码（vLLM/JetStream）在 16 个 CPU 上同时启动，跑同一份 Python 代码（**SPMD**）
- 通常用 **LeaderWorkerSet (LWS)** 或 **JobSet** 表达：1 个 Leader Pod 对外暴露 API，15 个 Worker Pod 配合
- Leader 收到请求后通过 DCN 广播给 Workers，每个 CPU 给自己脚下的 4 颗 TPU 下指令
- 64 颗 TPU 算出结果后由 Leader 汇总返回

没有"超级 CPU 管 64 颗 TPU"这回事。大切片本质上是多个"小 CPU + 小 TPU"通过 K8s gang scheduling 强行绑定的分布式联邦。

#### 5.4 NUMA：双路 CPU 的 PCIe 劈管

现代服务器主板通常是双路 CPU（CPU 0 和 CPU 1）。一半的 PCIe 通道连 CPU 0，另一半连 CPU 1。8 颗 TPU 主机上，TPU 0~3 挂 CPU 0，TPU 4~7 挂 CPU 1。

跨 NUMA 灾难：CPU 0 想写 TPU 4 的 HBM，数据必须先走 UPI 总线到 CPU 1，再走 PCIe 到 TPU 4。延迟剧增，可用带宽腰斩。

具体卡在哪些环节：

- **输入流水线**：Tokenize 在 CPU 0 但任务下到 TPU 4，每次 in-feed 跨 NUMA
- **KV Cache offload**：TPU 0 的 KV swap 到了 CPU 1 管的 DDR 槽条
- **权重加载**：百 GB 级的权重 DMA 跨 NUMA，冷启动慢

Google Cloud 的解法：

- **单 NUMA VM 切分**：申请 4 颗 TPU 实例时，hypervisor 直接把物理机切两半。给你的 VM **只**包含 CPU 0 + CPU 0 的内存 + 挂在 CPU 0 下的 4 颗 TPU。看不见 CPU 1 那半边
- **XLA 自动绑核**：8 颗 TPU 实例无法切分时，XLA Runtime（PJRT）读取 PCIe 树拓扑，自动把喂 TPU 0~3 的线程 pin 到 CPU 0 核心，喂 TPU 4~7 的 pin 到 CPU 1 核心

#### 5.5 ↔ GPU

| 维度 | TPU | GPU |
|---|---|---|
| 物理挂载 | PCIe 挂在 host CPU 旁 | 同上 |
| 比例 | 1:4 或 1:8 焊死 | HGX 通常 1:8 |
| Multi-host 编排 | LWS / JobSet + K8s gang | MPI Operator / Training Operator |
| NUMA 处理 | 单 NUMA VM 切分 + XLA PJRT 自动绑核 | 多用 `numactl` 手动绑 + NCCL 拓扑感知 |
| 双网 | DCN + ICI 严格分开 | 控制面 + 数据面通常都走 IB |

**Trade-off**：TPU 的 N:N 编排让单机故障域变小，但运维上看到的"一个推理服务"实际是多个 K8s 资源的协奏。

---

### 6. 先进封装：算力面积 vs 带宽周长

**一句话**：Die 上面积决定算力（FLOPs），周长决定带宽（HBM 接口）；2.5D / 3D 封装是在调和两者的根本张力。

#### 6.1 内存墙的物理起源

- **算力 ∝ 面积**：MXU 是二维的，面积稍大乘加单元就 $O(N^2)$ 爆炸增长（64×64 → 128×128，面积翻倍，算力翻 4 倍）
- **传统带宽 ∝ 周长**：HBM 带宽取决于芯片边缘引脚数量，是 $O(N)$ 线性增长

面积平方级、周长线性级——**带宽永远跟不上算力**。这就是"内存墙"的物理根源。

#### 6.2 封装演进的三步

| 代际 | 名字 | 解决了什么 |
|---|---|---|
| 第一代 | Wire Bonding（金线键合） | 芯片正面朝上，靠边缘金线接基板，受限于周长 |
| 第二代 | Flip-Chip（倒装芯片） | 芯片翻过来，整个面种满 C4 锡球，从"边长"扩到"面积"——但 PCB 主板走线精度跟不上（线宽几十微米） |
| 第三代 | **2.5D 硅中介层（Silicon Interposer / CoWoS）** | 在芯片和主板之间垫一块硅片，用 EUV 光刻在硅片上画**纳米级**走线 |
| 第四代 | **3D 封装 + TSV（Through-Silicon Via）** | 直接在硅片上垂直打几万个微米级孔灌铜，把多层芯片像盖楼一样叠起来 |

#### 6.3 硅中介层为什么有用

普通 PCB 板上 1 mm 宽度大概挤 10 根线；硅中介层上 1 mm 能挤上千根。GPU/TPU 与 HBM 的连接通道从"双向 4 车道"变成"双向 4096 车道"。HBM3 的位宽就是这么来的。

HBM 自己内部也是 3D 结构：多层内存 die 用 TSV 垂直叠起来，所以单个 HBM stack 能给到 Tbps 级带宽。

#### 6.4 算力过剩问题与硬件妥协

v4 时代算力涨太快，HBM 带宽跟不上，跑 Decode 时 MFU 经常跌到个位数，MXU 在等米下锅。Google 的硬件级补救：

- **v5e（推理芯片）故意缩小 MXU**，把 compute / bandwidth 比例调回更健康的区间
- 牺牲峰值 FLOPs 换性价比，承认 Decode 是 memory-bound 不是 compute-bound

> **[补充 — Claude 加]** 这一节里 v4 时代 MFU 跌到"个位数"是源对话原话，但没给出具体数据点和场景。我没有别的可信来源验证，建议你之后核实下要不要保留。

#### 6.5 ↔ GPU

NVIDIA H100/B100 也用 CoWoS（同样台积电方案），技术路线一样。差异在 die 比例：H100 有 50 MB L2 Cache、5 TB/s HBM 带宽；TPU 不堆硬件 cache，把那块面积让给 MXU。

**Trade-off**：H100 用大 cache 容忍随机访问；TPU 用大 MXU 跑稠密。两条路对应不同的工作负载假设。

---

## Part II — 编译与运行时：XLA

### 7. XLA 编译模型：把不确定性消灭在编译期

**一句话**：XLA 的核心打法是「把所有不确定性在编译期消掉」——算子融合、静态 padding、软件流水线、VLIW 指令包都是这个思路的不同切面。

#### 7.1 算子融合

最经典的例子：一层 `MatMul + Bias + ReLU`。

朴素执行（GPU eager 模式）会：
1. 算 MatMul → 写回 HBM
2. 读出来加 Bias → 写回 HBM
3. 读出来做 ReLU → 写回 HBM

XLA 把三步融合成一个计算块：

```
HBM → UB（MatMul 输入） → MXU 算 MatMul → 流出瞬间送进 VPU → VPU 做 Bias + ReLU → HBM
```

**节省了 2/3 的 HBM 读写**。这种融合在 LLM 里普遍存在，比如 Attention 后面紧跟的 dropout/scale/mask 通常都被融进同一个块。

#### 7.2 静态 Padding

MXU 是 128×128 的硬连线阵列。如果你的矩阵是 100×100，XLA 不会让硬件去处理边界——硬件根本不支持。XLA 直接在编译期把矩阵 padding 到 128×128（多余位置填 0）。

代价：白算 28 行 × 28 列 ≈ 50% 的边界算力。
收益：阵列保持满速流动，不用停下来做边界判断。

在脉动阵列里，**让硬件全速跑空格远比中途停下来快**。

#### 7.3 软件流水线

XLA 在编译期就算好每次 DMA 搬数需要多少 cycle，静态生成指令：MXU 算第 N 块时，DMA 已经在搬第 N+1 块的权重。计算和访存完美重叠。

#### 7.4 VLIW 五槽：单芯片 + 跨芯片同构

单芯片版指令包：

```
[ DMA ] | [ MXU ] | [ VPU ] | [ SPU 控制流 ]
```

多芯片协作版多了一槽——**ICI 网络槽**：

```
[ DMA ] | [ MXU ] | [ VPU ] | [ ICI 网络引擎 ] | [ SPU 控制流 ]
```

关键洞察：**对 TPU 来说，跨芯片传数据（ICI）和片内搬砖（DMA）是平级的**——都是 VLIW 指令字上的一个开关。XLA 可以做到 MXU 算乘法的同时，ICI 把上一层的梯度传给隔壁芯片，**计算和跨节点通信在时钟周期级别完美重叠**。

这是 GPU 体系做不到的。GPU 的跨节点通信要触发 CPU 中断、构建数据包、过 IB 交换机，是另一套子系统。

#### 7.5 一个具体的伪汇编例子

计算 $C = A \times B$，$A$ 是 256×128，$B$ 是 128×256。MXU 是 128×128。

XLA 在编译期把 $A$ 切上下两块 ($A_0, A_1$)，$B$ 切左右两块 ($B_0, B_1$)，分解成 4 个 128×128 子任务。所有 HBM 地址在编译期硬编码（无运行时指针计算）。

执行 $C_{00} = A_0 \times B_0$ 的 VLIW 流：

```
Instruction 1（预热加载权重）:
  [DMA] LOAD_HBM_TO_UB (Src: HBM_B0, Dst: UB_B)
  [MXU] NOP
  [VPU] NOP
  [SPU] WAIT_DMA_DONE

Instruction 2（权重驻留）:
  [DMA] NOP
  [MXU] LOAD_UB_TO_WEIGHT_REG (Src: UB_B)
  [VPU] NOP
  [SPU] WAIT_MXU_DONE

Instruction 3（核心计算 + 流水线预取）:
  [DMA] LOAD_HBM_TO_UB (Src: HBM_A0, Dst: UB_A)
  [DMA_ASYNC] LOAD_HBM_TO_UB (Src: HBM_B1, Dst: UB_B_Next)  ← 提前预取下一块
  [MXU] MATMUL_STREAM_ACT (Src: UB_A, Dest: Accumulator_C00)
  [VPU] NOP
  [SPU] WAIT_MXU_DONE

Instruction 4（融合 ReLU）:
  [DMA] NOP
  [MXU] NOP
  [VPU] READ_ACCUM_AND_RELU_AND_STORE (Src: Accumulator_C00, Dst: UB_C00)
  [SPU] WAIT_VPU_DONE

Instruction 5（写回 HBM）:
  [DMA] STORE_UB_TO_HBM (Src: UB_C00, Dst: HBM_C00)
  [MXU] NOP
  [VPU] NOP
  [SPU] JUMP_TO_NEXT_BLOCK  ← 准备算 C01
```

注意 Instruction 3：**DMA 在搬下一块的同时 MXU 在算**。如果 XLA 算错任何一个 cycle，要么 UB 溢出，要么 MXU 空转。这种精度只有静态编译才能给。

#### 7.6 ↔ GPU

| 维度 | TPU XLA | GPU |
|---|---|---|
| 调度 | 编译期静态 | 运行期动态（Warp Scheduler） |
| 隐藏访存延迟 | 静态流水线 | 海量并发线程切换 |
| 指令格式 | VLIW 多槽并发 | SIMT 单指令多线程 |
| 算子融合 | XLA 自动 | TorchInductor / TVM / Triton 半自动 |

**Trade-off**：XLA 的"全局上帝视角"在调度规整工作负载时无敌，但只要算子图里出现动态形状，就要重新编译。

---

### 8. 编译时机：JIT、AOT、bucketing、persistent cache

**一句话**：静态编译的代价是首跑慢，工业上靠 bucketing + AOT + cache 把这个代价摊薄。

#### 8.1 JIT 在 PyTorch/XLA 上的真实时间线

vLLM 等推理框架在 TPU 上启动后的真实流程：

| 阶段 | 动作 | XLA 状态 |
|---|---|---|
| 1. 初始化 | 加载权重到 HBM | **未编译**。只有 nn.Module 对象和权重张量 |
| 2. Tracing | 第一次请求触发 `model.forward()`，PyTorch/XLA 用 **Lazy Tensor** 不真正计算，只记录 DAG | 在画图，生成 HLO IR |
| 3. 触发编译 | 代码读 logits 做采样时遇到同步屏障（如 `xm.mark_step()`） | **立刻编译**：算子融合 + 静态地址 + 指令排布 → TPU Executable。耗时几秒到几十秒 |
| 4. 执行 + 缓存 | TPU 跑出结果（毫秒级），编译产物存内存中的 Compiler Cache | Cache key 包含计算图结构和所有输入 shape |
| 5. 后续请求 | 同 shape 命中 cache，跳过编译 | 直接复用 |

**关键澄清**：权重数值不进编译产物。XLA 只关心权重的 shape 和 dtype，把权重视为"静态显存地址指针"。换微调过的 LoRA 权重、换同架构的另一个模型都不需要重编。

#### 8.2 工业级解法：bucketing + AOT + 持久化 cache

生产环境绝不能让第一个用户等 30 秒编译。流程如下：

**Step 1：限制并离散化 bucket**

算法团队 profiling 后定一组离散桶：

```
BS_Buckets    = [1, 2, 4, 8, 16, 32, 64]
SeqLen_Buckets = [128, 512, 1024, 2048, 4096, 8192]
```

运行时来一个 BS=5 的请求，padding 到 BS=8 进对应桶。

**Step 2：CI/CD 阶段 AOT 预热**

构建 Docker 镜像或发布制品时加一个 warmup 环节：拉起含真实 TPU 拓扑的 CI 节点，遍历 `BS × SeqLen` 所有组合发假请求触发编译。

**Step 3：持久化 cache 打进镜像**

用 `XLA_FLAGS="--xla_dump_to=/path/to/cache"` 把编译产物落盘。流水线最后把这几百 MB 到几 GB 的 cache 文件**直接打进 release 镜像**或挂在分布式存储里。

线上 vLLM/JetStream 实例启动时读这个 cache，命中桶就**毫秒级下发硬件**。

#### 8.3 为什么没有"天下大同"的预编译库

一个 XLA Executable 绑定的不只是模型 shape，还包括以下**致命变量**——任意一个变了缓存就失效：

| 变量 | 影响 |
|---|---|
| **物理硬件拓扑** | v5e-8（一维环）的编译产物给不了 v5p-32（3D Torus），XLA 把走哪根光纤、延迟多少都算死了 |
| **并行切分策略** | TP 切 Attention 还是 FFN？PP 怎么切？这些 SPMD 注解必须编译前确定 |
| **编译器版本** | XLA / LLVM 后端更新频繁，旧 cache 大概率校验失败 |
| **模型结构微调** | 加一层 adapter、改 RoPE base 频率→常量折叠结果变→HLO hash 变→cache 全废 |

所以每个团队都得自己维护一套**模型分发 + 缓存预热**流水线。基础设施大版本发版背后必然伴随大规模自动化重新编译。

#### 8.4 ↔ GPU

GPU 也有这套问题（PyTorch 2.x 的 `torch.compile` / Inductor / TensorRT），但程度轻得多：因为 GPU 硬件能在运行时容忍 shape 变化（动态调度），编译失败时还能 fallback 到 eager。TPU 没有这个 fallback，编译失败 = 服务失败。

---

### 9. XLA 拓扑感知映射

**一句话**：编译器知道 3D Torus + OCS 的物理拓扑，所以能把高密度通信映射到短边、低频同步映射到长环。

第 4.6 节讲过 NUCA：跨机架的 8 跳大圈里，6 跳铜缆 + 2 跳光路，带宽同构延迟异构。XLA 在编译时把不同性质的并行策略塞进不同性质的拓扑。

#### 9.1 TP 走小圈，DP 走大圈

| 并行策略 | 通信特征 | 映射到 |
|---|---|---|
| **张量并行 (TP)** | 步步为营，每个线性层都要同步激活值。**对延迟极敏感** | X 轴或 Y 轴的 4 / 8 短铜环 |
| **数据并行 (DP)** | 秋后算账，每个 step（甚至累积几个 step）才同步一次梯度。梯度矩阵大，对带宽要求高，对单次延迟容忍 | Z 轴的 35 节点大光环 |
| **流水线并行 (PP)** | 阶段间传 activation，频次中等 | 通常分给中等长度的边 |
| **专家并行 (EP)** | 动态 All-to-All（MoE） | Torus 上吃亏，详见第 19 章 |

DP 走大圈时，35 跳的物理延迟被流水线稳态掩盖，并且底层计算单元可以利用同步等待时间做下一个 step 的前向计算（Compute-Communication Overlap）。

#### 9.2 拓扑信息怎么传给 XLA

K8s 给 Node 打的标签（如 `cloud.google.com/gke-tpu-topology: 4x4x4`）携带了切片的几何形状。XLA Runtime 启动时读这些 + PCIe sysfs 信息，构造拓扑图。然后根据用户的 SPMD 切分注解把通信组映射到具体的 ICI 链路。

**结论**：纯 K8s 调度只看节点存活，真正的高性能 AI 调度看的是微秒级的光电物理边界。

#### 9.3 ↔ GPU

GPU 体系里类似的事情靠手工 NCCL group 配置 + `torch.distributed` 的拓扑感知 API。NCCL 知道 NVLink/IB 的层级，但 fat-tree 本身近似对称，拓扑感知的优化空间没 Torus 那么大。

---

## Part III — 推理层适配（目标 C）

### 10. 软件栈分叉：vLLM、JetStream、Saxml、GKE

**一句话**：TPU 上推理框架不止一个，三家定位不同；GKE 是把它们都装进集群的胶水。

#### 10.1 GKE 为什么死磕 vLLM on TPU

vLLM 是开源推理事实上的"Linux"——绝大多数客户在 GPU 上用 PyTorch + vLLM 写好了业务代码（API 封装、调度、自定义 prompting）。GKE 想卖 TPU（v5e/v5p 极具性价比），最大阻力是迁移成本：

> 如果客户得改代码才能上 TPU，他们就跑了。

所以 Google 的策略是 **Lift and Shift**：让客户原本的 `vllm serve` 命令换个基础镜像就在 TPU 上跑起来，PyTorch 调用被自动路由到 PyTorch/XLA。

技术底座：vLLM 官方代码库已包含 TPU backend，PagedAttention 在 TPU 静态图上的水土不服由 Google 工程师用 Pallas 写的自定义 kernel 补齐。

#### 10.2 三家的位置

| 框架 | 定位 | 目标客群 |
|---|---|---|
| **vLLM** | 生态兼容王，"代码不想改" | 创业公司、多云客户、GPU 迁移 |
| **JetStream** | TPU 性能榨汁机 | 大厂、高并发推理，愿意为性能改框架 |
| **Saxml** | JAX 生态历史重型武器 | 深度绑 JAX 的存量大客户、特殊大规模切分 |

#### 10.3 JetStream 凭什么比 vLLM 快 20%-50%

JetStream 是 Google Cloud + XLA 团队联合主导，专为 v5 系列定制。它**不去硬凑动态分页**，而是完美契合 TPU 的静态编排哲学：

- 极深度的连续 Batching 优化
- 大量 XLA 算子融合
- 直接跟编译器协同设计，没有 PyTorch 这层间接

代价：API 不像 vLLM 那么开箱即用，对 PyTorch 生态的支持需要专门做。

#### 10.4 Saxml 为什么靠后

最早伴随 Pax / Seqio 一起诞生，深度绑 JAX。带浓厚的 Google 内部基础设施味道，外部上手门槛高，PyTorch 支持滞后。在公有云推广优先级靠后。

#### 10.5 ↔ GPU

| TPU | GPU |
|---|---|
| vLLM-TPU（带 Pallas） | vLLM 原生 |
| JetStream | TensorRT-LLM（NVIDIA 自家 + 特化） |
| Saxml | (无完全对应；DeepSpeed Inference 算半个) |
| Pallas 写 kernel | Triton / CUDA C++ |

---

### 11. PagedAttention 与连续批处理在 TPU 上的适配

**一句话**：GPU 上的动态内存管理（PagedAttention、Continuous Batching、Radix Tree）天生不适合静态编译；TPU 上靠 Pallas 写自定义 kernel + 把动态切到张量层来适配。

#### 11.1 没有 Pallas 之前：静态连续分配的低效

早期 TPU 推理（T5、早期 Pax）走的是"强迫症"路线：

- XLA 在编译期按 `[Max_BS, Max_SeqLen, Hidden_Dim]` 一次性挖出连续 KV Cache 池
- 假设 max_seq=4096，每个请求锁死 4096 个 Token 的 HBM 空间
- 用户请求只有 100 个 Token？剩下 3996 个 slot 全部空跑（浪费 97% 显存）
- HBM 被无效 padding 占满 → batch size 上不去 → MXU 计算裕量充足但请求接不进 → **被内存墙卡死了算力**

Google 早期靠"钞能力"扛——Pod 总 HBM 大、任务长度可控（翻译/搜索）、算法团队把桶切得极细——硬挺过去了。但长文本和多轮对话普及后这条路走不通。

#### 11.2 现代分治：XLA 建池子，vLLM 记账，Pallas 按图索骥

vLLM on TPU 现代架构是**控制面（CPU）+ 数据面（TPU）分离**：

| 角色 | 在哪 | 干什么 |
|---|---|---|
| **XLA** | TPU | 在 HBM 里分配一个一维化的巨大块张量 `[Num_Total_Blocks, Block_Size, Head_Dim]`（比如 10 万个物理块，每块 16 个 Token）。**XLA 不知道里面装的是谁的数据** |
| **vLLM** | CPU | 维护 Radix Tree 和所有内存页表（Block Tables）。每个 step 把当前活跃请求的页表打包成整数 Tensor 喂给 XLA 图 |
| **Pallas Kernel** | TPU | XLA 图里的一个 Custom Call 节点，接收页表后执行底层间接寻址（Gather），把零散物理块拼进 Unified Buffer 做 Attention |

**物理 HBM 还是 XLA 圈的全局张量，但 XLA 不再管理里面的内容。** vLLM 当调度员每个 cycle 把"寻址地图"发过去，TPU 上的 Pallas 算子照着地图取数据。

#### 11.3 三把刀的逐项落地

**PagedAttention（分页注意力）**

- TPU 阵痛：XLA 极度讨厌动态指针寻址，每次 Attention 都查页表会让 DMA 剧本乱套
- 解法：Pallas Kernel 在硬件寄存器层面手写页表查找 + 离散 Gather
- 结论：完全支持，HBM 碎片问题解决，batch size 提升

**Continuous Batching（连续批处理 / Inflight Batching）**

- GPU 玩法：1D 展平，调度器随时踢掉完成的、塞进新的，绝对动态
- TPU 玩法（**静态大巴模式**）：
  - XLA 预编译 `Batch_Size = 256` 的固定图，相当于 256 座大巴永远绕圈
  - 某请求生成到 EOS → vLLM 把座位标空 → 下个 step 把新请求的第一个 Decode Token 塞进刚空出来的索引
  - TPU 只看到完美 `[256, 1, D]` 张量，不知道索引 5 上一毫秒是 A、这一毫秒是 B
  - 不够 256 真实请求时用 Dummy Token（全零）填满

**Radix Tree（前缀缓存）**

- 在 TPU 上**完美适用**：本质是 CPU 端的调度算法
- 命中前缀时 vLLM 只需修改下发的 Block Table 让逻辑块指针指向已有物理块
- TPU 底层 Pallas 不知道是复用，按地图正常取数即可

#### 11.4 Google 自家也用同样的思想

JetStream / Saxml 也实现了等价机制（内部叫 **Blocked Attention** 或内置在底层的 **FlashAttention-TPU** 算子里），同样用 Pallas 写。所以 GKE 上跑 vLLM 还是 JetStream，**显存管理思想殊途同归**：HBM 维护离散 Block Pool + CPU 维护页表 + 计算时把页表传给底层 Kernel 做 Gather。

#### 11.5 一个核心哲学：FLOPs 换 Control Flow

**用极其廉价的 FLOPs（算力），去消除极其昂贵的 Control Flow（控制流）。**

这条贯穿 TPU 整个推理适配。Tree Attention（第 14 章）也是同一思想——把 if-else 编码成 Mask 矩阵，宁可多算多扔，也不让硬件停下来做分支判断。

#### 11.6 ↔ GPU

| 维度 | TPU | GPU |
|---|---|---|
| KV 分页 | XLA 池子 + Pallas Custom Call | vLLM 原生 PagedAttention |
| 调度灵活度 | 固定 batch 桶 + Dummy padding | 动态 1D 展平 |
| Custom kernel 工具 | Pallas | Triton / CUDA |

---

### 12. Prefill / Decode 协同与 Chunked Prefill

**一句话**：TPU 在 Prefill 强、Decode 弱（HBM 带宽瓶颈），混合执行 + chunked prefill 是用算法补硬件。

#### 12.1 算术强度决定 TPU 体感

判断硬件适合不适合一个 workload，看 **算术强度 = FLOPs / Byte**（每读写一个字节内存能做多少次浮点运算）。

| 阶段 | 数学形式 | 算术强度 | 瓶颈 | TPU 体感 |
|---|---|---|---|---|
| **Prefill** | GEMM（矩阵 × 矩阵，权重被几千 token 复用） | 极高 | Compute-Bound | MXU 跑得极爽 |
| **Decode** | GEMV（矩阵 × 向量，权重读出来只算 1 个 token 就丢） | 极低 | Memory-Bound | MXU 大量饥饿停机 |

所以 TPU 骨子里是 **Prefill 怪物**，Decode 阶段是被按在地上摩擦后强行优化出来的。

#### 12.2 Decode 阶段 TPU 怎么搞 Continuous Batching

**Decode 阶段不需要给 token 设桶**——每个请求当前要算的新 token 长度恒为 1。设的是 **Batch Size 的桶**。

预编译 BS=256 的 Decode 图：

- 静态输入矩阵 `[256, 1, Hidden_Dim]`
- 200 个真实请求 → 前 200 槽位放真 token，后 56 槽位塞 Dummy Token（全零）
- TPU MXU 算出 256 个结果
- CPU 调度器只把前 200 个真实结果拿走发用户，后 56 个丢弃

**核心难点：256 个请求的历史长度都不一样怎么办？**

CPU 还要传两个静态大小的整数数组：

```
context_lengths : 形状 [256]，记录每个请求真实历史长度，如 [105, 3042, 12, ...]
                  Dummy 槽位的长度填 0
block_tables    : 形状 [256, Max_Blocks]，每个请求的 KV 页表
```

Pallas 算子按 `context_lengths` 做循环边界（或 mask），按 `block_tables` 去 HBM gather 对应历史 KV，做 Attention。

#### 12.3 Prefill 的难处：每个请求 Prompt 长度差异巨大

Decode 可以拼成 `[256, 1]` 的整齐方块，但 Prefill 不行：有人 prompt 100，有人 3000，怎么塞进静态图？

两种方案：

- **分桶**：长度 100 → padding 到 128 桶；3000 → 切到 4096 桶
- **Chunked Prefill**：编译固定长度的 Prefill 图（如 chunk_size=512）。长 1000 的 prompt 切两块 512，分两次塞进同一个 `[1, 512]` 槽位计算

#### 12.4 Prefill / Decode 混合：静态 1D 展平

最前沿的做法：把 Prefill 长序列和 Decode 单 token 揉进**同一个 step**。

XLA 编译时设两个静态上限：

```
Max_Total_Tokens = 1024   # 一个 step 最多吞 1024 个 token
Max_Seqs        = 256     # 最多 256 个并发序列
```

输入张量从 3D 的 `[Batch, Seq, D]` 展平成 2D 的 `[1024, D]`。

CPU 端拼装：

```
请求 A (Prefill, 切下 chunk=512)  →  数组前 512 位
请求 B~Z (Decode, 200 个 token)   →  紧接 200 位
                                     共 712 位
Dummy Token × 312                    →  补齐到 1024
```

**MXU 阶段**：对脉动阵列来说不存在身份差别。一个巨大 `[1024, D] × [D, 4D]` 矩阵乘法全速冲过去，1024 个 token 的 Q/K/V 一次性算出。

**Attention 阶段**：到这里就糊弄不过去了——

| Token 类型 | 该怎么 Attend |
|---|---|
| Prefill 的 512 个 | **互相**做 Attention，生成新 KV 写回页表 |
| Decode 的 200 个 | 各自用自己的 1 个 Token 去 attend 自己的历史 KV Cache |
| Dummy 的 312 个 | 跳过 |

CPU 同时传 metadata：`seq_lens = [512, 1, 1, ..., 0, 0]`。Pallas 算子在底层解析 metadata，对 Prefill 块走 FlashAttention 逻辑（Q 向量在 UB 里互相点乘 + 写新 KV），对 Decode 块走 PagedAttention（gather 历史 KV），对 Dummy 块跳过。

#### 12.5 GPU vs TPU 混合的差异

| | GPU | TPU |
|---|---|---|
| 拼装方式 | 动态：712 → kernel 收 712；下次 850 → 收 850 | 静态：712 → 强行加 312 个 dummy → 1024 |
| 硬件成本 | 调度器开销 | Padding 的 MXU cycle |
| 软件成本 | kernel 灵活性高 | Pallas metadata 路由复杂 |

**Trade-off**：Padding 浪费一小部分 MXU cycle 是可控的，但避免了重新编译 XLA 图的灾难，同时控制延迟抖动。两害相权取其轻。

#### 12.6 ↔ GPU

GPU 玩"网约车"：712 个乘客就发 712 座的车，绝不拉空座。
TPU 玩"高铁直达专列"：班次到点就发，不够人就放假人。

---

### 13. KV / 内存层次

**一句话**：GPU 体系里的 RDMA / GDS / KV offload 在 TPU 上有的天生支持、有的不支持、有的只能走 PCIe 后备。

#### 13.1 三件事的对照

| 优化技术 | GPU | TPU |
|---|---|---|
| **跨节点通信绕开 CPU**（GPU: RDMA over IB） | GPUDirect RDMA + IB 网卡 | **天生集成**：ICI 网络控制器直接做进 TPU 硅片，根本不需要外部网卡。CPU 完全不知情，零 CPU 周期 |
| **直读存储到加速卡**（GPU: GPUDirect Storage） | NVMe → PCIe → GPU VRAM，绕过 CPU 内存 | **不支持**。TPU 没有直接读硬盘 / 外部网络的接口，必须 CPU 中转：GCS / PD → Host VM DDR → PCIe DMA → TPU HBM |
| **KV Cache offload 到 host DRAM**（GPU: vLLM CPU swap） | HBM → PCIe → CPU DDR | **完全适用**。vLLM Block Manager 在 host CPU 上跑，HBM 满了触发 PCIe DMA 把 KV Block 搬到 host DDR |

#### 13.2 ICI 比 RDMA 更彻底

GPU 的 RDMA：数据 → PCIe → NIC → IB 网络 → NIC → PCIe → VRAM。绕过 CPU 内存，但还是要离开 GPU 芯片走外部网卡。

TPU 的 ICI：数据 → 芯片边光模块 → 光纤 → 对端光模块 → 芯片。**根本不离开芯片硅片体系**。每秒几百 GB 的网络风暴里 host CPU 完全不知情。

#### 13.3 直读存储缺失为什么能忍

冷启动加载权重要把几百 GB 数据从 GCS 拉到 TPU HBM，必须 host CPU 当搬运工。但因为大模型部署是 multi-host 的（如 16 台 VM），16 台机器 CPU **并发**从 GCS 下载权重的不同 shard，总网络带宽极大，工程上可接受。

#### 13.4 KV Offload 性能特征

PCIe 带宽相对 ICI 来说**很窄**（v4 PCIe Gen4 x16 ≈ 64 GB/s 双向，而 ICI 单链路就能上几百 GB/s）。所以频繁 swap 会显著拖性能。这是防 OOM 的保底策略，不是首选。

#### 13.5 ↔ GPU

GPU 的优势在 GDS（直读存储），TPU 的优势在 ICI（更彻底的网络旁路）。两边各占一半。

> **[补充 — Claude 加]** 业界开始有 **Mooncake 类的"分离式 KV pool"**（KV Cache 单独服务化、跨节点共享），目前主要在 GPU 体系。TPU 上对应的工作没看到公开方案。这条不写进正文，仅供你 cluster TL 视角参考。

---

### 14. Gemini 在 TPU 上的实战妥协

**一句话**：MoE 和投机解码这两个推理优化，在 TPU 上都得改算法去迁就硬件。

#### 14.1 静态化 MoE：Capacity Factor

MoE 的天然问题是**动态路由**——你不知道下一个 token 会去找哪个 expert。XLA 不允许动态形状。

Gemini 团队的解法：

- 给每个 expert 设一个严格的 **Capacity Factor**（容量因子 / 静态槽位大小），比如规定每个 expert 一个 step 最多接 64 个 token
- 路由过来的 token 不足 64 个 → 塞 Dummy Padding 凑齐
- 路由过来的 token 超过 64 个 → 多出的**直接丢弃**（Token Dropping），或者强制走兜底通用网络

通过这种粗暴的截断 + 填充，MoE 的动态网络被强制拍平成 XLA 喜欢的静态计算图。

代价：偶尔丢 token，模型质量会受影响。Google 在 Gemini 训练时调 Capacity Factor 平衡丢弃率和算力开销。

#### 14.2 张量化投机解码：Tree Attention

传统投机解码：小模型猜 K 个 token → if-else 判断大模型是否同意 → 拒绝则回退。这种 if-else 流让 TPU VLIW 流水线崩溃。

Gemini 的解法（**并行验证**）：

- 小模型生成 K 个猜测 token 后，大模型把这 K 个拼成一个 1D 向量
- 设计一个特殊的 **Tree Attention Mask**（树状注意力掩码），让不相关的节点互相看不到（Mask 乘 0）
- 大模型在一次前向传播里**用一个矩阵乘法一口气把 K 个 token 的概率全部验证完**
- 把猜对的那条路径挑出来，其他废弃

这把"分支代码"变成了"小规模 Prefill 矩阵运算"。TPU 的 MXU 又狂喜了。

#### 14.3 为什么这些妥协在 GPU 上不是必须

- GPU 的 SIMT 调度器擅长 if-else（虽然分支发散会浪费 lane，但比 TPU 强得多）
- GPU 的动态显存可以容忍 expert 容量浮动
- 所以 GPU 上跑原始版 MoE 路由 + 原始版投机解码也能工作

GPU 路线图里也在朝 Tree Attention 等张量化方向走，但不是被硬件逼的，是为了榨更多性能。

#### 14.4 一个有趣的趋势

正因为硬件极度讨厌分支（TPU 完全不能容忍，GPU 也不喜欢 host-device 高频同步），算法工程师在**绞尽脑汁把控制流（逻辑分支）改写成数据流（矩阵 mask）**。Tree Attention、Masked Attention、谓词执行 (Predication) 都是这个思路的不同形态。

核心哲学：**全算再扔比 if-else 便宜**。计算规模有限的浅层分支（投机解码 3-5 步、Causal Mask 半个矩阵）这个套路超划算；但深层嵌套（10+ 层条件树）就 $O(2^N)$ 爆炸。所以新一代芯片设计在追 **硬件原生稀疏支持**——掩码全 0 时电路在物理层面跳过乘加运算（时钟门控断电），既不写 if-else 也不耗能。

#### 14.5 ↔ GPU

| 优化 | TPU 必须改 | GPU 可以不改 |
|---|---|---|
| MoE | Capacity Factor 静态化 | 动态路由可工作 |
| 投机解码 | Tree Attention 张量化 | if-else 也能跑（性能差点） |

---
