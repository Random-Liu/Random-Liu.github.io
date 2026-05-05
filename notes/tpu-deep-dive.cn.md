# TPU 深入笔记：从单芯片到生产集群

> **状态**：骨架阶段，章节结构已确认，正文内容待源对话可读后填入。  
> **目的**：给自己留一份从硬件原理一路到推理 / 集群层适配的端到端参考，重点在**为什么这么设计**而非术语解释。  
> **对照对象**：相同问题在 GPU 体系下是怎么做的——每章末尾会有 `↔ GPU` 小节。

---

## Part I — 硬件层：芯片、互联、封装

### 1. 单芯片：MXU、VPU、SPU 与脉动阵列

一句话：TPU 把矩阵乘法做成专用电路（MXU），用脉动阵列在芯片内部「传递数据」而不是「传递地址」，省掉了缓存层级。

- TODO：MXU / VPU / SPU 三者的角色分工
- TODO：CISC 指令集的取舍
- TODO：脉动阵列在矩阵乘法里逐 cycle 的数据流转图（用 ASCII 或 Mermaid）
- TODO：↔ GPU（SM / Tensor Core / 寄存器堆 的对比）

### 2. 多芯片互联：ICI 与 3D Torus

一句话：单芯片很强，但真正让 TPU 成为 TPU 的是片间网络——ICI 是物理层、3D Torus 是逻辑拓扑。

- TODO：ICI 的位置（链路层、带宽、延迟数量级）
- TODO：3D Torus 的几何（每芯片 6 个邻居、X/Y/Z 三轴）
- TODO：为什么选 Torus 而不是 fat-tree
- TODO：↔ GPU（NVLink / NVSwitch / IB 的位置）

### 3. OCS：光路交换让拓扑可重配

一句话：TPU Pod 在物理层是 star（每机架到 OCS 的若干光纤），逻辑层却被 OCS 接成 torus；MEMS 小镜子在微秒级切换光路。

- TODO：MEMS 镜子的工作原理（机械翻转 vs 半导体）
- TODO：4×4×4 机架为什么有 96 根外出光纤（角×3 + 棱×2 + 面心×1 的几何约束）
- TODO：OCS 切分粒度（机架 / Block 而非单芯片）的工程原因
- TODO：↔ GPU（packet-switched IB 的世界对应不到这个）

### 4. 集合通信：Ring All-Reduce 在 3D Torus 上的降维

一句话：3D Torus 上做 All-Reduce 的诀窍是把它拆成 X/Y/Z 三个 1D 圈分别做，再合起来——这是几何换算法。

- TODO：1D 圈的 All-Reduce 步骤
- TODO：3D 降维的合成顺序与时间复杂度
- TODO：「圈是 1D 但连线是 3D，剩下的线不就闲置了？」这个问题的回答
- TODO：NUCA：跨机架大圈中，铜缆 vs 光纤路径的延迟异构
- TODO：XLA 怎么把 TP（密集通信）映射到短边铜缆、把 DP（稀疏同步）映射到长光环
- TODO：↔ GPU（NCCL ring/tree、NVLink 全互联下的差异）

### 5. Host ↔ TPU：PCIe、NUMA 与 multi-host slice

一句话：TPU 不是独立机器，是挂在 CPU host 上的 PCIe 设备；slice 一旦跨 host，就是天然的分布式系统。

- TODO：1:4 / 1:8 的 CPU:TPU 比例从哪来
- TODO：multi-host slice 的 N:N 映射、SPMD 进程是怎么起的
- TODO：NUMA 1:4 PCIe 劈管、单 NUMA VM 的切分策略、XLA 自动绑核
- TODO：LWS / JobSet 在 K8s 层怎么把这套 N:N 表达出来
- TODO：↔ GPU（HGX 8 卡 + NUMA、GDS 直连存储）

### 6. 先进封装：算力面积 vs 带宽周长

一句话：Die 上面积决定算力（FLOPs），周长决定带宽（HBM 接口）；2.5D / 3D 封装是在调和两者的根本张力。

- TODO：硅中介层（CoWoS 类）的角色
- TODO：TSV（Through-Silicon Via）解决了什么
- TODO：HBM 带宽 vs 算力增长的不对称（这条会在 Ch 19 再呼应）
- TODO：↔ GPU（H100 / B100 的封装方案）

---

## Part II — 编译与运行时：XLA

### 7. XLA 编译模型：静态调度做减法

一句话：XLA 的核心打法是「把所有不确定性在编译期消掉」——算子融合、静态 padding、软件流水线、VLIW 指令包都是这个思路的不同切面。

- TODO：HLO → LLO 的层次
- TODO：算子融合的边界（什么能融、什么不能融）
- TODO：静态 padding 的代价与收益
- TODO：软件流水线在 systolic array 上的体现
- TODO：VLIW 指令包的五槽：DMA / MXU / VPU / SPU / **ICI**——为什么 ICI 也是一槽
- TODO：↔ GPU（动态调度器、SM warp scheduler、CUDA Graph 的对比位置）

### 8. 编译时机：JIT、AOT、bucketing、persistent cache

一句话：静态编译的代价是首跑慢，工业上靠 bucketing + AOT + cache 把这个代价摊薄。

- TODO：JIT 触发条件与冷启动延迟
- TODO：bucketing 的 shape 桶设计
- TODO：AOT 预编译的部署链路
- TODO：persistent cache 的实际形态
- TODO：↔ GPU（PyTorch 动态图 / TorchInductor / Triton AOT 的对照）

### 9. XLA 拓扑感知映射

一句话：编译器知道 3D Torus + OCS 的物理拓扑，所以能把高密度通信映射到短边、低频同步映射到长环。

- TODO：拓扑信息怎么传给 XLA
- TODO：TP（短边铜缆）、DP（长环光纤）的自动决策
- TODO：与 Ch 4 NUCA 部分的呼应
- TODO：↔ GPU（手工 NCCL group、torch.distributed 的拓扑感知能力）

---

## Part III — 推理层适配（目标 C）

### 10. 软件栈分叉：vLLM、JetStream、Saxml、GKE

一句话：TPU 上推理框架不止一个，三家定位不同；GKE 是把它们都装进集群的胶水。

- TODO：vLLM-TPU 的位置与裁剪
- TODO：JetStream 的角色（Google 内外用法的差异）
- TODO：Saxml 的服务模型
- TODO：三者在 GKE 上的部署形态对比
- TODO：↔ GPU（vLLM / TGI / TRT-LLM 的对应）

### 11. PagedAttention 与连续批处理在 TPU 上的适配

一句话：GPU 上的动态内存管理（PagedAttention、Continuous Batching、Radix Tree）天生不适合静态编译；TPU 上只能靠 Pallas 写自定义 kernel + 把动态切到张量层。

- TODO：PagedAttention 的核心难点：动态 indirection
- TODO：Pallas kernel 在哪一层做适配
- TODO：连续批处理的实现取舍
- TODO：Radix Tree（前缀缓存）的适配现状
- TODO：「用极其廉价的 FLOPs 去消除极其昂贵的 Control Flow」这个原则的展开
- TODO：↔ GPU（vLLM 原生的 paged 实现）

### 12. Prefill / Decode 协同与 Chunked Prefill

一句话：TPU 在 Prefill 强、Decode 弱（HBM 带宽瓶颈），混合执行 + chunked prefill 是用算法补硬件。

- TODO：Prefill / Decode 在硬件资源上的差异
- TODO：Chunked Prefill 的切片策略
- TODO：静态 1D 展平怎么把变长 batch 套进静态 shape
- TODO：↔ GPU（vLLM 的 continuous batching、SARATHI）

### 13. KV Cache 与内存层次

一句话：GPU 体系里的 RDMA / GDS / KV offload 在 TPU 上有的天生支持、有的不支持、有的只能走 PCIe 后备。

- TODO：ICI 天生 bypass host（相当于免费 RDMA）的语义
- TODO：GDS 类直连存储在 TPU 上为什么没对应物
- TODO：KV cache offload 的实际路径（HBM → PCIe → host DRAM / SSD）
- TODO：↔ GPU（NCCL over IB、GDS、Mooncake 类 KV pool）

### 14. Gemini 在 TPU 上的实战妥协

一句话：MoE 和投机解码这两个推理优化，在 TPU 上都得改算法去迁就硬件。

- TODO：MoE 的 All-to-All 与 3D Torus 的张力（这条会在 Ch 19 再呼应硬件原因）
- TODO：Capacity Factor 的作用（控制每专家上限以变成静态 shape）
- TODO：投机解码 Tree Attention 的 mask 设计
- TODO：为什么这些妥协在 GPU 上不是必须

---

## Part IV — 集群层适配（目标 D）

### 15. K8s 上的 TPU 抽象

一句话：K8s 看不见光，所以 OCS 切分必须由独立组件负责，TPU device plugin + 拓扑标签 + Kueue + TPU Provisioner 组成完整链路。

- TODO：device plugin 暴露给 kubelet 的资源粒度
- TODO：拓扑标签（node label）携带的 3D 坐标信息
- TODO：Kueue gang scheduling 为什么必要（slice 必须整组上）
- TODO：TPU Provisioner 调用 OCS API 的时机
- TODO：↔ GPU（NVIDIA device plugin、Volcano gang、Topology Aware Scheduling）

### 16. Multi-host slice 的编排

一句话：一个 slice 跨多 host 时，K8s 看到的是 N 个 pod 的协同启动，每个 pod 内的 TPU 又是 4 / 8 个芯片的本地组——这是两层 N:N。

- TODO：LWS（LeaderWorkerSet）的语义
- TODO：JobSet 的角色与 LWS 的关系
- TODO：调度耦合点：哪一层 fail 会拖整个 slice
- TODO：↔ GPU（MPI Operator、Training Operator、Ray on K8s 的对应）

---

## Part V — 系统对比与权衡（目标 B 集中点）

### 17. 编程模型链：从单卡到多机的指令链

一句话：GPU 是「单卡 CUDA → 多卡 NCCL → 多机 IB/RDMA」三段；TPU 是「SPMD → ICI（VLIW 第五槽）」一段，编译器统管。

- TODO：CUDA → NCCL → IB/RDMA 的语义跳转点
- TODO：SPMD + ICI 的「无缝」是怎么做到的
- TODO：两套模型对故障半径、调度灵活度的影响

### 18. 成本 / 能效

一句话：MFU 和 Tokens/$ 这两个指标是衡量真实账面差异的杠杆，不是芯片峰值算力。

- TODO：MFU（Model FLOPs Utilization）的定义与典型水位
- TODO：Tokens per dollar 的计算口径
- TODO：Midjourney 案例（$2.1M → <$700K，对话里给出的数字）
- TODO：Character.AI 案例
- TODO：「峰值 TFLOPs 不等于实际产出」这个 trade-off 集中在这

### 19. TPU 的硬件劣势与权衡

一句话：每个静态调度的优势都对应一个不擅长的工作负载——MXU 粒度大、SPU 弱、3D Torus All-to-All 拥塞、HBM 带宽 vs 算力失衡。

- TODO：MXU 128×128 粒度对小矩阵的浪费
- TODO：SPU 弱在哪些场景下成为瓶颈
- TODO：MoE All-to-All 与 3D Torus 的根本不匹配
- TODO：HBM 带宽 vs 峰值算力增长的剪刀差（呼应 Ch 6、Ch 12）
- TODO：每条劣势的 trade-off：换来了什么

---

## 附录

### A. Trade-off 速查表

按设计决策维度横切，每条 trade-off 链接回原章节。计划维度：

- 静态 vs 动态（编译 vs 调度）
- 密度 vs 灵活（MXU 大 vs 小、Torus vs fat-tree）
- 算力 vs 带宽（封装 / HBM）
- 集中 vs 分布（编译器 vs 运行时调度器）

### B. 数字 / 参数清单

所有数字都注明「源自原对话」。计划条目：

- TPU v4: 4096 芯片
- TPU v5p: 8960 芯片
- MXU: 128×128
- HBM 带宽数量级：TODO
- Midjourney: $2.1M → <$700K
- Character.AI: TODO
- CPU:TPU 比例：1:4 或 1:8
- 4×4×4 机架: 96 根光纤

### C. 术语 ↔ GPU 等价物对照

纯查询用。计划条目：

- ICI ↔ NVLink + IB（合并对应）
- SPMD ↔ NCCL collective
- Pallas ↔ Triton
- OCS ↔ （无对应）
- XLA ↔ TorchInductor / TensorRT
- MXU ↔ Tensor Core
- VPU ↔ CUDA Core
- HLO ↔ FX Graph
- TPU Provisioner ↔ （无对应，最接近的是 Slurm topology + 手工 NCCL）

---

## 写作日志（让作者验收用）

### 主动取舍（待原文读取后填）

- TODO

### 外部补充（Claude 加，原文未提及）

- TODO，每条会用 `> **[补充 — Claude 加]** ...` 在正文中标注
