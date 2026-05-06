# TPU Deep Dive: From Single Chip to Production Cluster

> **Purpose**: A reference for myself spanning hardware principles all the way to inference and cluster-layer adaptation, focused on **why each thing is designed the way it is** rather than terminology explanation.
> **Comparative target**: Each chapter ends with a `↔ GPU` subsection for nearby comparison.
> **About "additions"**: Every block tagged `> **[Note — added by Claude]** ...` is content not present in the source conversation. Please review and decide whether to keep.

---

## Part I — Hardware: Chip, Interconnect, Packaging

### 1. The Single Chip: MXU, VPU, SPU, and the Systolic Array

**One sentence**: TPU bakes matrix multiplication into a dedicated circuit (MXU); the systolic array "passes data" inside the chip rather than "passing addresses", which removes the cache hierarchy.

#### 1.1 The Systolic Array: Stepping Around the von Neumann Bottleneck

In a general-purpose architecture, each ALU op fetches operands from a register or L1 and writes back. LLM inference demands hundreds of GB/s of matmul throughput, and this "compute once, hit memory once" pattern gets pinned by bandwidth.

TPU's break: arrange compute units into a 2D grid (the v4 MXU is 128×128 MAC cells) and let data flow through like a pumping heart:

- **Weight stationary**: Before compute starts, weights are locked into each MAC cell.
- **Data systole**: Activations stream in from the left and from above, advancing right and down each cycle.
- **Hard-wired neighbors**: After each cell does its multiply-accumulate, the result is **passed directly through physical wires** to the adjacent cell — no register write.

Across the array, data is reused hundreds to thousands of times without ever touching SRAM. That's why, on the same silicon area, TPU packs far more pure compute units than a GPU.

#### 1.2 A Concrete 2×2 Walk-Through

Let $Y = X \times W$, with both activation and weight as 2×2, and the MXU as a 2×2 grid in **weight-stationary** mode:

```
Weight layout:  PE(1,1)=W11   PE(1,2)=W12
                PE(2,1)=W21   PE(2,2)=W22

Activation in:  Rows of X enter from the left; row 2 lags row 1
                by one cycle ("skewed" entry)
Partial sums:   Flow downward
```

Cycle by cycle:

- **Cycle 1**: $X_{11}$ enters PE(1,1), computes $X_{11} \times W_{11}$, passes the partial sum down and $X_{11}$ right.
- **Cycle 2**: $X_{11}$ flows to PE(1,2); $X_{21}$ enters PE(2,1), adds the incoming $X_{11} \times W_{11}$ to its own $X_{21} \times W_{21}$ — at this moment, what PE(2,1) outputs downward is the final $Y_{11}$.
- **Cycle 3**: Bottom-right PE(2,2) collects both inputs and outputs $Y_{12}$.
- **Cycle 4+**: Remaining results "drip" out from the array's bottom in sequence; no register lookup at any step.

#### 1.3 Macro Architecture: Who Else Lives Beside the MXU

The MXU alone can't run a full network. A TPU core also has:

| Component | Role |
|---|---|
| **MXU** | Dense matrix multiply-accumulate; the algebraic muscle |
| **VPU** | Vector ops: activations (GeLU/ReLU/Softmax), LayerNorm — anything that doesn't compress to a matmul |
| **SPU** | Scalar and control flow: loops, branches, address computation. Weak in raw ALU, but the conductor |
| **Unified Buffer** | Tens of MB of on-chip SRAM, holds activations staged from HBM and intermediate results |

Instructions follow a **CISC** style: a single `MatrixMultiply` triggers thousands of MACs in the MXU, keeping decode/dispatch overhead extremely low.

#### 1.4 SPU and VLIW: With Hardware Static, Who Runs Control Flow?

The MXU on its own doesn't know what to compute or where data comes from. That coordination falls to SPU + **VLIW (Very Long Instruction Word)**:

XLA packs multiple non-conflicting operations into one long instruction. A single instruction encodes four things:

```
[ DMA ] | [ MXU ] | [ VPU ] | [ SPU control flow ]
```

Example: `[DMA stage next activation block] | [MXU compute current block] | [VPU apply ReLU on previous result] | [SPU update address]`.

How the SPU works:

- **Fires before results land**: Pushes control signals into the MXU's FIFO; as long as the queue isn't full, SPU runs ahead.
- **Synchronization via barriers**: When you must wait for the MXU before continuing, the compiler inserts `WAIT_MXU_DONE` — a hardware-level stall.
- **Doesn't compute**: SPU's compute power is intentionally tiny; with no animation duties beyond loops, addresses, and barriers, its overhead is negligible.

The entire timing is determined 100% before the program runs.

#### 1.5 ↔ GPU

| Dimension | TPU | GPU |
|---|---|---|
| Matrix unit | One giant 128×128 MXU | Many small Tensor Cores (16×8×16) scattered across SMs |
| Scalar/vector unit | Weak SPU, mid-tier VPU | Many CUDA Cores, strong branch prediction |
| Data buffer | Unified Buffer (software-managed via DMA) | L1/L2 Cache (hardware prefetch) |
| Instruction model | VLIW, scheduled at compile time | SIMT, switched at runtime by Warp Scheduler |
| Hiding memory latency | Static pipelining | Massive thread-level concurrency |

**Trade-off**: TPU hands flexibility to the compiler and lets hardware execute literally; GPU keeps flexibility in hardware and accepts that a chunk of silicon goes to scheduling.

---

### 2. Inter-Chip Interconnect: ICI and the 3D Torus

**One sentence**: A single chip is fast, but what makes a TPU a TPU is the inter-chip fabric — ICI is the physical layer, 3D Torus is the logical topology.

#### 2.1 ICI: Networking Built Into the Silicon Edge

Each TPU integrates a set of high-speed interfaces, **ICI (Inter-Core Interconnect)**, right at the silicon edge:

- **No external switches**: Inside the Pod, data movement does not pass any traditional networking gear; it hops directly from one chip's optical module to the next.
- **Low and deterministic latency**: With switch buffering, route lookup, and congestion control all skipped, XLA can compute, in nanoseconds, how long data takes to traverse the Pod and use that for global scheduling.

> **[Note — added by Claude]** The source doesn't give a concrete ICI link bandwidth. Public material puts v4 at the order of 4.5 TB/s aggregated across all 6 directions per chip. Decide whether to keep this number in the note.

#### 2.2 3D Torus: Six Neighbors, Wrapped at the Edges

Each chip has a coordinate $(X, Y, Z)$ in 3D space and connects via ICI to neighbors in $\pm X$, $\pm Y$, $\pm Z$. **The wrap-around is key**: at any dimension's boundary, the link loops back to the start to form a ring, so the Pod has no "edge node".

Why a Torus over a fat-tree:

- **Redundant paths**: A broken link can be detoured via another dimension.
- **Collective-communication friendly**: Operations like All-Reduce / All-Gather map directly onto rings — no route table computation.

The cost:

- **Network diameter is linear**: As $N$ scales, max hops grow as $O(N^{1/3})$, vs $O(\log N)$ for fat-tree.
- **Irregular point-to-point causes congestion**: MoE All-to-All on a Torus produces hot links — see Chapter 19.

#### 2.3 ↔ GPU

| Dimension | TPU | GPU |
|---|---|---|
| Within node | ICI (directly on chip edge) | NVLink / NVSwitch (separate switch chip) |
| Across node | ICI + OCS (one fabric, all optical) | InfiniBand + NIC + switches |
| Topology | 3D Torus | Fat-tree / multi-tier Spine-Leaf |
| Latency | Statically predictable | Dynamic with jitter |

**Trade-off**: A Torus is dominant on regular collective comms but bleeds on dynamic sparse traffic (MoE); fat-tree is the inverse.

---

### 3. OCS: Optical Switching Makes the Topology Reconfigurable

**One sentence**: A TPU Pod is a star at the physical layer (each rack runs fibers into the OCS), but a Torus at the logical layer; MEMS micromirrors switch optical paths in microseconds.

#### 3.1 MEMS Mirrors: Pure Reflection, No Packet Inspection

A regular Ethernet/InfiniBand switch is **O-E-O** (optical → electrical → optical): fiber comes in → converted to electrical → ASIC reads packet headers, looks up routes, queues into buffers → converted back to optical → out. This adds latency, jitter, and the switch chip itself burns serious power.

OCS (Google internal codename Palomar), introduced in v4, is purely **O-O-O**:

- Inside is a sealed cavity filled with inert gas.
- The laser from the input fiber hits a MEMS mirror (the size of a hair, deflected by static electricity).
- It reflects to a second MEMS mirror on the opposite side.
- The second mirror "corrects" the beam to a flat angle and shoots it into the fiber leading to the destination TPU.
- End-to-end: **no digital chip, no buffer, no packet inspection, no bandwidth ceiling**.

The mirrors stay **completely still during transmission** — they only nudge briefly when switching "tracks". OCS is **Data Agnostic**: it doesn't care whether the laser blinks at 100 Gbps or 800 Gbps, it only reflects. Upgrade the optics and the OCS doesn't change.

#### 3.2 Slicing Granularity: Rack-Level, Not Chip-Level

**OCS cannot pick individual chips at will.** Within a rack, the 64 TPUs (a 4×4×4 base block) are wired together with cheap, short DAC copper cables. Only the rack's outward-facing interfaces get optical modules and feed into the OCS.

So OCS's "Lego brick" minimum unit is a 4×4×4 rack. Want 256 chips? Pin together 4 racks. Want 1024? 16 racks. And so on.

#### 3.3 96 Fibers: The Geometry of a 4×4×4 Rack

This is a clean little geometry exercise. How many outward-facing interfaces does the surface (56 chips) expose? Break down by position:

| Position | Count | Fibers per chip | Subtotal |
|---|---|---|---|
| 8 corners | 8 | 3 (exposes X, Y, Z) | 24 |
| 12 edges (each minus its 2 endpoints, leaves 2) | 24 | 2 | 48 |
| 6 faces (each minus edges/corners, leaves 2×2=4) | 24 | 1 | 24 |

Total: **96 fibers**. Cross-check via faces: each face has 4×4 = 16 outward interfaces, 6 faces × 16 = **96**. Matches.

In the data center, those 96 fibers aren't pulled one by one; they go through high-density parallel optical cables (MPO/MTP, 16 or 32 fibers per cable), bundled "waterfall-style" from the top of the rack to a central network rack containing the OCS.

#### 3.4 OCS and 3D Torus: How They Relate

**3D Torus is the logical topology shape; OCS is the joint of the transformer.**

- v2/v3 era: topology was hard-wired physical cabling, rack A → B → C → A welded in place. One bad chip and the whole region was offline.
- v4/v5 era: physical cabling becomes a **star** (every rack's fibers fan into the OCS); the OCS internally "folds" a 3D Torus loop using mirror angles.

Slicing scenarios:

| You ask for | What OCS does |
|---|---|
| Single rack, 64-chip closed loop | Reflects the 96 fibers among themselves (left 16 ↔ right 16, front ↔ back, up ↔ down) |
| 4×4×8 (128 chips, 2 racks) | X- and Y-axes wrap inside each rack; on the Z-axis, rack A's top face (16 fibers) cross-connects to rack B's bottom face, and vice versa |
| Bypass a bad chip | Borrows a fiber from a neighboring rack to maintain the 3D-torus logical integrity |

#### 3.5 The Physical Boundary: One Pod

Lasers attenuate over fibers and free space. The OCS network covers at most one Pod. A v5p Pod is 8960 chips. **The Pod is the absolute boundary of optical interconnect.**

Cross-Pod "communication" must go over standard datacenter ethernet (DCN), with latency and jitter that no longer match ICI and that would shred XLA's static clock. So in practice:

- **Inside one computation (Model Parallelism)**: never crosses Pods.
- **Service-level scheduling (Load Balancing)**: cross-Pod is fine — user A is fully routed to Pod 1, user B to Pod 2, with only an HTTP/gRPC load balancer between them.

#### 3.6 ↔ GPU

GPU clusters **don't have anything like this**. NVIDIA's NVSwitch and InfiniBand are both packet-switched electrical interconnects; OCS is a circuit-switched optical interconnect — a fundamentally different philosophy. This is one of the most distinct designs in the TPU lineage.

> **[Note — added by Claude]** Microsoft Azure has begun trialing Lumen-supplied OCS in some AI clusters, but the volume and tier is nowhere near Google TPU's. Not in the main text — for context only.

---

### 4. Collective Communication: Dimension-Partitioned Ring All-Reduce on a 3D Torus

**One sentence**: The trick to All-Reduce on a 3D Torus is splitting it into independent X / Y / Z 1D rings and running them sequentially — geometry traded for algorithm.

#### 4.1 1D Ring All-Reduce: Reduce-Scatter + All-Gather

Suppose 4 TPUs form a ring head-to-tail, each producing a 400 MB tensor; we need to element-wise-add the 4 of them and let every TPU end up with the full sum.

Naively forwarding everything to TPU 0 instantly chokes its network. XLA's approach: **chunk + pipeline relay**. Cut 400 MB into four 100 MB blocks (A, B, C, D).

**Phase 1: Reduce-Scatter (each ends up holding the full sum of one block)**

| Cycle | Action |
|---|---|
| 1 | Every chip ships one of its blocks rightward: TPU0→A→TPU1, TPU1→B→TPU2, TPU2→C→TPU3, TPU3→D→TPU0. **All 4 wires (including wrap-around) at full load** |
| 2 | TPU 1 adds incoming A to its own A via VPU → "partial sum A", forwards right |
| 3 | One more hop. TPU 3 receives the now-three-way partial-sum A, adds its own A → full-network sum of A |

End of phase: TPU 0 holds full B, TPU 1 holds full C, TPU 2 holds full D, TPU 3 holds full A.

**Phase 2: All-Gather (each broadcasts its own 1/4 of the answer)**

3 hops of relay broadcast; the four full blocks fly around the ring in parallel. Once each chip's Unified Buffer holds A/B/C/D, DMA flushes the final 400 MB to HBM.

#### 4.2 Hardware Detail: The Network Is an Extension of the Cache

TPU turns the network into a direct extension of SRAM:

- Data flows out of the sender's Unified Buffer → optical signal across ICI → the receiver's Unified Buffer.
- VPU pulls operands directly from the Unified Buffer to add — **HBM is never touched along the way**.
- A tiny **Sync Token** is appended to the data on send.

GPU's contrasting flow: HBM → PCIe → NIC → switch → NIC → PCIe → HBM → compute core. Every hop touches memory, and HBM bandwidth is consumed by the comm path itself.

#### 4.3 No Global Clock — Hardware Semaphores Instead

At datacenter scale, keeping hundreds of chips synchronized to a single nanosecond-level physical clock is physically impossible (light-speed delay, clock drift). TPU uses **XLA static schedule + hardware-semaphore async handshake**.

Receiver-side chain:

1. After ICI hardware finishes physical receipt, it **automatically** increments the corresponding hardware semaphore (SPU isn't involved at all).
2. SPU, reading XLA's instructions, sees `WAIT Semaphore_X >= 1` and stalls the VPU.
3. The instant the semaphore flips, WAIT releases — VPU springs like a coiled spring and starts the addition.
4. After computing, it kicks DMA to ship results forward with a fresh Sync Token, and resets its own semaphore to zero.

At the macro scale it looks like thousands of chips locked to the same gear, but actually each chip only watches a few hardware lights right in front of it. XLA pre-aligns the compute timing perfectly, so no deadlocks and minimal idle waiting.

#### 4.4 3D Decomposition: Splitting Into X, Y, and Z 1D Rings

If you string all 64 chips of a 4×4×4 rack into one giant 64-hop loop, disaster:

- Data needs 63 hops to circumnavigate.
- Each chip has 6 wires; only 1 receives + 1 sends, **the other 4 idle**.

The right play is **multi-dimensional orthogonal ring synchronization**: split a 3D task into three parallel 1D tasks.

| Phase | Work | Parallelism |
|---|---|---|
| X-axis sync | 16 parallel X-axis rings (length 4) running Reduce-Scatter + All-Gather | 16 |
| Y-axis sync | The synced data is re-chunked, 16 Y-axis rings run | 16 |
| Z-axis sync | 16 Z-axis rings run | 16 |

Total hops: 4 + 4 + 4 = **12**, vs 64.

Each phase looks like only 1/3 of the wires are active, but XLA uses **pipelining**: while matrix shard 1 is on the Y-axis, shard 2 is already on the X-axis. Macro-view, all 6 wires light up at full load.

#### 4.5 Big Rings Are Inevitable: How Cross-Rack Z-Axis Stitches

In a 4×4×8 (128-chip, 2-rack) slice, X and Y stay as length-4 small rings, but Z becomes a **length-8 big ring**. Physical form:

```
[ Rack A's Z axis ]                                [ Rack B's Z axis ]
TPU(Z=0) — TPU(Z=1) — TPU(Z=2) — TPU(Z=3)          TPU(Z=4) — TPU(Z=5) — TPU(Z=6) — TPU(Z=7)
   ^                                  |                                              |
   |          ← OCS cross-rack splice ←   +— OCS — TPU(Z=4)                          |
   +———————————————————————————————————————— OCS wrap ——————————————————————————————+
```

At full v5p Pod scale (8960 chips, possibly arranged 16×16×35), the longest Z-axis edge becomes a **35-hop big ring**. Pure physical transmission delay alone is enough to starve the upstream MXU.

#### 4.6 NUCA: Heterogeneous Latency Between Copper and Optical

The 8-hop Z-axis big ring contains 2 optical hops (cross-rack) and 6 copper hops (intra-rack). Two media:

- **Bandwidth must be strictly equal**: pipeline throughput is bottlenecked by the thinnest pipe segment. So TPU designs match the optical module modulation rate to the copper SerDes rate.
- **Latency is necessarily unequal**: copper is in the nanoseconds (single to low double digits); optical must do E-O conversion → tens of meters of fiber → OCS reflection → O-E conversion, into the hundreds of nanoseconds.

Bandwidth homogeneous + latency heterogeneous → the ring isn't perfectly symmetric in physics. This phenomenon is **NUCA (Non-Uniform Communication Architecture)**.

The business layer dissolves this with two tools:

1. **Steady-state pipeline masking**: At ring start, those 2 optical hops cause a small pipeline bubble, but in steady-state throughput is bandwidth-limited, and the few-hundred-nanosecond startup penalty disappears in the high-throughput data flow.
2. **XLA topology-aware mapping**: see Section 9.1.

#### 4.7 ↔ GPU

| Dimension | TPU 3D Torus | GPU NCCL on NVLink+IB |
|---|---|---|
| Algorithm | Multi-dim Ring All-Reduce | Ring or Tree (selectable) |
| Network diameter | $O(N^{1/3})$ | $O(\log N)$ |
| Synchronization | Hardware semaphore + static schedule | Software spin-wait + flag in HBM |
| Data path | UB → ICI → UB → VPU | HBM → PCIe → NIC → IB → NIC → PCIe → HBM → SM |
| Compute resource cost | VPU does the addition for free, MXU never stalls | NCCL kernel competes for SM resources |

**Trade-off**: Rings + static scheduling are unbeatable on regular collective comms but sacrifice arbitrary point-to-point flexibility.

---

### 5. Host ↔ TPU: PCIe, NUMA, and Multi-Host Slice

**One sentence**: TPU isn't a standalone machine — it's a PCIe device hanging next to a CPU host; once a slice spans hosts, you're running a distributed system.

#### 5.1 Physical Form: CPU as Foreman, TPU as Worker

Each rack runs standard x86 server boards (Intel/AMD) with ordinary DDR memory. TPUs attach via PCIe next to the CPU. **The typical ratio is 1 CPU host managing 4 or 8 TPUs**, hard-wired physically.

Division of labor:

- **CPU does**: Linux, Kubelet, accepting HTTP/gRPC, Python/PyTorch, vLLM scheduler (Radix Tree, PagedAttention page tables), XLA compilation (CPU-bound)
- **TPU does**: Pure execution of CPU-compiled machine code, doing matrix multiply-accumulate

A single Decode step from end to end:

1. CPU prepares metadata in system memory (page table pointer arrays, etc.).
2. PCIe DMA copies data + new token embedding into TPU HBM.
3. CPU sends the TPU an "execute compiled graph #5" instruction.
4. TPU goes into seclusion and computes.
5. PCIe pulls logits back to CPU memory.
6. CPU samples (Argmax, Top-P, etc.) on the result.

#### 5.2 Two-Plane Isolation: DCN vs ICI

Inside a Pod live two **completely independent** physical networks:

| Network | Used by | Medium | Purpose |
|---|---|---|---|
| **DCN (Datacenter Network)** | Host CPUs | Standard ethernet switches | CPU-to-CPU coordination, the control plane |
| **ICI (Inter-Chip Interconnect)** | TPUs | OCS + 3D Torus dedicated fibers | TPU-to-TPU data flood, the data plane |

Control plane (K8s coordination, Pod start/stop) goes over DCN with gRPC between CPUs. Data plane (All-Reduce, KV sync) bypasses CPU entirely.

#### 5.3 Multi-Host Slice: N:N Mapping

Many people assume a 64-chip v4-64 slice request lands on one giant VM with a super-CPU. **It doesn't.**

Physically it's **16 Host VMs** (4 TPUs each), 16×4=64:

- 16 VMs are connected by DCN.
- 64 TPUs are connected via OCS into a 3D Torus.
- Each VM runs one Kubelet.
- K8s schedules 16 Pods, one per VM.
- The inference code (vLLM/JetStream) starts in 16 CPUs simultaneously running the same Python code (**SPMD**).
- Typically expressed via **LeaderWorkerSet (LWS)** or **JobSet**: 1 Leader Pod exposes the API, 15 Worker Pods coordinate.
- The Leader broadcasts each request to Workers over DCN; each CPU drives the 4 TPUs underneath it.
- After the 64 TPUs compute the result, the Leader aggregates and returns it.

There is no "super CPU managing 64 TPUs". A large slice is a federation of many "small CPU + small TPU" nodes, gang-scheduled together by K8s.

#### 5.4 NUMA: PCIe Lanes Split Between Two Sockets

Modern server motherboards typically have two CPU sockets (CPU 0 and CPU 1). Half the PCIe lanes go to CPU 0, half to CPU 1. On an 8-TPU host, TPU 0~3 hang off CPU 0; TPU 4~7 off CPU 1.

The cross-NUMA disaster: CPU 0 wants to write into TPU 4's HBM — data must first cross the UPI bus to CPU 1, then PCIe to TPU 4. Latency spikes; usable bandwidth halves.

Where it bites in practice:

- **Input pipeline**: Tokenize on CPU 0, but task issued to TPU 4 — every in-feed crosses NUMA.
- **KV Cache offload**: TPU 0's KV swapped to a DDR slot owned by CPU 1.
- **Weight loading**: hundreds of GB of weights DMA-ing across NUMA — slow cold start.

Google Cloud's mitigations:

- **Single-NUMA VM partitioning**: When you request a 4-TPU instance, the hypervisor splits the physical machine in half. The VM you get **only contains** CPU 0 + CPU 0's memory + the 4 TPUs hanging off CPU 0. CPU 1's half is invisible.
- **XLA auto-pinning**: For 8-TPU instances that can't be split, XLA Runtime (PJRT) reads the PCIe-tree topology and automatically pins threads feeding TPU 0~3 to CPU 0 cores, threads feeding TPU 4~7 to CPU 1 cores.

#### 5.5 ↔ GPU

| Dimension | TPU | GPU |
|---|---|---|
| Physical attach | PCIe next to host CPU | Same |
| Ratio | 1:4 or 1:8, hard-wired | HGX usually 1:8 |
| Multi-host orchestration | LWS / JobSet + K8s gang | MPI Operator / Training Operator |
| NUMA handling | Single-NUMA VM split + XLA PJRT auto-pin | Often manual `numactl` + NCCL topology-aware |
| Two planes | DCN + ICI strictly separated | Usually both control and data planes share IB |

**Trade-off**: TPU's N:N orchestration shrinks the per-host failure blast radius, but operationally the "single inference service" you see is actually a coordinated dance of multiple K8s resources.

---

### 6. Advanced Packaging: Compute as Area, Bandwidth as Perimeter

**One sentence**: Die area governs FLOPs; perimeter governs bandwidth (HBM interfaces); 2.5D / 3D packaging is reconciling that fundamental tension.

#### 6.1 The Physical Origin of the Memory Wall

- **Compute ∝ area**: An MXU is 2D — a slightly larger MXU has $O(N^2)$ MAC cells (64×64 → 128×128 doubles area, quadruples compute).
- **Traditional bandwidth ∝ perimeter**: HBM bandwidth depends on the count of edge pins; growth is $O(N)$ linear.

Area is square-law, perimeter is linear — **bandwidth permanently lags compute**. That's the physical origin of the "memory wall".

#### 6.2 Three Generations of Packaging Evolution

| Generation | Name | What it solved |
|---|---|---|
| Gen 1 | Wire Bonding | Chip face-up, edge gold wires to the substrate; capped by perimeter |
| Gen 2 | Flip-Chip | Flip the chip, plant C4 micro-bumps across the whole face — moves from "edge" to "area"; but PCB trace precision is too coarse (line widths in tens of micrometers) |
| Gen 3 | **2.5D Silicon Interposer / CoWoS** | Lay a silicon slice between the chip and the substrate; use EUV lithography to draw **nanometer-scale** traces on it |
| Gen 4 | **3D Packaging + TSV (Through-Silicon Via)** | Drill tens of thousands of micrometer-scale holes vertically through the silicon, fill with copper; stack chip layers like floors of a building |

#### 6.3 Why the Silicon Interposer Helps

A regular PCB packs about 10 traces per millimeter; a silicon interposer packs over 1000 per millimeter. The connection between GPU/TPU and HBM expands from "bidirectional 4-lane" to "bidirectional 4096-lane". That's where HBM3's bus width comes from.

HBM itself is internally 3D too — multiple layers of memory dies stacked with TSV vertical interconnect, so a single HBM stack can deliver Tbps-class bandwidth.

#### 6.4 Compute-Surplus Problem and Hardware Compromise

In the v4 era, compute scaled too quickly and HBM bandwidth couldn't keep up; Decode MFU dropped to single digits, MXUs starved waiting for data. Google's hardware mitigation:

- **The v5e (inference chip) intentionally shrinks the MXU**, restoring a healthier compute/bandwidth ratio.
- Sacrificing peak FLOPs for cost-effectiveness — accepting that Decode is memory-bound, not compute-bound.

> **[Note — added by Claude]** The "single-digit MFU in v4 era" claim is direct from the source but lacks a specific data point or context. I have no other reliable source. Decide whether to keep it.

#### 6.5 ↔ GPU

NVIDIA H100/B100 also use CoWoS (same TSMC line) — same tech path. Difference is in die budget: H100 has 50 MB L2 Cache, 5 TB/s HBM; TPU skips hardware cache and gives that area to the MXU.

**Trade-off**: H100 uses big caches to tolerate random access; TPU uses big MXUs to crunch dense ops. Two paths matched to different workload assumptions.

---

## Part II — Compiler and Runtime: XLA

### 7. The XLA Compilation Model: Erase Uncertainty at Compile Time

**One sentence**: XLA's core play is to "erase all uncertainty at compile time" — operator fusion, static padding, software pipelining, and VLIW packs are all facets of that idea.

#### 7.1 Operator Fusion

Classic example: a layer of `MatMul + Bias + ReLU`.

Naive execution (GPU eager mode) does:
1. Compute MatMul → write back to HBM.
2. Read it back, add Bias → write back to HBM.
3. Read it back, do ReLU → write back to HBM.

XLA fuses the three steps into one compute block:

```
HBM → UB (MatMul input) → MXU does MatMul → result flows out → VPU
       does Bias + ReLU → HBM
```

**Saves 2/3 of HBM read/write traffic.** This kind of fusion is everywhere in LLMs — for example dropout/scale/mask after Attention is typically fused into the same block.

#### 7.2 Static Padding

The MXU is a hard-wired 128×128 array. If your matmul is 100×100, XLA does not let hardware handle the boundary — the hardware doesn't support that. XLA pads to 128×128 at compile time, filling the rest with zeros.

Cost: roughly 50% wasted compute on the boundary 28 rows × 28 columns.
Benefit: the array stays at full speed without stopping for boundary checks.

In a systolic array, **letting hardware run full-speed through blank cells beats stopping mid-stream**.

#### 7.3 Software Pipelining

XLA pre-computes how many cycles each DMA takes and statically generates instructions: while the MXU is computing block N, DMA is already staging block N+1. Compute and memory transfer overlap perfectly.

#### 7.4 VLIW 5-Slot: Same Shape, Single-Chip and Cross-Chip

Single-chip variant:

```
[ DMA ] | [ MXU ] | [ VPU ] | [ SPU control flow ]
```

Multi-chip cooperation adds one slot — the **ICI network slot**:

```
[ DMA ] | [ MXU ] | [ VPU ] | [ ICI network ] | [ SPU control flow ]
```

Key insight: **From the TPU's perspective, sending data across chips (ICI) and moving it within a chip (DMA) sit at the same level** — both are switches in a VLIW instruction word. XLA can have the MXU multiplying while ICI ships the previous layer's gradient to a neighbor, achieving **compute-and-cross-node-communication overlap at the cycle level**.

This is something GPU systems can't do. GPU cross-node comm fires CPU interrupts, builds packets, traverses IB switches — it's a separate subsystem.

#### 7.5 A Concrete Pseudo-Assembly Example

Compute $C = A \times B$, $A$ is 256×128, $B$ is 128×256. MXU is 128×128.

XLA at compile time splits $A$ into upper/lower halves ($A_0, A_1$), $B$ into left/right halves ($B_0, B_1$), decomposing into 4 sub-tasks of 128×128. All HBM addresses are hardcoded at compile (no runtime pointer math).

VLIW stream to compute $C_{00} = A_0 \times B_0$:

```
Instruction 1 (warm up: load weights):
  [DMA] LOAD_HBM_TO_UB (Src: HBM_B0, Dst: UB_B)
  [MXU] NOP
  [VPU] NOP
  [SPU] WAIT_DMA_DONE

Instruction 2 (weight stationary):
  [DMA] NOP
  [MXU] LOAD_UB_TO_WEIGHT_REG (Src: UB_B)
  [VPU] NOP
  [SPU] WAIT_MXU_DONE

Instruction 3 (core compute + pipeline prefetch):
  [DMA] LOAD_HBM_TO_UB (Src: HBM_A0, Dst: UB_A)
  [DMA_ASYNC] LOAD_HBM_TO_UB (Src: HBM_B1, Dst: UB_B_Next)  ← prefetch next block
  [MXU] MATMUL_STREAM_ACT (Src: UB_A, Dest: Accumulator_C00)
  [VPU] NOP
  [SPU] WAIT_MXU_DONE

Instruction 4 (fused ReLU):
  [DMA] NOP
  [MXU] NOP
  [VPU] READ_ACCUM_AND_RELU_AND_STORE (Src: Accumulator_C00, Dst: UB_C00)
  [SPU] WAIT_VPU_DONE

Instruction 5 (write back to HBM):
  [DMA] STORE_UB_TO_HBM (Src: UB_C00, Dst: HBM_C00)
  [MXU] NOP
  [VPU] NOP
  [SPU] JUMP_TO_NEXT_BLOCK  ← prep for C01
```

Note Instruction 3: **DMA stages the next block while the MXU is computing**. If XLA mis-times any cycle, either UB overflows or the MXU stalls. Only static compilation gives that precision.

#### 7.6 ↔ GPU

| Dimension | TPU XLA | GPU |
|---|---|---|
| Scheduling | Static at compile time | Dynamic at runtime (Warp Scheduler) |
| Memory-latency hiding | Static pipelining | Massive thread-level concurrency |
| Instruction format | VLIW with parallel slots | SIMT |
| Operator fusion | Automatic in XLA | Semi-automatic via TorchInductor / TVM / Triton |

**Trade-off**: XLA's "global view" is unbeatable on regular workloads, but any dynamic shape forces a recompile.

---

### 8. Compilation Timing: JIT, AOT, Bucketing, Persistent Cache

**One sentence**: The cost of static compilation is a slow first run; in production, bucketing + AOT + cache amortize the cost away.

#### 8.1 The Real JIT Timeline on PyTorch/XLA

Real flow when vLLM and similar frameworks start on TPU:

| Phase | Action | XLA state |
|---|---|---|
| 1. Init | Load weights to HBM | **Not compiled**. Just nn.Module objects and weight tensors |
| 2. Tracing | First request triggers `model.forward()`; PyTorch/XLA uses **Lazy Tensor** — no real compute, just records a DAG | Building the graph, generating HLO IR |
| 3. Trigger | Reading logits for sampling hits a sync barrier (e.g. `xm.mark_step()`) | **Compile now**: operator fusion + static address allocation + instruction layout → TPU Executable. Takes seconds to tens of seconds |
| 4. Execute + cache | TPU runs (in milliseconds), the compiled artifact is stored in an in-memory Compiler Cache | Cache key includes graph structure and all input shapes |
| 5. Subsequent requests | Same shape hits the cache, skips compile | Reuse |

**Key clarification**: Weight values do NOT enter the compiled artifact. XLA only cares about the weight tensor's shape and dtype, treating weights as a "static device-memory pointer". Swapping in a fine-tuned LoRA, or another model of the same architecture, doesn't trigger recompile.

#### 8.2 Production Approach: Bucketing + AOT + Persistent Cache

In production you absolutely cannot let the first user wait 30 seconds for compilation. The flow:

**Step 1: Restrict and discretize buckets**

After profiling, the team picks a discrete set of buckets:

```
BS_Buckets    = [1, 2, 4, 8, 16, 32, 64]
SeqLen_Buckets = [128, 512, 1024, 2048, 4096, 8192]
```

A request with BS=5 at runtime is padded to the BS=8 bucket.

**Step 2: AOT warmup at CI/CD**

Add a warmup stage when building the Docker image or release artifact: spin up a CI node with real TPU topology, iterate over every `BS × SeqLen` combination, and bombard the model with dummy requests to trigger compilation.

**Step 3: Bake the persistent cache into the image**

Use `XLA_FLAGS="--xla_dump_to=/path/to/cache"` to dump compiled artifacts. At the end of the pipeline, **bake those few hundred MB to a few GB of cache files directly into the release image** or attach them via shared storage.

When production vLLM/JetStream instances boot, they read the cache; bucket hits become **millisecond hardware dispatches**.

#### 8.3 Why There's No "One-Size-Fits-All" Pre-Compiled Library

An XLA Executable is bound not just to model shapes but to these other **fatal variables** — change any one and the cache is invalid:

| Variable | Impact |
|---|---|
| **Physical hardware topology** | A v5e-8 (1D ring) executable can't run on v5p-32 (3D Torus); XLA bakes in which fiber to use and the latency in nanoseconds |
| **Parallelism strategy** | Does TP cut Attention or FFN? How does PP cut? These SPMD annotations must be set before compilation |
| **Compiler version** | XLA / LLVM backend updates frequently; old caches usually fail validation |
| **Model structure tweaks** | Add a small adapter, tweak the RoPE base — constant-folding output changes → HLO hash changes → cache invalidated |

Every team has to maintain its own **model distribution + cache pre-warming pipeline**. A major infrastructure release is always paired with large-scale automated recompilation.

#### 8.4 ↔ GPU

GPU has the same problem space (PyTorch 2.x's `torch.compile` / Inductor / TensorRT), but to a far smaller degree: GPU hardware tolerates runtime shape changes (dynamic dispatch), and compile failures fall back to eager. TPU has no such fallback — compile failure equals service failure.

---

### 9. XLA Topology-Aware Mapping

**One sentence**: The compiler knows the physical topology of the 3D Torus + OCS, so it can map high-density communication onto short edges and low-frequency synchronization onto long rings.

Section 4.6 covered NUCA: the cross-rack 8-hop ring has 6 copper hops + 2 optical hops, bandwidth homogeneous, latency heterogeneous. XLA, at compile time, places different parallelism strategies onto different qualities of topology.

#### 9.1 TP on Small Rings, DP on Big Rings

| Parallelism | Comm characteristics | Mapped to |
|---|---|---|
| **Tensor Parallelism (TP)** | Step-by-step; every linear layer requires syncing activations. **Latency-sensitive** | The 4 / 8 short copper rings on X or Y |
| **Data Parallelism (DP)** | "After the dust settles"; each step (or after a few accumulated) syncs gradients. The matrix is large, demanding bandwidth, but tolerates one-shot latency | The 35-node large optical ring on Z |
| **Pipeline Parallelism (PP)** | Inter-stage activation transfer, medium frequency | Usually mid-length edges |
| **Expert Parallelism (EP)** | Dynamic All-to-All (MoE) | Suffers on Torus, see Chapter 19 |

When DP runs on the big ring, the 35-hop physical latency is masked by steady-state pipelining, and underlying compute units use the sync wait window to do the next step's forward (Compute-Communication Overlap).

#### 9.2 How Topology Information Reaches XLA

K8s labels on each Node (e.g., `cloud.google.com/gke-tpu-topology: 4x4x4`) carry the slice's geometry. XLA Runtime at startup reads those plus PCIe sysfs information to construct a topology graph. Then, per the user's SPMD partitioning annotations, it maps comm groups to specific ICI links.

**Conclusion**: Pure K8s scheduling only sees node liveness; high-performance AI scheduling sees microsecond-grained optical-electrical physical edges.

#### 9.3 ↔ GPU

GPU systems handle similar concerns with manual NCCL group configs + `torch.distributed`'s topology-aware APIs. NCCL knows NVLink/IB hierarchies, but fat-tree itself is roughly symmetric, so the optimization headroom isn't as large as on a Torus.

---

## Part III — Inference-Layer Adaptation (Goal C)

### 10. The Software Stack Forks: vLLM, JetStream, Saxml, GKE

**One sentence**: TPU has more than one inference framework — three players with three positions; GKE is the glue that puts them all into a cluster.

#### 10.1 Why GKE Is Hell-Bent on vLLM on TPU

vLLM is the de-facto "Linux" of open-source inference — most customers writing on GPU + PyTorch + vLLM have already built their business code (API wrappers, schedulers, custom prompting). For GKE selling TPU (v5e/v5p, sharply cost-effective), the biggest blocker is migration cost:

> If customers must rewrite code to use TPU, they leave.

So Google's strategy is **Lift and Shift**: let customers keep their `vllm serve` invocation, swap the base image, and PyTorch calls are auto-routed to PyTorch/XLA.

Foundation: vLLM's official repo already includes a TPU backend; the impedance-mismatch between PagedAttention and TPU's static graphs is patched with custom kernels written by Google engineers in Pallas.

#### 10.2 Three Players' Positions

| Framework | Position | Target users |
|---|---|---|
| **vLLM** | Ecosystem-compatibility king ("don't make me change code") | Startups, multi-cloud customers, GPU migrants |
| **JetStream** | TPU performance squeezer | Big tech, high-concurrency inference, willing to adapt framework for performance |
| **Saxml** | JAX legacy heavy artillery | Customers deeply tied to JAX, special large-scale partitioning |

#### 10.3 Why JetStream Beats vLLM by 20%-50%

JetStream is led jointly by Google Cloud + the XLA team, custom-built for the v5 series. It **doesn't try to bolt dynamic paging onto static graphs**; instead it embraces TPU's static-orchestration philosophy:

- Deeply optimized continuous batching
- Lots of XLA operator fusion
- Co-designed with the compiler, no PyTorch indirection

The cost: APIs aren't as plug-and-play as vLLM, and PyTorch ecosystem support requires dedicated work.

#### 10.4 Why Saxml Has Slipped to the Background

Born alongside Pax / Seqio, deeply tied to JAX. Carries the flavor of Google internal infrastructure, with a high external-developer learning curve and slower PyTorch support. In public-cloud advocacy it's been deprioritized.

#### 10.5 ↔ GPU

| TPU | GPU |
|---|---|
| vLLM-TPU (with Pallas) | Native vLLM |
| JetStream | TensorRT-LLM (NVIDIA's own + specialized) |
| Saxml | DeepSpeed Inference (partial) |
| Pallas for kernels | Triton / CUDA |

---

### 11. Adapting PagedAttention and Continuous Batching to TPU

**One sentence**: GPU-style dynamic memory management (PagedAttention, Continuous Batching, Radix Tree) is fundamentally hostile to static compilation; TPU adapts via Pallas custom kernels + pushing dynamism into the tensor layer.

#### 11.1 Before Pallas: The Inefficiency of Static Contiguous Allocation

Early TPU inference (T5, early Pax) ran on a "OCD" route:

- XLA at compile time pre-allocated a contiguous KV Cache pool sized `[Max_BS, Max_SeqLen, Hidden_Dim]`.
- With max_seq=4096, every request locked 4096 tokens of HBM.
- Real request only 100 tokens? The remaining 3996 slots were wasted (97% memory wasted).
- HBM saturated with useless padding → batch size couldn't grow → MXU had compute headroom, but the pool was full → **compute pinned by the memory wall**.

Google early on threw money at this — large total Pod HBM, controllable task lengths (translation/search), algorithm teams cutting buckets very finely — and survived. But long context and multi-turn dialogue popularized this approach to its limits.

#### 11.2 Modern Split: XLA Builds the Pool, vLLM Keeps the Books, Pallas Reads the Map

vLLM on TPU's modern architecture splits **control plane (CPU) and data plane (TPU)**:

| Role | Where | What it does |
|---|---|---|
| **XLA** | TPU | Allocates one giant 1D-flattened block tensor in HBM, shaped `[Num_Total_Blocks, Block_Size, Head_Dim]` (e.g. 100K physical blocks, each holding 16 tokens). **XLA doesn't know whose data is in there** |
| **vLLM** | CPU | Maintains a Radix Tree and all per-request page tables (Block Tables). Each step it bundles the active requests' page tables into an integer Tensor and feeds it to the XLA graph |
| **Pallas Kernel** | TPU | A Custom Call node in the XLA graph. After receiving the page table, executes low-level indirect addressing (Gather), pulling fragments into the Unified Buffer for the Attention compute |

**Physical HBM is still XLA's global tensor, but XLA no longer manages content.** vLLM acts as dispatcher each cycle, sending an "addressing map" over; the Pallas kernel on TPU follows the map to fetch data.

#### 11.3 Each of the Three Knives in Detail

**PagedAttention**

- TPU pain: XLA hates dynamic pointer addressing; querying the page table on each Attention call wrecks the DMA script.
- Solution: a Pallas kernel hand-written at the register level for page-table lookup + scatter/gather Gather.
- Result: Fully supported; HBM fragmentation problem solved, batch size goes up.

**Continuous Batching (Inflight Batching)**

- GPU's play: 1D flatten; the scheduler can boot a finished request and admit a new one at any time, fully dynamic.
- TPU's play (**static-bus mode**):
  - XLA pre-compiles a fixed-`Batch_Size = 256` graph — think of it as a 256-seat bus permanently looping the route.
  - When some request hits EOS → vLLM marks the seat empty → next step it slots a new request's first Decode token into that index.
  - TPU sees only a perfect `[256, 1, D]` tensor and doesn't know that index 5 was user A a millisecond ago and is user B now.
  - When fewer than 256 real requests, fill empty slots with Dummy Tokens (zeros).

**Radix Tree (Prefix Cache)**

- **A perfect fit on TPU**: The mechanism is essentially a CPU-side scheduling algorithm.
- On prefix hit, vLLM only needs to point logical block pointers in the dispatched Block Table to existing physical blocks.
- The TPU-side Pallas kernel doesn't know it's reuse — it follows the map and fetches HBM data to UB normally.

#### 11.4 Google's Own Frameworks Use the Same Approach

JetStream / Saxml implement the equivalent (called **Blocked Attention** internally, or built into the underlying **FlashAttention-TPU** kernel), all written in Pallas. So whether you run vLLM or JetStream on GKE, **the memory-management thinking has converged**: maintain a discrete Block Pool in HBM + page table on CPU + pass the page table to the underlying kernel for Gather at compute time.

#### 11.5 A Core Philosophy: FLOPs to Replace Control Flow

**Use very cheap FLOPs (compute) to eliminate very expensive Control Flow.**

This runs through all of TPU's inference adaptation. Tree Attention (Chapter 14) is the same idea — encode if-else into a Mask matrix, prefer to compute and discard rather than let hardware stop for branch decisions.

#### 11.6 ↔ GPU

| Dimension | TPU | GPU |
|---|---|---|
| KV paging | XLA pool + Pallas Custom Call | vLLM native PagedAttention |
| Scheduling flexibility | Fixed batch buckets + Dummy padding | Dynamic 1D flatten |
| Custom kernel tooling | Pallas | Triton / CUDA |

---

### 12. Prefill/Decode Coordination and Chunked Prefill

**One sentence**: TPU is strong on Prefill, weak on Decode (HBM bandwidth bottleneck); mixed execution + chunked prefill is algorithm patching for hardware.

#### 12.1 Arithmetic Intensity Determines TPU Feel

To judge whether hardware suits a workload, look at **Arithmetic Intensity = FLOPs / Byte** (how many floating-point ops you do per byte of memory access).

| Phase | Math form | Arithmetic intensity | Bottleneck | TPU feel |
|---|---|---|---|---|
| **Prefill** | GEMM (matrix × matrix; weights reused across thousands of tokens) | Very high | Compute-Bound | MXU runs blissfully |
| **Decode** | GEMV (matrix × vector; weights pulled out for one token then discarded) | Very low | Memory-Bound | MXU starves frequently |

So TPU is intrinsically a **Prefill beast**; Decode performance is bolted on after the fact.

#### 12.2 Continuous Batching on TPU Decode

**Decode does not bucket on token length** — every request's new-token length is fixed at 1. It buckets on **Batch Size**.

Pre-compile a BS=256 Decode graph:

- Static input matrix `[256, 1, Hidden_Dim]`.
- 200 real requests → first 200 slots hold real tokens, last 56 slots hold Dummy Tokens (zeros).
- The MXU computes 256 results.
- The CPU scheduler picks up only the first 200 real results to send to users; the 56 dummies are discarded.

**The hard part: the 256 requests have wildly different histories.**

The CPU also sends two static-sized integer arrays:

```
context_lengths : shape [256], real history length per request, e.g. [105, 3042, 12, ...]
                  (Dummy slots filled with 0)
block_tables    : shape [256, Max_Blocks], each request's KV page table
```

The Pallas kernel uses `context_lengths` as loop bounds (or mask) and `block_tables` to gather historical KV from HBM, doing Attention.

#### 12.3 The Prefill Difficulty: Wildly Different Prompt Lengths

Decode neatly stacks into `[256, 1]`, but Prefill can't: one prompt is 100, another is 3000. How to fit a static graph?

Two approaches:

- **Bucketing**: 100 → padded to bucket 128; 3000 → cut to bucket 4096.
- **Chunked Prefill**: Compile a fixed-length Prefill graph (e.g. chunk_size=512). A 1000-length prompt is split into two chunks of 512, fed twice into the same `[1, 512]` slot.

#### 12.4 Mixed Prefill/Decode: Static 1D Flatten

The most up-to-date approach: combine Prefill long sequences and Decode single tokens **in the same step**.

XLA at compile time sets two static upper bounds:

```
Max_Total_Tokens = 1024   # max tokens swallowed in one step
Max_Seqs        = 256     # max concurrent sequences
```

The input tensor is flattened from 3D `[Batch, Seq, D]` to 2D `[1024, D]`.

CPU-side composition:

```
Request A (Prefill, chunk=512)    →  first 512 of array
Requests B–Z (Decode, 200 tokens) →  next 200
                                     712 used
Dummy Tokens × 312                 →  pad to 1024
```

**MXU phase**: To the systolic array, identities don't matter. One giant `[1024, D] × [D, 4D]` matmul flies through, computing Q/K/V for all 1024 tokens in one shot.

**Attention phase**: Now identity matters:

| Token type | What Attention they need |
|---|---|
| The 512 Prefill tokens | Attend **to each other**, generating new KV that gets written into the page table |
| The 200 Decode tokens | Each uses its 1 token's Q to attend to its own historical KV Cache |
| The 312 Dummy tokens | Skip |

The CPU also passes metadata: `seq_lens = [512, 1, 1, ..., 0, 0]`. The Pallas kernel parses metadata at the register level — Prefill blocks go down a FlashAttention-style path (Q vectors mutually dot in UB + write new KV); Decode blocks go down PagedAttention (gather historical KV); Dummy blocks skip.

#### 12.5 GPU vs TPU Mixed-Batching Differences

| | GPU | TPU |
|---|---|---|
| Composition | Dynamic: 712 → kernel takes 712; next round 850 → 850 | Static: 712 → forced add 312 dummies → 1024 |
| Hardware cost | Scheduler overhead | Padded MXU cycles |
| Software cost | High kernel flexibility | Pallas metadata routing |

**Trade-off**: Wasting a bit of MXU cycles on padding is controllable, but it dodges the recompilation disaster and keeps tail latency tight. The lesser of two evils.

#### 12.6 ↔ GPU

GPU is the "ride-share" model: 712 passengers → a 712-seater dispatches, no empty seats.
TPU is the "high-speed direct express": departs on schedule whether full or not — fill empty seats with dummies if needed.

---

### 13. KV / Memory Hierarchy

**One sentence**: GPU's RDMA / GDS / KV offload counterparts on TPU are: some are native, some are not supported, some only fall back to PCIe.

#### 13.1 Three Things Compared

| Optimization | GPU | TPU |
|---|---|---|
| **Cross-node comm bypassing CPU** (GPU: RDMA over IB) | GPUDirect RDMA + IB NIC | **Native**: ICI network controllers are integrated directly into TPU silicon — no external NIC needed. The CPU is not in the path; zero CPU cycles consumed |
| **Direct read of storage into accelerator** (GPU: GPUDirect Storage) | NVMe → PCIe → GPU VRAM, bypassing CPU memory | **Not supported**. TPU has no direct storage / external network interface; CPU must mediate: GCS / PD → Host VM DDR → PCIe DMA → TPU HBM |
| **KV Cache offload to host DRAM** (GPU: vLLM CPU swap) | HBM → PCIe → CPU DDR | **Fully applicable**. vLLM's Block Manager runs on host CPU; when HBM fills, PCIe DMA copies KV blocks to host DDR |

#### 13.2 ICI Is More Thorough Than RDMA

GPU RDMA: data → PCIe → NIC → IB network → NIC → PCIe → VRAM. CPU memory bypassed, but data still leaves the GPU chip via an external NIC.

TPU ICI: data → chip-edge optical module → fiber → peer optical module → chip. **Never leaves the silicon-fabric world.** With hundreds of GB/s of network traffic flowing, the host CPU has zero awareness.

#### 13.3 Why Missing Direct Storage Is Tolerable

Cold-start weight loading needs hundreds of GB pulled from GCS to TPU HBM, requiring CPU as porter. But because deployment is multi-host (e.g. 16 VMs), 16 CPUs **download different weight shards in parallel** from GCS, total network bandwidth is large, and the workflow is acceptable in practice.

#### 13.4 KV Offload Performance Profile

PCIe bandwidth is **narrow** relative to ICI (v4 PCIe Gen4 x16 ≈ 64 GB/s bidirectional, while a single ICI link can do hundreds of GB/s). So frequent swap noticeably degrades performance. This is OOM insurance, not a primary mechanism.

#### 13.5 ↔ GPU

GPU advantage: GDS (direct storage). TPU advantage: ICI (more thorough comm bypass). They split the spoils.

> **[Note — added by Claude]** Industry has begun exploring **Mooncake-style "disaggregated KV pools"** (KV Cache as a separate service shared across nodes) — currently mostly on GPU systems. No public TPU-side equivalent that I've found. Not in main text — for cluster-TL perspective only.

---

### 14. Gemini's Practical Compromises on TPU

**One sentence**: Both MoE and speculative decoding have to be re-written algorithmically to suit TPU hardware.

#### 14.1 Static MoE: Capacity Factor

MoE's natural problem is **dynamic routing** — you don't know which expert the next token will pick. XLA forbids dynamic shapes.

The Gemini team's solution:

- Each expert is given a strict **Capacity Factor** (a static slot size). Say each expert can take up to 64 tokens per step.
- Fewer than 64 routed tokens → pad with Dummy Padding to fill the slot.
- More than 64 routed tokens → the surplus is **dropped directly** (Token Dropping), or routed through a fallback general network.

Through this brutal truncation + padding, the dynamic MoE network is forced into a static graph that XLA likes.

Cost: occasional token drops, with possible model-quality impact. Google tunes the Capacity Factor during Gemini training to balance drop rate against compute cost.

#### 14.2 Tensorized Speculative Decoding: Tree Attention

Traditional speculative decoding: a small model proposes K tokens → if-else checks acceptance by the large model → on rejection, roll back. That if-else flow trashes TPU's VLIW pipeline.

Gemini's solution (**parallel verification**):

- After the small model produces K candidate tokens, the large model packs them into a 1D vector.
- A specially designed **Tree Attention Mask** ensures unrelated nodes don't see each other (multiplied by 0).
- The large model **verifies all K tokens' probabilities in one matmul during a single forward pass**.
- The accepted path is selected; the others are discarded.

This converts "branch code" into "small Prefill matmul". The MXU rejoices.

#### 14.3 Why These Compromises Aren't Required on GPU

- GPU's SIMT scheduler handles if-else (branch divergence wastes lanes, but is much better than TPU).
- GPU's dynamic memory tolerates expert-capacity variability.
- So GPU runs vanilla MoE routing + vanilla speculative decoding fine.

GPU roadmaps are also moving toward Tree Attention and similar tensorization, but it's about extra performance — not survival.

#### 14.4 An Interesting Trend

Because hardware hates branches (TPU outright; GPU dislikes host-device sync at high frequency), algorithm engineers are **rewriting control flow as data flow** wherever possible. Tree Attention, Masked Attention, predicated execution — all variations of the same idea.

Core philosophy: **compute everything and discard is cheaper than if-else**. For shallow, bounded branches (speculative decoding 3-5 steps, Causal Mask half a matrix), this trick is wildly worth it. For deep nesting (10+ levels of conditional trees), it's $O(2^N)$ explosion. So next-gen chips are pursuing **native sparsity**: when a mask region is all zeros, the circuit physically skips MAC ops (clock-gated power off) — neither writes if-else nor consumes power.

#### 14.5 ↔ GPU

| Optimization | TPU must adapt | GPU need not |
|---|---|---|
| MoE | Static via Capacity Factor | Dynamic routing works |
| Speculative decoding | Tensorized via Tree Attention | if-else works (with perf hit) |

---

