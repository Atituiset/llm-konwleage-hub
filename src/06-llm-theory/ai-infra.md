# AI Infra：大模型基础设施（AI Infrastructure for LLMs）

> **适合人群**：希望理解大模型训练与部署底层基础设施的工程师
> **难度**：⭐⭐⭐⭐（较难）

---

## 1. GPU 集群架构（GPU Cluster Architecture）

### 1.1 单节点架构（Single-Node Architecture）

```
┌─────────────────────────────────────────────────────────────┐
│                        单节点（8×GPU）                        │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│  │GPU 0│←→│GPU 1│←→│GPU 2│←→│GPU 3│←→│GPU 4│←→│GPU 5│←→│GPU 6│←→│GPU 7│  │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘  │
│     └───────┴───────┴───────┘       └───────┴───────┴───────┘  │
│                    NVLink / NVSwitch                          │
│                              ↑                                │
│                         PCIe Switch                           │
│                              ↑                                │
│                           CPU + DRAM                          │
└─────────────────────────────────────────────────────────────┘
```

**关键组件**：
- **GPU**：NVIDIA H100（80GB HBM3）、A100（80GB HBM2e）、H200（141GB）
- **NVLink**：GPU 间高速互联，H100 上 NVLink 4.0 提供 900 GB/s 带宽
- **NVSwitch**：实现 8 路或 16 路 GPU 全互联，任意两 GPU 可直接通信
- **PCIe**：GPU 与 CPU/存储的连接，PCIe 5.0 x16 提供 64 GB/s

**典型配置（DGX H100）**：
- 8× H100 GPU，通过 4× NVSwitch 实现全互联
- 2× Intel Xeon CPU
- 2 TB System RAM
- 8× 3.84 TB NVMe SSD

### 1.2 多节点集群架构（Multi-Node Cluster Architecture）

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  节点 1     │      │  节点 2     │      │  节点 N     │
│  8× GPU     │←────→│  8× GPU     │←────→│  8× GPU     │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                    ┌───────┴───────┐
                    │  InfiniBand   │
                    │   交换机       │
                    │  (400 Gbps)   │
                    └───────────────┘
```

**网络拓扑选择**：

| 网络类型 | 带宽 | 延迟 | 适用场景 |
|---------|------|------|---------|
| **InfiniBand (IB)** | 400-800 Gbps | ~1μs | 大规模训练集群（首选） |
| **RoCEv2 (RDMA over Ethernet)** | 100-400 Gbps | ~2μs | 成本敏感的中等规模集群 |
| **NVLink + NVSwitch** | 900 GB/s | ~0.5μs | 节点内 GPU 互联 |
| **PCIe** | 64 GB/s | ~1μs | GPU-CPU 通信 |

**为什么 InfiniBand 是训练集群的首选？**
- RDMA（Remote Direct Memory Access，远程直接内存访问）：CPU 不参与数据传输，GPU 直接读写远端 GPU 内存
- 低延迟（Low Latency）：~1μs，比 TCP/IP 低两个数量级
- 高带宽（High Bandwidth）：NDR 400 Gbps，XDR 800 Gbps
- 网络内计算（In-Network Computing）：支持 SHARP（Scalable Hierarchical Aggregation and Reduction Protocol），在交换机层面完成 All-Reduce（全归约）

### 1.3 集群拓扑设计（Cluster Topology Design）

**胖树（Fat-Tree）拓扑**：

```
        ┌─────────┐
        │ 核心交换机 │ ←── 全带宽互联
        └───┬─┬─┬─┘
            │ │ │
    ┌───────┼─┼─┼───────┐
    │       │ │ │       │
┌───┴───┐ ┌─┴─┴─┐ ┌───┴───┐
│汇聚交换机│ │汇聚交换机│ │汇聚交换机│
└───┬─┬─┘ └──┬┬┬┘ └───┬─┬─┘
    │ │      │││      │ │
   GPU节点   GPU节点   GPU节点
```

- 任意两节点间的通信带宽相等
- 无拥塞（非阻塞）
- 扩展性好，但交换机数量随节点数平方增长

**Dragonfly+ 拓扑**：
- 减少交换机数量，降低成本
- 通过高带宽链路连接多个组
- 适合超大规模集群（数千节点）

---

## 2. 存储系统（Storage Systems）

### 2.1 训练数据存储架构（Training Data Storage）

```
┌──────────────────────────────────────────────────────────┐
│                      存储层次结构                          │
│                                                          │
│  热数据（活跃训练）                                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  节点本地 NVMe SSD（数TB）                            │  │
│  │  └─→ 预热到 GPU HBM（数百GB）                         │  │
│  └────────────────────────────────────────────────────┘  │
│                    ↑ 异步预加载                            │
│  温数据（近期使用）                                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  并行文件系统（Lustre/GPFS/BeeGFS）                   │  │
│  │  └─→ 数百TB ~ PB 级别                                 │  │
│  └────────────────────────────────────────────────────┘  │
│                    ↑ 批量导入                              │
│  冷数据（归档）                                            │
│  ┌────────────────────────────────────────────────────┐  │
│  │  对象存储（S3/OSS/Ceph）                              │  │
│  │  └─→ PB ~ EB 级别                                     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 2.2 并行文件系统（Parallel File Systems）

| 文件系统 | 特点 | 适用场景 |
|---------|------|---------|
| **Lustre** | 最常用，高并发 | 大型超算中心 |
| **GPFS/Spectrum Scale** | IBM 出品，稳定 | 企业级集群 |
| **BeeGFS** | 开源，易部署 | 中小型集群 |
| **WEKA** | 高性能，GPU 直连 | AI 训练专用 |
| **JuiceFS** | 基于对象存储，弹性 | 云原生场景 |

**数据加载瓶颈**：

预训练时数据加载是常见瓶颈。解决方案：

1. **多级缓存**：对象存储 → 并行文件系统 → 本地 NVMe → CPU DRAM → GPU HBM
2. **异步数据加载**：PyTorch DataLoader 的 `num_workers > 0`，预取（`prefetch_factor`）
3. **数据分片与局部性**：每个节点只读取本地存储的数据分片，减少网络传输
4. **数据格式优化**：用 WebDataset、TFRecord、Arrow 等格式替代原始文本

```python
from torch.utils.data import DataLoader

# 优化后的 DataLoader 配置
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,        # 根据 CPU 核心数调整
    prefetch_factor=4,    # 每个 worker 预取 4 个 batch
    pin_memory=True,      # 将数据固定到 CPU 锁页内存，加速 GPU 传输
    persistent_workers=True  # 保持 worker 进程，避免每 epoch 重建
)
```

---

## 3. 训练框架对比（Training Framework Comparison）

### 3.1 主流分布式训练框架（Distributed Training Frameworks）

| 框架 | 出品方 | 核心特性 | 适用场景 |
|------|--------|---------|---------|
| **Megatron-LM** | NVIDIA | 张量并行、流水线并行、序列并行 | GPT 类大模型预训练 |
| **DeepSpeed** | 微软 | ZeRO 优化、3D 并行、Offload | 显存受限场景 |
| **FSDP** | PyTorch 官方 | 全分片数据并行 | PyTorch 原生推荐 |
| **Colossal-AI** | 潞晨科技 | 统一并行接口、易用 | 国产化、教育 |
| **JAX/Flax** | Google | XLA 编译、TPU 友好 | Google TPU 生态 |
| **MindSpore** | 华为 | 自动并行、昇腾支持 | 国产替代 |

### 3.2 框架选择决策树（Framework Selection）

```
使用 PyTorch？
  ├── 是 → 模型能否放入单卡？
  │          ├── 是 → DDP / FSDP
  │          └── 否 → 需要 3D 并行？
  │                    ├── 是 → Megatron-LM / DeepSpeed
  │                    └── 否 → DeepSpeed ZeRO / FSDP
  └── 否 → 使用 TPU？
            ├── 是 → JAX/Flax
            └── 否 → MindSpore（昇腾）/ Paddle
```

### 3.3 Megatron-LM 详解

Megatron-LM 是 NVIDIA 开发的针对 Transformer 大模型优化的训练框架。

**张量并行（Tensor Parallelism, TP）**：

将 Transformer 层内的矩阵运算切分到多个 GPU：

```
输入 X ──→ ┌─────────┬─────────┐
            │  GPU 0  │  GPU 1  │  ← Attention QKV 投影按列切分
            │  QKV_0  │  QKV_1  │
            └────┬────┴────┬────┘
                 │         │
                 └── All-Gather → 完整 Attention 输出
```

**流水线并行（Pipeline Parallelism, PP）**：

将模型按层切分：

```
输入 ──→ [Layer 1-4] ──→ [Layer 5-8] ──→ [Layer 9-12]
         GPU 0/1/2/3   GPU 4/5/6/7   GPU 8/9/10/11
```

**序列并行（Sequence Parallelism, SP）**：

将序列维度切分，与 TP 结合使用：
- LayerNorm、Dropout 在序列维度切分
- Attention 在 head 维度切分（TP）
- 减少 activation 内存占用

**3D 并行组合**：

```python
# Megatron 的并行配置
parallel_config = {
    "tensor_model_parallel_size": 4,    # TP: 每层切 4 份
    "pipeline_model_parallel_size": 8,   # PP: 8 个 stage
    "data_parallel_size": 8,             # DP: 8 路数据并行
}
# 总 GPU 数 = 4 × 8 × 8 = 256
```

### 3.4 DeepSpeed 详解

**ZeRO（Zero Redundancy Optimizer）**：

解决数据并行（Data Parallelism）中每卡都存储完整模型状态（Model States）的冗余问题。

```
标准数据并行（4卡）:
GPU0: [模型参数] [梯度] [优化器状态]
GPU1: [模型参数] [梯度] [优化器状态]
GPU2: [模型参数] [梯度] [优化器状态]
GPU3: [模型参数] [梯度] [优化器状态]
      ↑ 4×冗余

ZeRO-3（4卡）:
GPU0: [参数0] [梯度0] [优化器状态0]
GPU1: [参数1] [梯度1] [优化器状态1]
GPU2: [参数2] [梯度2] [优化器状态2]
GPU3: [参数3] [梯度3] [优化器状态3]
      ↑ 通信时 All-Gather/Reduce-Scatter
```

**DeepSpeed 配置示例**：

```json
{
  "train_batch_size": 512,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "Adam",
    "params": { "lr": 1e-4 }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "fp16": { "enabled": true },
  "gradient_clipping": 1.0
}
```

---

## 4. 通信原语（Communication Primitives）

### 4.1 集合通信操作（Collective Communication）

分布式训练的核心是 GPU 之间的集合通信（Collective Communication）。

| 操作 | 功能 | 图示 | 典型应用 |
|------|------|------|---------|
| **All-Reduce** | 所有 GPU 的数据求和/平均，结果分发到所有 GPU | `[a,b,c,d] → [avg,avg,avg,avg]` | 梯度同步（DDP） |
| **All-Gather** | 收集所有 GPU 的分片数据，拼接后分发 | `[a] [b] [c] [d] → [a,b,c,d]` | 张量并行输出聚合 |
| **Reduce-Scatter** | 先求和，再按分片分发 | `[a,b,c,d] → [sum0] [sum1] [sum2] [sum3]` | ZeRO-3 梯度同步 |
| **Broadcast** | 从根节点发送数据到所有节点 | root:[a] → [a] [a] [a] [a] | 参数初始化 |
| **All-to-All** | 每个 GPU 发送不同数据给其他 GPU | 复杂路由 | MoE 专家路由 |

**NCCL（NVIDIA Collective Communications Library）**：
- NVIDIA 提供的高性能 GPU 通信库
- 自动选择最优通信路径（NVLink > InfiniBand > PCIe）
- 支持树状和环状 All-Reduce 算法

### 4.2 通信-计算重叠（Communication-Computation Overlap）

```
时间轴 →

无重叠:
├─────────┬─────────┤
│ Forward │ All-Reduce│
└─────────┴─────────┘

有重叠（bucket gradient）:
├─────────┤
│ Forward │
└────┬────┘
     ├─→ 梯度分桶
     │   ├─→ Bucket 1 All-Reduce（与 backward 重叠）
     │   ├─→ Bucket 2 All-Reduce（与 backward 重叠）
     │   └─→ Bucket 3 All-Reduce
```

PyTorch DDP 的 `bucket_cap_mb` 参数控制梯度分桶大小，较小的桶可以更早开始通信，增加重叠机会。

---

## 5. 训练监控与实验管理（Monitoring & Experiment Management）

### 5.1 关键监控指标（Key Metrics）

| 指标 | 含义 | 健康范围 | 异常处理 |
|------|------|---------|---------|
| **TFLOPS/GPU** | 每 GPU 的计算吞吐量 | > 50% 峰值 | 检查数据加载、通信瓶颈 |
| **GPU 利用率** | GPU 计算单元忙碌程度 | > 90% | 检查 CPU 瓶颈、数据加载 |
| **显存占用** | HBM 使用量 | < 95% | 调小 batch size、启用 ZeRO |
| **通信带宽** | 实际 IB/NVLink 带宽 | > 80% 理论值 | 检查网络配置、拓扑 |
| **Loss 曲线** | 训练损失 | 平滑下降 | 发散→降低 lr；平坦→增大 lr |
| **Grad Norm** | 梯度范数 | 稳定，无突变 | 过大→梯度裁剪；NaN→检查数值稳定性 |

### 5.2 实验管理工具（Experiment Tracking）

**Weights & Biases (W&B)**：
```python
import wandb

wandb.init(project="llm-pretrain", config={"lr": 1e-4, "batch_size": 512})

for step in range(total_steps):
    loss = train_step(...)
    wandb.log({"loss": loss, "lr": scheduler.get_last_lr()[0]}, step=step)
```

**TensorBoard**：
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")
writer.add_scalar("Loss/train", loss, step)
writer.add_histogram("Weights/embedding", model.embed.weight, step)
```

**MLflow**：
- 开源，支持模型版本管理（MLflow Model Registry）
- 与云厂商集成好
- 适合企业级部署

### 5.3 检查点策略（Checkpointing Strategy）

**检查点内容**：
```
checkpoint/
├── model_states/          # 模型参数（FP16 + FP32 副本）
├── optimizer_states/      # 优化器状态（Adam 的一阶/二阶矩）
├── rng_states/            # 随机数生成器状态
├── training_args/         # 训练配置
└── latest_checkpoint      # 符号链接指向最新检查点
```

**检查点保存策略**：
- **定期保存**：每 N 步或每 epoch
- **最佳保存**：保存验证集 loss 最低的模型
- **异步保存**：使用后台进程写入存储，不阻塞训练
- **增量保存**：只保存变化的参数（如 LoRA adapter）

**故障恢复**：
```python
# 从检查点恢复
checkpoint = torch.load("checkpoint/step_100000.pt")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_step = checkpoint["step"]

# 继续训练
for step in range(start_step, total_steps):
    ...
```

---

## 6. AI Infra 成本优化（Cost Optimization）

### 6.1 训练成本构成（Training Cost Breakdown）

```
总成本 = GPU 算力成本（70%）+ 网络成本（10%）+ 存储成本（10%）+ 人力运维（10%）
```

**算力成本优化**：
- **混合实例**：训练用 Spot/Preemptible 实例（便宜 60-90%，但可能中断）
- **自动扩缩容**：根据队列长度动态调整 GPU 数量
- **训练效率优化**：提升 MFU（Model FLOPs Utilization），减少空闲时间

**MFU 计算**：

$$\text{MFU} = \frac{\text{实际吞吐量} \times \text{每样本计算量}}{\text{GPU 峰值算力} \times \text{GPU 数量}}$$

- GPT-3 175B 在 A100 上的 MFU 约为 30-50%
- 优秀的大规模训练 MFU 可达 50-60%

### 6.2 云服务 vs 自建集群（Cloud vs On-Premise）

| 维度 | 云服务商（AWS/Azure/GCP） | 自建集群 |
|------|------------------------|---------|
| **初期投入** | 低，按需付费 | 高，硬件采购 |
| **灵活性** | 高，随时扩缩容 | 低，固定容量 |
| **长期成本** | 高（2-3× 硬件成本） | 低 |
| **运维复杂度** | 低 | 高 |
| **网络质量** | 中等 | 高（定制 IB） |
| **适用** | 中小规模、实验性 | 大规模、长期 |

**DeepSeek 的选择**：自建集群 + 2048 张 H800，训练成本约 $5.6M，比云服务商低 5-10 倍。

---

## 7. 面试高频考点（Interview Q&A）

1. **NVLink vs PCIe vs InfiniBand 的区别？**
   - NVLink：GPU 直连，带宽最高（900 GB/s），延迟最低
   - PCIe：通用总线，带宽较低（64 GB/s），用于 GPU-CPU 通信
   - InfiniBand：跨节点网络，支持 RDMA，用于多机通信

2. **ZeRO-1/2/3 的区别？**
   - ZeRO-1：切分优化器状态
   - ZeRO-2：切分优化器状态 + 梯度
   - ZeRO-3：切分优化器状态 + 梯度 + 参数

3. **为什么需要张量并行 + 流水线并行 + 数据并行？**
   - 单一并行方式无法扩展到超大模型
   - TP 解决层内放不下的问题（受限于单节点 GPU 数）
   - PP 解决层数过多的问题（受限于流水线气泡）
   - DP 解决 batch size 不够的问题（受限于显存）

4. **All-Reduce 的 Ring 算法和 Tree 算法有什么区别？**
   - Ring：带宽最优，2×(N-1) 步，但延迟高
   - Tree：延迟低，log(N) 步，但带宽稍差
   - NCCL 根据数据大小和网络拓扑自动选择

5. **如何诊断训练中的通信瓶颈？**
   - 使用 nsys / nvprof 分析 GPU 时间线
   - 检查通信-计算重叠比例
   - 监控 IB 端口的带宽利用率
   - 使用 PyTorch Profiler 的 communication 视图
