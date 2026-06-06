# 推理优化技术（Inference Optimization）

> **适合人群**：希望理解大模型推理加速原理和工程实践的开发者
> **难度**：⭐⭐⭐⭐（较难）

---

## 1. 推理性能指标（Inference Performance Metrics）

### 1.1 核心指标（Key Metrics）

| 指标 | 全称 | 含义 | 优化目标 |
|------|------|------|---------|
| **TTFT** | Time To First Token | 首 token 生成延迟 | < 500ms |
| **TPOT** | Time Per Output Token | 每输出 token 的间隔 | < 50ms |
| **Throughput** | 吞吐量 | 每秒生成的 token 数 | 最大化 |
| **Latency** | 端到端延迟 | 从请求到完整响应的时间 | 最小化 |

**TTFT 与 TPOT 的关系**：

```
输入: "北京今天天气怎么样？"
   │
   ▼  ┌─────┐  TTFT
   └──→│ Prefill │──→ 生成第一个 token（计算密集型）
       └─────┘
          │
          ▼  ┌─────┐  TPOT
          └──→│ Decode │──→ 生成第二个 token（内存受限）
             └─────┘
                │
                ▼  ┌─────┐
                └──→│ Decode │──→ 生成第三个 token
                   └─────┘
```

**Prefill vs Decode 阶段**：
- **Prefill（预填充）**：处理输入 prompt，计算所有 token 的 KV Cache（键值缓存），计算密集型（Compute-bound）
- **Decode（解码）**：自回归（Autoregressive）生成，每次只处理一个新 token，内存带宽受限（Memory-bandwidth-bound，受限于加载模型权重的速度）

### 1.2 Roofline 模型（Roofline Model）

推理性能受限于两个资源：

```
    计算峰值
       ▲
       │    ╱ 计算受限区
       │   ╱
       │  ╱
       │ ╱     ╱ 内存带宽受限区
       │╱     ╱
       └─────╱──────────→ 计算强度
            拐点
```

- **小 batch / 短序列**：计算强度低，处于内存带宽受限区 → 量化、KV Cache 优化更有效
- **大 batch / 长序列**：计算强度高，处于计算受限区 → 批处理、并行化更有效

---

## 2. 模型量化（Model Quantization）

### 2.1 量化原理（Quantization Principles）

将模型权重从高精度（FP32/FP16）转换为低精度（INT8/INT4），减少内存占用和计算量。

```
FP32:  [0.234, -1.567, 0.891, ...]  ← 32 bit × N
        ↓ 量化
INT8:  [30,   -128,  122,  ...]    ← 8 bit × N  (scale=0.0078, zero_point=0)
        ↓ 反量化
FP32:  [0.234, -1.000, 0.952, ...]  ← 有精度损失
```

**量化公式**：

$$w_{quant} = \text{round}\left(\frac{w}{scale}\right) + zero\_point$$
$$w_{dequant} = scale \times (w_{quant} - zero\_point)$$

其中 $scale = \frac{w_{max} - w_{min}}{2^n - 1}$，$n$ 为量化位数。

### 2.2 量化方法对比（Quantization Methods）

| 方法 | 精度 | 权重格式 | 激活格式 | 特点 |
|------|------|---------|---------|------|
| **RTN** | 最简单 | INT8/INT4 | FP16 | Round-To-Nearest，快速但质量差 |
| **GPTQ** | 中等 | INT4/INT3 | FP16 | 逐层量化，最小化输出误差 |
| **AWQ** | 好 | INT4 | FP16 | 保护对激活敏感的权重通道 |
| **SmoothQuant** | 好 | INT8 | INT8 | 将量化难度从激活迁移到权重 |
| **GGUF** | 中等 | INT4-Q8 | 混合 | llama.cpp 格式，CPU 友好 |
| **FP8** | 最好 | FP8 | FP8 | H100 原生支持，几乎无损 |

### 2.3 GPTQ（Generative Pre-trained Transformer Quantization）（Generative Pre-trained Transformer Quantization）

**核心思想**：逐层量化，考虑量化误差对后续层的影响。

**OPTQ 算法步骤**：
1. 对每层线性层，收集输入激活的 Hessian 矩阵
2. 按重要性排序权重（对输出影响大的权重用更高精度）
3. 逐权重量化，同时更新未量化的权重来补偿误差

```python
# 使用 AutoGPTQ 进行量化
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,      # 每 128 个权重共享一组 scale/zero_point
    desc_act=True,       # 激活顺序描述（减少内存使用）
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantize_config=quantize_config
)
model.quantize(examples)  # 用校准数据量化
model.save_quantized("Llama-2-7B-GPTQ")
```

### 2.4 AWQ（Activation-aware Weight Quantization）（Activation-aware Weight Quantization）

**核心洞察**：不是所有权重通道对输出质量的影响都相同。保护"重要"通道可以显著提升量化质量。

```
输入激活 X ──→ [权重矩阵 W]
                │通道0│通道1│通道2│...│通道N│
                │ 0.1 │ 1.5 │ 0.02│   │ 0.8 │
                │ ... │ ... │ ... │   │ ... │
                      ↑
                通道1 的激活值大 → 该通道权重更重要
                → 量化时给更高精度（或更小的 group_size）
```

**AWQ 相比 GPTQ 的优势**：
- 更好的 perplexity（困惑度更低）
- 更快的量化速度（不需要逐层迭代优化）
- 支持 fused kernel，推理更快

```python
# 使用 AutoAWQ
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b"
)
model.quantize(
    tokenizer,
    quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4}
)
model.save_quantized("Llama-2-7B-AWQ")
```

### 2.5 量化显存对比（Memory Comparison）

| 精度 | 7B 模型显存 | 13B 模型显存 | 70B 模型显存 | 质量损失 |
|------|------------|-------------|-------------|---------|
| FP32 | 28 GB | 52 GB | 280 GB | 无 |
| FP16/BF16 | 14 GB | 26 GB | 140 GB | 几乎无损 |
| INT8 (SmoothQuant) | 7 GB | 13 GB | 70 GB | 微小 |
| INT4 (AWQ/GPTQ) | 4 GB | 7.5 GB | 40 GB | 可接受 |
| FP8 (H100) | 7 GB | 13 GB | 70 GB | 几乎无损 |

---

## 3. KV Cache 优化（KV Cache Optimization）

### 3.1 KV Cache 原理（KV Cache Principles）

Decoder 推理时，每个新 token 需要与所有历史 token 计算 Attention。为避免重复计算，缓存历史 token 的 Key 和 Value。

```python
# 无 KV Cache：每次重新计算所有位置
for i in range(seq_len):
    Q = W_q(x[:, :i+1, :])      # 当前 token 的 Query
    K = W_k(x[:, :i+1, :])      # 所有历史 token 的 Key（重复计算！）
    V = W_v(x[:, :i+1, :])      # 所有历史 token 的 Value（重复计算！）
    attn = softmax(Q @ K.T / sqrt(d))
    out = attn @ V

# 有 KV Cache：只计算新 token
past_k, past_v = None, None
for i in range(seq_len):
    q = W_q(x[:, i:i+1, :])     # 只计算当前 token 的 Query
    k = W_k(x[:, i:i+1, :])     # 只计算当前 token 的 Key
    v = W_v(x[:, i:i+1, :])     # 只计算当前 token 的 Value

    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)  # 拼接历史 Key
        v = torch.cat([past_v, v], dim=2)  # 拼接历史 Value

    past_k, past_v = k, v
    attn = softmax(q @ k.transpose(-2, -1) / sqrt(d))
    out = attn @ v
```

**KV Cache 显存占用**：

$$\text{KV Cache} = 2 \times batch\_size \times num\_layers \times num\_heads \times head\_dim \times seq\_len \times bytes\_per\_element$$

以 LLaMA-2 70B 为例：
- batch_size=1, layers=80, heads=64, head_dim=128, seq_len=4096, FP16
- KV Cache = 2 × 1 × 80 × 64 × 128 × 4096 × 2 bytes = **10 GB**

### 3.2 PagedAttention（vLLM）

**问题**：传统 KV Cache 预分配连续内存（Contiguous Memory），导致：
- 内存碎片（Memory Fragmentation，不同序列长度）
- 无法共享（如 beam search（束搜索）、并行采样（Parallel Sampling））

**PagedAttention 的解决方案**：

借鉴操作系统的虚拟内存分页机制：
- 将 KV Cache 分成固定大小的 "blocks"（如每 block 16 tokens）
- 按需分配，不连续存储
- 支持共享（多个序列引用同一 block）

```
传统方式（连续内存）:
序列A: [K1 K2 K3 K4 K5 _ _ _]  预分配8个位置，3个空闲
序列B: [K1 K2 _ _ _ _ _ _]    预分配8个位置，6个空闲
       ↑ 碎片，无法给其他序列使用

PagedAttention（分页）:
Block Table:
序列A: [Block0] → [Block1] → [Block2]
              物理内存:
              Block0: [K1 K2 K3 K4]  ← Block1 也可以引用 Block0
              Block1: [K5 K6 K7 K8]     （如并行采样时共享前缀）
              Block2: [K9 K10]
```

**vLLM 核心优势**：
1. **内存高效**：减少 20-40% 的 KV Cache 内存浪费
2. **高吞吐量**：相同的 GPU 内存可以服务更多并发请求
3. **支持共享**：beam search、并行解码时共享 KV Cache

### 3.3 KV Cache 压缩技术（KV Cache Compression）

| 方法 | 原理 | 压缩比 | 质量损失 |
|------|------|--------|---------|
| **H2O（Heavy-Hitter Oracle）** | 只保留对 Attention 贡献大的 token | 50% | 微小 |
| **StreamingLLM** | 保留初始 token + 滑动窗口 | 可变 | 小 |
| **Scissorhands** | 基于 Attention 模式压缩 | 30-50% | 微小 |
| **量化 KV Cache** | INT8/FP8 存储 KV | 50% | 微小 |

**StreamingLLM 详解**：

观察发现：Attention 中初始的几个 token（如 system prompt）和最近的 token 最重要，中间的 token 关注度低。

```
序列: [System] [Context] ... [Recent] [New]
       ↑                        ↑
     保留（Attention Sink）    保留（滑动窗口）
       └───── 丢弃 ─────────────┘
```

- 保留前 4 个 token（Attention Sink）
- 保留最近的 1020 个 token（滑动窗口）
- 可以在无限长序列上保持稳定的生成质量

---

## 4. 长上下文架构（Long Context Architectures）

从 4K 到 128K 再到 1M+ token，长上下文是 2024-2025 年的关键战场。

### 4.1 长上下文的核心挑战

**Attention 的复杂度瓶颈**：
- 标准 Self-Attention 复杂度为 $O(n^2)$，$n$ 为序列长度
- 4K 上下文：$4096^2 = 16M$ 操作
- 128K 上下文：$131072^2 = 17B$ 操作（1000 倍增长）
- KV Cache 显存：128K × 80 层 × 2 (K/V) × FP16 = 数十 GB

**三大挑战**：
1. **计算**：$O(n^2)$ Attention 不可扩展
2. **显存**：KV Cache 随长度线性增长
3. **训练**：长序列的并行训练效率低

### 4.2 位置编码外推技术

将训练时的短上下文（如 4K）扩展到推理时的长上下文（如 128K）。

#### NTK-Aware 扩展

**问题**：RoPE 的旋转角度 $	heta_i = 10000^{-2i/d}$ 在超长序列时频率分布不均匀。

**NTK（Neural Tangent Kernel）解决方案**：
- 不插值位置，而是**缩小旋转角度的基频**
- 修改 $	heta_i$ 的 base 从 10000 到更大的值（如 1000000）
- 高频分量（小 $i$）保持不变，低频分量（大 $i$）压缩
- 效果：无需微调即可外推 2-8 倍长度

```python
def apply_ntk_scaling(freqs, scale=8.0):
    """
    NTK-aware RoPE 扩展
    scale: 目标长度 / 训练长度
    """
    # 低频部分压缩
    low_freq_mask = freqs < scale * freqs_original
    freqs = torch.where(low_freq_mask, freqs / scale, freqs)
    return freqs
```

#### YaRN（Yet another RoPE extension method）

NTK 的改进版，更稳定的外推：
- 引入**温度缩放（Temperature Scaling）**：对 Attention 分数乘以温度系数
- 解决外推时的 Attention 分数爆炸问题
- 支持 2×、4×、8×、16× 外推
- LLaMA 2 从 4K 扩展到 128K 的常用方案

```
训练长度: 4K
YaRN 外推:
  - 不微调: 扩展到 8-16K
  - 少量微调（1B tokens）: 扩展到 32-64K
  - 充分微调: 扩展到 128K+
```

#### 位置插值（Position Interpolation, PI）

最简单的方法：将位置索引按比例缩小。

```
训练时: 位置 0, 1, 2, ..., 4095
推理时（扩展到 16K）: 位置 0, 0.25, 0.5, ..., 4095

即将 16K 的位置"压缩"到 4K 的范围内
```

**缺点**：所有位置都被压缩，包括近距离位置，影响短程依赖。

**适用**：PI 适合微调后使用，YaRN 更适合零样本外推。

### 4.3 高效 Attention 架构

#### 滑动窗口注意力（Sliding Window Attention）

Mixtral、Mistral 等模型采用的核心技术：

```
标准 Attention: 每个 token  attending to 所有前序 token
                复杂度: O(n²)

滑动窗口 Attention: 每个 token 只 attending to 最近的 W 个 token
                    例如 W=4096，序列长度 128K
                    复杂度: O(n × W) = O(n)（当 W 固定）
                    
实际实现:
  大多数层: 滑动窗口（局部注意力）
  少数层（每 4 层）: 全局注意力（保证长距离依赖）
```

**效果**：128K 序列的计算量接近 4K 序列。

#### Ring Attention

UC Berkeley 提出的分布式长上下文方案：

```
问题: 单 GPU 显存放不下 1M token 的 KV Cache

Ring Attention 解决方案:
  - 将序列分成多个 block，分配到多个 GPU
  - GPU 0 处理 block 0，GPU 1 处理 block 1，...
  - Attention 计算通过 ring-allreduce 在 GPU 间传递
  - 每个 GPU 只需存储一个 block 的 KV Cache
  
扩展性:
  - 8 GPU → 支持 8x 单 GPU 长度
  - 64 GPU → 支持 64x 单 GPU 长度
  - 已支持 10M+ token 的上下文
```

**代表实现**：
- Google DeepMind 的环形注意力实现
- 开源项目：Ring Attention、Striped Attention

#### Linear Attention / 线性注意力近似

用核技巧将 $O(n^2)$ 的 Attention 近似为 $O(n)$：

```
标准 Attention:
  Attention(Q, K, V) = softmax(QK^T / √d) V
  瓶颈: QK^T 是 n×n 矩阵

Linear Attention:
  将 softmax 替换为特征映射 φ:
  Attention(Q, K, V) ≈ φ(Q) (φ(K)^T V) / (φ(Q) φ(K)^T)
  
  关键: (φ(K)^T V) 可以增量计算，复杂度降为 O(n)
  
代表方法:
  - Performer: 用随机特征映射近似 softmax
  - Linformer: 低秩近似
  - RWKV: 用循环机制替代 Attention
  - Mamba: 状态空间模型（State Space Model）
```

**当前状态**：
- Linear Attention 在理论上很有吸引力
- 但在实际大模型（>7B）上的效果仍略逊于标准 Attention
- 主要用在特定场景（如长文档处理、DNA 序列分析）

### 4.4 KV Cache 的层次化存储

对于超长上下文，KV Cache 无法全部放在 GPU 显存中：

```
层次化存储方案:

GPU HBM（显存）:
  - 存储最近的 4K-8K token 的 KV Cache
  - 访问延迟: ~1μs
  
CPU DRAM（内存）:
  - 存储中期的 32K-128K token
  - 访问延迟: ~100μs
  - 通过 PCIe 传输到 GPU
  
NVMe SSD（磁盘）:
  - 存储历史 token（1M+）
  - 访问延迟: ~10ms
  - 按需加载到 CPU/GPU

预加载策略:
  - 预测下一步需要哪些历史 token
  - 提前从 SSD → CPU → GPU 加载
```

**实现案例**：
- vLLM 的 prefix caching + 磁盘 offloading
- Anthropic Claude 的 200K 上下文推测采用类似方案

### 4.5 长上下文模型对比

| 模型 | 上下文长度 | 核心技术 | 实际可用性 |
|------|-----------|---------|-----------|
| GPT-4o | 128K | 未知（推测 RoPE + 优化） | 全量开放 |
| Claude 3 | 200K | 推测 Sliding Window + 分层 KV | 全量开放 |
| Gemini 1.5 Pro | 1M-10M | TPU 高带宽 + 稀疏 Attention | 全量开放 |
| LLaMA 3 | 128K | RoPE + 长文本微调 | 开源 |
| Kimi (月之暗面) | 200K | 长文本专用优化 | 国内可用 |

### 4.6 面试高频考点：长上下文

1. **RoPE 外推的 NTK-Aware 和 YaRN 有什么区别？**
   > 答：NTK-Aware 通过增大 RoPE 的 base 值来扩展频率范围，无需微调即可外推 2-8 倍。YaRN 在 NTK 基础上加入温度缩放，解决外推时的 Attention 分数爆炸问题，更稳定，支持更大倍数的外推。

2. **滑动窗口 Attention 如何保证长距离依赖？**
   > 答：大多数层使用滑动窗口（只关注最近的 W 个 token），但每隔几层（如每 4 层）使用全局 Attention。这样既能保证局部注意力的效率，又能通过全局层传递长距离信息。

3. **Ring Attention 如何支持 10M token 上下文？**
   > 答：将序列分块分配到多个 GPU，每个 GPU 只存储一个 block 的 KV Cache。Attention 计算通过 ring-allreduce 在 GPU 间循环传递，实现线性扩展。64 个 GPU 可支持 64 倍于单 GPU 的上下文长度。

4. **长上下文推理的主要瓶颈是什么？**
   > 答：三大瓶颈：(1) Attention 计算复杂度 $O(n^2)$；(2) KV Cache 显存占用随长度线性增长；(3) 预填充（Prefill）阶段的计算时间随长度增加。优化方向包括稀疏 Attention、KV Cache 压缩、层次化存储。

5. **Linear Attention 为什么还没有取代标准 Attention？**
   > 答：Linear Attention 通过核近似将复杂度降为 $O(n)$，但在大模型上的实际效果仍略逊于标准 Attention。原因包括：近似误差累积、与现有训练基础设施不兼容、在某些任务（如精细的局部依赖）上表现不佳。目前主要用于特定长序列场景。

---

## 5. 批处理优化（Batching Optimization）

### 4.1 连续批处理（Continuous Batching）

**传统静态批处理（Static Batching）的问题**：

```
Batch 中的请求必须同时完成
请求A: [====]          快
请求B: [==========]    慢
请求C: [======]        中等
─────────────────────
GPU 空闲等待 B 完成
```

**连续批处理（In-flight Batching）**：

```
t1: A[====]  B[====]  C[====]  D[====]
t2: A[done]  B[====]  C[====]  D[====]  E[new]
t3:          B[====]  C[done]  D[====]  E[====]  F[new]
     ↑ A 完成后立即加入新请求 E，无需等待 B
```

**实现机制**：
- 每个生成步骤后，检查是否有请求完成
- 完成的请求从 batch 中移除
- 新的请求立即加入 batch
- 动态调整 batch 大小

**效果**：
- 吞吐量提升 10-20 倍（相比无批处理）
- GPU 利用率接近 100%

### 4.2 请求调度策略（Request Scheduling）

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **FCFS** | 先来先服务 | 公平 | 短请求等待长请求 |
| **最短作业优先** | 优先处理短请求 | 平均延迟低 | 长请求饥饿 |
| **分桶调度** | 按序列长度分桶 | 减少 padding 浪费 | 需要预测长度 |
| **抢占式** | 长请求可中断 | 平衡公平和效率 | 实现复杂 |

---

## 6. 推测解码（Speculative Decoding）

### 6.1 核心思想（Core Idea）

用小模型（Draft Model，草稿模型）快速生成候选 token，用大模型（Target Model，目标模型）并行验证，只接受正确的 token。

```
传统解码:
大模型: [t1]→[t2]→[t3]→[t4]→[t5]  串行，每步一个token

推测解码:
小模型快速草稿: [t1]→[t2]→[t3]→[t4]→[t5]  （假设生成5个token）
                  ↓
大模型并行验证:  检查 [t1,t2,t3,t4,t5] 是否正确
                  ↓
接受 [t1,t2,t3]，拒绝 t4（因为 t4 概率太低）
                  ↓
从 t4 开始重新草稿
```

**为什么能加速？**
- 小模型生成 5 个 token 的时间 ≈ 大模型生成 1 个 token 的时间
- 如果接受率 80%，相当于每步生成 4 个 token
- 整体加速比：2-3 倍

### 6.2 接受-拒绝机制（Accept-Reject Mechanism）

**Modified Rejection Sampling**：

对于每个候选 token $x'$ 和真实分布 $p(x)$、草稿分布 $q(x)$：

- 以概率 $\min(1, \frac{p(x')}{q(x')})$ 接受 $x'$
- 如果拒绝，从修正分布 $\text{normalize}(\max(0, p(x) - q(x)))$ 采样新 token

**关键保证**：输出分布与原始大模型完全一致（无质量损失）。

### 6.3 草稿模型选择（Draft Model Selection）

| 方案 | 草稿模型 | 适用场景 |
|------|---------|---------|
| **独立小模型** | 1B-7B 模型 | 通用场景 |
| **模型自推测** | 大模型浅层 | 单模型部署 |
| **Lookahead** | n-gram 匹配 | 简单重复模式 |
| **Medusa** | 多个解码头 | 训练开销大，接受率高 |

---

## 7. 推理引擎对比（Inference Engine Comparison）

### 6.1 主流推理引擎（Mainstream Engines）

| 引擎 | 出品方 | 核心特性 | 适用场景 |
|------|--------|---------|---------|
| **vLLM** | Berkeley | PagedAttention、连续批处理 | 高并发服务 |
| **TensorRT-LLM** | NVIDIA | 极致 GPU 优化、多机推理 | 生产部署 |
| **TGI** | HuggingFace | 易用、支持多模型 | 快速原型 |
| **llama.cpp** | 社区 | CPU/GPU 混合、量化 | 本地/边缘 |
| **sglang** | 社区 | RadixAttention、多轮对话 | 对话应用 |
| **DeepSpeed-Inference** | 微软 | ZeRO、张量并行 | 超大模型 |

### 6.2 性能对比（Performance Comparison: 7B on A100）

| 引擎 | Throughput (tokens/s) | 延迟 (ms/token) | 显存效率 |
|------|----------------------|----------------|---------|
| HuggingFace 原生 | ~20 | 50 | 低 |
| vLLM | ~80-120 | 8-12 | 高 |
| TensorRT-LLM | ~100-150 | 6-10 | 高 |
| TGI | ~60-90 | 10-15 | 中等 |
| llama.cpp (GPU) | ~40-60 | 15-20 | 中等 |

### 6.3 前缀缓存（Prefix Caching）

**应用场景**：多轮对话中 system prompt 和上下文不变，只变最后一句话。

```
对话1: [System] + [Context] + "你好"
对话2: [System] + [Context] + "谢谢"
对话3: [System] + [Context] + "再见"
        ↑ 公共前缀

前缀缓存：
- 计算并缓存 [System] + [Context] 的 KV Cache
- 新请求只需计算最后一个 token 的 KV
- TTFT 从 500ms 降至 50ms
```

**sglang 的 RadixAttention**：
- 自动识别并缓存公共前缀
- 树状管理缓存，支持 LRU 淘汰
- 特别适合对话、RAG 等多轮场景

---

## 8. 工业界推理实现（Industrial Inference Practices）

本节分析 OpenAI、Anthropic、Google 等公司在生产环境中如何部署推理服务，以及开源社区的最佳实践如何被工业界采纳。

### 7.1 OpenAI API 的推理架构

OpenAI 的 API 服务是全球最大规模的 LLM 推理部署之一，其架构选择对行业有标杆意义。

#### 延迟目标与实现

| 模型 | TTFT 目标 | TPOT 目标 | 典型上下文 |
|------|-----------|-----------|----------|
| GPT-3.5-turbo | < 300ms | < 20ms/tok | 4K-16K |
| GPT-4 | < 500ms | < 30ms/tok | 8K-32K |
| GPT-4o | < 300ms | < 15ms/tok | 128K |
| GPT-4o-mini | < 200ms | < 10ms/tok | 128K |

**实现这些目标的关键技术**：

1. **模型分片与张量并行（Tensor Parallelism）**
   - GPT-4 规模的模型（推测 1.8T 参数）无法放入单卡，必须跨多 GPU 张量并行
   - 典型的分片策略：8x A100/H100 处理一层，通过 NVLink 高速互联
   - 通信开销是关键瓶颈，FlashAttention 的 IO-aware 设计减少中间激活的跨卡传输

2. **动态批处理 + 优先级调度**
   - 实时请求（chat completion）与批量请求（batch API）共享同一 GPU 集群
   - 采用抢占式调度：高优先级请求可中断低优先级的批量推理
   - 连续批处理（Continuous Batching）是基础设施标配

3. **多副本负载均衡**
   - 同一模型部署多个副本，分散在不同 GPU 节点
   - 路由层根据当前队列深度、KV Cache 占用动态分配请求
   - 长上下文请求路由到专门的"长序列副本"，避免阻塞短请求

#### 成本优化策略

OpenAI 的 API 定价反映了其推理成本结构：

```
定价结构（每 1M tokens）：
  Input tokens:  $x/1M  (Prefill 阶段，计算密集型)
  Output tokens: $y/1M  (Decode 阶段，内存带宽受限，通常 y > x)

为什么 output 更贵？
  - Prefill: 一次并行计算所有 input tokens，摊薄了固定开销
  - Decode: 每个 token 串行生成，无法利用批处理优势
```

**Prompt Caching 的成本革命**：
- 2024 年底 OpenAI 推出 Prompt Caching：缓存过的前缀按 50% 价格计费
- 实现原理：自动识别请求的公共前缀，复用已计算的 KV Cache
- 对 RAG 应用影响巨大：system prompt + 文档上下文只需计算一次

### 7.2 Anthropic 的 Claude 推理优化

Anthropic 在 Claude 3 系列上展现了与 OpenAI 不同的优化重点。

#### 长上下文优化（200K tokens）

Claude 3 支持 200K 上下文，而延迟仍保持可接受水平（首token < 3s）：

**1. 稀疏注意力模式**
- 推测采用 **Sliding Window + Global Attention 混合**
- 大多数层使用局部注意力（Sliding Window，窗口 4K），减少计算量
- 少数关键层使用全局注意力，保证长距离依赖
- 效果：200K 序列的理论复杂度从 O(n²) 降至接近 O(n)

**2. 分页 KV Cache 管理**
- 超过 GPU 显存容量的长序列，将 KV Cache 卸载到 CPU 内存或 SSD
- 采用 LRU + 热度预测：即将用到的 KV 预加载到 GPU
- 对 "大海捞针"（Needle in Haystack）任务的优化：自动识别关键位置，优先保留

**3. 输出长度预测**
- 在 Prefill 阶段同时运行轻量级预测器，估计输出长度
- 根据预测长度选择解码策略：短输出用 greedy，长输出用 sampling
- 优化调度：长输出请求分配到有更多空闲显存的副本

#### Computer Use 的流式架构

Claude 3.5 Sonnet 的 Computer Use 功能需要极低的交互延迟：
- **流式生成**：token 一经生成立即返回，不等完整响应
- **工具调用交织**：模型在生成本文时同步判断是否需要调用工具
- **截断与恢复**：如果工具调用耗时过长，模型先返回部分结果，后续补全

### 7.3 Google Gemini 的 TPU 推理优势

Google 的 Gemini 系列在 TPU 上部署，与 GPU 方案有本质差异。

#### TPU vs GPU 的推理差异

| 维度 | GPU (NVIDIA) | TPU (Google) |
|------|-------------|--------------|
| **内存带宽** | H100: 3.35 TB/s | TPU v5p: ~4.8 TB/s |
| **片间互联** | NVLink (900 GB/s) | TPU Pod 网络 (高速) |
| **量化支持** | FP8/INT8/INT4 灵活 | 主要 BF16/INT8 |
| **矩阵单元** | Tensor Core | MXU (Matrix Multiply Unit) |
| **软件生态** | CUDA, vLLM, TensorRT | JAX, XLA, Pathways |

**对推理的影响**：
- TPU 的高内存带宽使 **标准 MHA（而非 GQA）** 在推理时也可接受
- TPU 的脉动阵列（Systolic Array）对特定矩阵形状优化更好
- Gemini 1.5 Pro 的 1M-10M token 上下文：TPU Pod 的大内存（数百 GB HBM）是关键 enabler

#### Pathways 系统的推理优势

Pathways 最初为训练设计，但同样优化了推理：
- **多租户共享**：同一 TPU Pod 可同时运行多个模型副本
- **动态资源分配**：根据流量自动增减分配的 TPU 芯片
- **故障透明迁移**：TPU 故障时请求自动路由到健康芯片，无需用户感知

### 7.4 开源方案在工业界的落地

#### vLLM 的工业级部署

vLLM 从 Berkeley 研究项目成长为工业标准，其关键设计被各大公司采纳：

**生产部署模式**：
```yaml
# 典型的 Kubernetes + vLLM 部署配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-70b
spec:
  replicas: 3  # 3 个副本分散在不同节点
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model=meta-llama/Meta-Llama-3-70B
        - --tensor-parallel-size=4  # 每张卡处理 1/4 权重
        - --max-num-seqs=256        # 最大并发序列数
        - --max-model-len=8192      # 最大上下文长度
        resources:
          limits:
            nvidia.com/gpu: 4
```

**实际性能数据**（LLaMA-3 70B on 4x A100）：
| 指标 | 单请求 | 满负载 (256 seqs) |
|------|--------|-------------------|
| TTFT | 120ms | 800ms |
| TPOT | 15ms | 25ms |
| Throughput | 60 tok/s | 2,400 tok/s |

#### TensorRT-LLM 的生产优化

NVIDIA 的 TensorRT-LLM 在延迟敏感场景（如实时对话）中表现最优：

**核心优化**：
1. **Kernel Fusion**：将多个 CUDA kernel 合并为单个，减少 launch overhead
2. **Custom CUDA Kernels**：为特定 GPU 架构（Ampere/Hopper）手写优化 kernel
3. **INT4/FP8 推理**：在 H100 上实现几乎无损的 FP8 推理，吞吐提升 2x

**适用场景**：
- 需要极致 TTFT 的场景（如语音助手）
- 固定模型的长期部署（编译时间较长，不适合频繁切换模型）

### 7.5 工业界的量化实践

#### 各公司的量化策略

| 公司/产品 | 量化方案 | 精度 | 部署场景 |
|-----------|---------|------|---------|
| OpenAI GPT-4 | 未公开 | 推测 FP8/INT8 | API 服务端 |
| Anthropic Claude 3 | 未公开 | 推测 FP8/BF16 | API 服务端 |
| Meta LLaMA 3 | 官方 FP8/INT8 支持 | FP16 (基线) | 开源 + 自托管 |
| Google Gemini | TPU BF16/INT8 | BF16 | TPU Pod |
| DeepSeek-V3 | FP8 (训练+推理) | FP8 | 全链路 FP8 |

**关键观察**：
- **FP8 成为新标准**：H100/Blackwell 原生支持 FP8，质量几乎无损，推理速度提升 1.5-2x
- **权重量化 vs 激活量化**：工业界更倾向于仅权重量化（如 AWQ），因为激活量化对分布敏感，容易损失质量
- **动态量化**：根据输入序列的统计特性动态选择量化参数，比静态量化质量更好但开销更大

### 7.6 边缘与端侧推理

#### 手机端部署（iOS/Android）

**Meta LLaMA 的端侧实践**：
- LLaMA 3 8B 经过 INT4 量化后约 4.5GB，可部署在高端手机（iPhone 15 Pro, 24GB 统一内存）
- 使用 llama.cpp 的 Metal backend（iOS）或 Vulkan backend（Android）
- 性能：~10-15 tokens/s（iPhone 15 Pro），足够流畅的聊天体验

**Apple 的 Core ML 优化**：
- iOS 18 引入专门的 LLM runtime，支持 ANE（Apple Neural Engine）加速
- 推测采用类似的 KV Cache 管理和量化策略
- 隐私优势：端侧运行，数据不出设备

#### 推理芯片专用化趋势

| 芯片 | 厂商 | 定位 | 特点 |
|------|------|------|------|
| H100/H200 | NVIDIA | 数据中心 | 通用，生态成熟 |
| TPU v5p | Google | 数据中心 | 高带宽，与 Google 模型深度优化 |
| Gaudi 3 | Intel | 数据中心 | 性价比导向，支持 PyTorch |
| MTIA | Meta | 数据中心 | 自研，专为推荐和推理优化 |
| ANE | Apple | 端侧 | 低功耗，隐私优先 |

### 7.7 面试高频考点：工业界推理

1. **OpenAI 如何实现 GPT-4 的低延迟？**
   > 答：张量并行将模型分片到多 GPU；连续批处理提高 GPU 利用率；Prompt Caching 复用 KV Cache；动态路由根据副本负载分配请求。

2. **为什么 Claude 3 能在 200K 上下文下保持低延迟？**
   > 答：推测使用 Sliding Window Attention 降低计算复杂度；分页 KV Cache 管理支持显存+内存分层存储；输出长度预测优化调度。

3. **TPU 推理相比 GPU 有何优劣？**
   > 答：优势：更高内存带宽，片间互联更好，Pathways 系统支持动态资源分配。劣势：生态锁定（JAX/XLA），灵活性不如 CUDA，量化选择较少。

4. **vLLM 的 PagedAttention 为什么能提升吞吐量？**
   > 答：将 KV Cache 分页管理，消除内存碎片；支持序列间共享（如并行采样）；相同显存可容纳更多并发请求；连续批处理充分利用 GPU。

5. **FP8 在工业界的采用现状？**
   > 答：H100 原生支持 FP8 后，成为新的事实标准。DeepSeek-V3 首次全链路 FP8 训练+推理。质量几乎无损（vs FP16），速度提升 1.5-2x，显存减半。

---

## 9. 面试高频考点（Interview Q&A）

1. **为什么 Decode 阶段是内存带宽受限而非计算受限？**
   - 每次只处理 1 个新 token
   - 需要加载全部模型权重（14GB for 7B FP16）
   - 计算量小（矩阵-向量乘法），但内存读取量大
   - 量化可以减少内存读取，从而加速

2. **GPTQ 和 AWQ 的核心区别？**
   - GPTQ：逐层优化，考虑误差传播，用更高精度保护重要权重
   - AWQ：基于激活值大小判断权重重要性，保护激活大的通道
   - AWQ 通常更快、更容易部署

3. **PagedAttention 如何解决内存碎片？**
   - 将 KV Cache 分页管理，非连续存储
   - 按需分配，用完后回收
   - 支持不同长度序列共享同一批物理 block

4. **推测解码为什么不会降低输出质量？**
   - 使用修正的拒绝采样
   - 接受的 token 分布与原模型一致
   - 拒绝时从修正分布采样，保证最终输出分布不变

5. **连续批处理 vs 静态批处理？**
   - 静态：一个 batch 内所有请求一起完成，快的等慢的
   - 连续：每个 step 后动态调整 batch，完成的请求立即退出
   - 连续批处理显著提升 GPU 利用率和吞吐量
