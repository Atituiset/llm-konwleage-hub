# 大模型基础（LLM Foundations）

> **资料来源**：毛玉仁、高云君等《大模型基础》完整版
> **适合人群**：希望系统学习 LLM 原理的读者
> **难度**：⭐⭐⭐⭐（较难）

---

## 1. Transformer 架构详解（Transformer Architecture）

Transformer 是几乎所有现代大语言模型（Large Language Model, LLM）的基础架构。理解它的每个组件是掌握 LLM 原理的第一步。

### 1.1 整体架构（Overall Architecture）

Transformer 由**编码器（Encoder）**和**解码器（Decoder）**两部分组成，每部分都是多层堆叠的相同结构：

- **Encoder**：将输入序列映射为连续表示，每层包含 Self-Attention + Feed-Forward
- **Decoder**：基于 Encoder 输出和已生成内容自回归地生成目标序列

原始 Transformer 用于机器翻译（Encoder-Decoder），而 GPT 系列仅使用 Decoder，BERT 仅使用 Encoder。

```mermaid
graph TB
    subgraph Encoder['Encoder Stack (×N)']
        E1['Input Embedding + Positional Encoding']
        E2['Multi-Head Self-Attention']
        E3['Add & Norm']
        E4['Feed Forward']
        E5['Add & Norm']
        E1 --> E2 --> E3 --> E4 --> E5
    end

    subgraph Decoder['Decoder Stack (×N)']
        D1['Output Embedding + Positional Encoding']
        D2['Masked Multi-Head Self-Attention']
        D3['Add & Norm']
        D4['Cross-Attention (Q from decoder, K/V from encoder)']
        D5['Add & Norm']
        D6['Feed Forward']
        D7['Add & Norm']
        D1 --> D2 --> D3 --> D4 --> D5 --> D6 --> D7
    end

    E5 --> D4
```

### 1.2 Self-Attention（自注意力机制）

Self-Attention 是 Transformer 的核心，它让序列中的每个位置都能关注到所有其他位置，从而捕捉长距离依赖。

#### 1.2.1 完整数学推导（Complete Mathematical Derivation）（Complete Mathematical Derivation）

给定输入序列的嵌入矩阵（Embedding Matrix） $X \in \mathbb{R}^{n \times d}$，其中 $n$ 是序列长度（Sequence Length），$d$ 是嵌入维度：

**Step 1: 生成 Q、K、V**

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ 是可学习的投影矩阵（Projection Matrix）。

**直观理解 Q/K/V 的设计**：

这三个投影并非凭空设计，而是源于**信息检索（Information Retrieval）**的类比：

- **Query ($Q$)**："我正在处理第 $i$ 个词，我需要什么信息来理解它？"
- **Key ($K$)**："第 $j$ 个词包含什么信息？它的'标签'是什么？"
- **Value ($V$)**："第 $j$ 个词的实际语义（Semantics）内容是什么？"

Attention 的本质是：**用 Query 去匹配 Key，根据匹配程度加权聚合 Value**。

这种设计的精妙之处在于：
1. **解耦了"匹配什么"和"获取什么"**：Key 负责匹配（相似度计算），Value 负责提供内容
2. **每个 token 动态决定关注谁**：不同位置的 Query 会与不同的 Key 产生高相似度（Similarity）
3. **并行计算**：所有位置的 Attention 同时计算，高度适合 GPU

**Step 2: 计算注意力分数（Attention Score）**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Step 3: 缩放因子 $\sqrt{d_k}$ 的统计学原因**

为什么必须除以 $\sqrt{d_k}$？这是理解 Attention 的关键细节。

假设 $Q$ 和 $K$ 的每个元素是独立随机变量，均值（Mean）为 0，方差（Variance）为 1。那么点积（Dot Product） $Q_i \cdot K_j$ 是 $d_k$ 个独立随机变量之和：

$$Q_i \cdot K_j = \sum_{m=1}^{d_k} q_m \cdot k_m$$

根据中心极限定理（Central Limit Theorem, CLT），这个和的：
- **均值（Mean）** = 0
- **方差（Variance）** = $d_k \times \text{Var}(q_m) \times \text{Var}(k_m) = d_k$

因此，点积（Dot Product）的分布是 $\mathcal{N}(0, d_k)$，标准差（Standard Deviation）为 $\sqrt{d_k}$。

如果不做缩放，当 $d_k$ 很大时（如 64、128），点积（Dot Product）的绝对值会很大，导致 softmax 输入进入**饱和区**（梯度几乎为 0），造成梯度消失（Vanishing Gradient）。

除以 $\sqrt{d_k}$ 后，分布变为 $\mathcal{N}(0, 1)$，softmax 的输入在合理范围内，梯度流动正常。

**Step 4: Softmax（归一化指数函数） 与加权求和**

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}$$

$A_{ij}$ 表示位置 $i$ 对位置 $j$ 的注意力权重（Attention Weight）。然后：

$$\text{Output}_i = \sum_{j=1}^{n} A_{ij} V_j$$

即：每个位置的输出是其对所有位置 Value 的加权平均，权重由 Query-Key 相似度（Similarity）决定。

**完整的矩阵形式**：

$$\text{SelfAttention}(X) = \text{softmax}\left(\frac{XW^Q (XW^K)^T}{\sqrt{d_k}}\right) XW^V$$

#### 1.2.2 Causal (Masked) Self-Attention（因果/掩码（Mask）自注意力（Self-Attention））

Decoder-only 模型（如 GPT（Generative Pre-trained Transformer））使用因果 Attention，确保位置 $i$ 只能看到 $\leq i$ 的位置：

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M$ 是下三角 mask 矩阵：

$$M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$$

加入 $-\infty$ 后，softmax 会将上三角位置的权重置为 0，实现"只能看前面"的约束。

### 1.3 Multi-Head Attention（多头注意力 / MHA）

#### 1.3.1 数学定义（Mathematical Definition）

将注意力机制（Attention Mechanism）并行执行 $h$ 次，每次使用不同的投影矩阵（Projection Matrix），最后拼接并线性变换：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个 head：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

参数维度：
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_k}$，其中 $d_k = d/h$
- $W^O \in \mathbb{R}^{hd_k \times d} = \mathbb{R}^{d \times d}$

#### 1.3.2 为什么多头？（Why Multi-Head?）

**表达能力分析**：

单头 Attention 只有一个投影空间，所有信息交互都在同一语义（Semantics）空间中完成。多头相当于将 $d$ 维空间划分为 $h$ 个子空间，每个子空间学习不同的关系模式。

经验观察表明不同 head 确实学到了不同功能：
- **局部 head**：关注相邻词（语法依赖）
- **全局 head**：关注远距离词（指代消解、语义（Semantics）关联）
- **位置 head**：关注特定位置（如句首、标点）

**并行计算优势**：

虽然概念上是 $h$ 个独立的 Attention，实际实现中可以将 $h$ 个头的投影合并：

```
Q = X @ W_Q    # W_Q: (d, d) → 实际等价于 h 个 (d, d_k) 拼接
Q = Q.view(batch, n, h, d_k).transpose(1, 2)  # (batch, h, n, d_k)
```

这样 $h$ 个头的计算可以通过一个矩阵乘法（Matrix Multiplication）完成，然后在 GPU 上并行计算所有头的 Attention。

#### 1.3.3 输出合并过程（Output Concatenation）

```
每个 head 输出: (batch, h, n, d_k)
合并: transpose(1, 2) → (batch, n, h, d_k)
     reshape → (batch, n, h*d_k) = (batch, n, d)
线性投影: @ W_O → (batch, n, d)
```

### 1.4 Position-wise Feed-Forward Network（位置前馈网络（Feed-Forward Network, FFN） / FFN）

每个位置独立应用相同的前馈网络（Feed-Forward Network, FFN）：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

或写成：
$$\text{FFN}(x) = \text{ReLU（Rectified Linear Unit，线性整流单元）}(xW_1 + b_1)W_2 + b_2$$

**维度分析**：
- 输入/输出维度：$d$（如 512、768、4096）
- 中间层维度：$d_{ff} = 4d$（原始 Transformer），远大于 $d$
- 参数量：$W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$，共 $8d^2$

**为什么中间层要升维？**

FFN 的作用是为模型提供**逐位置的非线性变换和记忆能力**。升维到 $4d$ 提供了更多的"存储单元"来记忆不同的语义（Semantics）模式。可以证明：FFN 的两个线性层（Linear Layer）之间，模型学习到了大量可解释的语义（Semantics）原型（如"国家-首都"、"动词-名词"等关系）。

**特点**：
- 每个 token 独立计算，不共享跨位置信息（位置间的交互已在 Attention 中完成）
- 升维 → 非线性激活 → 降维，增加模型非线性表达能力

### 1.5 Normalization：LayerNorm vs RMSNorm vs BatchNorm（归一化方法对比）

#### 1.5.1 Batch Normalization（批归一化（Normalization） / BN）

$$\text{BN}(x) = \gamma \odot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

其中 $\mu_B, \sigma_B^2$ 是**当前 batch**中所有样本同一特征（Feature）的统计量。

**为什么不适用于 NLP？**
1. **序列长度（Sequence Length）可变**：不同句子的长度不同，难以对齐统计
2. **Batch size 小**：NLP 中 batch size 通常较小（显存限制），统计量不稳定
3. **推理（Inference）复杂**：需要维护 running mean/variance，增加工程复杂度

#### 1.5.2 Layer Normalization（层归一化（Normalization） / LN）

对每个样本的所有特征（Feature）进行归一化（Normalization）：

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} + \beta$$

其中：
$$\mu_L = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad \sigma_L^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu_L)^2$$

**优势**：
- 不依赖 batch size，每个样本独立归一化（Normalization）
- 训练（Training）和推理（Inference）行为一致
- 适合序列数据（每个位置的特征（Feature）分布可能不同）

#### 1.5.3 RMSNorm（Root Mean Square Layer Normalization）

LLaMA（Large Language Model Meta AI） 等现代模型采用 RMSNorm，去除均值（Mean）中心化：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

**与 LayerNorm 的对比**：

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 去均值（Mean） | ✅ | ❌ |
| 可学习参数 | $\gamma, \beta$ | 仅 $\gamma$ |
| 计算量 | 略高（需计算均值（Mean）） | 略低 |
| 效果 | 标准选择 | 在 LLM 中相当或更优 |
| 采用模型 | BERT（Bidirectional Encoder Representations from Transformers）、GPT（Generative Pre-trained Transformer）-2 | LLaMA（Large Language Model Meta AI）、Qwen、DeepSeek |

**为什么 LLM 中 RMSNorm 足够好？**

研究表明，在深层 Transformer  中，各层输出的均值（Mean）本身就接近 0（由于残差连接（Residual Connection）的累加效应），因此显式去均值（Mean）的收益有限。去掉均值（Mean）计算后：
1. 计算更快（少一次 reduce mean）
2. 参数量减半（少一个 bias 项）
3. 数值稳定性（Numerical Stability）相当

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq_len, dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
```

#### 1.5.4 Pre-LN vs Post-LN（前置/后置归一化（Normalization））

**Post-LN（原始 Transformer）**：
```
x → Sublayer → x + Sublayer(x) → LayerNorm → output
```

**Pre-LN（现代标准）**：
```
x → LayerNorm → Sublayer → x + Sublayer(LayerNorm(x)) → output
```

**为什么 Pre-LN 训练（Training）更稳定？**

在 Post-LN 中，LayerNorm 位于残差路径上，会缩放残差的大小。深层时，残差路径的梯度需要经过多个 LayerNorm 的缩放，梯度传播不稳定。

在 Pre-LN 中，LayerNorm 在子层之前，残差连接（Residual Connection） $x + \text{Sublayer}(\text{LN}(x))$ 中 $x$ 的梯度是干净的（恒等映射），不受 LayerNorm 影响。这使得深层模型（30+ 层）的训练（Training）成为可能。

**代价**：Pre-LN 的最终输出范数较大，需要在最后加一个 LayerNorm 来稳定。

### 1.6 Residual Connections（残差连接 / Residual）

每个子层（Attention 或 FFN）的输出都加上输入：

$$\text{Output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

**梯度流分析**：

假设网络有 $L$ 层，第 $l$ 层的输出为 $x_l$。对于 Post-LN：

$$x_l = \text{LN}(x_{l-1} + f_l(x_{l-1}))$$

反向传播（Backpropagation）时，梯度需要经过 $L$ 个残差连接（Residual Connection）。由于残差连接（Residual Connection）的梯度是：

$$\frac{\partial x_l}{\partial x_{l-1}} = I + \frac{\partial f_l}{\partial x_{l-1}}$$

即使 $\frac{\partial f_l}{\partial x_{l-1}}$ 很小，$I$ 保证了梯度至少不会消失。这是深层网络可训练（Training）的根本原因。

### 1.7 Positional Encoding（位置编码 / PE）详解

Transformer  没有递归或卷积结构，需要显式注入位置信息。

#### 1.7.1 正弦/余弦位置编码（Positional Encoding）的旋转矩阵解释（Sinusoidal PE Rotation Matrix）

原始 Transformer  使用：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**为什么这种形式能表达相对位置？**

将相邻维度 $(2i, 2i+1)$ 看作一个二维向量：

$$\begin{pmatrix} PE_{pos, 2i} \\ PE_{pos, 2i+1} \end{pmatrix} = \begin{pmatrix} \sin(pos \cdot \omega_i) \\ \cos(pos \cdot \omega_i) \end{pmatrix}$$

其中 $\omega_i = 10000^{-2i/d_{model}}$。

对于位置 $pos + k$，有三角恒等式：

$$\sin((pos+k)\omega_i) = \sin(pos\omega_i)\cos(k\omega_i) + \cos(pos\omega_i)\sin(k\omega_i)$$
$$\cos((pos+k)\omega_i) = \cos(pos\omega_i)\cos(k\omega_i) - \sin(pos\omega_i)\sin(k\omega_i)$$

写成矩阵形式：

$$\begin{pmatrix} PE_{pos+k, 2i} \\ PE_{pos+k, 2i+1} \end{pmatrix} = \begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix} \begin{pmatrix} PE_{pos, 2i} \\ PE_{pos, 2i+1} \end{pmatrix}$$

这意味着：**位置 $pos+k$ 的编码可以通过对位置 $pos$ 的编码进行旋转得到**！旋转角度只依赖于相对距离 $k$，与绝对位置 $pos$ 无关。

这是正弦编码能表达相对位置的本质原因。

#### 1.7.2 学习式位置编码（Learnable PE）

直接将位置编码（Positional Encoding）作为可学习参数：

$$PE \in \mathbb{R}^{L_{max} \times d}$$

其中 $L_{max}$ 是最大序列长度（如 512、2048）。

**问题**：无法处理超过 $L_{max}$ 的序列（外推性差）。

#### 1.7.3 RoPE（旋转位置编码（Positional Encoding） / Rotary Position Embedding）

LLaMA（Large Language Model Meta AI） 采用的 RoPE 是正弦编码思想的直接推广，但直接应用于 Attention 的 Q、K：

**核心思想**：不将位置编码（Positional Encoding）加到输入上，而是将 Q、K 的每对维度视为二维向量，根据位置旋转：

$$\begin{pmatrix} q'_{m, 2i} \\ q'_{m, 2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{m, 2i} \\ q_{m, 2i+1} \end{pmatrix}$$

其中 $\theta_i = 10000^{-2i/d}$，$m$ 是位置索引。

**为什么 RoPE 更优？**

1. **相对位置编码（Positional Encoding）的内禀性**：
   计算 $\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle$，结果只依赖于 $m-n$，不依赖于绝对位置。

2. **长距离外推性**：
   通过位置插值（Position Interpolation）或 NTK-aware 扩展，可将训练（Training）时的长度（如 4096）扩展到更长（如 128K）。

3. **与 Attention 的融合**：
   位置信息直接编码在 Q、K 中，Attention 分数自然带有位置感知。

**RoPE 的矩阵形式**：

定义旋转矩阵 $R_m$，则：

$$q_m = R_m W_q x_m, \quad k_n = R_n W_k x_n$$

$$\text{Attention}(q_m, k_n) \propto q_m^T k_n = x_m^T W_q^T R_{m-n} W_k x_n$$

其中 $R_{m-n} = R_m^T R_n$ 只与相对位置有关。

```python
def apply_rope(x, cos, sin):
    """
    x: (batch, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim/2)
    """
    # 将相邻维度拆分为两列
    x1, x2 = x[..., ::2], x[..., 1::2]  # (..., head_dim/2)

    # 构建旋转后的表示: [-x2, x1]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)

    # x * cos + rotated * sin
    return x * cos + rotated * sin
```

#### 1.7.4 ALiBi（线性偏置注意力 / Attention with Linear Biases）

BLOOM 采用的方案，不在输入中加位置编码（Positional Encoding），而是在 Attention 分数中加上一个与距离成比例的惩罚项：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot \text{mask}\right)V$$

其中 $m$ 是预设的斜率，$\text{mask}_{ij} = -(i - j)$ 对于 $i \geq j$。

**优势**：天然支持超长序列外推，无需位置插值。

### 1.8 训练（Training）稳定性技术

#### 1.8.1 权重初始化

Transformer  使用 Xavier/Glorot 初始化：

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}\right)$$

或 He 初始化（对于 ReLU（Rectified Linear Unit，线性整流单元））：

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{fan_{in}}}\right)$$

**为什么好的初始化很重要？**

深层网络中，如果每层输出方差（Variance）逐渐增大或减小，会导致：
- 梯度爆炸（方差逐层放大）
- 梯度消失（方差逐层衰减）

Xavier 初始化保证了前向和反向传播（Backpropagation）时，信号方差（Variance）保持恒定（在特定假设下）。

#### 1.8.2 学习率（Learning Rate）预热（Learning Rate Warmup）

Transformer  训练（Training）的前 $warmup\_steps$ 步，学习率（Learning Rate）从 0 线性增长到峰值：

$$lr_t = lr_{max} \cdot \min\left(\frac{t}{warmup\_steps}, 1.0\right)$$

**为什么需要 Warmup？**

训练（Training）初期，模型参数随机初始化，输出分布非常尖锐。如果此时使用大学习率（Learning Rate），梯度更新会过大，导致：
- Loss spike（损失突然飙升）
- 后续难以恢复

Warmup 让小步长"探索"参数空间，待模型初步稳定后再增大学习率（Learning Rate）。

**典型设置**：$warmup\_steps \approx 1\% \sim 2\%$ 的总步数，如 4000 步（对于 200K 总步数）。

#### 1.8.3 梯度裁剪（Gradient Clipping）

限制梯度范数不超过阈值：

$$g \leftarrow \frac{g}{\max(1, \|g\| / \text{threshold})}$$

**作用**：防止偶尔的 loss spike 导致梯度爆炸（Exploding Gradient），破坏已学到的知识。

#### 1.8.4 Dropout（随机失活） 与正则化（Regularization）

- **Attention Dropout（随机失活）**：对 Attention 权重矩阵（Weight Matrix）随机置零
- **Hidden Dropout（随机失活）**：对 FFN 和 Attention 输出随机置零
- **现代趋势**：大模型（>1B）训练（Training）时 dropout 通常设为 0 或很小（0.1），数据本身提供了足够的正则化（Regularization）

---

## 2. 语言模型（Language Model）采样方法（Language Model Sampling）

训练（Training）好的语言模型（Language Model）输出的是下一个 token 的概率分布（Probability Distribution） $P(w_t | w_1, ..., w_{t-1})$。采样方法决定如何从这个分布中选择下一个 token。

### 2.1 Greedy Search（贪心搜索）

每步选择概率最高的 token：

$$w_t = \arg\max_w P(w | w_1, ..., w_{t-1})$$

**优点**：简单、确定性输出
**缺点**：容易陷入局部最优（Local Optimum），生成的文本通常单调、重复

### 2.2 Beam Search（束搜索）

维护 $k$ 个最优候选序列（beam width = $k$），每步扩展所有候选并保留 top-$k$。

**特点**：
- $k=1$ 时退化为贪心搜索
- $k$ 越大，搜索空间越大，计算成本越高
- 适合机器翻译（Machine Translation）、摘要等需要准确性的任务
- **不适合开放式生成**：容易生成通用、安全的回复（因为高概率序列往往平庸）

### 2.3 Temperature Sampling（温度采样）

通过温度参数 $T$ 调节概率分布（Probability Distribution）的"尖锐程度"：

$$P'(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

其中 $z_i$ 是模型输出的 logits。

- $T \to 0$：接近贪心搜索（确定性）
- $T = 1$：原始分布
- $T > 1$：分布更平坦，采样更多样
- $T < 1$：分布更尖锐，采样更保守

**常用设置**：$T \in [0.7, 1.0]$ 用于文本生成

### 2.4 Top-K Sampling（Top-K 采样）

每步只从概率最高的 $K$ 个 token 中采样，其余概率置零后重新归一化（Normalization）。

**问题**：$K$ 固定时，不同分布"平坦度"不同。平坦分布中 Top-K 可能包含很多低概率 token；尖锐分布中 Top-K 可能太少。

### 2.5 Top-P (Nucleus) Sampling（核采样）

每步从累积概率达到 $P$ 的最小 token 集合（nucleus）中采样：

$$V^{(p)} = \{w | \sum_{w' \in V^{(p)}, P(w') \geq P(w)} P(w') \geq p\}$$

- 动态调整候选集合大小
- 分布平坦时选更多 token，尖锐时选更少
- **Top-P 通常比 Top-K 更鲁棒**，$p \in [0.9, 0.95]$ 是常用设置

### 2.6 采样策略组合

实际使用通常组合多种方法：

```
temperature = 0.8
top_p = 0.95
top_k = 50
```

先应用 temperature，再取 top_k，最后取 top_p，在剩余集合中采样。

---

## 3. 语言模型（Language Model）评测指标（Evaluation Metrics）

### 3.1 Perplexity（困惑度 / Perplexity, PPL）

衡量语言模型（Language Model）对测试（Testing）集的"困惑"程度，即模型预测下一个词时的不确定性：

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1 ... w_{i-1})\right)$$

**理解**：
- PPL = 100 相当于每次面对 100 个等概率选择
- PPL 越低越好
- 只衡量语言建模能力，不衡量生成质量或有用性

### 3.2 BLEU（双语评估替补 / Bilingual Evaluation Understudy）

用于评估生成文本与参考文本的 n-gram 重叠度：

$$\text{BLEU（Bilingual Evaluation Understudy）} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

其中 $p_n$ 是 n-gram 精确率（Precision），BP 是简短惩罚（Brevity Penalty）。

**特点**：
- 最初用于机器翻译（Machine Translation），现广泛用于文本生成
- 关注精确率（生成文本中有多少 n-gram 出现在参考中）
- 对同义词不敏感

### 3.3 ROUGE（面向召回的摘要评估 / Recall-Oriented Understudy for Gisting Evaluation）

主要用于评估摘要质量，关注召回率（Recall）：

- **ROUGE（Recall-Oriented Understudy for Gisting Evaluation）-N**：n-gram 召回率（Recall） = 参考与生成共现的 n-gram / 参考中所有 n-gram
- **ROUGE（Recall-Oriented Understudy for Gisting Evaluation）-L**：最长公共子序列（LCS）的 F1 分数

### 3.4 BERT（Bidirectional Encoder Representations from Transformers）Score（BERT 评分）

利用预训练（Pre-training） BERT（Bidirectional Encoder Representations from Transformers） 的上下文嵌入计算语义（Semantics）相似度（Similarity）：

1. 将参考文本和生成文本的每个 token 映射为 BERT（Bidirectional Encoder Representations from Transformers） 嵌入
2. 计算 token 级别的余弦相似度（Similarity）
3. 通过贪心匹配计算 Precision、Recall、F1

**优势**：
- 捕捉语义（Semantics）相似性，不仅字面匹配
- 对同义词、 paraphrase 更鲁棒

### 3.5 G-EVAL（大模型评估）

使用大模型自身作为评估器：

1. 设计评估 prompt（含评估维度和评分标准）
2. 将生成文本输入大模型
3. 大模型输出分数（通常 1-5 分）

**维度**：连贯性、一致性、流畅性、相关性、事实准确性等

**局限**：大模型评估可能存在偏见，且需要调用 API

---

## 4. 大语言模型（Language Model）架构分类（Architecture Classification）

### 4.1 Encoder-only：以 BERT（Bidirectional Encoder Representations from Transformers） 为代表

**结构**：双向注意力，每个 token 都能看到所有其他 token
**预训练（Pre-training）任务**：
- MLM（Masked Language Model）：随机 mask 15% 的 token，预测被 mask 的内容
- NSP（Next Sentence Prediction，已被证明效果有限，后续版本移除）
**代表模型**：BERT（Bidirectional Encoder Representations from Transformers）、RoBERT（Bidirectional Encoder Representations from Transformers）a、ALBERT（Bidirectional Encoder Representations from Transformers）、DeBERT（Bidirectional Encoder Representations from Transformers）a
**适用任务**：文本分类、NER、问答（提取式）等理解任务
**不适用的原因**：没有自回归（Autoregressive）生成能力

### 4.2 Decoder-only：以 GPT（Generative Pre-trained Transformer） 为代表

**结构**：因果（causal）注意力，每个 token 只能看到前面的 token（上三角 mask）
**预训练（Pre-training）任务**：Next Token Prediction（自回归语言建模）
$$\mathcal{L} = -\sum_{t} \log P(w_t | w_1, ..., w_{t-1})$$
**代表模型**：GPT（Generative Pre-trained Transformer） 系列、LLaMA（Large Language Model Meta AI）、Claude、PaLM、Qwen
**适用任务**：文本生成、对话、代码生成等绝大多数生成任务
**成为主流的原因**：
1. 自回归（Autoregressive）预训练（Pre-training）简单高效
2. 与生成任务天然对齐
3. 可扩展性好（GPT-3 证明 scale 后涌现能力）
4. 推理（Inference）时只需维护 KV Cache，效率高

### 4.3 Encoder-Decoder：以 T5 为代表

**结构**：Encoder 双向编码（Bidirectional Encoding）输入，Decoder 自回归（Autoregressive）生成输出
**预训练（Pre-training）任务**：Span Corruption（将输入中连续片段替换为哨兵 token，Decoder 预测被替换内容）
**代表模型**：T5、BART、UL2
**适用任务**：翻译、摘要、结构化预测（输入和输出有明显区分的任务）

### 4.4 架构对比总结

| 特性 | Encoder-only (BERT) | Decoder-only (GPT) | Encoder-Decoder (T5) |
|------|---------------------|---------------------|----------------------|
| 注意力方向 | 双向 | 单向（因果） | Encoder 双向 + Decoder 单向 |
| 预训练（Pre-training）任务 | MLM | Next Token Prediction | Span Corruption |
| 主要用途 | 理解任务 | 生成任务 | 翻译/摘要 |
| 推理（Inference）方式 | 一次性前向 | 自回归（Autoregressive）逐步生成 | Encoder 一次 + Decoder 自回归（Autoregressive） |
| 代表模型 | BERT（Bidirectional Encoder Representations from Transformers）, RoBERT（Bidirectional Encoder Representations from Transformers）a | GPT（Generative Pre-trained Transformer）, LLaMA（Large Language Model Meta AI）, Claude | T5, BART |
| 当前主流程度 | 较低 | 极高 | 中等 |

---

## 5. Scaling Laws（缩放定律 / Scaling Laws）

Scaling Laws 描述模型性能如何随模型规模（参数量 $N$）、数据量（$D$）和计算量（$C$）增长而变化。

### 5.1 Kaplan-McCandlish 定律（OpenAI, 2020）

通过训练（Training）不同规模的模型发现：

- **损失与参数量的关系**：$L(N) \propto N^{-0.073}$
- **损失与数据量的关系**：$L(D) \propto D^{-0.095}$
- **最优参数-数据分配**：在固定计算预算 $C$ 下：
  $$N_{opt} \propto C^{0.73}, \quad D_{opt} \propto C^{0.27}$$

**含义**：计算预算增加时，大部分增量应用于扩大模型规模（而非数据量）。

### 5.2 Chinchilla 定律（DeepMind, 2022）

重新分析后发现之前的实验受限于训练（Training）不充分（模型大但数据少）。在充分训练（Training）条件下：

$$N_{opt} \propto C^{0.46}, \quad D_{opt} \propto C^{0.54}$$

**关键结论**：
1. 模型参数量与训练（Training）数据量应该**等比例增长**
2. 理想数据量 ≈ **20 × 参数量**（以 token 计）
3. 同等计算预算下，较小模型 + 更多数据 往往优于 较大模型 + 较少数据

**实例**：
- Chinchilla（70B 参数，1.4T tokens）vs Gopher（280B 参数，300B tokens）
- Chinchilla 计算量只有 Gopher 的 1/4，但性能全面超越

### 5.3 对工程实践的指导

1. **预训练（Pre-training）时不要欠训练（Training）**：确保数据量与模型规模匹配
2. **数据质量同样重要**：高质量数据可以弥补数量不足
3. **推理（Inference）成本考量**：同等性能下，更小的模型推理（Inference）更快、成本更低
4. **当前趋势**：高质量数据越来越稀缺，数据工程成为关键竞争点

---

## 6. Emergent Abilities（涌现能力 / Emergent Abilities）

涌现能力是指模型规模达到某个阈值后**突然获得**的能力，在小模型中不存在。

### 6.1 主要涌现能力

| 能力 | 描述 | 典型规模 |
|------|------|----------|
| **In-Context Learning（上下文学习）** | 通过 prompt 中的示例学习新任务，无需梯度更新 | ~1B-10B |
| **Chain-of-Thought（思维链）** | 给出逐步推理（Inference）过程后，复杂推理（Inference）能力显著提升 | ~100B |
| **Commonsense Reasoning（常识推理（Inference））** | 利用世界知识进行推理（Inference） | ~10B-100B |
| **Code Generation（代码生成）** | 理解并生成可执行代码 | ~10B+ |
| **Instruction Following（指令遵循）** | 理解并执行自然语言指令 | ~100B+（经 SFT 后） |

### 6.2 In-Context Learning 详解

**Zero-shot**：直接描述任务，不给示例
```
翻译以下句子到法语：
Hello world →
```

**One-shot**：给一个示例
```
翻译以下句子到法语：
Good morning → Bonjour
Hello world →
```

**Few-shot**：给多个示例（通常 3-5 个）

**原理假说**：
- 大模型在预训练（Pre-training）中见过大量任务-输出对，In-Context Learning 实际上是检索和组合已有知识
- 注意力机制（Attention Mechanism）使得模型可以利用 prompt 中的示例作为"工作记忆"

### 6.3 Chain-of-Thought（CoT）

在回答复杂推理（Inference）问题前，让模型先输出推理（Inference）过程：

```
问：一个农场有 35 只鸡和 20 只兔子，一共有多少条腿？
答：每只鸡有 2 条腿，35 只鸡有 35 × 2 = 70 条腿。
每只兔子有 4 条腿，20 只兔子有 20 × 4 = 80 条腿。
总共有 70 + 80 = 150 条腿。答案是 150。
```

**触发条件**：
- 模型规模足够大（通常 100B+）
- prompt 中包含 "Let's think step by step" 或类似引导
- 提供 few-shot 示例时包含推理（Inference）链

### 6.4 涌现的争议

- **定性跃变 vs 定量平滑**：有研究认为涌现只是评估指标选择造成的假象，用连续指标（如 token 编辑距离）看，能力是平滑提升的
- **无论如何**，从应用角度，大模型确实在某些规模点后变得"可用"

---

## 7. 非 Transformer  架构（Non-Transformer  Architectures）

虽然 Transformer  是主流，但新架构也在探索中：

### 7.1 状态空间（State Space）模型（SSM / Mamba）

**动机**：Transformer  的 Attention 计算复杂度（Computational Complexity）为 $O(n^2)$，长序列效率低。

**核心思想**：
- 将序列建模为状态转移过程
- 通过结构化状态空间（State Space）方程压缩历史信息
- 推理（Inference）时可并行化训练（Training），自回归（Autoregressive）推理（Inference）时复杂度为 $O(n)$

**代表模型**：Mamba、Mamba-2、Jamba（Mamba + Attention 混合）

### 7.2 TTT（Test-Time Training）

**核心思想**：在推理（Inference）时通过梯度更新调整模型参数，使模型能针对当前输入"即时学习"。

**挑战**：推理（Inference）时训练（Training）的计算开销；如何设计高效的更新机制。

---

## 8. 工业界实现对比（Industrial Architecture Comparison）

本节对比 OpenAI、Anthropic、Meta、Google、DeepSeek 等公司在 Transformer 架构上的**工程选择差异**。理解这些选择是成为高级工程师的关键——相同的理论在不同公司会演化出不同的实现路径。

### 8.1 各公司核心架构决策对比

| 设计维度 | OpenAI (GPT 系列) | Anthropic (Claude) | Meta (LLaMA) | Google (Gemini/PaLM) | DeepSeek |
|---------|-------------------|-------------------|--------------|---------------------|----------|
| **基础架构** | Decoder-only Dense (推测 GPT-4 为 MoE) | Decoder-only | Decoder-only | Decoder-only + 多模态原生 | Decoder-only MoE |
| **位置编码** | 早期: 可学习 PE<br>GPT-3: 改进的 ALiBi<br>GPT-4: 未公开 | RoPE + 长上下文扩展 | RoPE (LLaMA 2/3) | RoPE (Gemini) | RoPE |
| **归一化** | Pre-LN | Pre-LN + 特定初始化 | Pre-RMSNorm | Pre-LN | Pre-RMSNorm |
| **激活函数** | GELU (GPT-3) | SwiGLU (Claude 2+) | SwiGLU (LLaMA 2/3) | SwiGLU | SwiGLU |
| **注意力变体** | 标准 MHA / 推测 GQA | GQA (Claude 3) | GQA (LLaMA 3) | MQA/GQA | MLA (Multi-head Latent Attention) |
| **上下文长度** | GPT-3: 2K → GPT-4: 8K/32K → GPT-4o: 128K | Claude 2: 100K → Claude 3: 200K | LLaMA 2: 4K → LLaMA 3: 8K/128K | Gemini 1.5 Pro: 1M-10M token | DeepSeek-V2: 128K |
| **模型规模** | GPT-3: 175B<br>GPT-4: ~1.8T (推测) | Claude 3 Opus: ~? (未公开) | LLaMA 3: 8B/70B/405B | Gemini Ultra: 未公开 | DeepSeek-V3: 671B (37B 激活) |
| **关键创新** | Scale 验证涌现<br>RLHF 普及 | Constitutional AI<br>Computer Use | 开源高性能基座<br>LLaMA 3 数据质量 | 原生多模态架构<br>TPU 训练优化 | MLA 显存优化<br>无辅助损失 MoE<br>FP8 训练 |

### 8.2 OpenAI 的架构演进路径

#### GPT-3 (2020) → GPT-3.5 (2022)

**核心工程决策**：
- **Dense 架构的极限探索**：证明 175B 参数 Dense Decoder-only 模型在足够数据下可涌现 In-Context Learning
- **ALiBi 变体**：GPT-3 使用修改版的位置编码，通过调整 Attention 分数的偏置项来外推更长序列
- **训练数据清洗**：从 Common Crawl 的 45TB 原始数据清洗出 570GB 高质量数据（约 300B tokens），质量比数量更重要

#### GPT-4 (2023) 的推测架构

OpenAI 未公开 GPT-4 架构细节，但业界通过 API 行为分析和论文线索推测：

**MoE 架构证据**：
- API 的 token 定价与 GPT-3.5 相比并非线性增长（1.8T 参数的 Dense 模型推理成本将 prohibitive）
- 某些任务输出风格在不同调用间有差异，推测是不同 expert 组合导致
- 估计：~1.8T 总参数，~200B 激活参数，16 个 experts，Top-2 路由

**工程权衡**：
```
Dense 模型 (GPT-3):
  优点: 实现简单，确定性输出
  缺点: 推理成本 ∝ 总参数量

MoE 模型 (GPT-4 推测):
  优点: 推理成本 ∝ 激活参数量 (1/9)
  缺点: 负载均衡复杂，All-to-All 通信开销，专家共置约束
```

#### GPT-4o (2024) 的多模态原生架构

**关键突破**：端到端多模态训练（非拼接式）
- 传统方案：分别训练视觉 Encoder + 文本 Decoder，后期拼接
- GPT-4o 方案：**统一 Transformer 处理文本、音频、图像 token**
- 音频直接输出：绕过文本中间层，降低语音对话延迟（平均 320ms 响应）

**对工程师的启示**：
1. 延迟敏感场景下，端到端架构优于流水线
2. 多模态融合发生在早期层比晚期层更自然
3. 训练数据需要精确时间对齐（音频帧 ↔ 文本 token）

### 8.3 Anthropic 的 Constitutional AI 与架构选择

#### Claude 系列的技术特色

**1. 更长的有效上下文**
- Claude 2 (2023)：100K 上下文，采用 **高效的 RoPE 外推技术**
- Claude 3 (2024)：200K 上下文，引入 **提示缓存 (Prompt Caching)** 降低长上下文成本
- 实现细节：通过对 RoPE 的 base 频率进行微调（NTK-aware 扩展），在训练长度外实现平滑外推

**2. Constitutional AI (CAI) 的训练流程**

与 OpenAI 的 RLHF 不同，Anthropic 开发了**无需人类标注的反馈机制**：

```
Standard RLHF:
  1. 收集人类偏好数据（ expensive，难以规模化）
  2. 训练 RM (Reward Model)
  3. PPO 优化

Constitutional AI:
  1. 定义一组原则（Constitution）如"选择最诚实的回答"
  2. 模型自我批评：用 SFT 模型生成回答，再用同一模型根据原则评估和改进
  3. 从自我改进的数据训练偏好模型
  4. RL from AI Feedback (RLAIF)
```

**工程优势**：
- 反馈循环可自动化，迭代成本远低于人类标注
- 原则可精确定义和修改（如针对特定行业的合规要求）
- Claude 3.5 Sonnet 在编码任务上的突出表现部分归因于 CAI 对"有用性"的精细控制

**3. 安全与能力的平衡**

Anthropic 在 Claude 3 中引入了 **多层安全分类器**：
- 输入层：检测恶意 prompt（jailbreak、prompt injection）
- 模型层：通过训练使模型本身拒绝有害请求
- 输出层：后处理过滤器确保输出安全

这种" defense in depth"（纵深防御）架构是当前工业界的标准实践。

### 8.4 Meta LLaMA：开源基座的设计哲学

#### LLaMA 1/2/3 的架构演进

| 版本 | 关键改进 | 工业界影响 |
|------|---------|-----------|
| LLaMA 1 (2023) | 证明小模型(7B/13B/33B/65B)+大数据(1.4T tokens)可匹敌大模型 | 引爆开源微调生态 |
| LLaMA 2 (2023) | 上下文从 2K → 4K，引入 GQA，发布 Chat 版本 | 企业私有化部署首选 |
| LLaMA 3 (2024) | 15T tokens 训练数据，8B/70B/405B 三档，原生多语言 | 开源 SOTA，性能接近闭源模型 |

#### LLaMA 3 的核心工程选择

**1. 数据质量 > 数据数量**
- 15T tokens 的筛选流程：
  - 原始网页数据 → 去重(MinHash) → 质量评分(分类器) → 退火阶段高质量数据上采样
  - 代码比例显著提升（从 LLaMA 2 的 ~8% 提升到 ~15%）
  - 多语言数据占比增加，非英语比例从 ~5% 提升到 ~30%

**2. GQA (Grouped-Query Attention) 的引入**
- LLaMA 2 70B 和 LLaMA 3 全系列采用 GQA
- 动机：标准 MHA 的 KV Cache 随层数和 head 数线性增长，推理显存压力大
- GQA 将多个 query head 共享同一组 K/V，减少 KV Cache 30-50%

**3. 训练稳定性工程**
- 405B 模型训练使用了 **16K H100 GPU**，遇到的核心挑战：
  - **Loss spike 恢复**：大规模训练中偶尔出现损失飙升，需要自动检测和回滚机制
  - **硬件故障容忍**：平均每小时有 GPU 故障，需要 checkpoint 和快速恢复
  - **数值稳定性**：在 FP16/BF16 混合精度下保持训练稳定

### 8.5 Google：原生多模态与 TPU 生态

#### PaLM (2022) 的训练工程

**Pathways 系统**：
- Google 专为 PaLM 开发的分布式训练框架
- 支持 **跨 TPU Pod 的异步分布式训练**
- 关键创新：单一控制器管理数千个 TPU，故障时自动迁移计算

**架构选择**：
- 使用标准 MHA（而非 GQA），TPU 的高内存带宽缓解了 KV Cache 压力
- SwiGLU 激活函数 + RoPE 位置编码
- 540B 参数，6144 TPU v4 芯片训练

#### Gemini 的多模态原生架构

**统一多模态设计**：
- 从预训练阶段就混合文本、图像、音频、视频数据
- **任意模态到任意模态**：文本→图像、图像→文本、视频→文本 均通过同一模型处理
- Gemini 1.5 Pro 的 1M-10M token 上下文：通过 **高效的注意力机制优化**（推测为局部+全局注意力混合）

**对工程师的启示**：
- TPU 与 GPU 的架构选择会影响模型设计（TPU 高带宽 → 可用标准 MHA；GPU 带宽受限 → 需要 GQA/MLA）
- 多模态训练的数据配比是关键超参数（Google 论文中不同模态 token 的混合比例经过大量实验）

### 8.6 DeepSeek：效率优先的中国方案

DeepSeek 系列代表了**在硬件约束下最大化效率**的工程路线。

#### DeepSeek-V2 的 MLA (Multi-head Latent Attention)

**问题**：标准 GQA 虽然减少了 KV head 数量，但每个 head 的维度仍然较大，KV Cache 压缩有限。

**MLA 的核心思想**：
- 将 Key 和 Value 压缩到一个 **低维 latent 向量** 中（如从 512 维压缩到 64 维）
- 推理时只缓存 latent 向量，而非完整 K/V
- 通过额外的上投影矩阵在 Attention 计算时恢复完整维度

```
标准 MHA KV Cache: (batch, num_heads, seq_len, head_dim)
  例如: (1, 32, 4096, 128) = 67M elements

GQA KV Cache: (batch, num_kv_heads, seq_len, head_dim)
  例如: (1, 8, 4096, 128) = 16.7M elements (减少 4x)

MLA KV Cache: (batch, seq_len, latent_dim)
  例如: (1, 4096, 64) = 262K elements (减少 256x!)
```

**效果**：DeepSeek-V2 在 128K 上下文下，显存占用与 4K 上下文的普通模型相当。

#### DeepSeek-V3 的 FP8 训练

**工程突破**：
- 首次在超大规模模型（671B 参数）训练中**全量使用 FP8 精度**
- 传统做法：FP8 仅用于矩阵乘法中间结果，主权重仍存 FP16/BF16
- DeepSeek-V3：主权重、优化器状态、激活值全面 FP8
- 配合细粒度量化（per-1x128 tile 的 scaling factor），精度损失可控

**对行业的影响**：
- 证明 FP8 已可支撑生产级训练，下一代 GPU（Blackwell）将全面转向 FP8
- 训练成本降低约 50%，使中小团队也能训练大模型

### 8.7 工程决策的底层逻辑

#### 为什么不同公司做出不同选择？

| 约束条件 | 典型公司 | 架构倾向 |
|---------|---------|---------|
| **算力充裕，追求极致性能** | OpenAI, Google | 大 Dense/MoE，标准 MHA |
| **推理成本敏感，服务 C 端** | Anthropic, Meta | GQA，量化友好设计 |
| **硬件受限，效率优先** | DeepSeek | MLA，FP8，极致通信优化 |
| **开源生态，社区适配** | Meta | 简单架构，广泛兼容 |

#### 位置编码选择的工业实践

```
RoPE 的采用路径：
  - LLaMA 1/2/3 → 成为开源标准
  - Claude → 改进版 RoPE 支持超长上下文
  - Gemini → RoPE 基础 + 局部注意力扩展
  - GPT-4 → 未公开（推测为 ALiBi 或 RoPE 变体）

选择逻辑：
  - RoPE 的相对位置特性使长上下文外推更自然
  - 外推技术（NTK-aware, YaRN）成熟，社区支持好
  - ALiBi 的显存优势在长序列下不再显著
```

#### 激活函数演进

```
ReLU (原始 Transformer) → GELU (GPT-2/3) → SwiGLU (LLaMA/Claude/Gemini)

SwiGLU 成为工业标准的原因：
  1. GLU 门控机制增加表达能力
  2. Swish 平滑激活优于 ReLU 的硬阈值
  3. 经验上在所有评测集上稳定优于 GELU
  4. 计算开销增加约 50%，但效果提升值得
```

### 8.8 面试高频考点：工业界架构

1. **为什么 LLaMA 选择 RMSNorm 而非 LayerNorm？**
   > 答：在深层 Transformer 中残差连接的累加效应使输出均值天然接近 0，去均值收益有限。RMSNorm 减少一次 reduce 操作，参数量减半，速度和效果均相当。

2. **GQA vs MHA 的取舍？**
   > 答：GQA 减少 KV Cache，适合长上下文和高并发推理。代价是轻微性能下降（LLaMA 论文中 <1%）。当显存/带宽是瓶颈时选 GQA，追求极致性能选 MHA。

3. **GPT-4 为什么推测是 MoE？**
   > 答：1.8T Dense 模型推理成本 prohibitive（需 ~20× GPT-3.5 算力）；API 定价和延迟不线性；输出风格有轻微差异（不同 expert 组合）。

4. **DeepSeek MLA 的核心创新？**
   > 答：将 KV 压缩到低维 latent 空间缓存，推理时通过投影恢复。相比 GQA 进一步减少 KV Cache 数十倍，使超长上下文（128K+）的推理成本大幅降低。

5. **Constitutional AI 相比 RLHF 的优势？**
   > 答：RLHF 依赖昂贵的人类偏好标注，难以迭代；CAI 通过预定义原则和模型自我批评实现自动化反馈循环，迭代更快，原则可精确控制。

---

## 学习路径建议

1. **先彻底理解 Self-Attention**：手写一遍 Attention 计算流程
2. **对比三种架构**：理解为什么 Decoder-only 成为主流
3. ** Scaling Laws 是工程决策的基础**：指导模型和数据量的配比
4. **涌现能力解释了大模型的价值**：也是为什么 "scale" 是核心策略
5. **深入一家公司的技术路线**：阅读 LLaMA 3 或 DeepSeek-V3 的技术报告，理解工程权衡
