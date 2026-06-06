# 李宏毅深度学习教程

> **资料来源**：Datawhale 整理《李宏毅深度学习教程 LeeDL Tutorial》
> **适合人群**：喜欢中文讲解、需要视频辅助的学习者
> **难度**：⭐⭐⭐（中等）

---

## 1. 为什么推荐李宏毅

李宏毅是台湾大学教授，其《机器学习》课程是**中文世界最受欢迎的深度学习入门课**：

- **幽默风趣**：用宝可梦等有趣例子讲解复杂概念
- **中文讲解**：母语学习，效率更高
- **内容全面**：覆盖深度学习的绝大多数领域
- **公式详细**：涉及公式的知识点都给出详细推导

---

## 2. 核心内容精讲

### 2.1 回归与分类基础

**回归 vs 分类**：

```mermaid
graph LR
    A[机器学习] --> B[回归]
    A --> C[分类]

    B --> B1[输出连续值<br/>房价预测<br/>股票预测]
    C --> C1[输出离散类别<br/>图像分类<br/>文本分类]
```

**线性回归的梯度下降可视化**：

```
损失函数等高线图：
        w
        ↑
        │    ╭────╮
        │   ╱      ╲
        │  ╱   ●    ╲
        │ ╱   最优   ╲
        │╱            ╲
        └──────────────→ b
```

**分类的关键：Softmax**

为什么分类不直接用回归？
- 回归输出范围无界，分类需要概率（和为 1）
- Softmax 将任意实数转换为概率分布

### 2.2 反向传播详解

李宏毅的讲解特色：**用计算图和链式法则逐步推导**

```mermaid
graph TB
    A[输入 x] --> B[线性变换 z=wx+b]
    B --> C[激活 a=sigmoid(z)]
    C --> D[输出 y]
    D --> E[损失 L]

    E -.->|∂L/∂y| D
    D -.->|∂y/∂a| C
    C -.->|∂a/∂z| B
    B -.->|∂z/∂w| A
```

**核心步骤**：
1. 前向传播：计算每层的输出
2. 反向传播：从损失开始，逐层计算梯度
3. 参数更新：用梯度下降更新权重

### 2.3 卷积神经网络（CNN）

**李宏毅的 CNN 讲解特色**：

```mermaid
graph LR
    A[图像输入] --> B[卷积层<br/>提取局部特征]
    B --> C[池化层<br/>降维]
    C --> D[卷积层]
    D --> E[池化层]
    E --> F[全连接层<br/>分类]
```

**关键洞察**：
- 卷积 = 滤波器在图像上滑动检测模式
- 浅层检测边缘/颜色，深层检测复杂模式（如眼睛、轮胎）
- 权值共享大幅减少参数量

**经典架构对比**：

| 架构 | 特点 | 参数量 |
|------|------|--------|
| LeNet (1998) | 5层，手写数字 | 60K |
| AlexNet (2012) | 8层，ReLU+Dropout | 60M |
| VGGNet (2014) | 16-19层，3×3小卷积 | 138M |
| ResNet (2015) | 残差连接，可训练 152+ 层 | 60M |

### 2.4 循环神经网络（RNN）

**RNN 的核心思想**：引入"记忆"，处理序列数据

```mermaid
graph LR
    A[x₁] --> B[h₁]
    B --> C[h₂]
    C --> D[h₃]
    D --> E[h₄]

    B --> B1[y₁]
    C --> C1[y₂]
    D --> D1[y₃]
    E --> E1[y₄]

    style B fill:#3498db,color:#fff
    style C fill:#3498db,color:#fff
    style D fill:#3498db,color:#fff
    style E fill:#3498db,color:#fff
```

**公式**：
$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = g(W_{hy} h_t + b_y)$$

**RNN 的问题**：

```mermaid
graph TB
    A[RNN问题] --> B[梯度消失]
    A --> C[梯度爆炸]
    A --> D[长程依赖]

    B --> B1[梯度呈指数衰减<br/>前面层几乎学不到]
    C --> C1[梯度呈指数增长<br/>参数更新不稳定]
    D --> D1[远距离信息<br/>难以传递]
```

**解决方案：LSTM**

LSTM 引入"门控机制"：
- **遗忘门**：决定丢弃多少旧信息
- **输入门**：决定加入多少新信息
- **输出门**：决定输出什么

```python
# LSTM 核心逻辑（简化）
def lstm_step(x, h_prev, c_prev):
    # 遗忘门
    f = sigmoid(W_f @ [h_prev, x] + b_f)
    # 输入门
    i = sigmoid(W_i @ [h_prev, x] + b_i)
    # 候选记忆
    c_tilde = tanh(W_c @ [h_prev, x] + b_c)
    # 更新记忆
    c = f * c_prev + i * c_tilde
    # 输出门
    o = sigmoid(W_o @ [h_prev, x] + b_o)
    # 隐藏状态
    h = o * tanh(c)
    return h, c
```

### 2.5 Transformer（重点）

李宏毅课程中 Transformer 是**最核心章节**，直接衔接大模型。

**Self-Attention 的直觉**：

```
句子："我 喜欢 深度 学习"

每个词都要看其他所有词，决定"关注"谁：

"我" → 关注"喜欢"（主谓关系）
"深度" → 关注"学习"（修饰关系）
```

**Q/K/V 的解释**：

```mermaid
graph LR
    A[输入X] --> B[Query<br/>我要查什么]
    A --> C[Key<br/>我有什么]
    A --> D[Value<br/>实际内容]

    B --> E[注意力分数<br/>Q·K^T]
    C --> E
    E --> F[加权求和<br/>Softmax × V]
    D --> F
    F --> G[输出]
```

**详细公式**：

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

**除以 $\sqrt{d_k}$ 的原因**：防止点积过大导致 Softmax 梯度消失。

**Multi-Head Attention**：

```mermaid
graph TB
    A[输入] --> B1[Head1]
    A --> B2[Head2]
    A --> B3[Head3]
    A --> B4[Head4]

    B1 --> C[Concat]
    B2 --> C
    B3 --> C
    B4 --> C

    C --> D[Linear]
    D --> E[输出]
```

- 多个头学习不同的"关注模式"
- 如：一个头学语法关系，一个头学语义关系

---

## 3. 生成模型

### 3.1 VAE（变分自编码器）

```mermaid
graph LR
    A[输入x] --> B[编码器]
    B --> C[潜在变量z<br/>分布参数]
    C --> D[采样]
    D --> E[解码器]
    E --> F[重建x']
```

**损失函数**：
$$L = \underbrace{\|x - x'\|^2}_{重建损失} + \underbrace{D_{KL}(q(z|x) \| p(z))}_{KL散度}$$

### 3.2 GAN（生成对抗网络）

```mermaid
graph LR
    A[噪声z] --> B[生成器G]
    B --> C[假样本]
    D[真实样本] --> E[判别器D]
    C --> E
    E --> F[真假判断]

    F -.->|反馈| B
    F -.->|反馈| E
```

**博弈过程**：
- G 试图骗过 D
- D 试图区分真假
- 达到纳什均衡时，G 生成以假乱真的样本

### 3.3 Diffusion Model（扩散模型）

**核心思想**：逐步加噪再逐步去噪

```mermaid
graph LR
    A[清晰图像x₀] --> B[加噪x₁]
    B --> C[加噪x₂]
    C --> D[...]
    D --> E[纯噪声x_T]

    E --> F[去噪x_{T-1}]
    F --> G[去噪x_{T-2}]
    G --> H[...]
    H --> I[清晰图像]
```

**与大模型的关联**：
- DALL-E、Stable Diffusion 的底层技术
- 扩散模型 + Transformer 是当前图像生成主流

---

## 4. 自监督学习与大模型

### 4.1 BERT（Encoder）

**训练任务**：

```mermaid
graph TB
    A[输入句子] --> B[Mask 15%的token]
    B --> C[Transformer Encoder]
    C --> D[预测被mask的token]

    A --> E[两个句子]
    E --> F[判断是否是连续句子]
```

**特点**：
- 双向编码：每个词都能看到左右上下文
- 适合：文本理解任务（分类、NER、问答）

### 4.2 GPT（Decoder）

**训练任务**：自回归预测下一个 token

```mermaid
graph LR
    A['我 喜欢'] --> B[预测'深度']
    A --> C[预测下一个token]
    B --> D['我 喜欢 深度']
    D --> E[预测'学习']
```

**特点**：
- 单向生成：只能看左边上下文
- 适合：文本生成任务

### 4.3 从 BERT/GPT 到大模型

```mermaid
graph LR
    A[BERT/GPT<br/>1亿参数] --> B[GPT-2<br/>15亿]
    B --> C[GPT-3<br/>1750亿]
    C --> D[GPT-4<br/>估计万亿级]

    A --> A1[预训练+微调]
    B --> B1[Zero-shot能力]
    C --> C1[Few-shot学习]
    D --> D1[涌现能力]
```

---

## 5. 学习路径建议

```mermaid
graph LR
    A[李宏毅课程] --> B[重点章节]
    A --> C[可以跳过]

    B --> B1[反向传播<br/>必须彻底理解]
    B --> B2[Transformer<br/>大模型基础]
    B --> B3[自监督学习<br/>BERT/GPT]

    C --> C1[强化学习<br/>如需再学]
    C --> C2[旧版RNN<br/>了解即可]
```

**推荐配合资料**：

| 李宏毅章节 | 配合资料 | 目的 |
|-----------|----------|------|
| 神经网络基础 | 《深度学习入门：基于Python》 | 代码实践 |
| CNN | 经典论文 + PyTorch 练习 | 图像任务 |
| Transformer | 《大模型基础》 | 衔接大模型 |
| BERT/GPT | HuggingFace 文档 | 实际使用 |
