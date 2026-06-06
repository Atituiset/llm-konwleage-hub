# 第四阶段：深度学习核心

> **资料来源**：综合《深度学习入门：基于Python的理论与实现》《深度学习小书》《李宏毅深度学习教程》《动手学深度学习》《麻省理工：深入理解深度学习》《Python神经网络编程》
> **适合人群**：需要理解神经网络底层原理的读者
> **难度**：⭐⭐⭐（中等）

---

## 1. 神经网络基础

### 1.1 从感知机到多层感知机

**感知机（Perceptron）**：最简单的线性分类器

$$y = \sigma(w^T x + b)$$

其中 $\sigma$ 是阶跃函数，$w$ 是权重，$b$ 是偏置。

**多层感知机（MLP）**：多个全连接层堆叠

$$h^{(l)} = f(W^{(l)} h^{(l-1)} + b^{(l)})$$

- 输入层 → 隐藏层 → 输出层
- 隐藏层引入非线性，使网络可以拟合复杂函数
-  Universal Approximation Theorem：足够大的 MLP 可以逼近任意连续函数

### 1.2 激活函数

激活函数引入非线性，没有非线性，多层网络等价于单层：

**Sigmoid**：
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- 输出范围 (0, 1)，适合二分类输出层
- **缺点**：梯度饱和（|x| > 5 时梯度 ≈ 0），导致梯度消失

**Tanh**：
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
- 输出范围 (-1, 1)，均值为 0
- 仍面临梯度饱和问题

**ReLU（Rectified Linear Unit）**：
$$\text{ReLU}(x) = \max(0, x)$$
- **优点**：计算简单、不会饱和（正区间梯度为 1）、稀疏激活
- **缺点**：Dead ReLU 问题（负区间永远输出 0，梯度为 0，神经元"死亡"）

**Leaky ReLU**：
$$\text{LeakyReLU}(x) = \max(\alpha x, x), \quad \alpha = 0.01$$
- 解决 Dead ReLU，负区间有小梯度

**GELU（GPT 系列使用）**：
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$
- 平滑、处处可导
- Transformer 类模型的默认选择

**Swish / SiLU**：
$$\text{Swish}(x) = x \cdot \sigma(x)$$
- 自门控机制，DeepMind 发现效果优于 ReLU
- LLaMA 的 SwiGLU 基于此

**选择建议**：
- 隐藏层：ReLU / GELU / Swish
- 输出层（二分类）：Sigmoid
- 输出层（多分类）：Softmax
- 现代 LLM：GELU / SwiGLU

### 1.3 损失函数

**均方误差（MSE）**：
$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- 用于回归任务

**交叉熵损失（Cross-Entropy）**：
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$
- 用于分类任务
- 语言模型本质上也是分类（预测词表中的下一个 token），因此使用交叉熵

**为什么语言模型用交叉熵？**
- 将词表中的每个词视为一个类别
- 模型输出词表大小的概率分布
- 目标是最小化正确词的负对数概率

---

## 2. 反向传播与优化

### 2.1 反向传播（Backpropagation）

反向传播是训练神经网络的核心算法，基于**链式法则**高效计算梯度。

**核心思想**：
1. 前向传播：计算预测输出和损失
2. 反向传播：从输出层到输入层，逐层计算梯度
3. 参数更新：用梯度下降更新权重

**链式法则示例**：

对于复合函数 $z = f(g(x))$：
$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial g} \cdot \frac{\partial g}{\partial x}$$

在神经网络中：
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial h^{(l)}} \cdot \frac{\partial h^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

其中 $z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$ 是第 $l$ 层的线性变换。

**计算图视角**：
- 每个运算是一个节点
- 前向：计算每个节点的输出
- 反向：计算每个节点对损失的梯度

### 2.2 梯度下降变体

**批量梯度下降（BGD）**：
$$W_{t+1} = W_t - \eta \nabla_W \mathcal{L}(W_t; \text{全部数据})$$
- 每次用全部数据计算梯度，稳定但慢

**随机梯度下降（SGD）**：
$$W_{t+1} = W_t - \eta \nabla_W \mathcal{L}(W_t; \text{一个样本})$$
- 每次用一个样本，快但噪声大

**小批量梯度下降（Mini-batch SGD）**：
$$W_{t+1} = W_t - \eta \nabla_W \mathcal{L}(W_t; \text{batch})$$
- 折中方案，实际中最常用
- batch size 通常为 32、64、128、256

**带动量的 SGD**：
$$v_t = \beta v_{t-1} + \nabla_W \mathcal{L}$$
$$W_{t+1} = W_t - \eta v_t$$
- 动量积累历史梯度方向，加速收敛，减少震荡

### 2.3 Adam 优化器

Adam（Adaptive Moment Estimation）结合了动量和自适应学习率：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(一阶矩估计，梯度均值)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(二阶矩估计，梯度方差)}$$

偏差修正：
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

参数更新：
$$W_{t+1} = W_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**默认超参数**：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$

**优势**：
- 自适应学习率：梯度大的参数学习率自动减小
- 适合大多数场景，默认选择

**AdamW**：
- 将权重衰减（L2 正则）与梯度更新解耦
- 现在训练 Transformer 的标准选择

### 2.4 学习率调度

**Warmup + Cosine Decay**（LLM 训练标准）：

```
学习率
  │    ╭────╮
  │   ╱      ╲_____
  │  ╱        \
  │ ╱          \
  │╱            \
  └──────────────→ 步数
    warmup      cosine decay
```

- **Warmup**：前几步线性增大学习率，防止早期更新过大破坏预训练权重
- **Cosine Decay**：按余弦曲线衰减到最小值
- **为什么重要**：大模型训练不稳定，好的调度策略决定能否收敛

---

## 3. 核心架构

### 3.1 卷积神经网络（CNN）

**卷积操作**：用滑动窗口（卷积核）提取局部特征

$$Y[i, j] = \sum_{m} \sum_{n} X[i+m, j+n] \cdot K[m, n]$$

**核心概念**：
- **卷积核（Kernel）**：可学习的特征检测器（边缘、纹理等）
- **步幅（Stride）**：卷积核移动的步长
- **填充（Padding）**：在输入边缘补零，控制输出尺寸
- **池化（Pooling）**：降采样，减少参数量，增加平移不变性

**经典架构**：
- LeNet（1998）：5 层，手写数字识别
- AlexNet（2012）：8 层，ReLU + Dropout，深度学习复兴
- VGG（2014）：16-19 层，小卷积核 (3×3) 堆叠
- ResNet（2015）：残差连接，可训练 152+ 层

**与大模型的关系**：
- Vision Transformer 正在取代 CNN 成为视觉主流
- 但 CNN 的局部性、平移不变性思想仍有价值
- 多模态模型（如 GPT-4V）中 CNN 仍用于视觉编码

### 3.2 循环神经网络（RNN）

**核心思想**：维护隐藏状态，处理序列数据

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

**问题**：
- **梯度消失/爆炸**：长序列时梯度呈指数衰减或增长
- **长期依赖困难**：难以捕捉远距离的关联

**LSTM（Long Short-Term Memory）**：
通过门控机制解决长期依赖：
- **遗忘门**：决定丢弃多少历史信息
- **输入门**：决定加入多少新信息
- **输出门**：决定输出多少信息

**GRU（Gated Recurrent Unit）**：
LSTM 的简化版，合并遗忘门和输入门，参数量更少。

**与大模型的关系**：
- RNN/LSTM/GRU 是 NLP 的前 Transformer 时代主流
- Transformer 的 Self-Attention 解决了 RNN 的并行化和长依赖问题
- 理解 RNN 有助于理解序列建模的演进

### 3.3 Transformer（详见第六阶段）

Transformer 是深度学习最重要的架构创新：
- **Self-Attention**：直接建模任意位置间的关系
- **并行化**：不像 RNN 需要顺序计算
- **可扩展性**：堆叠更多层、更多头即可扩大模型

所有现代大语言模型（GPT、LLaMA、Claude 等）都是基于 Transformer Decoder。

---

## 4. 训练技巧

### 4.1 正则化

**目标**：防止过拟合，提高泛化能力

**L2 正则化（权重衰减）**：
$$\mathcal{L}' = \mathcal{L} + \lambda \sum_{w} w^2$$
- 惩罚大权重，使权重趋于小而分散
- 等价于权重的先验分布为高斯分布（MAP 估计）

**Dropout**：
- 训练时随机丢弃（置零）一部分神经元（通常 0.2-0.5）
- 测试时所有神经元参与，输出按比例缩放
- **原理**：相当于训练多个子网络的集成，强制冗余表示
- **现代 LLM 中的使用**：Attention 和 FFN 后常用 dropout=0.1

**注意**：大模型预训练时 dropout 较少使用（数据量极大，过拟合风险低），微调时可能使用。

### 4.2 归一化

**Batch Normalization（BN）**：
$$\text{BN}(x) = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$
- 对一个 batch 的数据，按特征维度归一化
- 适合 CNN，不适合序列长度变化的 NLP

**Layer Normalization（LN）**：
$$\text{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
- 对每个样本，按特征维度归一化
- **Transformer 的标准选择**，不依赖 batch size

**RMSNorm**（LLaMA 使用）：
- 去除均值中心化，只按 RMS 缩放
- 计算更简单，效果相当

### 4.3 残差连接（Residual Connection）

$$\text{Output} = \text{Layer}(x) + x$$

**解决的问题**：
- 深层网络的梯度消失/爆炸
- 退化问题：深层网络比浅层网络训练误差更大（不是过拟合，是优化困难）

**原理**：
- 残差连接提供梯度高速公路，梯度可以直接回传
- 至少保证深层网络不差于浅层网络（学不会就恒等映射）
- Transformer 中每层都有两次残差连接（Attention 后、FFN 后）

### 4.4 权重初始化

**Xavier / Glorot 初始化**：
$$W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]$$
- 保持前向和反向传播时激活值和梯度的方差稳定

**He 初始化**（ReLU 专用）：
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$
- 考虑 ReLU 截断负半区的影响

**现代实践**：
- Transformer 通常使用较小的初始化标准差（如 0.02）
-  embedding 层和输出层可能需要特殊处理
- 预训练模型加载时不需要关心初始化

### 4.5 早停（Early Stopping）

监控验证集性能，当验证损失不再下降时停止训练：

```python
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(max_epochs):
    train(model, train_loader)
    val_loss = validate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

## 5. 大模型训练的特殊考量

### 5.1 梯度累积

显存不足时，用多步小 batch 模拟大 batch：

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**有效 batch size** = per_device_batch × accumulation_steps × num_gpus

### 5.2 混合精度训练

FP16（半精度）减少显存和加速，但需处理数值稳定性：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()

    with autocast():  # 自动在 FP16 和 FP32 间切换
        loss = model(x, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**BF16 vs FP16**：
- BF16：范围与 FP32 相同，精度较低，更稳定
- FP16：范围和精度都较低，需要 loss scaling
- A100+ 支持 BF16，推荐优先使用

### 5.3 梯度裁剪

限制梯度范数，防止 loss spike：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**大模型训练必备**：Transformer 训练经常出现 loss 突然增大，梯度裁剪是标准操作。

---

## 6. 面试高频考点

1. **反向传播为什么高效？**
   - 利用链式法则，复用中间计算结果
   - 复杂度从指数级降到线性级

2. **ReLU 为什么比 Sigmoid 好？**
   - 不会梯度饱和（正区间梯度恒为 1）
   - 计算简单，没有指数运算

3. **BatchNorm 和 LayerNorm 的区别？**
   - BN：跨 batch 对同一特征归一化（适合 CNN）
   - LN：对单个样本所有特征归一化（适合序列/Transformer）

4. **Dropout 训练和测试的区别？**
   - 训练：随机丢弃，输出缩放 1/(1-p)
   - 测试：全部使用，不缩放（或训练时 inverted dropout）

5. **为什么 Transformer 比 RNN 好？**
   - 并行计算：RNN 必须顺序，Transformer 可并行
   - 长距离依赖：Attention 直接连接任意位置
   - 可扩展性：堆叠更多层即可扩大

6. **残差连接为什么有效？**
   - 梯度高速公路，缓解梯度消失
   - 至少保证深层网络不差于浅层

---

## 学习路径建议

1. **手写 MLP + 反向传播**：建立最直观的理解
2. **用 PyTorch 实现 CNN 和 RNN**：理解框架如何抽象
3. **可视化训练过程**：观察 loss 曲线、梯度分布、激活值分布
4. **调参实验**：修改学习率、batch size、初始化，观察影响
5. **阅读 Transformer 论文**：Attention Is All You Need（2017）
