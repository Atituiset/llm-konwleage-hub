# 动手学深度学习

> **资料来源**：《动手学深度学习》（Dive into Deep Learning，李沐 等著，d2l.ai）
> **适合人群**：需要系统性代码实践的学习者
> **难度**：⭐⭐⭐（中等）

---

## 1. 多层感知机（MLP）

### 1.1 从线性回归到 MLP

**线性回归**假设输出是输入的线性组合：

$$\hat{y} = W^T x + b$$

但现实世界大多是非线性的。**MLP** 通过堆叠多个全连接层并引入非线性激活函数来解决：

$$h^{(1)} = \sigma(W^{(1)} x + b^{(1)})$$
$$h^{(2)} = \sigma(W^{(2)} h^{(1)} + b^{(2)})$$
$$\hat{y} = W^{(3)} h^{(2)} + b^{(3)}$$

其中 $\sigma$ 是非线性激活函数（如 ReLU）。没有激活函数，多层网络等价于单层线性变换。

### 1.2 MLP 的 PyTorch 实现

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)           # (B, C, H, W) → (B, C*H*W)
        x = self.fc1(x)               # (B, input_dim) → (B, hidden_dim)
        x = self.relu(x)              # 非线性激活
        x = self.dropout(x)           # 正则化
        x = self.fc2(x)               # (B, hidden_dim) → (B, num_classes)
        return x

# 用于 Fashion-MNIST (28×28=784 像素，10 类)
model = MLP(input_dim=784, hidden_dim=256, num_classes=10)
```

### 1.3 激活函数对比

| 激活函数 | 公式 | 优点 | 缺点 | 使用场景 |
|---------|------|------|------|---------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 输出 (0,1)，可解释 | 梯度饱和 | 二分类输出层 |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 均值为 0 | 仍梯度饱和 | 早期 RNN |
| ReLU | $\max(0, x)$ | 计算快、不饱和 | Dead ReLU | 隐藏层默认 |
| Leaky ReLU | $\max(\alpha x, x)$ | 解决 Dead ReLU | 需调 $\alpha$ | ReLU 替代 |
| GELU | $x \cdot \Phi(x)$ | 平滑、处处可导 | 计算稍慢 | **Transformer 默认** |
| Swish/SiLU | $x \cdot \sigma(x)$ | 自门控 | 计算稍慢 | LLaMA 使用 |

> **大模型提示**：GPT 系列用 GELU，LLaMA 用 SwiGLU（Swish + Gated Linear Unit）。

---

## 2. 卷积神经网络（CNN）

### 2.1 卷积操作原理

卷积用可学习的滤波器（卷积核）在输入上滑动，提取局部特征：

```
输入图像 (5×5)          卷积核 (3×3)         输出特征图 (3×3)
┌───┬───┬───┬───┬───┐    ┌───┬───┬───┐
│ 1 │ 1 │ 1 │ 0 │ 0 │    │ 1 │ 0 │ 1 │      ┌───┬───┬───┐
├───┼───┼───┼───┼───┤    ├───┼───┼───┤      │ 4 │ 3 │ 4 │
│ 0 │ 1 │ 1 │ 1 │ 0 │    │ 0 │ 1 │ 0 │  *   ├───┼───┼───┤
├───┼───┼───┼───┼───┤    ├───┼───┼───┤  =   │ 2 │ 4 │ 3 │
│ 0 │ 0 │ 1 │ 1 │ 1 │    │ 1 │ 0 │ 1 │      ├───┼───┼───┤
├───┼───┼───┼───┼───┤    └───┴───┴───┘      │ 2 │ 3 │ 4 │
│ 0 │ 0 │ 1 │ 1 │ 0 │                         └───┴───┴───┘
├───┼───┼───┼───┼───┤
│ 0 │ 1 │ 1 │ 0 │ 0 │
└───┴───┴───┴───┴───┘
```

**关键超参数**：
- **Kernel Size**：卷积核尺寸（常用 3×3）
- **Stride**：步幅，控制滑动间隔
- **Padding**：边缘填充，控制输出尺寸
- **Channels**：输入/输出通道数

### 2.2 经典 CNN 架构演进

#### LeNet（1998）— 开山之作

```python
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 6@28×28
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # 6@14×14
            nn.Conv2d(6, 16, kernel_size=5),             # 16@10×10
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # 16@5×5
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
```

**特点**：5 层，手写数字识别，使用 Sigmoid 和平均池化。

#### AlexNet（2012）— 深度学习复兴

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # 输入: 3×224×224
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # 96×55×55
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),       # 96×27×27
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # 256×27×27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),       # 256×13×13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),       # 256×6×6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**关键创新**：
- ReLU 替代 Sigmoid（训练快 6 倍）
- GPU 并行训练
- Dropout 正则化
- 数据增强

#### VGG（2014）— 小卷积核堆叠

```python
def vgg_block(num_convs, in_channels, out_channels):
    """VGG 块：num_convs 个 3×3 卷积 + MaxPool"""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG11(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            vgg_block(1, 3, 64),     # 64×112×112
            vgg_block(1, 64, 128),   # 128×56×56
            vgg_block(2, 128, 256),  # 256×28×28
            vgg_block(2, 256, 512),  # 512×14×14
            vgg_block(2, 512, 512),  # 512×7×7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        )
    def forward(self, x):
        return self.fc(self.conv(x))
```

**核心思想**：用多个 3×3 卷积替代大卷积核（2 个 3×3 感受野 = 1 个 5×5，参数更少，非线性更多）。

#### ResNet（2015）— 残差连接革命

**问题**：深层网络（>20 层）训练误差反而比浅层高——不是过拟合，是优化困难。

**解决方案**：残差连接（Skip Connection）

```python
class ResidualBlock(nn.Module):
    """ResNet 基础块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 下采样（当维度不匹配时）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # 残差连接！
        out = self.relu(out)
        return out
```

**为什么有效**：
1. **梯度高速公路**：梯度可直接通过 shortcut 回传，缓解梯度消失
2. **恒等映射保证**：如果某层学不到有用信息，可以退化为恒等映射（$F(x)=0$，输出=$x$）
3. **深层网络可训练**：ResNet-152、ResNet-1000+ 都能训练

> **大模型提示**：Transformer 中的残差连接直接继承自 ResNet。每个 Transformer 层有两次残差连接（Attention 后、FFN 后）。

### 2.3 CNN 架构对比表

| 模型 | 年份 | 层数 | 参数量 | 关键创新 | ImageNet Top-5 |
|------|------|------|--------|---------|----------------|
| LeNet | 1998 | 5 | 60K | 首个成功 CNN | - |
| AlexNet | 2012 | 8 | 60M | ReLU、Dropout、GPU | 84.7% |
| VGG16 | 2014 | 16 | 138M | 3×3 小卷积核堆叠 | 92.5% |
| ResNet-50 | 2015 | 50 | 25.6M | 残差连接 | 93.3% |
| ResNet-152 | 2015 | 152 | 60.2M | 更深的残差网络 | 93.8% |
| DenseNet | 2017 | 121 | 8M | 密集连接 | 93.6% |

---

## 3. 循环神经网络（RNN）

### 3.1 RNN 基础

RNN 维护隐藏状态，处理序列数据：

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

```python
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # x: (batch, input_size)
        # h_prev: (batch, hidden_size)
        h = torch.tanh(self.W_xh(x) + self.W_hh(h_prev))
        return h
```

**问题**：
- **梯度消失/爆炸**：长序列时梯度呈指数变化
- **长期依赖困难**：难以捕捉远距离关联

### 3.2 LSTM（长短期记忆网络）

通过门控机制解决长期依赖：

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 门控：输入门、遗忘门、输出门、候选记忆
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        # 拼接输入和上一时刻隐藏状态
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)

        # 分割四个门
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)   # 输入门：多少新信息进入
        f = torch.sigmoid(f)   # 遗忘门：多少旧信息保留
        g = torch.tanh(g)      # 候选记忆
        o = torch.sigmoid(o)   # 输出门：多少信息输出

        # 细胞状态更新
        c = f * c_prev + i * g    # 核心：加法更新，缓解梯度消失
        h = o * torch.tanh(c)     # 隐藏状态

        return h, c
```

**三个门的直觉**：
- **遗忘门**："昨天的新闻，今天还重要吗？"
- **输入门**："今天有什么新信息需要记住？"
- **输出门**："基于当前记忆，应该输出什么？"

### 3.3 GRU（门控循环单元）

LSTM 的简化版，合并遗忘门和输入门：

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.new_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)

        r = torch.sigmoid(self.reset_gate(combined))   # 重置门
        z = torch.sigmoid(self.update_gate(combined))  # 更新门

        # 候选隐藏状态
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.new_gate(combined_reset))

        # 更新隐藏状态
        h = (1 - z) * h_prev + z * h_tilde
        return h
```

| 特性 | RNN | LSTM | GRU |
|------|-----|------|-----|
| 参数量 | 少 | 多（4倍） | 中（3倍） |
| 训练速度 | 快 | 慢 | 中等 |
| 长期依赖 | 差 | 好 | 较好 |
| 计算效率 | 高 | 低 | 中等 |
| 现代使用 | 很少 | 特定场景 | 特定场景 |

> **大模型提示**：RNN/LSTM/GRU 已被 Transformer 取代。但理解它们有助于理解序列建模的演进——Transformer 的 Self-Attention 正是为了解决 RNN 的并行化和长依赖问题。

---

## 4. 注意力机制

### 4.1 注意力机制的直觉

**核心思想**：在处理序列的每个位置时，动态地"关注"输入序列的不同部分。

类比翻译：
- 翻译"猫"时，重点关注"cat"
- 翻译"坐在"时，需要关注"cat"和"mat"的关系
- 传统 Seq2Seq：编码器压缩所有信息到一个向量，信息瓶颈
- Attention：解码器每一步都可以"回看"编码器的所有输出

### 4.2 注意力分数计算

**Query-Key-Value** 框架：

```python
def attention(query, key, value, mask=None):
    """
    query: (batch, seq_len_q, d_k)
    key:   (batch, seq_len_k, d_k)
    value: (batch, seq_len_k, d_v)
    """
    d_k = query.size(-1)

    # 1. 计算相似度分数: Q @ K^T
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # (batch, seq_len_q, seq_len_k)

    # 2. 可选：掩码（如防止看到未来信息）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. Softmax 归一化得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)

    # 4. 加权求和
    output = torch.matmul(attn_weights, value)
    # (batch, seq_len_q, d_v)

    return output, attn_weights
```

**三种注意力类型**：

| 类型 | Query 来源 | Key/Value 来源 | 应用场景 |
|------|-----------|---------------|---------|
| Encoder Self-Attention | 编码器输出 | 编码器输出 | 编码器内部 |
| Masked Decoder Self-Attention | 解码器输出 | 解码器输出 | 解码器内部（带掩码） |
| Cross Attention | 解码器输出 | 编码器输出 | 连接编码器-解码器 |

---

## 5. 优化与训练技巧

### 5.1 优化器对比

```python
# SGD + Momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam（最常用）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                             betas=(0.9, 0.999), eps=1e-8)

# AdamW（Transformer 标准，解耦权重衰减）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

| 优化器 | 自适应学习率 | 动量 | 权重衰减 | 适用场景 |
|--------|-----------|------|---------|---------|
| SGD | 否 | 可选 | 耦合 | CNN 图像任务 |
| Adam | 是 | 是 | 耦合 | 通用，默认选择 |
| AdamW | 是 | 是 | 解耦 | **Transformer/LLM 训练** |

### 5.2 学习率调度

```python
# Warmup + Cosine Decay（LLM 训练标准）
from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

# 使用
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,      # 前 1000 步 warmup
    num_training_steps=100000   # 总训练步数
)
```

### 5.3 批归一化 vs 层归一化

```python
# BatchNorm：对一个 batch 的同一特征归一化
nn.BatchNorm2d(num_features=64)   # CNN 中使用

# LayerNorm：对单个样本的所有特征归一化
nn.LayerNorm(normalized_shape=512)  # Transformer 中使用
```

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | 跨 batch，同特征 | 同样本，跨特征 |
| 依赖 batch size | 是 | 否 |
| 适合序列 | 否（序列长度变化） | 是 |
| 使用场景 | CNN | **Transformer/RNN** |

### 5.4 训练技巧速查

```python
# 1. 权重初始化
nn.init.xavier_uniform_(m.weight)      # 用于 Sigmoid/Tanh
nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # 用于 ReLU

# 2. 梯度裁剪（防止梯度爆炸）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 标签平滑（防止过拟合）
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 4. 学习率预热（大模型训练必备）
# 见上方 warmup 代码
```

---

## 6. 与大模型的直接关联

| 本书内容 | 大模型应用 |
|---------|-----------|
| MLP | Transformer 中的 FFN 就是 MLP |
| CNN | 视觉编码器（CLIP、GPT-4V） |
| 残差连接 | Transformer 每层都有 |
| LayerNorm | Transformer 的标准归一化 |
| 注意力机制 | **Self-Attention 是 Transformer 核心** |
| AdamW | 训练 LLM 的标准优化器 |
| 学习率 Warmup | 大模型训练不稳定，warmup 必备 |

---

## 7. 面试高频考点

1. **为什么 ReLU 比 Sigmoid 好？**
   - ReLU 正区间梯度恒为 1，不会饱和；Sigmoid 两端梯度 ≈ 0

2. **ResNet 残差连接为什么有效？**
   - 梯度高速公路 + 恒等映射保证

3. **LSTM 如何解决梯度消失？**
   - 细胞状态用加法更新（而非乘法），梯度可直接传播

4. **BatchNorm 和 LayerNorm 的区别？**
   - BN 跨 batch 归一化，LN 单样本内归一化；LN 适合变长序列

5. **Attention 的 Q、K、V 分别是什么？**
   - Query：当前要查询什么；Key：每个位置提供什么信息；Value：实际传递的内容

---

## 8. 实践建议

1. **在线运行代码**：[d2l.ai](https://d2l.ai) 支持 Jupyter Notebook
2. **重点实践**：Transformer 章节的完整实现
3. **修改实验**：改变网络深度、宽度、激活函数，观察影响
4. **可视化**：用 TensorBoard 观察特征图、注意力权重
