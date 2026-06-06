# 从零构建大语言模型

> **资料来源**：Sebastian Raschka《Build a Large Language Model (From Scratch)》中文版
> **适合人群**：希望深入理解 LLM 每个细节的学习者
> **难度**：⭐⭐⭐⭐⭐（很难）

---

## 1. 什么是大语言模型

### 1.1 定义与本质

大语言模型（Large Language Model, LLM）是参数量巨大（通常数十亿到数千亿）的神经网络，经过海量文本训练后，能够理解和生成人类语言。

**核心本质**：一个**概率模型**，对给定的 token 序列预测下一个 token 的概率分布：

$$P(w_t | w_1, w_2, ..., w_{t-1})$$

通过自回归方式，逐 token 生成完整文本：

$$P(w_1, w_2, ..., w_n) = \prod_{t=1}^{n} P(w_t | w_1, ..., w_{t-1})$$

### 1.2 大模型 vs 小模型

| 特性 | 传统 NLP 模型 | 大语言模型 |
|------|-------------|-----------|
| 参数量 | 百万级 | 十亿 ~ 万亿级 |
| 训练数据 | 标注数据集（MB~GB） | 未标注文本（TB~PB） |
| 任务方式 | 每个任务单独训练 | 预训练 + 通用能力 |
| 使用方式 | 特征提取 + 分类器 | Prompt / 微调 |
| 能力 | 单一任务 | 通用语言理解与生成 |

### 1.3 LLM 的能力谱

**基础能力**：
- 文本补全：给定前缀，生成后续内容
- 语法纠错：识别并修正语法错误
- 翻译：在不同语言间转换
- 摘要：压缩长文本

**高级能力（大模型特有）**：
- **In-Context Learning**：通过 prompt 中的示例学习新任务
- **Zero-shot/Few-shot**：无需训练即可处理未见过的任务
- **Chain-of-Thought**：复杂问题的逐步推理
- **代码理解与生成**：阅读、解释、编写程序

---

## 2. 训练阶段

### 2.1 两阶段范式

现代 LLM 的训练通常分为两个阶段：

```
海量无标注文本 ──→ 预训练 ──→ 基础模型（Base Model）
                                      │
标注指令数据 ──→ 微调 ──→ 指令模型 / 聊天模型（Instruct/Chat Model）
```

### 2.2 预训练（Pre-training）

**目标**：让模型学习语言的统计规律和通用知识
**数据**：互联网文本、书籍、代码、论文等（去重、清洗后）
**任务**：Next Token Prediction
**规模**：
- GPT-3（175B）：300B tokens
- LLaMA-3（70B）：15T tokens
- DeepSeek-V3（671B MoE）：14.8T tokens
- 计算量：数千 GPU 运行数周至数月

**自监督学习**：
预训练不需要人工标注，文本本身就是监督信号：

```
输入:  [猫, 坐, 在, 地, 毯, 上, 睡, 觉]
目标1:  猫 → [坐]
目标2:  猫, 坐 → [在]
目标3:  猫, 坐, 在 → [地]
...
```

### 2.3 微调（Fine-tuning）

预训练后的模型虽然"懂"语言，但不会按照指令行动。微调将其转化为有用的助手。

**指令微调（Instruction Fine-tuning）**：
使用（指令，回答）格式的数据训练：

```
输入:  请将以下句子翻译成英文：你好世界
目标:  Hello world
```

**对话微调**：
使用多轮对话数据：

```json
[
  {"role": "user", "content": "什么是机器学习？"},
  {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
  {"role": "user", "content": "能举个例子吗？"},
  {"role": "assistant", "content": "当然。比如垃圾邮件过滤..."}
]
```

**为什么需要微调？**
- 预训练模型只学会了"续写"，不会"回答"
- 微调让模型学会对话格式和指令遵循
- 可以注入安全约束（拒绝有害请求）

### 2.4 分类微调（Classification Fine-tuning）

除了生成任务，LLM 也可以用于分类：

```python
# 最后一层改为分类头
class GPTForClassification(nn.Module):
    def __init__(self, gpt, num_classes):
        super().__init__()
        self.gpt = gpt
        self.classifier = nn.Linear(gpt.embed_dim, num_classes)

    def forward(self, x):
        # 取最后一个 token 的隐藏状态
        hidden = self.gpt(x)[:, -1, :]
        return self.classifier(hidden)
```

**应用场景**：情感分析、垃圾检测、意图识别

---

## 3. GPT 架构详解

GPT（Generative Pre-trained Transformer）采用 **Decoder-only** 架构，是当前 LLM 的主流范式。

### 3.1 整体结构

```
输入 Token IDs
    ↓
Token Embedding + Positional Embedding
    ↓
[Transformer Block] × N 层
    ↓
LayerNorm
    ↓
Linear → Vocab Size（输出 logits）
    ↓
Softmax → 概率分布
```

### 3.2 Token Embedding

将离散 token 映射为连续向量：

```python
self.token_emb = nn.Embedding(vocab_size, embed_dim)
```

**特点**：
- 每个 token 有一个唯一的嵌入向量
- 嵌入维度通常 768（小模型）~ 8192（大模型）
- 训练开始时随机初始化，训练后学到语义表示

### 3.3 位置编码

GPT 使用**可学习的位置嵌入**（Learnable Positional Embeddings）：

```python
self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
```

与固定正弦编码不同，位置嵌入也是可训练参数。模型自己学习每个位置该加什么偏移。

**输入表示**：
$$x = \text{TokenEmb}(token) + \text{PosEmb}(position)$$

### 3.4 Transformer Block 内部

每个 Block 包含：

```
Input
  ↓
LayerNorm
  ↓
Masked Multi-Head Self-Attention  ← 因果 mask，只关注前面
  ↓
Dropout
  ↓
Residual Connection (+ Input)
  ↓
LayerNorm
  ↓
Feed-Forward Network (FFN)
  │   Linear(embed_dim → 4×embed_dim)
  │   GELU activation
  │   Linear(4×embed_dim → embed_dim)
  ↓
Dropout
  ↓
Residual Connection
  ↓
Output
```

**与原始 Transformer Decoder 的区别**：
- 没有 Encoder-Decoder Cross Attention（因为没有 Encoder）
- 只有 Masked Self-Attention + FFN
- 使用 Pre-LayerNorm（原始 Transformer 是 Post-LayerNorm）

### 3.5 GELU 激活函数

GPT-2/3 使用 GELU 替代 ReLU：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

或使用近似：
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

**优势**：平滑可导，在 0 附近梯度不为 0，训练更稳定。

### 3.6 输出层

最后一个 Block 的输出经过 LayerNorm，然后通过线性层映射到词表维度：

```python
self.out_head = nn.Linear(embed_dim, vocab_size, bias=False)
```

**权重共享**：输出层的权重通常与 Token Embedding 层共享（Tied Embeddings），减少参数量并提升性能。

### 3.7 GPT-2 配置对比

| 模型 | 层数 | 隐藏维度 | 注意力头数 | 参数量 | 上下文长度 |
|------|------|----------|-----------|--------|-----------|
| GPT-2 Small | 12 | 768 | 12 | 124M | 1024 |
| GPT-2 Medium | 24 | 1024 | 16 | 355M | 1024 |
| GPT-2 Large | 36 | 1280 | 20 | 774M | 1024 |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B | 1024 |

---

## 4. 文本生成

### 4.1 自回归生成流程

```python
def generate(model, prompt, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    tokens = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        # 截断到最大上下文长度
        input_tokens = tokens[-model.max_seq_len:]
        input_tensor = torch.tensor([input_tokens])

        with torch.no_grad():
            logits = model(input_tensor)

        # 取最后一个位置的 logits
        next_token_logits = logits[:, -1, :]

        # 应用 temperature
        next_token_logits = next_token_logits / temperature

        # Top-K 采样
        if top_k is not None:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        tokens.append(next_token)

    return tokenizer.decode(tokens)
```

### 4.2 不同采样策略的效果

| 策略 | 参数 | 输出特点 |
|------|------|----------|
| 贪心 | 无 | 确定性、保守、可能重复 |
| Temperature=0.7 | T=0.7 | 较确定，有一定变化 |
| Temperature=1.0 | T=1.0 | 平衡 |
| Temperature=1.5 | T=1.5 | 更随机、更有创意 |
| Top-K=50 | K=50 | 限制候选范围，减少荒谬输出 |

---

## 5. 文本处理与 Tokenization

### 5.1 文本到 Token 的转换

原始文本不能直接输入神经网络，需要转换为数字序列：

```
文本 → Tokenization → Token IDs → Embedding → 模型输入
```

### 5.2 BPE（Byte-Pair Encoding）算法

GPT-2/GPT-3 使用 BPE 分词。核心思想：**从字符开始，迭代合并最常见的字符对**。

**算法步骤**：
1. 初始化词表为所有单个字符
2. 统计语料中所有相邻 token 对的频率
3. 合并频率最高的对，加入词表
4. 重复 2-3 步直到词表达到目标大小

**示例**：
```
初始语料:  low, lower, lowest, new, newer
初始词表:  {l, o, w, e, r, n, s, t}

第1轮: "er" 出现3次 → 合并 → 词表加入 "er"
第2轮: "est" 出现1次但高频... 实际按频率
...
最终词表:  {l, o, w, e, r, n, s, t, er, es, est, lo, low, ...}
```

**特点**：
- 词表大小固定（GPT-2: 50257）
- 未知词（OOV）通过子词拆分处理
- 中文通常按字或子词切分

### 5.3 Tokenizer 使用示例

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Hello, world!"
tokens = tokenizer.encode(text)
print(tokens)  # [15496, 11, 995, 0]

print(tokenizer.decode(tokens))  # "Hello, world!"

# 查看每个 token
for token_id in tokens:
    print(f"{token_id}: '{tokenizer.decode([token_id])}'")
```

### 5.4 特殊 Token

| Token | 作用 |
|-------|------|
| `<|endoftext|>` | 文本结束标记，也用于填充 |
| `<|pad|>` | 填充（对齐序列长度） |
| `<|unk|>` | 未知 token（BPE 中较少出现） |

---

## 6. 从零构建的完整流程

### 6.1 三阶段构建法

Raschka 的书按以下三阶段组织：

**阶段一：数据准备与理解**
- 加载文本语料
- 实现 BPE Tokenizer
- 构建数据集和 DataLoader
- 理解 batching 和 padding

**阶段二：模型实现**
- 实现 GPT 的所有组件（Embedding、Attention、FFN、Block）
- 组装完整模型
- 加载预训练权重（可选，用于验证实现正确性）
- 文本生成

**阶段三：训练**
- 预训练：在通用语料上训练
- 微调：在指令数据上微调
- 分类微调：用于下游任务

### 6.2 数据加载示例

```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.tokens = tokenizer.encode(text)
        self.max_length = max_length
        self.stride = stride

    def __len__(self):
        return (len(self.tokens) - self.max_length) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length
        input_chunk = self.tokens[start:end]
        target_chunk = self.tokens[start+1:end+1]
        return torch.tensor(input_chunk), torch.tensor(target_chunk)

# 使用
dataset = GPTDataset(text, tokenizer, max_length=256, stride=128)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### 6.3 训练循环

```python
def train(model, dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            # 计算交叉熵损失
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten()
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}")
```

### 6.4 损失函数详解

交叉熵损失等价于最大似然估计：

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, ..., w_{i-1})$$

**理解**：
- 模型给正确 token 分配的概率越高，loss 越低
- Perplexity = $e^{\text{loss}}$，表示模型"有多困惑"
- 训练目标就是让模型"不困惑"，即准确预测下一个 token

---

## 7. 关键概念深入

### 7.1 为什么 Next Token Prediction 有效

看似简单的任务如何产生智能？

1. **压缩即智能**：预测下一个 token 需要理解语法、语义、逻辑、世界知识
2. **多任务学习**：语料中蕴含了翻译、问答、推理等各种任务的隐式示例
3. **规模效应**：足够大的模型 + 足够多数据 = 涌现通用能力

### 7.2 Catastrophic Forgetting（灾难性遗忘）

微调时模型可能遗忘预训练学到的通用知识。

**缓解方法**：
- **混合预训练数据**：SFT 时混入 5-10% 预训练数据
- **LoRA**：只训练低秩适配器，保持主模型参数冻结
- **渐进式微调**：逐步增加新任务数据比例

### 7.3 上下文窗口与位置编码

- GPT-2：1024 tokens
- GPT-3：2048 tokens
- GPT-4：128K tokens
- LLaMA-3：128K tokens

**长上下文技术**：
- **位置插值（Positional Interpolation）**：将位置编码缩放，使模型适应更长序列
- **RoPE 外推**：利用 RoPE 的相对位置特性
- **NTK-aware 扩展**：非线性位置编码插值

---

## 8. 扩展：与 HuggingFace 的对比

实现完自己的 GPT 后，建议与 HuggingFace 的实现对比：

```python
from transformers import GPT2LMHeadModel, GPT2Config

# 自定义模型
my_model = MyGPT(vocab_size=50257, embed_dim=768, num_layers=12, num_heads=12)

# HuggingFace 模型
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')

# 对比输出
test_input = torch.randint(0, 50257, (1, 10))
my_out = my_model(test_input)
hf_out = hf_model(test_input).logits

print("输出形状是否一致:", my_out.shape == hf_out.shape)
print("最大差异:", (my_out - hf_out).abs().max().item())
```

**预期差异**：
- 初始化不同导致数值不同，但形状应一致
- 加载相同预训练权重后，输出应几乎相同（误差 < 1e-5）

---

## 学习路径建议

1. **不要跳过代码实现**：每一行都很重要，特别是维度变换
2. **打印张量形状**：在关键位置打印 `.shape`，验证理解
3. **从小模型开始**：先用 GPT-2 Small（124M）验证，再扩展
4. **对比不同实现**：Raschka 的实现、HuggingFace、nanoGPT
5. **修改超参数**：观察层数、维度对生成质量的影响
