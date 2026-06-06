# 大规模语言模型：从理论到实践（Large Language Models: Theory to Practice）

> **资料来源**：张奇、桂韬、郑锐、黄萱菁《大规模语言模型：从理论到实践》
> **适合人群**：希望从工程角度理解 LLM 的读者
> **难度**：⭐⭐⭐⭐（较难）

---

## 1. 四阶段训练流程（Four-Stage Training Pipeline）

现代大语言模型的构建通常分为四个阶段。理解每个阶段的目标、数据和难点，是 LLM 工程实践的核心。

### 1.1 预训练（Pre-training）

**目标**：学习通用的语言表示和世界知识
**数据**：大规模未标注文本（网页、书籍、代码等）
**任务**：自回归 Next Token Prediction
**算力消耗**：占整个流程的 90% 以上

#### 1.1.1 完整数据 Pipeline（Data Pipeline）

预训练的数据质量直接决定模型上限。完整的数据处理流程包括：

**Step 1: 数据采集**
- **Common Crawl**：网页抓取数据，量大但质量参差不齐
- **GitHub**：代码数据，提升代码能力
- **Wikipedia**：高质量百科知识
- **Books**：书籍数据，长文本连贯性好
- **ArXiv**：学术论文，提升推理能力
- **对话数据**：Reddit、论坛等，提升对话能力

**Step 2: 质量过滤**
```python
# 典型过滤规则
filters = {
    "min_length": 100,        # 剔除过短文档
    "max_length": 100000,     # 剔除过长文档（可能是爬虫错误）
    "min_words_per_line": 5,  # 剔除代码/表格残留
    "language_ratio": 0.8,    # 目标语言占比
    "perplexity_threshold": 1000,  # 语言模型困惑度过滤（剔除乱码）
}
```

**Step 3: 去重（Deduplication）**

重复数据会导致：
- 模型过拟合到特定内容
- 训练效率降低（重复学习相同信息）
- 生成时倾向于复制训练数据（版权/隐私风险）

去重方法：
- **精确去重**：基于哈希（MD5/SHA256）的完全重复
- **模糊去重**：MinHash + LSH（局部敏感哈希），检测近似重复
- **子串去重**：在文档级别检测长重复子串

**Step 4: 敏感内容过滤**
- 隐私信息（PII）：邮箱、电话、身份证号
- 有毒内容：仇恨言论、成人内容
- 个人可识别信息（使用正则或分类器检测）

**数据配比示例（GPT-3）**：
- Common Crawl（过滤后）：60%
- WebText2：22%
- Books1+Books2：16%
- Wikipedia：3%

**现代模型的数据配比趋势**：
- **增加代码比例**：GitHub 数据从 GPT-3 的接近 0% 增加到 LLaMA-3 的 15%+，显著提升代码和推理能力
- **增加多语言比例**：如 LLaMA-3 支持 30+ 种语言，Qwen 支持 100+ 种
- **数据质量 > 数量**：用分类器筛选高质量数据（如 C4 → RefinedWeb）

#### 1.1.2 分词器训练（Tokenizer Training）

预训练前需要训练分词器，将文本转化为 token 序列。

**BPE（Byte-Pair Encoding）算法**：

1. 初始化词汇表为所有单个字符（或 bytes）
2. 统计所有相邻字符对的频率
3. 合并频率最高的字符对，加入词汇表
4. 重复步骤 2-3 直到词汇表达到目标大小

**示例**：
```
初始语料：['l o w </w>', 'l o w e r </w>', 'n e w e s t </w>']
合并 ('e', 's') → ['l o w </w>', 'l o w e r </w>', 'n e w es t </w>']
合并 ('es', 't') → ['l o w </w>', 'l o w e r </w>', 'n e w est </w>']
合并 ('est', '</w>') → ...
```

**词汇表大小选择**：

| 词汇表大小 | 代表模型 | 特点 |
|-----------|---------|------|
| 32K | GPT-2 | 较小，英文为主 |
| 50K | BERT | 标准选择 |
| 100K | LLaMA-3/Qwen | 多语言，平衡压缩率和词表大小 |
| 128K+ | GPT-4, LLaMA-3 | 更大的词表，更好的多语言支持 |

**词汇表大小的影响**：
- **太小**：常见词被拆成多个 token，序列变长，增加计算量
- **太大**：词嵌入矩阵巨大，增加参数量和内存
- **经验**：100K 左右是较好的平衡点

#### 1.1.3 分布式训练策略（Distributed Training Strategies）

大模型训练需要分布式策略将计算分散到多张 GPU/多个节点。

**数据并行（Data Parallelism, DP）**：

```
每张 GPU 保存完整模型副本
将 batch 分成 N 份，每份给一张 GPU
每张 GPU 独立计算前向+反向
使用 All-Reduce 同步梯度
每张 GPU 独立更新参数
```

- 优点：实现简单，通信量小（只需同步梯度）
- 缺点：每张 GPU 都要存完整模型，模型大时单卡放不下

**模型并行（Model Parallelism / Tensor Parallelism, TP）**：

将模型的层内参数切分到多张 GPU：

```
Linear 层: Y = XW
将 W 按列切分: W = [W1 | W2]
GPU1 计算 XW1, GPU2 计算 XW2
拼接结果得到 Y
```

- 优点：可以训练单卡放不下的模型
- 缺点：通信量大（每层的激活值需要 all-gather）

**流水线并行（Pipeline Parallelism, PP）**：

将模型按层切分，每张 GPU 负责若干连续层：

```
GPU1: Layer 1-4
GPU2: Layer 5-8
GPU3: Layer 9-12
GPU4: Layer 13-16
```

- 优点：通信量小（只需传递层间激活值）
- 缺点：存在流水线气泡（bubble），GPU 利用率不高

**优化：GPipe / PipeDream**：

将 batch 进一步切分为 micro-batches，让不同 micro-batch 在不同 stage 并行：

```
t1: [micro1 on GPU1]
t2: [micro2 on GPU1, micro1 on GPU2]
t3: [micro3 on GPU1, micro2 on GPU2, micro1 on GPU3]
```

**3D 并行（数据 + 张量 + 流水线）**：

GPT-3（175B）的训练配置：
- 96 层 Transformer
- 使用 3D 并行：数据并行度 × 张量并行度 × 流水线并行度 = 总 GPU 数
- 例如：8（DP）× 8（TP）× 4（PP）= 256 张 GPU

#### 1.1.4 混合精度训练（Mixed Precision Training）

**动机**：FP32 精度下，每个参数占 4 bytes。175B 模型仅参数就需要 700GB，训练时需要存储参数、梯度、优化器状态、激活值，总内存需求超过 2TB。

**方案**：在大多数计算中使用 FP16，关键计算保持 FP32：

```python
# PyTorch Automatic Mixed Precision (AMP)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)      # 前向用 FP16
    loss = criterion(output, target)

scaler.scale(loss).backward()  # 反向用 FP16
scaler.step(optimizer)         # 参数更新用 FP32
scaler.update()
```

**Loss Scaling**：

FP16 的表示范围小（最小正数 ~6×10⁻⁸），反向传播时梯度容易下溢为 0。

解决：将 loss 乘以一个较大的 scale factor（如 2¹⁶），反向传播后梯度也相应放大，然后再 unscale。

```
scaled_loss = loss * scale_factor
scaled_loss.backward()   # 梯度也被放大
gradients = gradients / scale_factor  # 恢复
```

**BF16（BFloat16）**：

Google Brain 提出的格式：
- 与 FP32 相同的 8 位指数范围，但尾数只有 7 位（FP16 是 5 位指数 + 10 位尾数）
- 动态范围更大，不容易溢出
- 精度略低于 FP16，但训练稳定性更好
- A100/H100 原生支持

#### 1.1.5 梯度累积（Gradient Accumulation）

当单卡 batch size 太小（显存限制）时，可以通过梯度累积模拟大 batch：

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()  # 梯度累加

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()   # 每 accumulation_steps 步更新一次
        optimizer.zero_grad()
```

**有效 batch size** = per_device_batch_size × gradient_accumulation_steps × num_gpus

**为什么大 batch 有帮助？**
- 梯度估计更稳定，噪声更小
- 可以使用更大的学习率
- 但在一定阈值后收益递减（BERT: 256-512, GPT-3: 3.2M tokens）

#### 1.1.6 ZeRO 优化器（Zero Redundancy Optimizer）

DeepSpeed 提出的 ZeRO 解决了数据并行中每个 GPU 都存储完整优化器状态的冗余问题。

**ZeRO 的三个阶段**：

| 阶段 | 切分内容 | 内存节省 | 通信开销 |
|------|---------|---------|---------|
| **ZeRO-1** | 优化器状态 | 4× | 与 DP 相同 |
| **ZeRO-2** | 优化器状态 + 梯度 | 8× | 与 DP 相同 |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | 与数据并行度成正比 | 增加 |

**ZeRO-3 的原理**：

数据并行有 N 张 GPU 时：
- 每张 GPU 只存储 1/N 的参数、梯度和优化器状态
- 前向传播时，需要某层参数时通过 all-gather 从其他 GPU 收集
- 计算完成后立即释放

**ZeRO-Offload**：

将优化器状态和计算 offloading 到 CPU/NVMe：
- 参数和梯度在 GPU
- 优化器状态（Adam 需要 4× 参数量的内存）在 CPU
- 甚至可以将计算放在 CPU，进一步减少 GPU 内存

**效果**：用 ZeRO-Offload，可在单张消费级 GPU（如 24GB 3090）上训练 10B+ 参数模型（速度较慢）。

#### 1.1.7 激活检查点（Activation Checkpointing）

以计算换内存：前向传播时不保存中间激活值，反向传播时重新计算。

**内存分析**：
- 不启用：存储所有层的激活值，内存 = O(L × batch × seq × d)
- 启用：只保存输入，反向时重计算每层，内存 = O(L) → O(1)
- 代价：约 30% 额外计算时间

```python
from torch.utils.checkpoint import checkpoint

# 将 Transformer Block 包装为 checkpoint
class CheckpointedTransformerBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self.block, x)
```

#### 1.1.8 预训练核心工程难点总结

- **数据构建**：如何清洗、去重、过滤低质量数据
- **分布式训练**：模型大到无法放入单卡，需要数据并行、模型并行、流水线并行
- **数值稳定性**：大模型训练容易出现 loss spike、梯度爆炸
- **检查点管理**：训练中断恢复、多版本管理

### 1.2 有监督微调（Supervised Fine-Tuning / SFT）

**目标**：让模型学会遵循指令、完成特定任务
**数据**：(指令, 回答) 形式的标注数据
**任务**：在给定指令下生成期望的回答

#### 1.2.1 指令数据格式

标准格式包含 system prompt、user message、assistant response：

```json
{
  "messages": [
    {"role": "system", "content": "你是一位专业的技术文档工程师。"},
    {"role": "user", "content": "请解释什么是 RAG？"},
    {"role": "assistant", "content": "RAG（Retrieval-Augmented Generation）..."}
  ]
}
```

**System Prompt 的作用**：
- 设定模型的角色和风格
- 注入全局约束（如"不要生成有害内容"）
- 可以在 SFT 时学习，也可以在推理时动态指定

#### 1.2.2 样本构造策略

**覆盖度**：
- **任务多样性**：问答、摘要、翻译、代码、推理、创意写作等
- **领域多样性**：科技、医疗、法律、教育、金融等
- **难度多样性**：简单、中等、复杂问题
- **语言多样性**：多语言覆盖

**质量原则**：
- 多样性：覆盖尽可能多的场景
- **质量 > 数量**：1 万条高质量指令数据往往比百万条低质量数据效果更好
- 格式一致：统一的 system/user/assistant 格式
- 避免污染：确保测试集数据不在训练集中

#### 1.2.3 课程学习（Curriculum Learning）

让模型从简单样本开始学，逐步增加难度：

```
阶段 1: 简单问答（事实性，单步推理）
阶段 2: 中等复杂度（多步推理，简单代码）
阶段 3: 复杂任务（长推理链，复杂代码，创意写作）
```

**实现方式**：
- 按困惑度排序：模型在预训练阶段已经熟悉的样本排在前面
- 按回答长度排序：短回答 → 长回答
- 按任务类型排序：分类 → 生成 → 推理

**效果**：加速收敛，提升最终性能，尤其在小数据集上。

#### 1.2.4 混合预训练数据

SFT 时通常混入 5-10% 的预训练数据，防止模型遗忘通用能力：

```python
# 数据配比
data_mixture = {
    "instruction_data": 0.9,   # SFT 数据
    "pretrain_data": 0.1,      # 原始语料
}
```

**原因**：
- 纯 SFT 数据量通常只有几十万到几百万条
- 模型容易过拟合到 SFT 数据的分布和风格
- 混入预训练数据保持语言建模能力

#### 1.2.5 核心难点

- 如何构建覆盖广泛场景的高质量指令数据
- 数据分布与真实使用分布的对齐
- 避免过拟合到特定数据格式

### 1.3 奖励建模（Reward Modeling）

**目标**：训练一个打分模型，评估生成的质量
**数据**：同一 prompt 的多个回答，人工标注好坏排序
**任务**：学习人类偏好

#### 1.3.1 Bradley-Terry 模型

假设人类偏好的概率与分数差的 sigmoid 成正比。对于一对回答 $(y_w, y_l)$，其中 $y_w$ 是标注者偏好的（win），$y_l$ 是较差的（lose）：

$$P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是 sigmoid 函数，$r_\theta$ 是奖励模型，输出一个标量分数。

**损失函数**：

$$\mathcal{L} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

**理解**：
- 当 $r_\theta(x, y_w) > r_\theta(x, y_l)$ 时，loss 趋近于 0
- 当 $r_\theta(x, y_w) < r_\theta(x, y_l)$ 时，loss 趋近于 $r_\theta(x, y_l) - r_\theta(x, y_w)$（线性惩罚）
- 这保证了好回答的分数高于差回答

#### 1.3.2 奖励模型的训练

通常基于 SFT 模型初始化，将最后的输出层替换为标量输出头：

```python
# 基于 SFT 模型初始化
reward_model = AutoModel.from_pretrained("sft-model")

# 替换输出头
reward_model.lm_head = nn.Linear(hidden_size, 1)

# 训练：对 (prompt, chosen, rejected) 三元组
chosen_reward = reward_model(prompt + chosen)
rejected_reward = reward_model(prompt + rejected)
loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
```

#### 1.3.3 核心难点

- **Reward Hacking**：奖励模型只在训练分布内可靠，分布外可能给出错误分数。策略模型可能找到奖励模型的"漏洞"，生成奖励高但实际质量差的回答。
- 人类标注成本高、主观性强
- 需要限定奖励模型的应用场景

### 1.4 强化学习（Reinforcement Learning / RL, PPO）

**目标**：用奖励模型指导语言模型生成更高质量的回答
**算法**：PPO（Proximal Policy Optimization）

#### 1.4.1 PPO 核心组件

- **策略模型**（Policy，即要训练的 LLM）：生成回答，参数为 $\theta$
- **价值模型**（Value Model）：估计状态价值 $V(s)$，用于计算优势函数
- **奖励模型**（Reward Model，固定）：给出生成质量的外在奖励 $r_\theta(x, y)$
- **参考模型**（Reference Model，固定）：SFT 后的模型，参数为 $\theta_{ref}$，用于计算 KL 散度惩罚

#### 1.4.2 PPO 的数学推导

**Step 1: 从奖励模型获取奖励**

对于生成的回答 $y$，奖励为：
$$R(x, y) = r_\phi(x, y) - \beta \cdot \text{KL}[\pi_\theta(y|x) \| \pi_{ref}(y|x)]$$

其中：
- $r_\phi(x, y)$：奖励模型给出的分数
- KL 惩罚项：防止策略模型偏离参考模型太远
- $\beta$：KL 惩罚系数

**KL 散度的近似计算**：

对于每个生成的 token：
$$\text{KL}_t = \log \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{ref}(y_t|x, y_{<t})}$$

总 KL = 对所有 token 求和。

**Step 2: 优势函数估计（GAE）**

PPO 使用 GAE（Generalized Advantage Estimation）计算优势函数：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + ...$$

其中：
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\gamma$：折扣因子（通常 0.99）
- $\lambda$：GAE 参数（通常 0.95），平衡偏差和方差

**Step 3: PPO 裁剪目标**

定义概率比：
$$r_t(\theta) = \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\theta_{old}}(y_t|x, y_{<t})}$$

裁剪后的目标：
$$\mathcal{L}^{CLIP} = \mathbb{E}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

其中 $\epsilon$ 是超参数（通常 0.1 或 0.2）。

**裁剪的作用**：
- 当 $r_t(\theta)$ 超过 $[1-\epsilon, 1+\epsilon]$ 时，目标函数停止增加
- 防止策略在单次更新中变化过大，保证训练稳定性

**Step 4: 完整 PPO 损失**

$$\mathcal{L}_{PPO} = \mathcal{L}^{CLIP} - c_1 \mathcal{L}^{VF} + c_2 \mathcal{H} + c_3 \cdot \text{KL}[\pi_\theta \| \pi_{ref}]$$

其中：
- $\mathcal{L}^{CLIP}$：裁剪后的策略梯度损失
- $\mathcal{L}^{VF}$：价值函数损失（MSE  between $V(s_t)$ 和实际回报）
- $\mathcal{H}$：策略熵，鼓励探索
- KL 项：防止策略模型与参考模型偏离过大

#### 1.4.3 PPO 训练流程

```python
for iteration in range(num_iterations):
    # 1. 用当前策略生成回答
    responses = policy_model.generate(prompts)

    # 2. 计算奖励
    rewards = reward_model(prompts, responses)
    kl_penalty = compute_kl(policy_model, ref_model, prompts, responses)
    final_rewards = rewards - beta * kl_penalty

    # 3. 计算优势函数（使用 GAE）
    advantages = gae(final_rewards, value_model)

    # 4. 多次梯度更新（PPO 的 epoch 参数）
    for epoch in range(ppo_epochs):
        # 计算 PPO loss
        loss = ppo_loss(policy_model, old_policy, advantages, responses)
        loss.backward()
        optimizer.step()
```

#### 1.4.4 RLHF 的问题与替代方案

**PPO 的问题**：
1. **训练不稳定**：超参数敏感（learning rate、clip epsilon、KL coefficient）
2. **需要训练四个模型**：Policy、Value、Reward、Reference，内存开销大
3. **Reward Hacking**：策略模型可能利用奖励模型的盲点
4. **样本效率低**：需要大量在线采样

**DPO（Direct Preference Optimization）**：

DPO 的核心洞察：PPO 中的 RL 过程可以被一个更简单的偏好学习目标替代。

**理论推导**：

RLHF 的目标是找到最优策略：
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

其中 $Z(x)$ 是配分函数（归一化常数）。

将上式变形，得到奖励函数与策略的关系：
$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

代入 Bradley-Terry 偏好模型的损失：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

**DPO 的优势**：
1. **无需显式奖励模型**：直接从偏好数据优化
2. **无需强化学习**：只是一个带参考模型的分类损失
3. **训练更稳定**：没有 PPO 的超参数调优问题
4. **内存需求减半**：只需策略模型和参考模型

**DPO 的劣势**：
1. 对数据质量要求更高
2. 容易过拟合到偏好数据
3. 泛化到新分布的能力可能不如 PPO

**DPO 代码实现**：

```python
# DPO 损失计算
def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    prompts, chosen, rejected = batch

    # 策略模型的 log prob
    policy_chosen_logps = get_logprobs(policy_model, prompts, chosen)
    policy_rejected_logps = get_logprobs(policy_model, prompts, rejected)

    # 参考模型的 log prob
    ref_chosen_logps = get_logprobs(ref_model, prompts, chosen)
    ref_rejected_logps = get_logprobs(ref_model, prompts, rejected)

    # DPO 损失
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    ref_ratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_ratio - ref_ratio)
    loss = -F.logsigmoid(logits).mean()

    return loss
```

#### 1.4.5 现代对齐方法对比（2024-2025）

DPO 之后，学术界提出了一系列改进方法，各有适用场景：

| 方法 | 核心思想 | 数据要求 | 训练稳定性 | 当前主流程度 |
|------|---------|---------|-----------|-------------|
| **PPO** | 在线 RL + 价值模型 | 偏好对 + 在线采样 | 低 | 下降 |
| **DPO** | 离线偏好优化 | 偏好对 (chosen/rejected) | 中 | 高 |
| **IPO** | DPO 的正则化版本 | 偏好对 | 高 | 中 |
| **KTO** | 二分类（好/坏），无需配对 | 二元标签 | 高 | 上升 |
| **ORPO** | SFT + 偏好优化一步完成 | 偏好对 | 高 | 上升 |
| **SimPO** | 去掉参考模型，长度归一化 | 偏好对 | 高 | 高 |

**KTO（Kahneman-Tversky Optimization）**：

仅需二分类偏好（好/坏），无需配对数据：

$$\mathcal{L}_{KTO} = \mathbb{E}\left[ \lambda_y - \log \sigma\left( z_y - \mathbb{E}_{x' \sim D}[z_{x'}] \right) \right]$$

其中 $z_y = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$。

**为什么 KTO 更实用？**

```
DPO 的数据要求：
  Prompt: "解释量子力学"
  Chosen: "量子力学是..." (好回答)
  Rejected: "量子是..." (差回答)
  问题：需要为同一 prompt 生成多个回答并排序，标注成本高

KTO 的数据要求：
  样本1: Prompt + "解释量子力学" → Answer → 标签: "好"
  样本2: Prompt + "写首诗" → Answer → 标签: "坏"（格式错误）
  优势：不需要配对，任何带好坏标签的数据都可以用
```

**ORPO（Odds Ratio Preference Optimization）**：

**核心洞察**：SFT 和 DPO 可以合并为一步。

传统流程：
```
Base Model ──→ SFT ──→ SFT Model ──→ DPO ──→ Aligned Model
   两步训练，需要保存中间模型
```

ORPO 流程：
```
Base Model ──→ ORPO ──→ Aligned Model
   一步完成，同时优化 likelihood 和 preference
```

ORPO 损失函数：
$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} - \lambda \cdot \mathbb{E}\left[ \log \sigma\left( \log \frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)} \right) \right]$$

其中 odds = P(y|x) / (1 - P(y|x))。

**优势**：
- 训练速度快（一步 vs 两步）
- 不需要参考模型
- 适合资源有限的团队

**SimPO（Simple Preference Optimization）**：

**核心改进**：去掉 DPO 的参考模型，用长度归一化解决 DPO 的长度偏见。

DPO 的问题：
```
DPO 倾向于生成更长的回答：
  因为 log P(y|x) 随序列长度增加而减小（更多 token 的乘积）
  模型学会通过"多说"来提高 chosen 的 likelihood
```

SimPO 的解决方案：
$$\mathcal{L}_{SimPO} = -\mathbb{E}\left[ \log \sigma\left( \frac{\beta}{|y_w|} \log P_\theta(y_w|x) - \frac{\beta}{|y_l|} \log P_\theta(y_l|x) - \gamma \right) \right]$$

- 用平均 log prob（除以长度）替代总 log prob
- 加入 margin γ，强制好回答比差回答好一个固定 margin
- 去掉参考模型，进一步简化

**工业界选择建议**：

```
资源充足、追求极致效果：
  → PPO + 高质量 Reward Model（如 OpenAI）

有配对偏好数据、追求稳定训练：
  → DPO 或 SimPO（SimPO 更稳定，无需参考模型）

数据标注资源有限：
  → KTO（二元标签即可）

追求训练效率、一步完成：
  → ORPO（SFT + 偏好优化合并）
```

### 1.5 四阶段流程总结（Training Pipeline Summary）

```
原始语料 ──→ 预训练 ──→ 基础模型
                          │
指令数据 ──→ SFT ────────→ 指令模型
                          │
偏好数据 ──→ 奖励模型 ────→ 奖励模型 (可选)
                          │
PPO/DPO/KTO/ORPO/SimPO ──→ 对齐模型（最终产品）
```

**2024-2025 年的趋势变化**：
- **PPO 正在被放弃**：训练不稳定、超参数敏感，只有资源充足的大厂仍在使用
- **DPO/SimPO 成为主流**：稳定、简单、效果相当，开源社区首选
- **KTO 降低数据门槛**：二元标签即可，适合数据有限的场景
- **ORPO 一步完成**：SFT + 对齐合并，训练效率最高
- **预训练成本极高**：大多数团队基于开源模型（LLaMA、Qwen、DeepSeek）做 SFT + DPO
- **推理模型的 RL**：GRPO（DeepSeek-R1）正在重新定义 RL 在 LLM 中的作用

---

## 2. Transformer 的 PyTorch 实现（PyTorch Implementation）

理解代码实现是掌握原理的最佳方式。以下是基于 PyTorch 的简化版 Transformer。

### 2.1 Self-Attention 实现

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Attention scores: (batch, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return self.W_o(output)
```

### 2.2 Multi-Head Attention 实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        batch, seq_len, embed_dim = x.shape

        # 生成 Q, K, V 并分头: (batch, num_heads, seq_len, head_dim)
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (batch, num_heads, seq_len, head_dim)

        # 拼接多头并线性变换
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        return self.W_o(out)
```

### 2.3 完整 Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN: LayerNorm → Sublayer → Residual
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x
```

### 2.4 因果 Mask（Decoder-only）

```python
def create_causal_mask(seq_len):
    """生成下三角 mask，确保位置 i 只能看到 ≤ i 的位置"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # (seq_len, seq_len)
```

### 2.5 BERT 风格的 MLM 训练流程

```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

# 准备输入
text = "今天[MASK]气很好"
inputs = tokenizer(text, return_tensors='pt')

# 前向传播
outputs = model(**inputs)
predictions = outputs.logits

# 获取 [MASK] 位置的预测
mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
predicted_token_id = predictions[mask_idx].argmax(dim=-1)
print(tokenizer.decode(predicted_token_id))
```

---

## 3. LLaMA 架构改进（LLaMA Architecture Improvements）

LLaMA（Meta, 2023）是开源社区最重要的基础模型之一，其架构改进被广泛采用。

### 3.1 RMSNorm（Root Mean Square Layer Normalization）

替代 LayerNorm，去除均值中心化：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma$$

**优势**：
- 计算更简单（少一次均值计算）
- 在 LLM 中效果与 LayerNorm 相当甚至更优
- 现代 LLM 普遍采用

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
```

### 3.2 SwiGLU 激活函数

替代 Transformer 中的 ReLU/GELU：

$$\text{SwiGLU}(x) = \text{Swish}(xW + b) \otimes (xV + c)$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$，$\otimes$ 是逐元素乘法。

**为什么有效**：
- GLU（Gated Linear Unit）门控机制增加表达能力
- Swish（SiLU）平滑非线性，梯度性质更好
- 实践经验表明比 ReLU/GELU 效果更好

**维度调整**：使用 SwiGLU 时，中间维度通常设为 $\frac{2}{3} \times 4d$ 而非 $4d$，保持参数量相当。

### 3.3 RoPE（Rotary Position Embedding，旋转位置编码）

LLaMA 不使用绝对位置编码，而是将位置信息融入 Attention 的 Q、K 中。

**核心思想**：将每对维度 $(d_{2i}, d_{2i+1})$ 视为一个二维向量，根据位置 $m$ 旋转一定角度：

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

其中旋转角度 $\theta_i = 10000^{-2i/d}$。

**关键特性**：
- **相对位置编码**：$\text{RoPE}(q_m, k_n)$ 只依赖于相对距离 $m-n$
- **外推性**：可以处理比训练时更长的序列（配合位置插值等技术）
- **现代标准**：LLaMA、Qwen、Baichuan 等主流模型均采用

```python
import torch

def get_rotary_embedding(seq_len, dim, base=10000):
    """生成 RoPE 的旋转矩阵"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len)
    angles = torch.outer(positions, inv_freq)  # (seq_len, dim/2)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin):
    """将 RoPE 应用于输入张量 x"""
    # x: (batch, num_heads, seq_len, head_dim)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rotated * sin
```

---

## 4. 注意力机制优化（Attention Optimization）

### 4.1 Sparse Attention（稀疏注意力）

标准 Self-Attention 的复杂度为 $O(n^2)$，长序列时成为瓶颈。稀疏注意力通过限制每个位置的注意力范围来降低复杂度。

**常见模式**：

| 模式 | 结构 | 复杂度 | 特点 |
|------|------|--------|------|
| **Global** | 全局 token 可关注所有位置 | $O(n)$ | 适合分类任务，需选择哪些 token 是全局的 |
| **Band（滑动窗口）** | 每个位置只关注附近 $w$ 个位置 | $O(n \cdot w)$ | 适合局部依赖强的任务 |
| **Dilated** | 在 band 中跳步采样 | $O(n \cdot w / d)$ | 进一步扩大感受野 |
| **Random** | 随机选择若干位置关注 | $O(n \cdot r)$ | 补充长距离连接 |
| **Block Local** | 将序列分块，块内全连接 | $O(n \cdot b)$ | 适合长文档建模 |

**代表模型**：
- **Longformer**：Global + Band + Dilated 组合
- **BigBird**：Global + Band + Random，理论上可表达任意 Turing 机
- **Sparse Transformer**：跨步 Attention 模式

### 4.2 FlashAttention

**核心问题**：标准 Attention 的 $O(n^2)$ 内存需求导致长序列时 GPU HBM 内存不足。

**核心思想**：
1. **分块计算**：将 Q、K、V 分成小块，在 GPU SRAM（高速缓存）中完成 Attention 计算
2. **避免存储中间矩阵**：不存储完整的 $S = QK^T$ 和 $P = \text{softmax}(S)$
3. **在线 softmax**：通过统计量（max、sum）逐块计算 softmax

**算法流程**：
```
将 Q, K, V 按行分成块
对每个 Q 块:
    初始化 m = -inf, l = 0, o = 0
    对每个 K, V 块:
        S_ij = Qi * Kj^T          # 局部 attention scores
        m_new = max(m, max(S_ij))  # 更新 running max
        P_ij = exp(S_ij - m_new)   # 局部 softmax 分子
        l = l * exp(m - m_new) + sum(P_ij)  # 更新 running sum
        o = o * exp(m - m_new) + P_ij @ Vj  # 更新输出
        m = m_new
    o = o / l  # 最终归一化
```

**优势**：
- **内存高效**：从 $O(n^2)$ 降至 $O(n)$
- **计算等价**：数值结果与标准 Attention 完全一致
- **IO 感知**：最大化利用 GPU 内存层次结构
- **训练加速**：长序列训练显著提速

**限制**：
- 主要优化训练，推理时的 KV Cache 优化是另一个问题
- 需要特定硬件支持（GPU SRAM 足够大）

### 4.3 Multi-Query Attention（MQA）

**动机**：Decoder 推理时，每个新 token 都需要计算与所有历史 token 的 Attention。标准 Multi-Head Attention 需要缓存每个 head 的 K、V，内存开销大。

**核心思想**：所有 Attention Head 共享同一组 K、V 投影，只有 Q 保持多头：

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)      # 多头 Q
        self.W_k = nn.Linear(embed_dim, self.head_dim)  # 单头 K（共享）
        self.W_v = nn.Linear(embed_dim, self.head_dim)  # 单头 V（共享）
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, past_k=None, past_v=None):
        batch, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        V = self.W_v(x).unsqueeze(1)  # (batch, 1, seq_len, head_dim)

        # K, V 在所有 head 间广播
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_o(out)
```

**效果**：
- KV Cache 内存从 $O(batch \times num\_heads \times seq \times head\_dim)$ 降至 $O(batch \times seq \times head\_dim)$
- 推理速度显著提升，尤其是长序列
- 略微牺牲一点质量，但通常可接受

**变体 GQA（Grouped-Query Attention）**：
- 折中方案：将 heads 分成若干组，每组共享 K、V
- 例如 8 个 head 分成 2 组，每组 4 个 head 共享 K、V
- LLaMA-2/3、Qwen 等模型采用

---

## 5. 工程实践要点（Engineering Best Practices）

### 5.1 训练稳定性

- **梯度裁剪**：限制梯度范数，防止 loss spike
- **混合精度训练**：FP16/BF16 + loss scaling，加速且省内存
- **激活检查点**：以计算换内存，重计算中间激活值
- **Pre-LN**：LayerNorm 放在残差连接之前，训练更稳定

### 5.2 分布式训练策略

| 策略 | 切分维度 | 适用场景 |
|------|----------|----------|
| **数据并行（DP）** | 批次维度 | 模型可放入单卡 |
| **模型并行（MP/TP）** | 参数维度 | 模型太大，单卡放不下 |
| **流水线并行（PP）** | 层维度 | 超大规模模型 |
| **FSDP** | 参数 + 梯度 + 优化器状态 | 推荐方案，PyTorch 原生支持 |
| **3D 并行** | 上述组合 | 万亿参数模型 |

### 5.3 推理优化

- **KV Cache**：缓存历史 K、V，避免重复计算
- **量化**：INT8/INT4 权重，减少内存占用
- **连续批处理（Continuous Batching）**：动态调度，提高 GPU 利用率
- **推测解码（Speculative Decoding）**：用小模型草稿 + 大模型验证，加速生成

---

## 学习建议

1. **先动手实现简化版 Transformer**：理解维度变换是关键
2. **对比 Pre-Norm 和 Post-Norm**：观察训练稳定性差异
3. **使用 HuggingFace 调试**：加载真实模型，打印每层输出形状
4. **关注 FlashAttention 和 MQA**：现代推理优化的核心技术
