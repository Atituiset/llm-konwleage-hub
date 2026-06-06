# 大模型经典论文导读

> **资料来源**：本目录下的分类论文压缩包
> **适合人群**：希望深入理解大模型技术演进的研究者
> **难度**：⭐⭐⭐⭐⭐（很难）

---

## 论文阅读方法论

阅读顶会论文时，建议按以下框架分析：

1. **问题定义**：论文解决什么问题？为什么重要？
2. **方法创新**：核心思想是什么？与之前工作有何不同？
3. **实验验证**：在哪些数据集/任务上验证？结果如何？
4. **影响分析**：对后续工作的影响？当前是否仍适用？

---

## 01 语言模型基础

### A Neural Probabilistic Language Model (Bengio, 2003)

**问题定义**：传统 n-gram 语言模型受限于稀疏性和维度灾难，无法捕捉词之间的语义相似性。

**方法创新**：
- 首次提出用神经网络学习分布式词表示
- 词嵌入矩阵 $C \in \mathbb{R}^{|V| \times m}$，将每个词映射为 $m$ 维稠密向量
- 用前馈网络预测下一个词的概率：

$$P(w_t | w_{t-n+1}, ..., w_{t-1}) = f(w_t, w_{t-1}, ..., w_{t-n+1})$$

**影响**：开创了"神经网络 + 分布式表示"的语言建模范式，为后续 Word2Vec、Transformer 奠定基础。

---

### Word2Vec (Mikolov, 2013)

**问题定义**：Bengio 的模型太慢，无法处理大规模语料。如何在保持分布式表示优势的同时大幅提升训练速度？

**方法创新**：
- 提出两种简化架构：
  - **CBOW（Continuous Bag-of-Words）**：用上下文词预测中心词
  - **Skip-gram**：用中心词预测上下文词

- **负采样（Negative Sampling）**：将多分类问题转化为二分类问题：

$$\log \sigma(v_{w_O}'^T v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-v_{w_i}'^T v_{w_I}) \right]$$

其中 $w_O$ 是输出词，$w_I$ 是输入词，$w_i$ 是负样本。

- **层次 Softmax**：用霍夫曼树结构将复杂度从 $O(|V|)$ 降至 $O(\log |V|)$

**实验设计**：在 Google News 数据集（100B words）上训练，展示了著名的"king - man + woman ≈ queen"语义类比。

**影响**：
- 使词嵌入成为 NLP 标准组件
- 证明了无监督预训练的有效性
- 启发了后续句嵌入、文档嵌入的研究

---

### Attention Is All You Need (Vaswani, 2017)

**问题定义**：RNN/LSTM 的序列计算无法并行，长距离依赖捕捉困难。能否完全用 Attention 替代循环结构？

**方法创新**：

**核心：Self-Attention 机制**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**完整架构**：
- Encoder：6 层 × (Multi-Head Self-Attention + FFN)
- Decoder：6 层 × (Masked Self-Attention + Cross-Attention + FFN)
- 每个子层使用残差连接 + LayerNorm
- 位置编码使用正弦/余弦函数

**关键设计决策**：
1. **为什么是 $\sqrt{d_k}$？** 防止点积过大导致 softmax 饱和（见前面章节的统计学分析）
2. **为什么多头？** 不同子空间学习不同的关系模式
3. **为什么 FFN 中间层 4d？** 提供足够的记忆容量

**实验设计**：
- 数据集：WMT 2014 英德（4.5M 句对）、英法（36M 句对）
- 结果：英德 BLEU 28.4（SOTA），训练时间 12 小时（8 张 P100）
- 与基于 RNN 的模型相比：训练更快、质量更好、长距离依赖更强

**影响**：
- 彻底改变了 NLP 架构设计
- "Attention Is All You Need" 的标题本身成为经典
- Transformer 成为所有后续大模型的基础

---

## 02 大语言模型

### GPT-1: Improving Language Understanding by Generative Pre-Training (Radford, 2018)

**问题定义**：标注数据昂贵且有限，如何充分利用大量未标注文本？

**方法创新**：
- **两阶段训练**：
  1. **无监督预训练**：在大规模语料上用语言建模目标训练
  2. **有监督微调**：在下游任务上微调

- **预训练目标**：

$$\mathcal{L}_1 = \sum_{i} \log P(w_i | w_{i-k}, ..., w_{i-1}; \Theta)$$

- **微调目标**（增加任务相关的线性层）：

$$\mathcal{L}_2 = \sum_{(x, y)} \log P(y | x; \Theta) + \lambda \mathcal{L}_1$$

**关键洞察**：预训练学到的通用语言表示可以通过微调迁移到各种下游任务。

**影响**：确立了"预训练 + 微调"的范式。

---

### BERT: Pre-training of Deep Bidirectional Transformers (Devlin, 2018)

**问题定义**：GPT 使用单向（左到右）注意力，无法充分利用双向上下文。如何设计真正的双向预训练？

**方法创新**：
- **Masked Language Model（MLM）**：随机 mask 15% 的 token，模型预测被 mask 的内容
  - 80% 概率用 [MASK] 替换
  - 10% 概率用随机词替换
  - 10% 概率保持不变

- **Next Sentence Prediction（NSP）**：判断句子 B 是否是句子 A 的下一句

**架构**：
- 仅使用 Transformer Encoder（双向注意力）
- Base: 12 层，768 维，12 头，110M 参数
- Large: 24 层，1024 维，16 头，340M 参数

**输入表示**：
```
[CLS] 句子 A [SEP] 句子 B [SEP]
```

**实验设计**：
- 预训练：BooksCorpus（800M 词）+ Wikipedia（2,500M 词）
- 在 11 个 NLP 任务上微调，全面超越 SOTA

**影响**：
- 与 GPT 形成"Encoder vs Decoder"两大路线
- 双向预训练成为理解任务的标准
- 启发了 RoBERTa、ALBERT、DeBERTa 等改进

---

### GPT-2: Language Models are Unsupervised Multitask Learners (Radford, 2019)

**问题定义**：能否取消微调阶段，直接用预训练模型完成多种任务？

**方法创新**：
- **零样本能力（Zero-shot）**：在 prompt 中描述任务，模型直接生成答案
- **WebText 数据集**：从 Reddit 外链抓取的高质量网页，约 40GB

**规模扩展**：
- 参数量：1.5B（GPT-1 是 117M）
- 层数：48 层
- 上下文长度：1024

**关键发现**：
- 模型在没有针对任何任务训练的情况下，在多个任务上达到有竞争力的表现
- 证明了"模型越大，零样本能力越强"的趋势

**影响**：
- 开启了"更大模型 → 更强能力"的 Scaling 路线
- 展示了 In-Context Learning 的雏形

---

### GPT-3: Language Models are Few-Shot Learners (Brown, 2020)

**问题定义**：GPT-2 的零样本能力有限。扩大规模能否实现真正的少样本学习？

**方法创新**：
- **规模飞跃**：175B 参数（GPT-2 的 100 倍）
- **In-Context Learning**：在 prompt 中提供几个示例（few-shot），模型就能执行新任务，无需梯度更新
- **提示工程（Prompt Engineering）**：任务描述 + 示例的格式设计

**不同学习模式的对比**：

| 模式 | 格式 | 示例数量 |
|------|------|---------|
| Fine-tuning | 梯度更新 | 数千到数万 |
| Few-shot | Prompt 中给示例 | 10-100 |
| One-shot | Prompt 中给 1 个示例 | 1 |
| Zero-shot | 仅描述任务 | 0 |

**训练数据**：
- Common Crawl（过滤后）：410B tokens
- WebText2：19B
- Books1+Books2：12B
- Wikipedia：3B

**关键发现**：
- 175B 模型在翻译、问答、完形填空等任务上达到 SOTA
- 证明了"涌现能力"：某些能力只在足够大的模型中出现
- 为后续大模型发展指明了方向

**影响**：
- 确立了 LLM 的核心范式：大模型 + 提示工程
- 催生了 Prompt Engineering 的研究方向
- 直接推动了 ChatGPT 的诞生

---

### InstructGPT: Training Language Models to Follow Instructions (Ouyang, 2022)

**问题定义**：GPT-3 擅长续写文本，但不擅长遵循用户指令。如何对齐模型行为与人类意图？

**方法创新**：RLHF（Reinforcement Learning from Human Feedback）的三阶段流程：

```
阶段 1: SFT（Supervised Fine-Tuning）
   用标注者编写的 (prompt, response) 数据微调 GPT-3

阶段 2: Reward Modeling
   对同一 prompt 的多个回答排序，训练奖励模型

阶段 3: PPO
   用奖励模型指导策略模型优化
```

**核心洞察**：
- 人类偏好比绝对评分更容易标注（排序 > 打分）
- RL 可以让模型超越训练数据的质量上限

**实验结果**：
- 1.3B 的 InstructGPT 比 175B 的 GPT-3 更受人类偏好
- 证明了**对齐（Alignment）**比单纯扩大规模更重要

**影响**：
- 奠定了 ChatGPT/GPT-4 的技术基础
- RLHF 成为对齐大模型的标准方法

---

### LLaMA: Open and Efficient Foundation Language Models (Touvron, 2023)

**问题定义**：如何训练一个开源、高效、性能优异的基础模型？

**方法创新**：
- **架构优化**：
  - Pre-Norm（RMSNorm）
  - SwiGLU 激活
  - RoPE 位置编码
- **数据优化**：
  - 只用公开数据集（无版权争议）
  - 1.4T tokens（LLaMA-65B）
  - 高质量过滤和去重

**不同规模**：

| 模型 | 参数量 | 训练数据 | 训练时间 |
|------|--------|---------|---------|
| LLaMA-7B | 6.7B | 1.0T | 82,432 A100-hours |
| LLaMA-13B | 13.0B | 1.0T | 135,168 A100-hours |
| LLaMA-33B | 32.5B | 1.4T | 530,432 A100-hours |
| LLaMA-65B | 65.2B | 1.4T | 1,022,362 A100-hours |

**影响**：
- 推动了开源 LLM 生态的爆发
- 成为后续 Alpaca、Vicuna、WizardLM 等模型的基础
- 证明了数据质量和训练效率的重要性

---

### Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron, 2023)

**问题定义**：LLaMA 仅用于研究，无法商用。如何构建可商用的开源模型？

**方法创新**：
- **上下文长度从 2K 扩展到 4K**
- **分组查询注意力（GQA）**：提升推理速度
- **两阶段 SFT**：先用公开数据，再用高质量内部数据
- **RLHF**：多轮迭代，拒绝采样（Rejection Sampling）+ PPO

**模型系列**：
- 基础模型：7B、13B、34B、70B
- Chat 模型：经过 SFT + RLHF 的对齐版本

**影响**：
- 首个可商用的高性能开源 LLM
- 成为企业部署的首选开源模型

---

## 03 Prompt 工程

### Chain-of-Thought Prompting Elicits Reasoning in LLMs (Wei, 2022)

**问题定义**：大模型在复杂推理任务（数学、常识推理）上表现不佳。如何在不微调的情况下提升推理能力？

**方法创新**：
- **Chain-of-Thought（CoT）**：在 prompt 的示例中包含推理过程，引导模型逐步思考

```
Q: Roger 有 5 个网球，又买了 2 罐，每罐 3 个。他现在有几个？
A: Roger 原有 5 个。每罐 3 个，2 罐是 2*3=6 个。5+6=11。答案是 11。

Q: [新问题]
A:
```

**关键发现**：
- CoT 只在足够大的模型（~100B）上有效
- 与标准 few-shot 相比，在 GSM8K（数学）上准确率从 17.9% 提升到 58.1%

**影响**：
- 启发了后续多步推理、Self-Consistency、Tree of Thoughts 等研究

---

### ReAct: Synergizing Reasoning and Acting in Language Models (Yao, 2022)

**问题定义**：如何将推理与外部工具使用结合？

**方法创新**：
- **ReAct 模式**：交替进行"思考（Thought）"和"行动（Action）"

```
思考：我需要查找当前温度
行动：调用 weather_api(location="北京")
观察：25°C

思考：25°C 适合户外活动
行动：调用 final_answer("适合")
```

**影响**：
- 奠定了 Agent 框架的基础
- 启发了 LangChain、AutoGPT 等工具

---

## 04 参数高效微调

### LoRA: Low-Rank Adaptation of Large Language Models (Hu, 2021)

**问题定义**：全参数微调 LLM 需要大量显存和计算。如何高效地将 LLM 适配到下游任务？

**方法创新**：
- **低秩分解**：不训练全部参数，只训练低秩矩阵

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

**关键假设**：
- 预训练权重矩阵是过参数化的，其内在有效秩很低
- 低秩微调足以捕捉任务特定信息

**实现细节**：
- $A$ 用随机高斯初始化
- $B$ 用零初始化 → 初始时 $\Delta W = 0$
- 训练时只优化 $A$ 和 $B$
- 推理时 $W' = W + BA$ 可以合并，不增加延迟

**实验结果**：
- 在 GPT-3 175B 上，LoRA 将可训练参数量从 175B 降至 4.7M（0.0026%）
- 在多个任务上达到或超越全参数微调

**影响**：
- 成为参数高效微调的标准方法
- 催生了 QLoRA、AdaLoRA、DoRA 等改进

---

### QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers, 2023)

**问题定义**：LoRA 虽然减少了可训练参数，但基模型仍需全精度加载，单卡 24GB 无法微调 65B 模型。

**方法创新**：
1. **4-bit 量化**：将模型权重量化为 4-bit（Normal Float 4）
2. **双量化**：量化常数也量化
3. **分页优化器**：将优化器状态分页到 CPU

**Normal Float 4（NF4）**：
- 根据权重分布（近似高斯）优化量化 bin 的位置
- 比均匀 4-bit 量化信息损失更小

**实验结果**：
- 在单张 48GB GPU 上微调 65B 模型
- 达到 16-bit LoRA 的 99.3% 性能

**影响**：
- 使个人研究者也能微调大模型
- 推动了开源模型微调的普及

---

## 05 检索增强生成

### RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis, 2020)

**问题定义**：大模型参数中存储的知识有限且固定，如何动态引入外部知识？

**方法创新**：
- **端到端可训练**：检索器和生成器联合训练
- **稠密检索**：用 DPR（Dense Passage Retrieval）编码文档和查询
- **生成器**：BART 基于检索到的文档生成答案

**架构**：
```
输入 query
   ↓
[检索器] → Top-K 文档
   ↓
[生成器] → query + 文档 → 生成答案
```

**影响**：
- 开创了 RAG 范式
- 成为解决 LLM 幻觉和知识截止问题的主流方案

---

## 论文阅读路线图

### 入门（必读）：
```
Word2Vec (2013)
    ↓
Attention Is All You Need (2017)
    ↓
BERT (2018)  ←→  GPT-1/2 (2018/2019)
    ↓
GPT-3 (2020)
    ↓
InstructGPT (2022)
```

### 进阶（选读方向）：

| 方向 | 论文序列 |
|------|---------|
| **对齐方向** | InstructGPT → RLHF 系列 → DPO |
| **效率方向** | LoRA → QLoRA → 量化系列 |
| **应用方向** | RAG → ReAct → Agent 系列 |
| **架构方向** | Mamba → Mixture of Experts → 状态空间模型 |

---

## 演进逻辑总结

从 Word2Vec 到 GPT-4 的演进，核心驱动力有三条主线：

**主线一：表示学习**
```
Word2Vec (静态词向量)
  ↓
ELMo (上下文相关词向量)
  ↓
BERT/GPT (Transformer 编码上下文)
  ↓
GPT-3/4 (In-Context Learning，无需微调)
```

**主线二：规模扩展**
```
GPT-1 (117M) → GPT-2 (1.5B) → GPT-3 (175B) → GPT-4 (~1T?)
关键发现：规模达到阈值后，能力"涌现"
```

**主线三：人机对齐**
```
预训练模型（续写文本）
  ↓
SFT（学会遵循指令）
  ↓
RLHF/DPO（符合人类偏好）
  ↓
多轮迭代（持续改进）
```

理解这三条主线，就理解了大模型技术的核心发展逻辑。
