# 第七阶段：大模型应用开发

> **资料来源**：综合《OpenClaw 橙皮书》《DeepSeek 行业应用案例集》《DeepSeek 赋能职场》《吴恩达大模型通关手册》
> **适合人群**：希望将大模型技术落地为实际应用的开发者
> **难度**：⭐⭐⭐（中等）

---

## 1. 大模型应用开发全景

### 1.1 应用开发的三种范式

| 范式 | 原理 | 适用场景 | 开发难度 |
|------|------|----------|----------|
| **Prompt Engineering** | 优化输入提示，引导模型输出 | 快速原型、通用任务 | 低 |
| **RAG（检索增强）** | 外挂知识库，动态注入上下文 | 企业知识问答、专业领域 | 中 |
| **Fine-tuning（微调）** | 调整模型参数适配特定任务 | 特定风格、私有数据、性能要求高 | 高 |

**选择原则**：
- 先用 Prompt Engineering 验证可行性
- 知识需要动态更新 → 用 RAG
- 需要改变模型行为/风格 → 用 Fine-tuning
- 两者常组合使用：RAG 提供上下文 + Fine-tuned 模型生成

### 1.2 典型应用架构

```
用户请求
   ↓
[网关/API] ──→ 限流、鉴权、日志
   ↓
[编排层] ──→ 路由、记忆管理、多轮对话状态
   ↓
┌─────────────┬─────────────┬─────────────┐
│   RAG 模块   │   Agent 模块 │  直接调用    │
│ 向量检索+重排 │ 工具调用+推理 │  Prompt     │
└─────────────┴─────────────┴─────────────┘
   ↓
[模型服务] ──→ vLLM / TGI / 云 API
   ↓
后处理 ──→ 安全过滤、格式校验、输出
```

---

## 2. RAG（检索增强生成）

### 2.1 为什么需要 RAG

大语言模型的局限性：
- **知识截止**：训练后发生的事件不知道
- **幻觉**：可能生成看似合理但错误的内容
- **私有数据**：无法访问企业内部文档
- **可解释性**：无法追溯答案来源

RAG 通过**检索外部知识**解决上述问题。

### 2.2 RAG 核心流程

```
用户提问
   ↓
[查询理解] ──→ Query 改写、扩展
   ↓
[向量检索] ──→ 从向量数据库召回 Top-K 文档
   ↓
[重排序] ──→ 更精确的排序模型筛选
   ↓
[上下文构建] ──→ 将相关文档拼接为 Prompt 上下文
   ↓
[LLM 生成] ──→ 基于上下文生成回答
   ↓
[引用标注] ──→ 指出答案来源
```

### 2.3 文档处理 pipeline

**Step 1: 加载**
```python
from langchain.document_loaders import PyPDFLoader, TextLoader

# 支持 PDF、Word、Markdown、网页等多种格式
loader = PyPDFLoader("document.pdf")
docs = loader.load()  # List[Document]
```

**Step 2: 切分（Chunking）**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块大小（token 数或字符数）
    chunk_overlap=50,    # 相邻块重叠，保持上下文连贯
    separators=["\n\n", "\n", "。", " ", ""]  # 优先级切分
)
chunks = splitter.split_documents(docs)
```

**Chunking 策略对比**：

| 策略 | 特点 | 适用 |
|------|------|------|
| 固定长度 | 简单、均匀 | 通用场景 |
| 语义切分 | 按句子/段落边界 | 需要保持语义完整 |
| 递归切分 | 优先在大分隔符处切分 | 结构化文档 |
| Agentic | 让模型决定切分点 | 高质量要求 |

**Chunk 大小选择**：
- 太小（<200）：丢失上下文，检索碎片
- 太大（>1000）：噪声多，超出模型上下文限制
- 推荐：300-800 tokens，配合 10-20% 重叠

### 2.4 Embedding 与向量数据库

**Embedding 模型**：将文本映射为语义向量
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings = model.encode(["这是一段文本", "这是另一段"])
# embeddings.shape = (2, 1024)
```

**常用 Embedding 模型**：

| 模型 | 维度 | 语言 | 特点 |
|------|------|------|------|
| BGE-Large | 1024 | 多语言 | 开源 SOTA，推荐使用 |
| GTE-Large | 1024 | 多语言 | 阿里出品，效果优秀 |
| M3E | 768 | 中文 | 中文社区常用 |
| OpenAI text-embedding-3 | 3072 | 多语言 | API 调用，效果好 |
| E5-Mistral | 4096 | 多语言 | 大模型-based，效果顶级 |

**向量数据库选择**：

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| **Milvus/Zilliz** | 企业级、分布式 | 大规模生产环境 |
| **Pinecone** | 全托管、易用 | 快速上线、无运维 |
| **Weaviate** | 自带向量化、GraphQL | 需要语义搜索+过滤 |
| **Chroma** | 轻量、本地优先 | 开发测试、中小规模 |
| **pgvector** | PostgreSQL 扩展 | 已有 PG 基础设施 |
| **Qdrant** | Rust 编写、高性能 | 自托管、高性能要求 |
| **Faiss** | Meta 开源、纯检索 | 研究、自定义系统 |

### 2.5 检索策略

**基础检索**：向量相似度搜索
```python
# 余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 在向量数据库中检索
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where={"category": "技术文档"}  # 元数据过滤
)
```

**混合检索（Hybrid Search）**：
向量检索（语义匹配）+ 关键词检索（BM25/TF-IDF）结合：
```python
# 分别获取向量检索和关键词检索结果
vector_results = vector_search(query, k=20)
keyword_results = bm25_search(query, k=20)

# RRF (Reciprocal Rank Fusion) 融合
combined = rrf_fusion([vector_results, keyword_results], k=60)
```

**查询改写（Query Rewriting）**：
```python
# 原始查询可能不准确，用模型改写
rewrite_prompt = f"""
将以下用户问题改写为更适合向量检索的查询，
保持原意但使用更正式、更完整的表达。

用户问题：{query}
改写后："""

rewritten_query = llm.generate(rewrite_prompt)
```

### 2.6 重排序（Reranking）

召回的 Top-K 可能包含噪声，用更强的模型精排：

```python
from sentence_transformers import CrossEncoder

# Cross-Encoder 直接对 (query, doc) 对打分，更准确但慢
reranker = CrossEncoder('BAAI/bge-reranker-large')

pairs = [[query, doc.page_content] for doc in retrieved_docs]
scores = reranker.predict(pairs)

# 按分数重排，取 Top-5
reranked = sorted(zip(scores, retrieved_docs), reverse=True)[:5]
```

### 2.7 完整 RAG 代码示例

```python
from langchain import OpenAI, VectorDBQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

# 1. 加载文档
loader = DirectoryLoader('./docs', glob='**/*.md')
docs = loader.load()

# 2. 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 生成 Embedding 并存入向量库
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-zh-v1.5')
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory='./chroma_db')

# 4. 构建 RAG Chain
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type='stuff',  # 将所有文档塞入 prompt
    retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True
)

# 5. 查询
result = qa({'query': '公司的年假政策是什么？'})
print(result['result'])
print('来源：', [d.metadata['source'] for d in result['source_documents']])
```

### 2.8 RAG 评估

| 指标 | 含义 | 计算方法 |
|------|------|----------|
| **Context Precision** | 检索到的文档中有多少是相关的 | 相关文档数 / 检索文档数 |
| **Context Recall** | 相关文档有多少被检索到 | 检索到的相关文档 / 全部相关文档 |
| **Faithfulness** | 生成内容是否忠实于检索文档 | 用 LLM 判断每个陈述是否被文档支持 |
| **Answer Relevance** | 回答是否与问题相关 | LLM 评分 |
| **Answer Correctness** | 回答是否正确 | 与标准答案对比 |

**评估框架**：RAGAS、LangChain Evals、TruLens

### 2.9 RAG 进阶：GraphRAG

传统 RAG 的问题：无法捕捉实体间的关系，难以回答需要综合多个文档的复杂问题。

**GraphRAG 流程**：
1. 从文档中提取实体和关系，构建知识图谱
2. 查询时先在图谱中做社区发现/路径搜索
3. 将图谱子结构和相关文本一起送入 LLM

**工具**：Neo4j + LLM Graph Builder、Microsoft GraphRAG

---

## 3. Agent（智能体）

### 3.1 什么是 Agent

Agent = LLM + 工具调用 + 规划 + 记忆

与简单 Chatbot 的区别：
- Chatbot：一问一答，单轮交互
- Agent：可自主决策、调用工具、多步执行、使用记忆

### 3.2 ReAct 模式（Reasoning + Acting）

核心思想：让模型交替进行"思考"和"行动"，直到完成任务。

```
思考：我需要查找北京的天气
行动：调用 weather_api(location="北京")
观察：{"temperature": 25, "condition": "晴"}

思考：用户想知道北京今天是否适合出门，25度晴天很适合
行动：调用 final_answer("北京今天 25°C，晴天，非常适合出门！")
```

**Prompt 模板**：
```
你可以使用以下工具：
{tool_descriptions}

按以下格式响应：
思考：你当前的想法
行动：工具名称（参数）
观察：工具返回的结果
...（可重复多轮）

思考：我现在知道最终答案
最终答案：给用户的结果
```

### 3.3 Function Calling（工具调用）

现代 LLM 支持结构化工具调用：

```python
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "搜索产品数据库",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string", "enum": ["电子产品", "服装", "食品"]}
                },
                "required": ["query"]
            }
        }
    }
]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京明天天气怎么样？"}],
    tools=tools,
    tool_choice="auto"
)

# 模型决定调用 get_weather
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(tool_call.function.name)  # "get_weather"
    print(tool_call.function.arguments)  # '{"city": "北京", "date": "2024-01-16"}'
```

**主流模型的 Function Calling 支持**：

| 模型 | 原生支持 | 格式 |
|------|----------|------|
| GPT-4.1/GPT-4o | ✅ | OpenAI 标准 |
| Claude | ✅ | Anthropic 标准 |
| DeepSeek-V3 | ✅ | OpenAI 兼容 |
| Qwen | ✅ | OpenAI 兼容 |
| LLaMA | ⚠️ | 需通过 prompt 引导或微调 |

### 3.4 Agent 框架对比

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| **LangChain** | 生态最完善、组件丰富 | 快速原型、复杂工作流 |
| **LlamaIndex** | 数据层抽象强、RAG 友好 | 知识库应用、数据代理 |
| **AutoGPT** | 自主目标驱动 | 实验性、自动化任务 |
| **Dify** | 可视化编排、开源 | 低代码搭建、团队协作 |
| **CrewAI** | 多 Agent 协作 | 团队协作模拟 |
| **LangGraph** | 图结构 Agent 编排 | 复杂工作流、状态管理 |
| **OpenClaw** | 自托管、多平台接入 | 私有化部署、社交集成 |

### 3.5 记忆系统

Agent 需要记忆来维持上下文：

| 记忆类型 | 存储内容 | 实现方式 |
|----------|----------|----------|
| **短期记忆** | 当前对话历史 | 直接传入 messages |
| **长期记忆** | 跨对话的用户信息 | 向量数据库 + 摘要 |
| **实体记忆** | 提取的实体和关系 | 知识图谱 |

**记忆压缩**：
对话过长时，用 LLM 生成摘要替代原始消息：
```python
summary_prompt = f"""
将以下对话历史总结为关键信息，保留重要事实：
{conversation_history}
"""
summary = llm.generate(summary_prompt)
# 后续对话使用 summary 替代原始历史
```

### 3.6 多 Agent 协作

复杂任务可分解为多个 Specialist Agent：

```
[用户请求]
   ↓
[Router Agent] ──→ 分析意图，分配给对应 Agent
   ↓
┌─────────────┬─────────────┬─────────────┐
│ Researcher  │  Coder      │  Writer     │
│ 搜索信息    │  编写代码   │  撰写文档   │
└─────────────┴─────────────┴─────────────┘
   ↓
[Integrator Agent] ──→ 汇总结果，生成最终输出
```

---

## 4. 模型微调（Fine-tuning）

### 4.1 什么时候需要微调

| 场景 | 解决方案 | 说明 |
|------|----------|------|
| 需要特定领域知识 | RAG 优先 | 知识经常更新时更适合 |
| 需要特定输出风格 | Fine-tuning | 如客服语气、法律文书格式 |
| 需要极低延迟 | Fine-tuning + 小模型 | 蒸馏到更小模型 |
| 私有数据不能外传 | 本地 Fine-tuning | 数据安全要求 |
| Prompt 工程已达极限 | Fine-tuning | 突破性能天花板 |

### 4.2 LoRA（Low-Rank Adaptation）

**核心思想**：不训练全部参数，只在每层注入低秩矩阵，冻结原模型参数。

```python
# 原始权重 W ∈ R^{d×k}
# LoRA: W' = W + ΔW = W + BA
# 其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d, k)
```

**为什么有效**：
- 预训练权重矩阵是过参数化的，其有效秩很低
- 低秩微调足以捕捉任务特定信息
- 可训练参数量降至 0.1% ~ 1%

**PyTorch 实现**：
```python
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        self.rank = rank
        # 只训练这两个小矩阵
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        # A 用随机初始化，B 用零初始化 → 初始时 ΔW = 0

    def forward(self, x):
        return x @ self.A @ self.B

# 应用到 Transformer 的 Attention 和 FFN 层
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8):
        super().__init__()
        self.linear = linear  # 原始层，冻结
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank)
        # 冻结原始参数
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

**关键超参数**：

| 参数 | 含义 | 常用值 |
|------|------|--------|
| `rank` (r) | 低秩维度 | 4, 8, 16, 32, 64 |
| `alpha` | 缩放系数 | 通常 = rank 或 2×rank |
| `dropout` | LoRA 层的 dropout | 0.0 ~ 0.1 |
| `target_modules` | 应用 LoRA 的模块 | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |

**使用 PEFT 库**：
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 33M || all params: 7B || trainable%: 0.47
```

### 4.3 QLoRA（Quantized LoRA）

**问题**：7B 模型需要 ~14GB 显存（FP16），单卡 4090（24GB）几乎满载，无法微调。

**QLoRA 方案**：
1. **4-bit 量化**：将模型权重量化为 4-bit（NF4 或 FP4），显存降至 ~4GB
2. **双量化**：量化常数也量化，进一步省内存
3. **分页优化器**：CPU  offloading，避免 OOM

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"  # 自动分配层到 GPU/CPU
)

# 应用 LoRA（同上）
model = get_peft_model(model, lora_config)

# 现在 7B 模型可在 16GB 显存上微调！
```

**显存需求对比**：

| 方法 | 7B 模型显存 | 13B 模型显存 | 70B 模型显存 |
|------|------------|-------------|-------------|
| 全参数微调 (FP16) | ~28GB | ~52GB | ~280GB |
| LoRA (FP16) | ~14GB | ~26GB | ~140GB |
| QLoRA (4-bit) | ~6GB | ~10GB | ~48GB |

### 4.4 指令微调数据格式

标准对话格式：
```json
{
  "messages": [
    {"role": "system", "content": "你是一位专业的技术文档工程师。"},
    {"role": "user", "content": "请解释什么是 RAG？"},
    {"role": "assistant", "content": "RAG（Retrieval-Augmented Generation）是一种将检索系统与生成模型结合的技术..."}
  ]
}
```

**数据质量原则**：
- 多样性：覆盖尽可能多的场景
- 质量 > 数量：1 万条高质量 > 10 万条低质量
- 格式一致：统一的 system/user/assistant 格式
- 避免污染：确保测试集数据不在训练集中

### 4.5 训练参数

```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 有效 batch = 4×4 = 16
    learning_rate=2e-4,             # LoRA 通常用较大学习率
    warmup_ratio=0.03,
    lr_scheduler_type='cosine',
    logging_steps=10,
    save_strategy='epoch',
    fp16=True,
)
```

**学习率选择**：
- 全参数微调：1e-5 ~ 5e-5
- LoRA：1e-4 ~ 1e-3（可更高，因为参数量小）

---

## 5. 模型部署与服务

### 5.1 部署方式对比

| 方式 | 优点 | 缺点 | 适用 |
|------|------|------|------|
| **云 API** | 免运维、即开即用 | 数据外泄风险、按量计费 | 快速验证、非敏感数据 |
| **私有化部署** | 数据安全、可控 | 运维成本高 | 金融、医疗、政务 |
| **边缘部署** | 低延迟、离线可用 | 模型受限 | 移动端、IoT |

### 5.2 模型量化

降低推理显存占用和加速：

| 精度 | 显存占用 | 速度 | 质量损失 |
|------|----------|------|----------|
| FP32 | 100% | 基准 | 无 |
| FP16/BF16 | 50% | 1.5-2× | 几乎无损 |
| INT8 | 25% | 2-3× | 微小 |
| INT4 (GPTQ/AWQ) | 12.5% | 3-4× | 可接受 |

**GPTQ 量化示例**：
```python
from transformers import AutoModelForCausalLM

# 使用预量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-GPTQ",
    device_map="auto"
)
# 7B 模型仅需 ~4GB 显存！
```

**AWQ（Activation-aware Weight Quantization）**：
- 保护对激活值敏感的权重通道
- 通常比 GPTQ 质量更好
- 推荐用于生产环境

### 5.3 vLLM 推理引擎

**核心特性**：PagedAttention，解决 KV Cache 内存管理问题。

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(model="meta-llama/Meta-Llama-3-8B", tensor_parallel_size=1)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# 批量推理
prompts = [
    "北京是中国的",
    "机器学习是",
    "深度学习框架有"
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

**性能对比**（7B 模型，A100）：

| 引擎 | Throughput (tokens/s) | 显存效率 |
|------|----------------------|----------|
| HuggingFace 原生 | ~20 | 低，碎片化 |
| vLLM | ~80-120 | 高，PagedAttention |
| TensorRT-LLM | ~100-150 | 高，需编译 |
| TGI | ~60-90 | 中等 |

**连续批处理（Continuous Batching）**：
- 传统：一个 batch 内所有请求一起完成，快的等慢的
- vLLM：请求动态加入/离开 batch，GPU 利用率最大化

### 5.4 API 服务搭建

**FastAPI + vLLM**：
```python
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()
llm = LLM(model="meta-llama/Meta-Llama-3-8B")

class ChatRequest(BaseModel):
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    # 格式化消息为 prompt
    prompt = format_messages(request.messages)

    params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    outputs = llm.generate([prompt], params)
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": outputs[0].outputs[0].text
            }
        }]
    }
```

### 5.5 生产环境 checklist

- [ ] **模型热加载**：支持不停机更新模型版本
- [ ] **动态批处理**：最大化 GPU 利用率
- [ ] **流式输出**：首 token 延迟 < 500ms
- [ ] **限流与降级**：防止过载，超限时返回简化回复
- [ ] **监控指标**：QPS、延迟（P50/P95/P99）、GPU 利用率、显存
- [ ] **A/B 测试**：支持多版本模型并行
- [ ] **安全过滤**：输入/输出内容审核
- [ ] **成本追踪**：按用户/项目统计 token 消耗

---

## 6. 企业级应用场景

### 6.1 智能客服

**架构**：RAG + Fine-tuned 模型 + 多轮对话管理
- 知识库：产品文档、FAQ、历史工单
- 工具调用：查询订单状态、创建工单
- 兜底策略：置信度低时转人工

### 6.2 代码助手

**关键技术**：
- 代码补全：FIM（Fill-In-the-Middle）训练
- 代码检索：基于 AST 的代码搜索
- 多文件上下文：Repo-level 编码

**代表产品**：GitHub Copilot、Cursor、Codeium

### 6.3 文档生成与处理

**场景**：合同审查、财报分析、研报生成
**技术栈**：
- 长文档处理：分块 + 递归摘要
- 结构化输出：JSON Schema 约束
- 事实核查：与源文档交叉验证

### 6.4 多模态应用

**Vision-Language 模型**：
- GPT-4o、Claude 4、Qwen2.5-VL、LLaVA
- 场景：图片理解、图表分析、OCR+理解

**架构**：
```
图片 → Vision Encoder（ViT）→ 视觉 token
文本 → Text Embedding → 文本 token
           ↓
    融合 → Transformer → 输出
```

---

## 7. 面试高频考点

### 7.1 应用开发方向

1. **RAG 的检索召回率低怎么办？**
   - 混合检索（向量+关键词）
   - 查询改写/扩展
   - 重排序
   - 调整 chunk 大小和重叠

2. **如何减少 LLM 幻觉？**
   - RAG 提供事实依据
   - 温度调低 + Top-P 限制
   - 要求模型标注不确定性
   - 事实核查 pipeline

3. **7B 模型如何在单卡 4090 上部署？**
   - 4-bit 量化（AWQ/GPTQ）
   - vLLM 推理优化
   - 如果还需微调 → QLoRA

4. **LoRA 和全参数微调怎么选？**
   - 数据量 < 1万 → LoRA
   - 需要改变模型基础能力 → 全参数
   - 资源受限 → LoRA/QLoRA
   - 生产环境快速迭代 → LoRA（只存 adapter，几十 MB）

5. **Agent 调用工具失败如何处理？**
   - 重试机制（指数退避）
   - 备用工具
   - 错误信息反馈给模型，让它重试
   - 最大步数限制，防止无限循环

### 7.2 架构设计方向

1. **设计一个支持 10万 QPS 的 LLM 服务**
   - 模型并行 + 数据并行
   - 负载均衡（按序列长度分配）
   - 缓存层（相似查询结果缓存）
   - 自动扩缩容

2. **长上下文（100K+）如何优化？**
   - KV Cache 压缩（H2O、StreamingLLM）
   - 稀疏注意力
   - 分块处理 + 递归摘要

---

## 学习路径建议

1. **先做 RAG 项目**：这是最容易落地、需求最大的技能
2. **掌握一个 Agent 框架**：LangChain 或 LlamaIndex
3. **用 QLoRA 微调一次**：体验低成本微调流程
4. **部署一个本地模型**：vLLM + Docker，理解推理优化
5. **跟踪开源社区**：HuggingFace、LangChain、LlamaIndex 的更新
