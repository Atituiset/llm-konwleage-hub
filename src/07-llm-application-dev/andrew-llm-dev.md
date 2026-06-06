# 面向开发者的LLM入门课程（吴恩达大模型通关手册）

> **资料来源**：Datawhale 整理，吴恩达（Andrew Ng）与 OpenAI 合作推出的大模型系列课程中文整理版
> **原课程**：ChatGPT Prompt Engineering for Developers、Building Systems with the ChatGPT API、LangChain for LLM Application Development、LangChain Chat with Your Data
> **适合人群**：希望系统学习 LLM 应用开发的开发者
> **难度**：⭐⭐⭐（中等）

---

## 1. 课程概述与核心能力

吴恩达与 OpenAI 联合推出的这套课程是**大模型应用开发的标准入门教材**。与面向普通用户的 Prompt 教程不同，本课程完全从开发者视角出发，教授如何通过代码调用 LLM API 构建实际应用。

学完本课程后，你将具备以下能力：

| 能力层级 | 具体内容 |
|---------|---------|
| **基础** | 熟练调用 OpenAI API，理解 Token、温度、Top-P 等核心参数 |
| **进阶** | 设计高质量 Prompt，掌握迭代优化方法 |
| **应用** | 构建文本总结、推断、转换、扩展等实用工具 |
| **系统** | 搭建基于 ChatGPT 的完整问答系统，掌握提示链（Chaining） |
| **框架** | 使用 LangChain 开发复杂应用，实现 RAG 和 Agent |

---

## 2. LLM API 基础：从调用到理解

### 2.1 核心概念：Token

Token 是大模型处理文本的基本单位。理解 Token 对控制成本和输出至关重要。

```python
import openai

# 获取 Token 计数（估算）
# 英文：1 token ≈ 0.75 个单词
# 中文：1 个汉字 ≈ 1.5~2 tokens

def estimate_tokens(text):
    """粗略估算 Token 数（实际应使用 tiktoken）"""
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

# 示例
chinese_text = "你好，世界！"
print(f"'{chinese_text}' 的 Token 数: {estimate_tokens(chinese_text)}")
# 输出约 5-6 tokens
```

**Token 计费规则**：
- 输入 Token + 输出 Token 合计计费
- 中文内容成本通常高于英文（同信息量下 Token 更多）
- 上下文窗口包含历史对话，长对话成本累积很快

### 2.2 API 调用基础

```python
import openai

client = openai.OpenAI(api_key="your-api-key")

# 基础对话调用
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一位有帮助的助手。"},
        {"role": "user", "content": "请解释什么是 REST API"}
    ],
    temperature=0.7,  # 创造性：0=确定性，2=随机性最高
    max_tokens=500,   # 限制输出长度
    top_p=1.0         # 核采样概率阈值
)

print(response.choices[0].message.content)
```

**关键参数详解**：

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `temperature` | 控制随机性。低=保守/确定，高=创意/随机 | 事实任务: 0~0.3, 创意任务: 0.7~1.0 |
| `top_p` | 核采样，只从累积概率前 p% 的 Token 中采样 | 通常与 temperature 只调一个，设 0.9~1.0 |
| `max_tokens` | 最大输出 Token 数 | 根据任务设定，避免过长浪费 |
| `presence_penalty` | 惩罚已出现过的 Token，鼓励多样性 | 0~0.6，避免重复 |
| `frequency_penalty` | 惩罚高频 Token，降低重复 | 0~0.6 |

### 2.3 系统消息（System Message）的重要性

系统消息是塑造模型行为的"隐形指令"，不会出现在对话历史中。

```python
# 不同的系统消息产生完全不同的输出风格
system_prompts = {
    "formal": "你是一位严谨的学术写作助手。回答需使用正式语言，引用相关理论，结构清晰。",
    "casual": "你是一位友好的技术博主。用轻松幽默的语气解释技术概念，适当使用比喻。",
    "expert": "你是一位拥有20年经验的系统架构师。从技术选型、性能、可维护性角度分析问题。",
    "concise": "你是一位高效的信息提取助手。只输出关键要点，每点不超过20字，拒绝冗余。"
}

def chat_with_persona(persona, user_question):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompts[persona]},
            {"role": "user", "content": user_question}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# 同一问题，不同角色
question = "什么是微服务架构？"
for persona in ["formal", "casual", "concise"]:
    print(f"\n【{persona}风格】")
    print(chat_with_persona(persona, question))
```

---

## 3. 提示工程（Prompt Engineering）核心原则

### 3.1 两大核心原则

**原则一：编写清晰具体的指令**

| 不好的 Prompt | 好的 Prompt |
|--------------|-------------|
| "写一段关于Python的代码" | "写一个 Python 函数，接收一个整数列表，返回其中的最大值和最小值。要求时间复杂度 O(n)，包含类型注解和 docstring。" |
| "总结一下这篇文章" | "请用3句话总结以下文章的核心观点。每句话不超过30字。关注作者对AI安全性的立场。" |

**原则二：给模型思考的时间**

复杂任务不要期望一步完成，让模型分步思考。

```python
# 不好的做法：直接要答案
bad_prompt = """
判断以下评论的情感倾向（正面/负面/中性）：
"这款手机的电池续航真的很一般，拍照效果倒是出乎意料的好，但是价格有点贵。"
"""

# 好的做法：让模型先分析再判断
good_prompt = """
请按以下步骤分析评论的情感倾向：

步骤1：提取评论中提到的每个方面（如电池、拍照、价格）
步骤2：判断每个方面的情感（正面/负面/中性）
步骤3：综合所有方面，给出整体情感倾向

评论："这款手机的电池续航真的很一般，拍照效果倒是出乎意料的好，但是价格有点贵。"

请按上述步骤输出分析。
"""
```

### 3.2 迭代优化方法论

Prompt 开发是一个迭代过程，遵循"想法→编码→实验→分析→改进"的循环。

```
第1轮：写出初版 Prompt，测试几个样本
    ↓
第2轮：分析失败案例，找出模糊/缺失的部分
    ↓
第3轮：添加更具体的约束、示例或步骤
    ↓
第4轮：在更多样本上测试，确保鲁棒性
    ↓
第5轮：固化最终版本，编写文档
```

**迭代检查清单**：
- [ ] 输出格式是否符合预期？（JSON、Markdown、纯文本）
- [ ] 边界情况是否处理？（空输入、超长输入、特殊字符）
- [ ] 是否存在偏见或不一致的输出？
- [ ] 是否包含不需要的内容？（添加"不要..."的否定约束）

### 3.3 结构化输出技巧

让模型输出结构化数据，便于程序解析：

```python
# 使用 JSON 模式约束输出
json_prompt = """
从以下产品描述中提取信息，以 JSON 格式输出：

产品描述："Apple iPhone 15 Pro Max，256GB，钛金属原色，支持5G，售价9999元"

要求输出格式：
{
  "品牌": "...",
  "型号": "...",
  "存储容量": "...",
  "颜色": "...",
  "网络": "...",
  "价格": "..."
}

只输出 JSON，不要其他文字。
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": json_prompt}],
    response_format={"type": "json_object"}  # 强制 JSON 输出
)

import json
result = json.loads(response.choices[0].message.content)
print(result)
# {'品牌': 'Apple', '型号': 'iPhone 15 Pro Max', ...}
```

---

## 4. 四大核心应用场景

### 4.1 文本总结（Summarization）

将长文本压缩为精炼摘要，是 LLM 最成熟的应用之一。

```python
def summarize_text(text, max_words=100, focus=None):
    """
    通用文本总结函数

    Args:
        text: 待总结的长文本
        max_words: 摘要最大字数
        focus: 关注重点（如"技术细节"、"商业影响"）
    """
    focus_prompt = f"重点关注{focus}。" if focus else ""

    prompt = f"""
    请将以下文本总结为不超过{max_words}字的摘要。
    {focus_prompt}
    摘要应包含：核心观点、关键数据（如有）、结论。

    文本：
    ```
    {text}
    ```
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3  # 低温度，确保事实准确
    )
    return response.choices[0].message.content

# 使用示例：提取会议纪要的行动项
def extract_action_items(meeting_transcript):
    prompt = f"""
    从以下会议记录中提取所有行动项（Action Items）。
    对每个行动项，输出：任务描述、负责人（如有）、截止日期（如有）。

    以 Markdown 表格格式输出：
    | 任务 | 负责人 | 截止日期 | 优先级 |

    会议记录：
    ```
    {meeting_transcript}
    ```
    """
    # ... 调用 API
```

**总结策略对比**：

| 策略 | 适用场景 | Prompt 技巧 |
|------|---------|------------|
| 提取式 | 法律合同、技术文档 | "提取原文中的关键句，不要改写" |
| 抽象式 | 新闻、博客 | "用自己的话概括核心观点" |
| 多文档 | 研报对比、文献综述 | "对比以下文档的观点差异..." |
| 渐进式 | 超长文本（>100K tokens） | 分段总结→合并摘要→最终精炼 |

### 4.2 文本推断（Inference）

从文本中提取结构化信息、判断情感、识别实体等。

```python
# 情感分析 + 方面级情感（Aspect-Based Sentiment）
def aspect_sentiment_analysis(review):
    prompt = f"""
    分析以下产品评论中，每个方面的情感倾向。

    评论："{review}"

    要求：
    1. 识别评论中提到的所有产品方面（如质量、服务、价格、物流）
    2. 判断每个方面的情感：正面(😊)、负面(😞)、中性(😐)
    3. 给出置信度评分（1-10）

    输出格式：
    方面 | 情感 | 置信度 | 依据原文
    -----|------|--------|----------
    """
    # ... 调用 API

# 使用示例
review = "手机性能很强，打游戏完全不卡。但是充电器发热严重，客服态度也很差。"
# 预期输出识别出：性能(正面)、充电器/散热(负面)、客服(负面)
```

**推断任务类型大全**：

| 任务 | 说明 | 示例输出 |
|------|------|---------|
| 情感分析 | 判断文本情感极性 | 正面/负面/中性 + 置信度 |
| 命名实体识别(NER) | 识别人名、地名、机构名 | 实体列表 + 类型标签 |
| 主题分类 | 判断文本所属类别 | 单标签/多标签分类 |
| 意图识别 | 理解用户目的 | "查询订单"、"投诉"、"咨询" |
| 关系抽取 | 识别实体间关系 | "张三-就职于-阿里巴巴" |
| 事件抽取 | 识别事件及参与者 | "收购事件：收购方=A公司，被收购方=B公司" |

### 4.3 文本转换（Transformation）

将文本从一种形式转换为另一种形式：翻译、格式转换、风格迁移等。

```python
# 多语言翻译 + 术语一致性
def translate_with_glossary(text, target_lang="英文", glossary=None):
    """
    带术语表的专业翻译

    glossary: {"源术语": "目标术语", ...}
    """
    glossary_text = "\n".join([f"{k} → {v}" for k, v in (glossary or {}).items()])

    prompt = f"""
    将以下文本翻译为{target_lang}。

    术语对照表（必须严格遵守）：
    {glossary_text}

    翻译要求：
    - 保持专业、正式的语气
    - 技术术语使用行业通用译法
    - 保留原文的段落结构

    原文：
    ```
    {text}
    ```
    """
    # ... 调用 API

# 代码转换：Python → JavaScript
def convert_code(python_code, target_lang="JavaScript"):
    prompt = f"""
    将以下 Python 代码转换为 {target_lang}。
    要求：
    1. 保持功能完全一致
    2. 使用目标语言的最佳实践
    3. 添加必要的类型注解/注释
    4. 如果存在不兼容的特性，给出替代方案并标注

    Python 代码：
    ```python
    {python_code}
    ```
    """
    # ... 调用 API
```

### 4.4 文本扩展（Expansion）

基于简短输入生成更丰富的内容：邮件补全、内容创作、代码生成等。

```python
# 智能邮件生成
def compose_email(context):
    """
    context = {
        "recipient": "客户",
        "purpose": "延期通知",
        "key_points": ["需求变更导致", "延期2周", "已安排额外资源"],
        "tone": "专业且诚恳"
    }
    """
    prompt = f"""
    写一封{context['tone']}的邮件。

    收件人：{context['recipient']}
    目的：{context['purpose']}
    必须包含的要点：
    {chr(10).join('- ' + p for p in context['key_points'])}

    要求：
    - 开头礼貌问候
    - 正文清晰说明情况
    - 结尾提供解决方案或下一步行动
    - 总字数控制在200字以内
    """
    # ... 调用 API

# 温度控制示例：同一提示，不同创造性
def generate_creative_variants(prompt, n=3):
    """生成同一提示的多个变体，展示 temperature 的影响"""
    variants = []
    for temp in [0.2, 0.7, 1.2]:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=200
        )
        variants.append({
            "temperature": temp,
            "output": response.choices[0].message.content
        })
    return variants
```

---

## 5. 构建聊天机器人

### 5.1 对话状态管理

聊天机器人需要维护对话历史，同时控制上下文长度。

```python
class ChatBot:
    def __init__(self, system_prompt, max_history=10):
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.messages = []

    def chat(self, user_input):
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})

        # 控制历史长度（保留最近的 N 轮）
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history * 2:]

        # 构建完整消息列表
        full_messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.messages

        response = client.chat.completions.create(
            model="gpt-4",
            messages=full_messages,
            temperature=0.7
        )

        assistant_reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    def clear_history(self):
        self.messages = []

# 使用示例
bot = ChatBot(
    system_prompt="你是一位专业的Python编程导师。用中文回答，代码示例要完整可运行。",
    max_history=5
)

print(bot.chat("请解释列表推导式"))
print(bot.chat("那和 map 函数有什么区别？"))  # 能记住上文提到的"列表推导式"
```

### 5.2 带上下文的聊天机器人（RAG 雏形）

```python
class ContextualChatBot:
    """能引用外部文档回答问题的聊天机器人"""

    def __init__(self, system_prompt, knowledge_base=None):
        self.system_prompt = system_prompt
        self.knowledge_base = knowledge_base or {}
        self.messages = []

    def retrieve_context(self, query):
        """简单的关键词检索（实际可用向量检索）"""
        relevant_docs = []
        for doc_id, doc_content in self.knowledge_base.items():
            # 简化的相关性判断
            query_words = set(query.lower().split())
            doc_words = set(doc_content.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                relevant_docs.append((overlap, doc_content))

        # 按相关度排序，取前3条
        relevant_docs.sort(reverse=True)
        return [doc for _, doc in relevant_docs[:3]]

    def chat(self, user_input):
        # 检索相关知识
        contexts = self.retrieve_context(user_input)
        context_text = "\n\n".join(f"[文档{i+1}] {c}" for i, c in enumerate(contexts))

        # 构建增强的 Prompt
        augmented_prompt = f"""
        基于以下参考文档回答问题。如果文档中没有相关信息，请明确说明。

        参考文档：
        {context_text}

        用户问题：{user_input}
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.messages,
            {"role": "user", "content": augmented_prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3  # 低温度，忠实于文档
        )

        reply = response.choices[0].message.content
        self.messages.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": reply}
        ])
        return reply
```

---

## 6. 提示链（Chaining Prompts）：构建复杂系统

### 6.1 为什么需要提示链

单一 Prompt 的局限性：
- 上下文窗口有限，无法一次性处理大量输入
- 复杂任务一步完成容易出错
- 需要中间结果进行条件判断

**提示链的核心思想**：将复杂任务分解为多个子任务，每个子任务由一个 Prompt 处理，前一个任务的输出作为后一个任务的输入。

```
用户请求
    ↓
[Prompt 1: 意图识别] → 确定任务类型
    ↓
[Prompt 2: 信息提取] → 从用户输入提取关键参数
    ↓
[Prompt 3: 数据处理] → 调用工具/查询数据库
    ↓
[Prompt 4: 结果生成] → 基于处理结果生成最终回复
    ↓
用户收到回复
```

### 6.2 提示链实战：产品评价分析系统

```python
class ProductReviewAnalyzer:
    """多步骤分析产品评价"""

    def __init__(self):
        self.client = openai.OpenAI()

    def step1_extract_aspects(self, review):
        """步骤1：提取评价中提到的所有产品方面"""
        prompt = f"""
        从以下产品评价中提取所有被提及的产品方面（aspect）。
        只输出方面列表，每行一个。

        评价："{review}"

        示例输出格式：
        - 电池续航
        - 屏幕显示
        - 拍照效果
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        aspects = [line.strip("- ") for line in response.choices[0].message.content.split("\n") if line.strip()]
        return aspects

    def step2_analyze_sentiment(self, review, aspect):
        """步骤2：分析特定方面的情感"""
        prompt = f"""
        分析以下评价中关于"{aspect}"的情感倾向。

        评价："{review}"

        只输出以下之一：正面、负面、中性、未提及
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def step3_generate_summary(self, review, aspect_sentiments):
        """步骤3：生成综合分析报告"""
        aspects_text = "\n".join([f"- {aspect}: {sentiment}" for aspect, sentiment in aspect_sentiments])
        prompt = f"""
        基于以下方面级情感分析结果，生成一段综合评价摘要。

        原始评价："{review}"

        方面分析：
        {aspects_text}

        要求：
        1. 总结产品的优点和缺点
        2. 给出整体推荐意见（推荐/谨慎考虑/不推荐）
        3. 控制在100字以内
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content

    def analyze(self, review):
        """执行完整分析流程"""
        # 步骤1
        aspects = self.step1_extract_aspects(review)
        print(f"提取到的方面: {aspects}")

        # 步骤2（并行执行）
        aspect_sentiments = {}
        for aspect in aspects:
            sentiment = self.step2_analyze_sentiment(review, aspect)
            aspect_sentiments[aspect] = sentiment
            print(f"  {aspect}: {sentiment}")

        # 步骤3
        summary = self.step3_generate_summary(review, aspect_sentiments)
        print(f"\n综合摘要: {summary}")

        return {
            "aspects": aspects,
            "sentiments": aspect_sentiments,
            "summary": summary
        }

# 使用示例
analyzer = ProductReviewAnalyzer()
review = "这款手机性价比很高，屏幕色彩鲜艳，但电池续航一般，充电速度倒是很快。"
result = analyzer.analyze(review)
```

### 6.3 提示链与错误处理

```python
class RobustChain:
    """带错误处理和重试的提示链"""

    def run_with_retry(self, prompt, max_retries=3, validator=None):
        """
        执行 Prompt，带重试和输出验证

        validator: 函数，接收输出，返回 (是否有效, 错误信息)
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3 + attempt * 0.2  # 逐步增加随机性
                )
                output = response.choices[0].message.content

                # 验证输出
                if validator:
                    is_valid, error_msg = validator(output)
                    if not is_valid:
                        prompt += f"\n\n注意：之前的输出不符合要求。{error_msg}请重新输出。"
                        continue

                return output

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避

        raise Exception("Max retries exceeded")

# 使用示例：验证 JSON 输出
def validate_json(output):
    try:
        json.loads(output)
        return True, ""
    except:
        return False, "输出必须是有效的 JSON 格式。"
```

---

## 7. 思维链推理（Chain-of-Thought）

### 7.1 基础思维链

让模型展示推理过程，能显著提升复杂任务的准确率。

```python
# 数学问题：不展示推理 vs 展示推理

# 方式1：直接要答案（容易出错）
direct_prompt = """
问题：一个农场有鸡和兔共35只，脚共94只。鸡和兔各有多少只？
答案：
"""

# 方式2：思维链（更准确）
cot_prompt = """
问题：一个农场有鸡和兔共35只，脚共94只。鸡和兔各有多少只？

请逐步推理：
步骤1：设鸡有 x 只，兔有 y 只
步骤2：根据题意列出方程
步骤3：解方程
步骤4：验证答案

推理过程：
"""

# 零样本思维链：在 Prompt 末尾加 "Let's think step by step"
zero_shot_cot = """
问题：一个农场有鸡和兔共35只，脚共94只。鸡和兔各有多少只？

让我们一步步思考：
"""
```

### 7.2 自一致性解码（Self-Consistency）

对同一问题采样多条推理路径，取最一致的答案。

```python
def self_consistency_solve(problem, n_samples=5):
    """多次采样，投票选出最一致的答案"""
    answers = []

    for _ in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": problem + "\n让我们一步步思考："}],
            temperature=0.7,  # 较高的温度产生多样性
        )
        # 从推理过程中提取最终答案
        answer = extract_final_answer(response.choices[0].message.content)
        answers.append(answer)

    # 投票：选择出现次数最多的答案
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0]
    return {
        "answer": most_common[0],
        "confidence": most_common[1] / n_samples,
        "all_answers": answers
    }
```

---

## 8. 与 LangChain 框架的结合

### 8.1 LangChain 核心组件

```python
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

# 1. Prompt Template（可复用的 Prompt 模板）
template = """
你是一位{role}。请用{style}的风格回答以下问题。

历史对话：
{history}

用户：{input}
助手：
"""

prompt = PromptTemplate(
    input_variables=["role", "style", "history", "input"],
    template=template
)

# 2. Memory（记忆系统）
memory = ConversationBufferMemory(memory_key="history")

# 3. Chain（将组件串联）
llm = OpenAI(temperature=0.7)
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# 4. 运行
result = chain.predict(role="Python专家", style="通俗易懂", input="什么是装饰器？")
```

### 8.2 基于文档的问答（RAG 基础）

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 加载文档
loader = PyPDFLoader("company_handbook.pdf")
docs = loader.load()

# 切分文档
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 创建向量数据库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 构建 RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 提问
result = qa_chain({"query": "公司的年假政策是什么？"})
print(result["result"])
print("来源:", [d.metadata for d in result["source_documents"]])
```

---

## 9. 学习路径与进阶建议

### 9.1 实践项目建议

| 阶段 | 项目 | 技能覆盖 |
|------|------|---------|
| 入门 | 智能客服聊天机器人 | API 调用、对话管理、Prompt 设计 |
| 进阶 | 产品评论分析系统 | 提示链、文本推断、结构化输出 |
| 进阶 | 企业知识库问答 | RAG、文档处理、向量检索 |
| 高级 | 多 Agent 协作系统 | Agent、工具调用、任务分解 |

### 9.2 常见陷阱与最佳实践

| 陷阱 | 解决方案 |
|------|---------|
| Prompt 过长导致成本飙升 | 精简指令，使用变量替换重复内容 |
| 输出格式不稳定 | 提供明确的格式示例，使用 JSON mode |
| 模型幻觉严重 | 降低 temperature，使用 RAG 提供事实依据 |
| 对话历史无限增长 | 设置最大轮数，或使用摘要压缩历史 |
| 敏感信息泄露 | 在 system prompt 中设置安全约束，输出过滤 |

### 9.3 后续学习方向

- **深入 RAG**：学习高级检索策略（混合检索、重排序、GraphRAG）
- **Agent 开发**：掌握 ReAct、Plan-and-Execute 等 Agent 模式
- **模型微调**：用 LoRA/QLoRA 在特定任务上微调开源模型
- **生产部署**：学习 vLLM、模型量化、API 服务搭建
