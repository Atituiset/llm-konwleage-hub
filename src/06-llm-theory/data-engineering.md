# 数据工程基础（Data Engineering for LLMs）

> **适合人群**：需要理解大模型数据 pipeline 的工程师
> **难度**：⭐⭐⭐（中等）
> **前置知识**：Python、本书"预训练"章节

---

## 1. 数据生命周期（Data Lifecycle）

大模型的数据 pipeline 是一个完整的生命周期：

```
数据采集（Collection）
    │
    ▼
数据清洗（Cleaning）──→ 去重、过滤、去毒、去偏
    │
    ▼
数据标注（Annotation）──→ 分类、抽取、生成、偏好标注
    │
    ▼
数据存储（Storage）──→ 对象存储、数据湖、版本管理
    │
    ▼
数据消费（Consumption）──→ 训练、评估、对齐
    │
    ▼
数据反馈（Feedback）──→ 生产日志、用户反馈、错误案例
```

---

## 2. 数据采集（Data Collection）

### 2.1 数据来源

| 来源 | 示例 | 适用场景 |
|------|------|---------|
| **公开数据集** | Common Crawl、The Pile、C4、Wikipedia | 预训练 |
| **开源代码** | GitHub、StackOverflow、GitLab | 代码能力 |
| **书籍文献** | Gutenberg、arXiv、PubMed | 专业知识 |
| **对话数据** | ShareGPT、WildChat、LMSYS | 对话微调 |
| **合成数据** | Self-Instruct、Evol-Instruct | 指令微调 |
| **领域数据** | 金融报告、法律条文、医疗记录 | 领域模型 |
| **多模态数据** | LAION-5B、WebVid、AudioSet | 多模态训练 |

### 2.2 大规模数据采集架构

```python
# crawl_pipeline.py
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.parse import urljoin, urlparse

class WebCrawler:
    def __init__(self, max_workers=10, respect_robots=True):
        self.max_workers = max_workers
        self.respect_robots = respect_robots
        self.visited = set()
        self.results = []

    def fetch(self, url: str) -> dict:
        """获取单个页面"""
        try:
            response = requests.get(url, timeout=30, headers={
                "User-Agent": "LLM-DataBot/1.0 (Research Purpose)"
            })
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # 提取主要内容（去除导航、广告等）
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()

            # 使用 Readability 风格提取
            main_content = self._extract_main_content(soup)

            return {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "content": main_content,
                "timestamp": response.headers.get("Last-Modified"),
                "status": "success"
            }
        except Exception as e:
            return {"url": url, "status": "error", "error": str(e)}

    def _extract_main_content(self, soup) -> str:
        """提取页面主要内容"""
        # 优先选择 article、main 标签
        for selector in ["article", "main", "[role='main']", ".content", "#content"]:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text(separator="\n", strip=True)
        # 回退到 body
        return soup.body.get_text(separator="\n", strip=True) if soup.body else ""

    def crawl_parallel(self, urls: list) -> list:
        """并行爬取"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetch, url): url for url in urls}
            for future in futures:
                result = future.result()
                if result["status"] == "success":
                    self.results.append(result)
        return self.results

# 使用示例
crawler = WebCrawler(max_workers=20)
urls = ["https://example.com/article1", "https://example.com/article2"]
documents = crawler.crawl_parallel(urls)
```

### 2.3 数据去重（Deduplication）

```python
# deduplication.py
import hashlib
from datasketch import MinHash, MinHashLSH
import re

def exact_dedup(documents: list) -> list:
    """精确去重：基于内容 hash"""
    seen = set()
    unique = []
    for doc in documents:
        # 归一化后计算 hash
        normalized = re.sub(r"\s+", " ", doc.lower().strip())
        h = hashlib.md5(normalized.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique

def minhash_dedup(documents: list, threshold=0.85) -> list:
    """近似去重：基于 MinHash + LSH"""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    unique = []

    for i, doc in enumerate(documents):
        # 分词（按 n-gram）
        tokens = [doc[j:j+5] for j in range(len(doc) - 4)]

        m = MinHash(num_perm=128)
        for token in tokens:
            m.update(token.encode("utf8"))

        # 查询相似文档
        duplicates = lsh.query(m)
        if not duplicates:  # 无相似文档，保留
            lsh.insert(f"doc_{i}", m)
            minhashes[f"doc_{i}"] = m
            unique.append(doc)

    return unique

# 使用
from datasets import load_dataset

ds = load_dataset("oscar", "unshuffled_deduplicated_zh", split="train")
print(f"去重前: {len(ds)} 条")
unique_docs = minhash_dedup([item["text"] for item in ds])
print(f"去重后: {len(unique_docs)} 条")
```

---

## 3. 数据清洗（Data Cleaning）

### 3.1 质量过滤流水线

```python
# data_cleaning.py
import re
from collections import Counter
import langdetect

class DataCleaner:
    def __init__(self):
        self.stats = Counter()

    def clean(self, text: str) -> str | None:
        """清洗单条文本，返回清洗后文本或 None（丢弃）"""

        # 1. 长度过滤
        if len(text) < 100:
            self.stats["too_short"] += 1
            return None
        if len(text) > 100000:
            self.stats["too_long"] += 1
            text = text[:100000]

        # 2. 语言检测
        try:
            lang = langdetect.detect(text[:1000])
            if lang != "zh":
                self.stats["wrong_language"] += 1
                return None
        except:
            self.stats["lang_detect_fail"] += 1
            return None

        # 3. 去 HTML 标签
        text = re.sub(r"<[^>]+>", "", text)

        # 4. 去 URL
        text = re.sub(r"https?://\S+", "", text)

        # 5. 去邮箱
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # 6. 规范化空白
        text = re.sub(r"\s+", " ", text)

        # 7. 去重复段落（3次以上重复）
        paragraphs = text.split("\n")
        para_counts = Counter(paragraphs)
        paragraphs = [p for p in paragraphs if para_counts[p] <= 3]
        text = "\n".join(paragraphs)

        # 8. 质量评分：字符熵（去低熵/重复内容）
        if self._entropy(text) < 3.0:
            self.stats["low_entropy"] += 1
            return None

        # 9. 垃圾内容过滤
        if self._is_garbage(text):
            self.stats["garbage"] += 1
            return None

        self.stats["kept"] += 1
        return text.strip()

    def _entropy(self, text: str) -> float:
        """计算字符级信息熵"""
        from math import log2
        counts = Counter(text)
        length = len(text)
        entropy = -sum((c/length) * log2(c/length) for c in counts.values())
        return entropy

    def _is_garbage(self, text: str) -> bool:
        """判断是否为垃圾内容"""
        # 过多特殊字符
        special_ratio = len(re.findall(r"[^\w\s一-鿿]", text)) / len(text)
        if special_ratio > 0.5:
            return True

        # 过多重复字符
        if re.search(r"(.)\1{20,}", text):
            return True

        # 乱码检测
        if len(re.findall(r"[�\x00-\x08]", text)) > 10:
            return True

        return False

# 使用
import json

cleaner = DataCleaner()
with open("raw_data.jsonl") as f, open("cleaned.jsonl", "w") as out:
    for line in f:
        doc = json.loads(line)
        cleaned = cleaner.clean(doc["text"])
        if cleaned:
            out.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")

print(cleaner.stats)
```

### 3.2 毒性检测与过滤

```python
# toxicity_filter.py
from transformers import pipeline

class ToxicityFilter:
    def __init__(self, threshold=0.7):
        # 使用多语言毒性分类器
        self.classifier = pipeline(
            "text-classification",
            model="citizenlab/distilbert-base-multilingual-cased-toxicity",
            device=0
        )
        self.threshold = threshold

    def is_toxic(self, text: str) -> tuple[bool, float]:
        """返回 (是否毒性, 毒性分数)"""
        result = self.classifier(text[:512])[0]  # 截断到 512
        score = result["score"] if result["label"] == "toxic" else 1 - result["score"]
        return score > self.threshold, score

    def filter_dataset(self, texts: list) -> list:
        """批量过滤数据集"""
        results = self.classifier([t[:512] for t in texts], batch_size=32)
        clean = []
        for text, result in zip(texts, results):
            score = result["score"] if result["label"] == "toxic" else 1 - result["score"]
            if score <= self.threshold:
                clean.append(text)
        return clean

# 使用
filter = ToxicityFilter(threshold=0.8)
with open("cleaned.jsonl") as f:
    texts = [json.loads(line)["text"] for line in f]

safe_texts = filter.filter_dataset(texts)
print(f"过滤前: {len(texts)}, 过滤后: {len(safe_texts)}")
```

---

## 4. 数据标注（Data Annotation）

### 4.1 标注类型

| 类型 | 说明 | 示例 |
|------|------|------|
| **分类标注** | 给文本打标签 | 情感分类、主题分类 |
| **抽取标注** | 抽取结构化信息 | NER、关系抽取 |
| **生成标注** | 人工撰写答案 | SFT 数据、QA 对 |
| **偏好标注** | 比较两个输出 | RLHF 偏好对 |
| **质量评分** | 给输出打分 | Helpfulness、Safety |

### 4.2 偏好数据标注流程

```python
# preference_annotation.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class PreferencePair:
    prompt: str
    chosen: str      # 更好的回答
    rejected: str    # 较差的回答
    reason: str      # 选择理由
    annotator_id: str
    category: str    # 标注维度：helpfulness, safety, honesty, etc.

class AnnotationGuide:
    """标注指南：确保标注一致性"""

    CRITERIA = {
        "helpfulness": {
            "description": "回答是否直接、完整地解决了用户问题",
            "dimensions": [
                "相关性：回答是否针对问题",
                "完整性：是否覆盖了问题的所有方面",
                "正确性：信息是否准确",
                "可操作性：是否提供了可执行的步骤"
            ]
        },
        "safety": {
            "description": "回答是否安全、无害",
            "dimensions": [
                "无有害内容：不包含暴力、仇恨、非法内容",
                "无偏见：不强化刻板印象",
                "诚实：不误导用户"
            ]
        },
        "honesty": {
            "description": "回答是否诚实、承认不确定性",
            "dimensions": [
                "承认局限：不知道时明确说明",
                "不编造：不虚构事实",
                "引用来源：重要声明有依据"
            ]
        }
    }

    @classmethod
    def evaluate_pair(cls, prompt: str, response_a: str, response_b: str) -> PreferencePair:
        """由人类标注员使用此指南评估"""
        # 实际实现中，这会在标注平台（如 Label Studio、Argilla）中展示
        pass

# 标注质量控制
class QualityControl:
    def __init__(self):
        self.annotator_agreement = {}

    def calculate_kappa(self, annotations_a: list, annotations_b: list) -> float:
        """计算 Cohen's Kappa 一致性"""
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(annotations_a, annotations_b)

    def detect_spam_annotators(self, annotations: list) -> list:
        """检测敷衍标注者"""
        spam = []
        for annotator_id, data in annotations.items():
            # 所有选择都相同
            if len(set(data["choices"])) == 1:
                spam.append(annotator_id)
            # 标注速度过快（< 5秒/条）
            if data["avg_time"] < 5:
                spam.append(annotator_id)
        return list(set(spam))
```

---

## 5. 数据存储与版本管理

### 5.1 数据湖架构

```
数据湖（Data Lake）
├── raw/                    # 原始数据（不可变）
│   ├── 2024-01-crawl/
│   ├── 2024-02-crawl/
│   └── ...
├── cleaned/                # 清洗后数据
│   ├── deduped/
│   └── filtered/
├── processed/              # 处理后数据
│   ├── tokenized/
│   ├── instruction/
│   └── preference/
└── metadata/               # 元数据
    ├── statistics.json
    ├── quality_report.json
    └── lineage.json
```

### 5.2 DVC 数据版本管理

```bash
# 安装 DVC
pip install dvc dvc-s3

# 初始化
dvc init

# 跟踪数据集
dvc add data/processed/train.jsonl

# 推送到远程存储
dvc remote add -d myremote s3://mybucket/dvc
dvc push

# 打标签
git tag -a v1.0-data -m "训练数据 v1.0"
dvc tag create train.jsonl@v1.0

# 切换到特定版本
git checkout v1.0-data
dvc checkout
```

### 5.3 数据卡片（Data Card）

```markdown
# 数据集卡片：Chinese-Web-Text-v1

## 基本信息
- **名称**: Chinese-Web-Text-v1
- **版本**: 1.0.0
- **创建日期**: 2024-01-15
- **数据量**: 500GB, 约 2 亿文档
- **语言**: 中文（98%）、英文（2%）

## 数据来源
- Common Crawl (CC-MAIN-2023-50)
- 自建爬虫（1000 个高质量网站）

## 清洗流程
1. 语言检测（langdetect，保留 zh > 0.9）
2. 去重（MinHash，阈值 0.85）
3. 质量过滤（长度 100-10000，熵 > 3.0）
4. 毒性过滤（Toxic-BERT，阈值 0.7）
5. PII 脱敏（正则匹配 + NER）

## 统计信息
| 指标 | 数值 |
|------|------|
| 平均文档长度 | 2,500 字符 |
| 中位数文档长度 | 1,800 字符 |
| 去重率 | 35% |
| 过滤率 | 22% |

## 已知问题
- 部分技术文档包含代码片段，可能不是自然语言
- 少量早期互联网内容质量较低

## 使用建议
- 适合通用中文预训练
- 建议与专业领域数据混合使用
- 不适合直接用于微调（需进一步清洗）

## 许可
CC0 1.0 Universal
```

---

## 6. 合成数据生成（Synthetic Data Generation）

### 6.1 Self-Instruct

用模型自己生成指令数据：

```python
# self_instruct.py
import json
from openai import OpenAI

client = OpenAI()

# 种子指令
seed_instructions = [
    "解释什么是深度学习",
    "写一个 Python 函数计算斐波那契数列",
    "比较 CNN 和 RNN 的区别"
]

# 生成新指令的 prompt
GENERATE_PROMPT = """你需要创建 5 条新的指令-回答对，用于训练 AI 助手。

要求：
- 指令应该多样化，涵盖不同类型的问题
- 回答应该详细、准确、有帮助
- 避免与已有指令重复

已有指令示例：
{examples}

请输出 JSON 格式：
[
  {"instruction": "...", "input": "...", "output": "..."},
  ...
]
"""

def generate_instructions(seed_pool: list, num_rounds: int = 5) -> list:
    all_instructions = seed_pool.copy()

    for round in range(num_rounds):
        # 采样已有指令作为示例
        examples = "\n".join(f"- {i}" for i in all_instructions[-10:])

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": GENERATE_PROMPT.format(examples=examples)
            }],
            temperature=0.9
        )

        new_data = json.loads(response.choices[0].message.content)
        for item in new_data:
            all_instructions.append(item["instruction"])

        print(f"Round {round + 1}: {len(all_instructions)} total instructions")

    return all_instructions

# 过滤低质量生成
QUALITY_PROMPT = """评估以下指令-回答对的质量（1-5分）：

{instruction}
{output}

评分标准：
5分：指令清晰，回答准确、完整、有帮助
3分：指令或回答有小问题，但基本可用
1分：指令不清或回答错误/无用

只输出分数数字。"""

def filter_quality(data: list) -> list:
    high_quality = []
    for item in data:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": QUALITY_PROMPT.format(**item)
            }],
            temperature=0
        )
        score = int(response.choices[0].message.content.strip())
        if score >= 4:
            high_quality.append(item)
    return high_quality
```

### 6.2 Evol-Instruct（指令进化）

通过改写提升指令复杂度：

```python
# evol_instruct.py
EVOLUTION_PROMPTS = {
    "breadth": """改写以下指令，使其更加广泛适用（Breadth）：
原始指令：{instruction}
改写要求：
- 保持核心意图不变
- 增加更多应用场景或变体
- 输出新的指令""",

    "depth": """改写以下指令，使其更加深入（Depth）：
原始指令：{instruction}
改写要求：
- 增加复杂度或专业性
- 要求多步骤推理
- 输出新的指令""",

    "concretization": """改写以下指令，使其更加具体（Concretization）：
原始指令：{instruction}
改写要求：
- 添加具体场景、数字、约束条件
- 输出新的指令""",

    "reasoning": """改写以下指令，使其需要更多推理（Reasoning）：
原始指令：{instruction}
改写要求：
- 要求分析因果关系
- 或要求比较不同方案
- 输出新的指令"""
}

def evolve_instruction(instruction: str, evolution_type: str) -> str:
    """进化单条指令"""
    prompt = EVOLUTION_PROMPTS[evolution_type].format(instruction=instruction)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    return response.choices[0].message.content.strip()

# 进化流程
def evolve_dataset(instructions: list, num_evolutions: int = 3) -> list:
    evolved = []
    for inst in instructions:
        for _ in range(num_evolutions):
            etype = random.choice(list(EVOLUTION_PROMPTS.keys()))
            new_inst = evolve_instruction(inst, etype)
            evolved.append(new_inst)
    return evolved
```

---

## 7. 数据合规（Data Compliance）

### 7.1 版权与许可

| 许可类型 | 说明 | 商用限制 |
|---------|------|---------|
| **CC0** | 公有领域 | 无限制 |
| **CC-BY** | 署名 | 需注明来源 |
| **CC-BY-SA** | 署名+相同方式共享 | 衍生作品需同许可 |
| **CC-BY-NC** | 署名+非商用 | 禁止商用 |
| **GPL** | 开源 copyleft | 衍生作品需开源 |
| **MIT** | 宽松开源 | 需保留版权声明 |
| **自定义** | 各平台不同 | 需仔细阅读 |

### 7.2 GDPR 合规要点

1. **数据最小化**：只收集训练必需的数据
2. **目的限制**：不将训练数据用于其他目的
3. **去标识化**：移除 PII（个人身份信息）
4. **被遗忘权**：提供数据删除机制
5. **数据影响评估**：大规模数据处理前评估风险

### 7.3 PII 检测与脱敏

```python
# pii_detection.py
from transformers import pipeline
import re

class PIIDetector:
    def __init__(self):
        # 使用 NER 模型检测实体
        self.ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

    def detect(self, text: str) -> list:
        """检测文本中的 PII"""
        pii_items = []

        # 1. NER 检测
        entities = self.ner(text)
        for ent in entities:
            if ent["entity_group"] in ["PER", "ORG", "LOC"]:
                pii_items.append({
                    "type": ent["entity_group"],
                    "value": ent["word"],
                    "start": ent["start"],
                    "end": ent["end"]
                })

        # 2. 正则检测
        patterns = {
            "phone": r"1[3-9]\d{9}",
            "email": r"\S+@\S+\.\S+",
            "id_card": r"\d{17}[\dXx]",
            "bank_card": r"\d{16,19}"
        }

        for pii_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                pii_items.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

        return pii_items

    def anonymize(self, text: str) -> str:
        """脱敏处理"""
        pii_items = self.detect(text)

        # 从后往前替换，避免位置偏移
        for item in sorted(pii_items, key=lambda x: x["start"], reverse=True):
            placeholder = f"[<{item['type']}>]"
            text = text[:item["start"]] + placeholder + text[item["end"]:]

        return text

# 使用
detector = PIIDetector()
text = "张三的电话是13800138000，邮箱是zhangsan@example.com"
clean = detector.anonymize(text)
print(clean)  # [<PER>]的电话是[<phone>]，邮箱是[<email>]
```

---

## 8. 面试高频考点

1. **数据去重为什么重要？**
   - 防止模型过拟合到重复内容
   - 减少训练时间和计算成本
   - 避免数据泄露（训练集和测试集重复）
   - Common Crawl 原始数据重复率可达 30-50%

2. **MinHash 的原理是什么？**
   - 将文档表示为 n-gram 集合
   - 用多个 hash 函数对集合签名
   - 相似文档的 MinHash 签名相似度高
   - LSH 将相似签名映射到同一桶，加速查找

3. **Self-Instruct 和 Evol-Instruct 的区别？**
   - Self-Instruct：从种子指令生成新指令，扩展多样性
   - Evol-Instruct：通过改写进化指令，提升复杂度
   - 两者结合：先用 Self-Instruct 扩量，再用 Evol-Instruct 提质

4. **如何评估合成数据质量？**
   - 指令多样性：n-gram 分布、主题覆盖
   - 回答正确性：采样人工检查
   - 格式合规性：是否符合训练格式
   - 模型训练验证：用合成数据训练后评估下游任务

5. **数据版本管理为什么需要 DVC 而不是只用 Git？**
   - 数据集太大（GB/TB 级），Git 无法有效管理
   - DVC 支持多种远程存储（S3、GCS、Azure）
   - DVC 跟踪数据 pipeline 的依赖关系
   - Git 跟踪元数据，DVC 跟踪实际数据文件
