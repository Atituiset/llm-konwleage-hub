# 生产部署与 LLMOps（Production Deployment & LLMOps）

> **适合人群**：需要将 LLM 应用从原型推向生产的工程师
> **难度**：⭐⭐⭐⭐⭐（极难）
> **前置知识**：Docker、Linux、FastAPI、本书"推理优化技术"章节

---

## 1. 从原型到生产（From Prototype to Production）

### 1.1 原型与生产的差距

```
原型级（Prototype）                    生产级（Production）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FastAPI 单进程启动                      Gunicorn + Uvicorn 多 worker
本地单卡 GPU                           多卡并行 + 负载均衡
直接返回结果                           流式输出 + 超时控制
无状态服务                            会话保持 + 上下文缓存
手动部署                              CI/CD 自动化
裸机运行                              Docker + K8s 编排
无监控                                Prometheus + Grafana
无容错                                熔断 + 限流 + 重试
```

### 1.2 生产部署架构概览

```
用户请求
    │
    ▼
[CDN / WAF] ──→ [API Gateway / Nginx]
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   [vLLM Pod]   [vLLM Pod]   [vLLM Pod]   ← K8s Deployment (HPA)
        │             │             │
        └─────────────┴─────────────┘
                      │
              [Prometheus]
              [Grafana]
              [Langfuse]
```

---

## 2. 容器化模型服务（Containerized Model Serving）

### 2.1 vLLM Dockerfile

```dockerfile
# vLLM Dockerfile（多阶段构建）
# ─────────────────────────────────────────────

# 阶段1：构建阶段
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

WORKDIR /build
RUN apt update && apt install -y python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（利用缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 阶段2：运行阶段（更小的镜像）
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# 只复制必要的运行时文件
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/cuda /usr/local/cuda

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# 下载模型（或在运行时挂载）
# RUN huggingface-cli download meta-llama/Llama-2-7b-chat-hf

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "meta-llama/Llama-2-7b-chat-hf", \
     "--tensor-parallel-size", "1", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

构建与运行：
```bash
# 构建（约 10-20 分钟）
docker build -t vllm-llama2:prod .

# 运行（需要 GPU）
docker run -d \
  --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/models \
  --shm-size=8g \
  vllm-llama2:prod

# 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

> **关键参数**：
> - `--shm-size=8g`：vLLM 需要较大的共享内存用于 PagedAttention
> - `--gpus all`：传递所有 GPU 给容器
> - `-v`：挂载模型目录，避免每次下载

### 2.2 docker-compose 编排

```yaml
# docker-compose.yml
version: "3.8"

services:
  # 模型推理服务
  vllm:
    image: vllm-llama2:prod
    container_name: vllm-server
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    volumes:
      - ./models:/models:ro
      - ./logs:/logs
    ports:
      - "8000:8000"
    shm_size: "8gb"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Nginx 反向代理 + 负载均衡
  nginx:
    image: nginx:alpine
    container_name: nginx-gateway
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - vllm
    restart: unless-stopped

  # Redis 缓存层
  redis:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  # Prometheus 监控
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    restart: unless-stopped

  # Grafana 可视化
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

启动整个栈：
```bash
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f vllm
```

---

## 3. Kubernetes 部署（Kubernetes Deployment）

### 3.1 为什么需要 K8s？

| 能力 | Docker Compose | Kubernetes |
|------|---------------|------------|
| 单节点部署 | 适合 | 可以 |
| 多节点扩展 | 不支持 | 原生支持 |
| 自动扩缩容 | 不支持 | HPA/VPA |
| 服务发现 | 基础 | 完善 |
| 故障恢复 | 手动 | 自动 |
| 滚动更新 | 不支持 | 原生支持 |
| GPU 调度 | 困难 | NVIDIA GPU Operator |

### 3.2 核心资源清单

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-serving

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama2
  namespace: llm-serving
spec:
  replicas: 1
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: vllm-llama2
  template:
    metadata:
      labels:
        app: vllm-llama2
    spec:
      containers:
        - name: vllm
          image: vllm-llama2:prod
          ports:
            - containerPort: 8000
              name: http
          resources:
            limits:
              nvidia.com/gpu: 2  # 申请 2 张 GPU
              memory: "32Gi"
            requests:
              nvidia.com/gpu: 2
              memory: "16Gi"
          env:
            - name: CUDA_VISIBLE_DEVICES
              value: "0,1"
            - name: VLLM_MODEL
              value: "/models/Llama-2-7b"
          volumeMounts:
            - name: model-volume
              mountPath: /models
            - name: shm
              mountPath: /dev/shm
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
      volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-pvc
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
      nodeSelector:
        accelerator: nvidia-gpu  # 调度到 GPU 节点

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: llm-serving
spec:
  selector:
    app: vllm-llama2
  ports:
    - port: 80
      targetPort: 8000
      name: http
  type: ClusterIP

---
# hpa.yaml（HPA 自动扩缩容）
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama2
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_avg_generation_throughput
        target:
          type: AverageValue
          averageValue: "10"  # 吞吐量低于 10 tokens/s 时扩容
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60

---
# pvc.yaml（模型存储）
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: llm-serving
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd  # 使用 SSD 存储类
```

部署：
```bash
# 应用所有配置
kubectl apply -f k8s/

# 查看状态
kubectl get pods -n llm-serving -w

# 查看日志
kubectl logs -f deployment/vllm-llama2 -n llm-serving

# 端口转发测试
kubectl port-forward svc/vllm-service -n llm-serving 8080:80
```

### 3.3 GPU 节点调度

```bash
# 查看 GPU 节点
kubectl get nodes -l accelerator=nvidia-gpu

# 查看节点 GPU 资源
kubectl describe node <node-name> | grep nvidia.com/gpu

# 为节点打标签
kubectl label nodes <node-name> accelerator=nvidia-gpu
```

---

## 4. 模型服务网关（Model Serving Gateway）

### 4.1 为什么需要网关？

单体的 vLLM/FastAPI 服务缺少生产级能力：
- **统一认证**：API Key 管理
- **限流熔断**：防止过载
- **A/B 测试**：多模型流量分割
- **请求路由**：根据模型/负载分发
- **缓存**：避免重复计算

### 4.2 BentoML 架构

```python
# service.py
import bentoml
from bentoml.io import Text, JSON

@bentoml.service(
    name="llm-service",
    traffic={"timeout": 300},
    resources={"gpu": 1, "memory": "16Gi"}
)
class LLMService:
    def __init__(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

    @bentoml.api(route="/generate")
    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    @bentoml.api(route="/chat")
    def chat(self, messages: list) -> dict:
        # OpenAI-compatible chat API
        ...
```

部署：
```bash
bentoml build
bentoml containerize llm-service:latest
bentoml deploy llm-service:latest --scaling min=1,max=3
```

### 4.3 Triton Inference Server

NVIDIA 出品的高性能推理服务器，支持多框架并发：

```bash
# 启动 Triton
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /models:/models:ro \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

模型仓库结构：
```
/models/
├── llama2-7b/
│   ├── 1/
│   │   └── model.py  # Python backend
│   └── config.pbtxt
└── embeddings/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

---

## 5. A/B 测试与金丝雀发布（A/B Testing & Canary）

### 5.1 流量分割策略

```
                    100% 流量
                       │
              ┌────────┴────────┐
              ▼                 ▼
        [旧模型 v1]         [新模型 v2]
           90%                10%      ← 金丝雀阶段
              │                 │
              └────────┬────────┘
                       ▼
                  监控指标对比
                       │
              ┌────────┴────────┐
              ▼                 ▼
           通过              失败
              │                 │
        扩大 v2 流量      回滚到 v1
```

### 5.2 Istio 流量管理

```yaml
# virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llm-routing
spec:
  hosts:
    - llm-api.example.com
  http:
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: vllm-v2
          weight: 100
    - route:
        - destination:
            host: vllm-v1
          weight: 90
        - destination:
            host: vllm-v2
          weight: 10
```

### 5.3 影子模式（Shadow Mode）

新模型接收真实流量但不返回结果，仅用于验证：

```python
import asyncio

async def shadow_test(request, model_v1, model_v2):
    # v1 返回结果给用户
    response_v1 = await model_v1.generate(request)

    # v2 在后台处理，记录对比
    asyncio.create_task(
        log_comparison(request, response_v1, model_v2)
    )

    return response_v1
```

---

## 6. 缓存策略（Caching Strategies）

### 6.1 三层缓存架构

```
用户请求
    │
    ▼
[L1: 精确缓存] ──→ 完全相同请求？──→ 直接返回
    │ No
    ▼
[L2: 语义缓存] ──→ 语义相似请求？──→ 返回缓存结果
    │ No
    ▼
[L3: 前缀缓存] ──→ 前缀匹配？──→ 复用 KV Cache
    │ No
    ▼
  [模型推理]
```

### 6.2 语义缓存实现

```python
import redis
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    def __init__(self, redis_host="localhost", threshold=0.95):
        self.redis = redis.Redis(host=redis_host, decode_responses=False)
        self.encoder = SentenceTransformer("BAAI/bge-large-zh")
        self.threshold = threshold

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.encoder.encode(text, normalize_embeddings=True)

    def _hash_embedding(self, emb: np.ndarray) -> str:
        # 量化到桶，减少精确匹配需求
        bucket = (emb * 10).astype(np.int8)
        return hashlib.md5(bucket.tobytes()).hexdigest()[:16]

    def get(self, query: str) -> str | None:
        query_emb = self._get_embedding(query)
        query_hash = self._hash_embedding(query_emb)

        # 搜索相似键
        for key in self.redis.scan_iter(match="cache:*"):
            stored_emb = np.frombuffer(self.redis.hget(key, "embedding"), dtype=np.float32)
            similarity = np.dot(query_emb, stored_emb)
            if similarity > self.threshold:
                return self.redis.hget(key, "response").decode()
        return None

    def set(self, query: str, response: str, ttl=3600):
        emb = self._get_embedding(query)
        h = self._hash_embedding(emb)
        key = f"cache:{h}"
        self.redis.hset(key, mapping={
            "embedding": emb.astype(np.float32).tobytes(),
            "response": response,
            "query": query
        })
        self.redis.expire(key, ttl)
```

### 6.3 前缀缓存（Prefix Caching）

利用 vLLM 的 RadixAttention 自动实现：

```python
from vllm import LLM

# vLLM >= 0.4.0 自动支持前缀缓存
llm = LLM(
    model="meta-llama/Llama-2-7b",
    enable_prefix_caching=True  # 启用前缀缓存
)

# 第一次请求：计算并缓存 system prompt 的 KV
response1 = llm.generate([system_prompt + "问题1"])

# 第二次请求：复用 system prompt 的 KV Cache
response2 = llm.generate([system_prompt + "问题2"])  # 更快！
```

---

## 7. 监控体系（Monitoring）

### 7.1 LLM 专属指标设计

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# 请求指标
REQUEST_COUNT = Counter("llm_requests_total", "Total requests", ["model", "status"])
REQUEST_LATENCY = Histogram("llm_request_duration_seconds", "Request latency", ["model"])

# Token 指标
TOKENS_INPUT = Counter("llm_tokens_input_total", "Input tokens", ["model"])
TOKENS_OUTPUT = Counter("llm_tokens_output_total", "Output tokens", ["model"])
TOKEN_LATENCY = Histogram("llm_token_latency_seconds", "Per-token latency", ["model"])

# GPU 指标
GPU_UTILIZATION = Gauge("llm_gpu_utilization", "GPU utilization %", ["gpu_id"])
GPU_MEMORY = Gauge("llm_gpu_memory_used_mb", "GPU memory used MB", ["gpu_id"])

# 业务指标
QUEUE_SIZE = Gauge("llm_queue_size", "Pending request queue size")
ACTIVE_SESSIONS = Gauge("llm_active_sessions", "Active sessions")
```

### 7.2 FastAPI + Prometheus 集成

```python
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app
import time

app = FastAPI()

# 挂载 Prometheus 指标端点
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(
        model="llama2-7b",
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(model="llama2-7b").observe(duration)

    return response
```

### 7.3 Grafana 看板配置

核心面板：

| 面板 | 指标 | 告警阈值 |
|------|------|---------|
| QPS | `rate(llm_requests_total[1m])` | — |
| P99 延迟 | `histogram_quantile(0.99, llm_request_duration_seconds)` | > 5s |
| TTFT | `histogram_quantile(0.99, time_to_first_token)` | > 1s |
| TPOT | `histogram_quantile(0.99, llm_token_latency_seconds)` | > 100ms |
| GPU 利用率 | `llm_gpu_utilization` | < 50% |
| GPU 显存 | `llm_gpu_memory_used_mb / gpu_memory_total * 100` | > 90% |
| 队列长度 | `llm_queue_size` | > 100 |

---

## 8. 可观测性（Observability）

### 8.1 OpenTelemetry 链路追踪

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# 初始化
provider = TracerProvider()
otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4317")
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-service")

# 在推理代码中使用
async def generate_with_trace(request_id, messages):
    with tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("request.id", request_id)
        span.set_attribute("model.name", "llama2-7b")
        span.set_attribute("input.tokens", count_tokens(messages))

        # 预处理
        with tracer.start_span("llm.preprocess"):
            inputs = tokenizer.apply_chat_template(messages)

        # 推理
        with tracer.start_span("llm.inference"):
            outputs = model.generate(inputs)

        span.set_attribute("output.tokens", len(outputs))
        span.set_attribute("duration_ms", elapsed_ms)

        return outputs
```

### 8.2 Langfuse 集成

Langfuse 是开源的 LLM 可观测性平台：

```python
from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse(
    public_key="pk-lf-xxx",
    secret_key="sk-lf-xxx",
    host="http://localhost:3000"
)

@observe(as_type="generation")
def llm_generate(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 自动记录：prompt、response、token 用量、延迟、成本
```

---

## 9. CI/CD for ML（CI/CD for Machine Learning）

### 9.1 GitHub Actions 工作流

```yaml
# .github/workflows/llm-cicd.yml
name: LLM CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio

      - name: Run unit tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Prompt regression tests
        run: pytest tests/prompt_regression/ -v

      - name: RAG evaluation
        run: python scripts/eval_rag.py --dataset tests/rag_eval.json

      - name: Model performance benchmark
        run: python scripts/benchmark.py --model ./models --timeout 300

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t llm-app:${{ github.sha }} .

      - name: Push to registry
        run: |
          docker tag llm-app:${{ github.sha }} registry.example.com/llm-app:${{ github.sha }}
          docker push registry.example.com/llm-app:${{ github.sha }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/llm-app \
            app=registry.example.com/llm-app:${{ github.sha }} \
            -n staging
```

### 9.2 模型版本管理

```python
# model_registry.py
import mlflow
from datetime import datetime

def register_model(model_path, metrics, tags):
    mlflow.set_tracking_uri("http://mlflow-server:5000")

    with mlflow.start_run():
        # 记录参数
        mlflow.log_params({
            "model_name": "llama2-7b-sft",
            "base_model": "meta-llama/Llama-2-7b",
            "lora_rank": 64,
            "lora_alpha": 128
        })

        # 记录指标
        mlflow.log_metrics({
            "eval_loss": metrics["loss"],
            "perplexity": metrics["ppl"],
            "bleu": metrics["bleu"],
            "latency_p99": metrics["latency_p99"]
        })

        # 注册模型
        mlflow.transformers.log_model(
            transformers_model=model_path,
            artifact_path="model",
            registered_model_name="llm-chat-v1"
        )

# 使用
register_model(
    model_path="./output/checkpoint-1000",
    metrics={"loss": 1.23, "ppl": 3.42, "bleu": 0.65, "latency_p99": 120},
    tags={"stage": "staging", "reviewed": "false"}
)
```

---

## 10. 工业界部署实践（Industrial Deployment Practices）

本节分析 OpenAI、Anthropic 等公司在生产环境中部署 LLM 服务的实际架构和工程决策。

### 10.1 OpenAI API 的基础设施架构

OpenAI 的 API 服务是全球最大的 LLM 推理集群之一，其架构选择是行业的 de facto 标准。

#### 多区域部署与流量管理

```
全球流量
   ↓
[Cloudflare / AWS Route 53] ──→ 地理位置路由
   ↓
┌─────────────┬─────────────┬─────────────┐
│ 美国西部    │ 美国东部    │ 欧洲        │
│ (us-west)   │ (us-east)   │ (eu-west)   │
└─────────────┴─────────────┴─────────────┘
   ↓              ↓              ↓
[API Gateway]  [API Gateway]  [API Gateway]
   ↓              ↓              ↓
[推理集群]     [推理集群]     [推理集群]
   ↓              ↓              ↓
[模型权重存储]  [模型权重存储]  [模型权重存储]
   (S3/EFS)      (S3/EFS)      (S3/EFS)
```

**关键设计**：
1. **就近路由**：用户请求路由到最近的区域，降低网络延迟
2. **区域级故障转移**：某个区域故障时自动切换到其他区域
3. **模型分片存储**：大模型权重分布式存储，按需加载到推理节点

#### 动态模型加载与热切换

OpenAI API 支持同一 endpoint 访问不同模型（gpt-4, gpt-4-turbo, gpt-4o）：

```
请求: /v1/chat/completions
  {
    "model": "gpt-4o",  // 或 gpt-4-turbo, gpt-3.5-turbo
    ...
  }

路由层根据 model 字段分发:
  gpt-4o ──→ 加载 GPT-4o 权重的 GPU 池
  gpt-4 ──→ 加载 GPT-4 权重的 GPU 池
  gpt-3.5 ──→ 加载 GPT-3.5 权重的 GPU 池
```

**工程挑战**：
- 不同模型需要不同的 GPU 资源（显存、计算能力）
- 模型切换时的权重加载时间（数十 GB 的模型文件）
- 解决方案：预加载热门模型，冷模型按需加载

#### 速率限制与配额管理

OpenAI 的多层级速率限制：

| 层级 | 限制维度 | 典型值 |
|------|---------|--------|
| **Organization** | 组织级总 QPS | 根据合同定制 |
| **Project** | 项目级 QPS | 组织配额的分割 |
| **User/API Key** | 密钥级 RPM/TPM | 如 60 RPM, 150K TPM |
| **Model** | 模型级并发 | gpt-4 vs gpt-3.5 不同 |

**令牌桶算法实现**：
```python
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate      # 每秒补充的 token 数
        self.capacity = capacity  # 桶容量（突发流量）
        self.tokens = capacity
        self.last_update = time.time()

    def consume(self, tokens=1):
        now = time.time()
        # 补充 token
        self.tokens = min(
            self.capacity,
            self.tokens + (now - self.last_update) * self.rate
        )
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True  # 允许请求
        return False  # 拒绝请求，返回 429
```

#### 按量计费的实现

OpenAI 的计费精确到 token 级别：
- **Input tokens**：Prompt 中的文本 token 数
- **Output tokens**：生成响应的 token 数
- **不同定价**：input 和 output 分别计价（output 通常更贵）

**计费采集点**：
```
请求进入 → [tokenizer 计数 input tokens] → 推理 → [tokenizer 计数 output tokens]
                                              ↓
                                         记录到计费数据库
                                              ↓
                                         实时余额检查
```

### 10.2 Anthropic 的 Claude API 架构

Anthropic 在 Claude 3 的部署中展现了不同的工程重点。

#### 长上下文服务的特殊挑战

Claude 3 支持 200K token 上下文，这对基础设施提出独特要求：

**1. KV Cache 的显存管理**
- 200K 上下文的 KV Cache 占用巨大（LLaMA-3 70B 约 500GB+）
- 方案：分层存储（GPU HBM → CPU DRAM → NVMe SSD）
- 热数据在 GPU，温数据在 CPU，冷数据在 SSD
- LRU 淘汰策略 + 预加载预测

**2. 预填充阶段的优化**
- 长文档的预填充（Prefill）是计算密集型阶段
- 采用 ** chunked prefill**：将长输入分成多个 chunk 并行处理
- (chunk1, chunk2, chunk3) 并行计算 KV Cache，然后拼接

**3. 200K 的渐进式推出**
- Anthropic 没有一开始就开放 200K 给所有用户
- 先向企业客户开放，收集性能和稳定性数据
- 逐步扩展到更多用户，同时优化基础设施

#### 提示缓存（Prompt Caching）的成本优化

Anthropic 2024 年推出的 Prompt Caching 是降低长上下文成本的关键创新：

```
第一次请求:
  [System Prompt] + [Long Document] + [User Question 1]
     ↑ 全部计算 KV Cache，计费 100%

第二次请求（同一文档，不同问题）:
  [System Prompt] + [Long Document] + [User Question 2]
     ↑ 命中缓存！这部分只计费 50%
     ↑ 只需计算 User Question 2 的 KV Cache
```

**实现机制**：
- 自动识别请求的公共前缀（通常是最前面的 system prompt + context）
- 缓存前缀的 KV Cache，新请求只需计算后缀
- 缓存命中后，TTFT 从数秒降至数百毫秒

**对行业的影响**：
- RAG 应用的成本大幅降低（文档上下文只需计算一次）
- 多轮对话的成本降低（历史消息缓存）
- 成为大模型 API 的标配功能（OpenAI 随后推出类似功能）

### 10.3 企业私有化部署实践

#### 金融行业部署案例

某大型银行部署 LLaMA-3 70B 用于内部知识问答：

```
部署架构:
  私有云（不连接互联网）
     ↓
  [K8s 集群]
     ├── vLLM 推理服务 (4× A100 80GB)
     ├── RAG 管道 (向量数据库 + 重排序)
     ├── 安全网关（PII 检测 + 审计日志）
     └── 监控（Prometheus + Grafana）
     ↓
  [内部网络]
     ├── 网银系统 API 调用
     ├── 风控系统 API 调用
     └── 内部办公助手
```

**关键安全要求**：
1. **数据不出域**：所有计算在私有云完成
2. **PII 检测**：输入输出都经过脱敏处理
3. **审计追踪**：每个请求记录谁、何时、问了什么、得到什么回答
4. **权限控制**：不同部门访问不同知识库

#### 医疗行业部署案例

某医院部署医学大模型用于辅助诊断：

**特殊挑战**：
- **HIPAA 合规**：患者数据隐私保护
- **高可用性**：不能宕机，需要多副本
- **低延迟**：医生等待时间不能超过 3 秒
- **准确性**：幻觉可能导致严重后果

**解决方案**：
- 本地部署（on-premise），数据不离院
- RAG 接入医院知识库（药品说明、诊疗指南）
- 输出附带引用来源，医生可验证
- 明确标注"AI 建议，仅供参考"

### 10.4 大规模推理集群的管理

#### GPU 集群调度策略

在数百张 GPU 的集群中，调度策略直接影响成本和性能：

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **先来先服务 (FCFS)** | 按到达顺序处理 | 简单、公平 |
| **最短作业优先 (SJF)** | 优先处理短请求 | 降低平均延迟 |
| **分桶调度** | 按序列长度分桶 | 减少 padding 浪费 |
| **抢占式调度** | 长请求可被中断 | 保证短请求 SLA |
| **成本感知调度** | 优先使用低成本 GPU | 降低整体成本 |

**OpenAI 的推测策略**：
- 实时请求（chat）和批量请求（batch API）共享集群
- 批量请求在低峰期运行，利用闲置算力
- 实时请求优先级高，可抢占批量请求的算力

#### 模型并行与流水线并行

大规模模型（如 GPT-4 推测的 1.8T）需要多层并行：

```
张量并行 (Tensor Parallelism):
  单层权重分片到多 GPU
  例如: Linear(8192, 8192) → 4× Linear(8192, 2048)
  通信: All-Reduce（每层的激活值聚合）
  适用: 单节点多 GPU（NVLink 高速互联）

流水线并行 (Pipeline Parallelism):
  不同层分配到不同 GPU
  例如: Layer 1-10 → GPU0, Layer 11-20 → GPU1
  通信: 层间激活值传递
  适用: 跨节点扩展

3D 并行 (数据 + 张量 + 流水线):
  数据并行: 不同 batch 分配到不同节点
  张量并行: 单层分片到节点内多 GPU
  流水线并行: 层分配到不同节点
  
  总 GPU 数 = 数据并行度 × 张量并行度 × 流水线并行度
  例如: 8 × 8 × 4 = 256 GPU
```

#### 故障恢复与弹性

大规模集群中 GPU 故障是常态：

```
GPU 故障率:
  - H100: ~1-2% 月故障率
  - 256 GPU 集群: 平均每周 1-2 次故障

应对策略:
  1. 健康检查: 每 30 秒检测 GPU 状态
  2. 优雅降级: 故障 GPU 的请求路由到其他副本
  3. 自动重启: 故障节点自动重启并重新加入集群
  4. Checkpoint: 定期保存 KV Cache（长对话场景）
  5. 冗余部署: 关键模型至少 3 个副本
```

### 10.5 成本优化策略

#### 动态批处理与请求合并

```python
# 动态批处理的工业实现
class DynamicBatcher:
    def __init__(self, max_batch_size=64, max_wait_ms=10):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []

    async def add_request(self, request):
        self.queue.append(request)

        # 如果队列满或等待超时，执行批处理
        if len(self.queue) >= self.max_batch_size:
            return await self._process_batch()

    async def _process_batch(self):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]

        # 动态 padding 到同一长度
        max_len = max(len(r.tokens) for r in batch)
        padded = [r.pad_to(max_len) for r in batch]

        # 并行推理
        outputs = await model.generate(padded)
        return outputs
```

**效果**：
- 批大小从 1 提升到 16-64，吞吐量提升 10-50 倍
- 短请求等待时间增加（max_wait_ms 内），但总体延迟降低

#### 量化部署的成本收益

| 方案 | 显存占用 | 吞吐量 | 质量损失 | 适用场景 |
|------|---------|--------|---------|---------|
| FP16 基线 | 100% | 1x | 0% | 高精度要求 |
| FP8 (H100) | 50% | 1.8x | <1% | 生产首选 |
| INT8 (SmoothQuant) | 50% | 1.5x | 1-2% | 通用场景 |
| INT4 (AWQ) | 25% | 2.0x | 2-5% | 资源受限 |

**实际案例**：
- 某公司将 LLaMA-3 70B 从 FP16 切换到 FP8
- 同样 8×A100 集群，并发能力从 32 → 64
- 单请求成本降低 45%，质量无明显下降

### 10.6 面试高频考点：工业界部署

1. **OpenAI 如何实现同一 endpoint 支持多模型？**
   > 答：路由层根据请求中的 model 字段分发到不同的 GPU 池。每个模型有独立的权重加载和推理实例。热门模型常驻内存，冷门模型按需加载。

2. **Anthropic 的 Prompt Caching 如何降低长上下文成本？**
   > 答：自动识别请求的公共前缀（如 system prompt + 文档），缓存前缀的 KV Cache。新请求只需计算新后缀，缓存部分按折扣价（50%）计费。TTFT 从数秒降至数百毫秒。

3. **金融/医疗行业部署 LLM 的核心安全要求？**
   > 答：数据不出域（本地/私有云部署）；PII 检测与脱敏；完整的审计日志；权限控制（不同角色访问不同数据）；输出标注"AI 建议仅供参考"；RAG 接入权威知识库减少幻觉。

4. **大规模 GPU 集群中如何处理故障？**
   > 答：定期健康检查检测 GPU 状态；故障节点的请求路由到健康副本；自动重启并重新加入集群；关键模型至少 3 副本；长对话定期 checkpoint KV Cache。

5. **动态批处理为什么能提升吞吐量？**
   > 答：将多个短请求合并为一个 batch，GPU 并行计算。矩阵乘法在 batch 维度上天然并行，batch size 从 1 提升到 16-64，吞吐量提升 10-50 倍。代价是短请求需要等待凑齐 batch。

---

## 11. 生产 checklist（Production Checklist）

### 部署前必检项

| 检查项 | 标准 | 验证方式 |
|--------|------|---------|
| 模型加载时间 | < 60s | 多次重启测试 |
| 首 token 延迟 | < 500ms | 压力测试 |
| 并发支持 | >= 10 req/s | Locust/Apache Bench |
| 内存泄漏 | 24h 无增长 | 长时间运行监控 |
| 故障恢复 | 自动重启 < 30s | kill pod 测试 |
| 灰度发布 | 支持 1%→100% | 流量切换测试 |
| 回滚时间 | < 5min | 模拟回滚 |
| 日志完整 | 100% 请求可追踪 | 抽样检查 |
| 安全扫描 | 无高危漏洞 | Trivy/Clair 扫描 |
| 成本估算 | 单请求成本可接受 | 账单分析 |

---

## 11. 面试高频考点

1. **为什么 vLLM 部署需要 `--shm-size=8g`？**
   - vLLM 的 PagedAttention 使用共享内存存储 KV Cache
   - 默认 shm 太小会导致 OOM 或性能下降
   - 建议设置为 GPU 显存的 1/4 到 1/2

2. **K8s 中 GPU 调度如何实现？**
   - 安装 NVIDIA GPU Operator
   - 节点打上 `nvidia.com/gpu` 资源标签
   - Pod 通过 `resources.limits.nvidia.com/gpu` 申请
   - GPU Device Plugin 负责分配

3. **LLM 服务的缓存策略有哪些？**
   - 精确缓存：hash(request) → response
   - 语义缓存：embedding 相似度匹配
   - 前缀缓存：复用共同前缀的 KV Cache（vLLM RadixAttention）
   - 响应缓存：热门问题的固定答案

4. **如何设计 LLM 的监控指标？**
   - 系统层：GPU 利用率/显存/温度、QPS、延迟
   - 业务层：TTFT、TPOT、吞吐量、token 消耗
   - 质量层：输出长度分布、错误率、用户满意度
   - 成本层：单请求成本、GPU 利用率vs成本

5. **金丝雀发布 vs A/B 测试的区别？**
   - 金丝雀：验证新版本的稳定性，流量比例逐渐扩大
   - A/B 测试：对比两个版本的业务效果，流量固定比例
   - 金丝雀关注"不崩"，A/B 测试关注"哪个更好"
