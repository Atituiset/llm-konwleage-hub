# 第零阶段：环境准备（Stage 0: Environment Setup）

> **为什么叫"第零阶段"？** 因为它是所有后续学习的前置准备，不是核心知识本身，但缺了它你无法动手实践。

---

## 本阶段目标

1. **30 分钟内**搭建完整的 AI 开发环境
2. 学会使用 **Conda** 管理 Python 环境
3. 验证 **GPU** 能正常工作
4. 掌握 **Docker** 基础操作
5. 了解**云 GPU** 租用方式

---

## 学习路径

```
[环境搭建指南](environment-setup.md)
    ├── 系统要求与硬件选择
    ├── Python / Conda 环境管理
    ├── GPU 驱动与 CUDA 配置
    ├── PyTorch 安装与验证
    ├── 开发工具链（VS Code, Git, Jupyter）
    ├── 本书代码环境一键配置
    ├── Docker 基础与 GPU 容器
    ├── 云 GPU 平台使用
    └── 常见问题 FAQ
```

---

## 验证标准

完成本阶段后，你应该能运行以下命令并得到正确输出：

```bash
# 1. 环境激活
conda activate llm-book

# 2. Python 版本
python --version  # Python 3.10.x

# 3. PyTorch + GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}')"

# 4. 本书核心库
python -c "import transformers, datasets, langchain, fastapi"
```

全部通过 → 进入[第一阶段：AI 与大模型通识](../01-ai-overview/README.md)

---

## 没有 GPU 怎么办？

本书**大部分代码支持 CPU 运行**，只是训练/推理速度较慢。你仍然可以：

- 完整学习所有理论知识
- 运行小模型（如 GPT-2、BERT-base）
- 使用云 GPU 完成大模型实验

**推荐云 GPU 平台**：
| 平台 | 特点 | 价格参考 |
|------|------|---------|
| [AutoDL](https://www.autodl.com) | 国内，按小时计费，镜像丰富 | RTX 3090 ~￥1.5/时 |
| [恒源云](https://www.gpushare.com) | 国内，学生优惠 | RTX 3090 ~￥1.2/时 |
| [Featurize](https://featurize.cn) | 国内，界面友好 | 多种卡型可选 |
| [阿里云 PAI](https://pai.console.aliyun.com) | 企业级，稳定 | 按量/包年包月 |
| [Google Colab](https://colab.research.google.com) | 免费，T4 GPU | 免费但有使用限制 |

---

## 环境管理的长期建议

```
项目A ──→ conda env: llm-book
    ├── PyTorch 2.3 + CUDA 12.1
    ├── transformers 4.40
    └── 本书所有依赖

项目B ──→ conda env: stable-diffusion
    ├── PyTorch 2.1 + CUDA 11.8
    ├── diffusers 0.27
    └── xformers

项目C ──→ conda env: tensorflow
    ├── TensorFlow 2.16
    └── 不装 PyTorch
```

**原则**：一项目一环境，永远不混用。
