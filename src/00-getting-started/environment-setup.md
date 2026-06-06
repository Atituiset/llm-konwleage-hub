# 开发环境搭建指南（Development Environment Setup）

> **适合人群**：零基础或需要重新配置环境的读者
> **预计时间**：30-60 分钟
> **难度**：⭐（简单）

---

## 0. 快速验证清单（Quick Checklist）

搭好环境后，你应该能运行以下命令并得到正确输出：

```bash
# Python 版本 >= 3.9
python --version

# PyTorch 能识别 GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 本书核心依赖都能导入
python -c "import transformers, datasets, accelerate, peft"
```

如果以上全部通过，你可以跳过本章直接开始学习。

---

## 1. 系统要求（System Requirements）

### 1.1 操作系统选择

| 系统 | 推荐度 | 说明 |
|------|--------|------|
| **Linux (Ubuntu 22.04/24.04)** | ⭐⭐⭐⭐⭐ | AI开发首选，所有工具链原生支持 |
| **macOS** | ⭐⭐⭐⭐ | 适合CPU开发，M系列芯片支持PyTorch MPS |
| **Windows + WSL2** | ⭐⭐⭐ | 可用，但CUDA和Docker配置稍复杂 |
| **Windows 原生** | ⭐⭐ | 不推荐，很多库兼容性差 |

**建议**：如果条件允许，直接使用 Ubuntu 或云服务器（AutoDL/阿里云）。

### 1.2 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | 4核 | 8核+ |
| 内存 | 16 GB | 32 GB+ |
| 硬盘 | 50 GB SSD | 200 GB+ NVMe SSD |
| GPU | 无（CPU模式） | NVIDIA RTX 3090/4090 / A100 |
| 显存 | — | 24 GB+ |

> **没有GPU怎么办？** 本书大部分代码支持CPU运行（只是慢）。你也可以使用：
> - **Google Colab**（免费T4 GPU）
> - **AutoDL**（按小时租用，RTX 3090 约 ￥1.5/小时）
> - **阿里云PAI**、**恒源云**、**Featurize**

---

## 2. Python 环境管理（Python Environment）

### 2.1 安装 Miniconda（推荐）

Conda 是 Python 环境管理的事实标准，能隔离不同项目的依赖。

**Linux/macOS**：
```bash
# 下载安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 运行安装（按提示操作，建议同意初始化shell）
bash Miniconda3-latest-Linux-x86_64.sh

# 重启终端或source
source ~/.bashrc

# 验证
conda --version  # 应显示 conda 24.x.x
```

**Windows**：下载 [Miniconda 安装包](https://docs.conda.io/en/latest/miniconda.html) 并运行。

### 2.2 配置国内镜像源（中国大陆用户）

```bash
# Conda 使用清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes

# pip 使用清华镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.3 创建本书专属环境

```bash
# 创建 Python 3.10 环境（本书所有代码基于 3.9+）
conda create -n llm-book python=3.10 -y

# 激活环境
conda activate llm-book

# 验证 Python 路径
which python  # 应显示 .../miniconda3/envs/llm-book/bin/python
python --version  # Python 3.10.x
```

> **核心概念**：`conda activate` 和 `conda deactivate` 是你每天用得最多的命令。不同项目用不同环境，避免依赖冲突。

---

## 3. GPU 环境配置（GPU Setup）

### 3.1 检查 NVIDIA 驱动

```bash
nvidia-smi
```

如果显示类似以下内容，说明驱动已安装：
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   45C    P8              25W / 350W |    512MiB / 24576MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+
```

**关键信息**：
- **Driver Version**：驱动版本（应 >= 470）
- **CUDA Version**：驱动支持的最高 CUDA 版本（如 12.2）

### 3.2 驱动未安装？（Linux）

```bash
# Ubuntu 自动安装推荐驱动
ubuntu-drivers devices  # 查看推荐驱动
sudo ubuntu-drivers autoinstall  # 自动安装

# 或手动安装最新驱动
sudo apt update
sudo apt install nvidia-driver-535  # 替换为最新版本
sudo reboot
```

### 3.3 CUDA Toolkit 安装（可选但推荐）

PyTorch 会自带 CUDA runtime，但如果你需要编译自定义 CUDA 扩展，需要安装 CUDA Toolkit：

```bash
# 方式1：通过 conda 安装（推荐，简单）
conda install -c nvidia cuda-toolkit=12.1

# 方式2：通过 NVIDIA 官网下载 runfile 安装
# 访问 https://developer.nvidia.com/cuda-downloads
```

> **注意**：`nvidia-smi` 显示的 CUDA Version 是驱动支持的**最高**版本。PyTorch 实际使用的 CUDA 版本可以低于这个值。

---

## 4. PyTorch 安装（PyTorch Installation）

### 4.1 选择正确的版本

PyTorch 版本必须与 CUDA 版本匹配。访问 [pytorch.org](https://pytorch.org) 获取最新安装命令。

**常见组合**：

| CUDA 版本 | PyTorch 安装命令 |
|-----------|-----------------|
| CUDA 12.1 | `pip install torch torchvision torchaudio` |
| CUDA 11.8 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| CPU only | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |

### 4.2 本书推荐安装

```bash
# 确保在 llm-book 环境中
conda activate llm-book

# 安装 PyTorch 2.x + CUDA 12.1（当前最稳定组合）
pip install torch torchvision torchaudio

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU可用: {torch.cuda.is_available()}')"
```

**预期输出**：
```
PyTorch: 2.3.0+cu121
CUDA: 12.1
GPU可用: True
```

### 4.3 常见安装问题

**问题1：`torch.cuda.is_available()` 返回 False**

排查步骤：
```bash
# 1. 确认 nvidia-smi 能显示GPU
nvidia-smi

# 2. 确认 PyTorch CUDA 版本与驱动兼容
python -c "import torch; print(torch.version.cuda)"  # 应 <= nvidia-smi 的 CUDA Version

# 3. 如果装成了CPU版本，重新安装
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**问题2：`libcudart.so.x.x: cannot open shared object file`**

```bash
# 安装 CUDA 运行时库
conda install -c nvidia cudatoolkit=12.1

# 或设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

**问题3：OOM（Out of Memory）**

这是运行时的显存不足，不是安装问题。解决方案见本书"推理优化技术"章节。

---

## 5. 开发工具链（Development Tools）

### 5.1 VS Code（推荐 IDE）

```bash
# Ubuntu
sudo snap install code --classic

# macOS
brew install --cask visual-studio-code
```

**必装插件**：
| 插件 | 用途 |
|------|------|
| Python | Python 语言支持、调试、Linting |
| Jupyter | Notebook 支持 |
| Pylance | 类型检查、智能提示 |
| GitLens | Git 历史查看 |
| Markdown All in One | Markdown 编辑增强 |
| Rainbow CSV | CSV 文件高亮 |

### 5.2 Git 配置

```bash
# 安装
sudo apt install git  # Ubuntu
brew install git      # macOS

# 配置身份
git config --global user.name "你的名字"
git config --global user.email "your.email@example.com"

# 验证
git --version
```

### 5.3 Jupyter Lab

```bash
pip install jupyterlab

# 启动
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

---

## 6. 本书代码环境（Book Environment）

### 6.1 一键安装核心依赖

创建 `requirements.txt`：

```text
# 深度学习框架
torch>=2.0.0
transformers>=4.40.0
datasets>=2.14.0
accelerate>=0.25.0
peft>=0.8.0
bitsandbytes>=0.41.0

# 推理与部署
vllm>=0.4.0
fastapi>=0.109.0
uvicorn>=0.27.0

# RAG 与向量数据库
langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.0
sentence-transformers>=2.5.0

# Agent 与工具
openai>=1.10.0
crewai>=0.30.0

# 数据处理
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# 其他工具
tqdm>=4.65.0
wandb>=0.16.0
python-dotenv>=1.0.0
requests>=2.31.0
httpx>=0.26.0

# 开发工具
pytest>=7.4.0
black>=23.0.0
isort>=5.12.0
```

安装：
```bash
conda activate llm-book
pip install -r requirements.txt
```

### 6.2 环境验证脚本

创建 `check_env.py`：

```python
#!/usr/bin/env python3
"""验证本书开发环境是否配置正确"""

import sys

def check():
    print("=" * 60)
    print("本书环境验证脚本")
    print("=" * 60)

    # Python 版本
    py_version = sys.version_info
    assert py_version >= (3, 9), f"Python 版本需 >= 3.9，当前 {py_version.major}.{py_version.minor}"
    print(f"[OK] Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    # PyTorch & GPU
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[OK] CUDA: {torch.version.cuda}")
        # 运行一个简单的GPU操作
        x = torch.rand(100, 100).cuda()
        y = x @ x.T
        print(f"[OK] GPU 计算测试通过")
    else:
        print("[WARN] 未检测到 GPU，将以 CPU 模式运行（速度较慢）")

    # 核心库
    libs = [
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("langchain", "LangChain"),
        ("fastapi", "FastAPI"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence-Transformers"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
    ]

    for module, name in libs:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[FAIL] {name} 未安装")

    print("=" * 60)
    print("验证完成！如果显示 [FAIL]，请运行：pip install <包名>")
    print("=" * 60)

if __name__ == "__main__":
    check()
```

运行验证：
```bash
python check_env.py
```

---

## 7. Docker 基础（Docker Basics）

Docker 是生产环境部署的标配，建议提前熟悉。

### 7.1 安装 Docker

```bash
# Ubuntu 一键安装
curl -fsSL https://get.docker.com | sh

# 将当前用户加入 docker 组（免 sudo）
sudo usermod -aG docker $USER
newgrp docker

# 验证
docker --version
docker run hello-world
```

### 7.2 NVIDIA Docker（GPU 容器）

```bash
# 安装 nvidia-docker2
sudo apt install nvidia-docker2
sudo systemctl restart docker

# 测试 GPU 容器
docker run --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### 7.3 本书 Dockerfile 示例

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

WORKDIR /app

# 安装 Python 和基础工具
RUN apt update && apt install -y python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

CMD ["/bin/bash"]
```

使用：
```bash
# 构建镜像
docker build -t llm-book-env .

# 运行容器（挂载代码目录）
docker run --gpus all -it -v $(pwd):/app llm-book-env
```

---

## 8. 云 GPU 环境（Cloud GPU）

如果你没有本地 GPU，云服务器是最便捷的选择。

### 8.1 AutoDL（推荐，国内）

1. 注册 [AutoDL](https://www.autodl.com)
2. 选购实例：RTX 3090（24G）约 ￥1.5/小时，A100（80G）约 ￥8/小时
3. 选择镜像：**PyTorch 2.x + CUDA 12.x**
4. SSH 连接：
```bash
ssh -p <端口> root@<主机地址>
# 密码在控制台查看
```
5. 环境已预装，直接开始

### 8.2 阿里云 PAI-DSW

1. 进入 [PAI控制台](https://pai.console.aliyun.com)
2. 创建 DSW 实例，选择 GPU 规格
3. 选择镜像：PAI-PyTorch 2.x
4. 通过 JupyterLab 或 SSH 使用

### 8.3 Google Colab（免费）

```python
# Colab 中切换 GPU：Runtime -> Change runtime type -> GPU
# 验证
import torch
torch.cuda.is_available()  # True
```

> **注意**：Colab 免费版会话最长 12 小时，且 GPU 可能被回收。

---

## 9. 常见问题 FAQ

### Q1: Conda 和 pip 混用会出问题吗？

**原则**：先用 conda 装底层依赖（CUDA、编译器），再用 pip 装 Python 包。不要来回切换。

```bash
# 推荐顺序
conda install cudatoolkit  # conda 包
pip install torch transformers  # pip 包
```

### Q2: 多个项目依赖冲突怎么办？

每个项目创建一个独立环境：

```bash
conda create -n project-a python=3.10
conda create -n project-b python=3.11
# 使用时分别激活
```

### Q3: Windows 下能开发吗？

可以，但建议：
- 使用 **WSL2**（Windows Subsystem for Linux）
- 在 WSL2 中按本指南的 Linux 步骤操作
- WSL2 支持 GPU Passthrough（需 Windows 11 + 最新驱动）

### Q4: M1/M2 Mac 能用吗？

可以，但有限制：
- PyTorch 支持 MPS（Metal Performance Shaders）加速
- 部分库（如 vLLM、bitsandbytes）暂不支持 ARM
- 大模型需要量化才能跑

```python
# MPS 验证
import torch
print(torch.backends.mps.is_available())  # True
```

### Q5: 如何离线安装？

```bash
# 有网机器下载
pip download torch transformers -d ./packages

# 离线机器安装
pip install --no-index --find-links=./packages torch transformers
```

### Q6: 显存不够怎么办？

```bash
# 查看显存占用
nvidia-smi

# 清理 PyTorch 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 如果仍不够，使用量化模型（详见本书"推理优化技术"章节）
```

---

## 10. 下一步

环境验证通过后，你可以：

1. **按顺序学习**：从[第一阶段：AI 与大模型通识](../01-ai-overview/README.md)开始
2. **跳过基础**：如果你已有基础，直接从目标阶段开始
3. **动手实践**：每章代码都在环境中实际运行，不要只看不练

如果在环境搭建中遇到问题，先检查：
1. 是否在正确的 conda 环境中（`conda activate llm-book`）
2. `nvidia-smi` 是否正常输出
3. PyTorch 安装命令是否正确对应你的 CUDA 版本
