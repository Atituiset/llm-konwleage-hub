# 第五阶段：PyTorch 框架实践

> **资料来源**：综合《PyTorch 实用教程（第二版）》《PyTorch 保姆级中文教程》《动手学 PyTorch 建模与应用》《Programming PyTorch for Deep Learning》《Deep Learning with PyTorch》
> **适合人群**：需要掌握深度学习工程实现的开发者
> **难度**：⭐⭐⭐（中等）

---

## 1. PyTorch 核心概念

### 1.1 为什么选 PyTorch

| 特性 | PyTorch | TensorFlow |
|------|---------|------------|
| 计算图 | 动态图（Eager） | 静态图（Graph） |
| 调试 | 直接用 pdb，像普通 Python | 需要专用工具 |
| 灵活性 | 高，随时修改 | 中等，先定义后运行 |
| 论文复现 | 90%+ 论文用 PyTorch | 早期主流 |
| 生产部署 | torch.compile, ONNX | TFLite, TF Serving |
| 大模型生态 | HuggingFace 首选 | 相对较弱 |

**结论**：学习大模型，PyTorch 是唯一选择。

### 1.2 Tensor：PyTorch 的核心数据结构

```python
import torch

# 创建 Tensor
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
x = torch.zeros(2, 3)
x = torch.ones(2, 3)
x = torch.randn(2, 3)  # 标准正态分布
x = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# 关键属性
print(x.shape)   # torch.Size([2, 3])
print(x.dtype)   # torch.float32
print(x.device)  # cpu / cuda:0
```

**Tensor 操作**：
```python
# 形状操作
x.view(3, 2)      # 重塑（共享内存）
x.reshape(3, 2)   # 重塑（必要时拷贝）
x.unsqueeze(0)    # 增加维度: (2,3) → (1,2,3)
x.squeeze()       # 移除大小为1的维度
x.permute(1, 0)   # 转置/维度重排

# 数学运算
y = x + 2         # 广播
z = x * y         # 逐元素乘法
w = x @ y.T       # 矩阵乘法
v = x.sum(dim=1)  # 按维度求和
m = x.mean(dim=0) # 按维度求均值

# 索引
x[0, 1]           # 单个元素
x[:, 1:3]         # 切片
x[x > 0.5]        # 布尔索引
```

**广播（Broadcasting）**：
```python
# (3, 1) + (1, 3) → (3, 3)
a = torch.tensor([[1], [2], [3]])      # (3, 1)
b = torch.tensor([[10, 20, 30]])       # (1, 3)
print(a + b)
# [[11, 21, 31],
#  [12, 22, 32],
#  [13, 23, 33]]
```

### 1.3 GPU 加速

```python
# 检查 GPU
print(torch.cuda.is_available())  # True/False
print(torch.cuda.device_count())  # GPU 数量

# 创建 GPU Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(2, 3).to(device)
x = torch.randn(2, 3, device=device)  # 直接创建在 GPU 上

# 模型和数据必须在同一设备
model = model.to(device)
inputs = inputs.to(device)
```

**多 GPU**：
```python
# 数据并行（最简单）
model = torch.nn.DataParallel(model)

# 分布式数据并行（推荐，更高效）
model = torch.nn.parallel.DistributedDataParallel(model)
```

---

## 2. 自动求导（Autograd）

### 2.1 核心原理

PyTorch 自动构建计算图并计算梯度：

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 2 * x + 1
z = y.sum()

z.backward()  # 自动计算梯度
print(x.grad)  # ∂z/∂x = [6, 8]
```

**requires_grad**：
- 设置为 True 的 Tensor 会追踪所有操作
- 叶子节点的梯度保存在 `.grad` 属性中
- 非叶子节点的梯度默认不保留（可设置 `retain_graph=True`）

### 2.2 计算图

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b      # c = 6
d = c + a ** 2  # d = 6 + 4 = 10
e = d * 2      # e = 20

e.backward()

# 手动验证：
# de/da = de/dd * dd/da = 2 * (b + 2a) = 2 * (3 + 4) = 14
# de/db = de/dd * dd/db = 2 * a = 4
print(a.grad)  # 14.0
print(b.grad)  # 4.0
```

**关键规则**：
- 默认只计算叶子节点的梯度
- 梯度会累积，每次 backward 前需要 `optimizer.zero_grad()`
- 训练时用 `with torch.no_grad()` 禁用梯度计算（推理加速省内存）

### 2.3 自定义梯度

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input

# 使用
x = torch.tensor(3.0, requires_grad=True)
y = MyFunction.apply(x)
y.backward()
print(x.grad)  # 6.0
```

---

## 3. 构建神经网络

### 3.1 nn.Module

所有神经网络模块都继承自 `nn.Module`：

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 实例化
model = MLP(784, 256, 10)
print(model)

# 前向传播
x = torch.randn(32, 784)
output = model(x)  # (32, 10)
```

**nn.Module 的特性**：
- `forward()` 定义前向传播
- 所有子模块自动注册，可用 `model.parameters()` 遍历
- 自动处理设备迁移（`.to(device)` 会递归应用到所有子模块）

### 3.2 常用层

```python
# 全连接层
nn.Linear(in_features=784, out_features=256)

# 卷积层
nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# 循环层
nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)

# Embedding
nn.Embedding(num_embeddings=10000, embedding_dim=256)

# LayerNorm
nn.LayerNorm(normalized_shape=256)

# 激活函数
nn.ReLU()
nn.GELU()
nn.Sigmoid()
nn.Softmax(dim=-1)
```

### 3.3 模型参数

```python
# 查看参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")

# 冻结部分参数
for param in model.backbone.parameters():
    param.requires_grad = False
```

---

## 4. 数据加载

### 4.1 Dataset 与 DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

# 使用
dataset = MyDataset(data, labels)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # 多进程加载
    pin_memory=True  # 加速 GPU 数据传输
)

for batch_x, batch_y in dataloader:
    # batch_x: (32, ...)
    # batch_y: (32,)
    pass
```

### 4.2 图像数据预处理

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),  # PIL → Tensor, [0, 255] → [0, 1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 4.3 文本数据处理

```python
from torch.nn.utils.rnn import pad_sequence

# 将变长序列填充到相同长度
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return texts_padded, labels

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

---

## 5. 训练流程

### 5.1 完整训练循环

```python
import torch.optim as optim

# 1. 准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 2. 训练循环
num_epochs = 50
best_val_acc = 0.0

for epoch in range(num_epochs):
    # ---- 训练 ----
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---- 验证 ----
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    scheduler.step()

    # ---- 保存最佳模型 ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_val_acc,
        }, 'best_model.pth')

    print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
          f"Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}")
```

### 5.2 模型保存与加载

```python
# 保存完整检查点（推荐）
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

# 仅保存/加载模型权重（部署时使用）
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))
```

### 5.3 训练模式 vs 评估模式

```python
model.train()   # 启用 Dropout、BatchNorm 使用 batch 统计
model.eval()    # 禁用 Dropout、BatchNorm 使用运行统计

# 推理时必须调用 eval()
with torch.no_grad():  # 禁用梯度计算
    predictions = model(inputs)
```

---

## 6. 大模型训练关键技术

### 6.1 混合精度训练（AMP）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()

    with autocast():  # 自动在 FP16/BF16 和 FP32 间切换
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**收益**：
- 显存减少 ~40%
- 训练速度提升 1.5-3×（Tensor Core 加速）
- 几乎无损精度

### 6.2 分布式数据并行（DDP）

```python
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    torch.distributed.init_process_group(
        backend='nccl',  # NVIDIA GPU
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # 训练循环...

# 启动
mp.spawn(train, args=(world_size,), nprocs=world_size)
```

**DDP vs DataParallel**：
- DataParallel：单进程多 GPU，有 GIL 瓶颈
- DDP：多进程，每个 GPU 一个进程，效率更高
- 大模型训练必须使用 DDP

### 6.3 梯度累积

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 6.4 梯度裁剪

```python
# 限制梯度范数，防止爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 6.5 检查点与恢复

```python
# 定期保存
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)

# 中断恢复
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']
```

---

## 7. 模型部署

### 7.1 ONNX 导出

```python
# 导出为 ONNX（跨平台部署）
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

### 7.2 TorchScript

```python
# 脚本化（脱离 Python 依赖）
scripted_model = torch.jit.script(model)
scripted_model.save('model.pt')

# 加载
loaded_model = torch.jit.load('model.pt')
```

### 7.3 torch.compile（PyTorch 2.0+）

```python
# 编译加速，自动优化计算图
model = torch.compile(model)
# 后续使用与普通模型相同，但更快
```

---

## 8. 与 HuggingFace 集成

HuggingFace Transformers 是大模型开发的事实标准：

```python
from transformers import AutoModel, AutoTokenizer

# 加载预训练模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')

# 编码文本
inputs = tokenizer("你好，世界", return_tensors='pt')
outputs = model(**inputs)

# 获取最后一层隐藏状态
last_hidden_state = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

# 获取 [CLS] 向量用于分类
cls_vector = last_hidden_state[:, 0, :]  # (1, hidden_dim)
```

**Trainer API**（简化训练）：
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

---

## 9. 调试技巧

### 9.1 检查张量

```python
print(x.shape)   # 形状
print(x.dtype)   # 数据类型
print(x.device)  # 设备
print(x.min(), x.max(), x.mean())  # 统计信息
print(torch.isnan(x).any())  # 检查 NaN
print(torch.isinf(x).any())  # 检查 Inf
```

### 9.2 梯度检查

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### 9.3 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `RuntimeError: Expected all tensors to be on the same device` | 模型和数据在不同设备 | 都 `.to(device)` |
| `CUDA out of memory` | 显存不足 | 减小 batch size、使用梯度累积、混合精度 |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | 矩阵维度不匹配 | 检查输入形状和模型期望 |
| Loss 不下降 | 学习率过大/过小、梯度消失 | 检查学习率、初始化、梯度范数 |
| Loss 为 NaN | 梯度爆炸、学习率过大 | 梯度裁剪、降低学习率、检查数据 |

---

## 10. 面试高频考点

1. **PyTorch 动态图 vs TensorFlow 静态图**
   - 动态图：每次前向传播时构建图，灵活易调试
   - 静态图：先定义图再执行，可全局优化，部署效率高

2. **`backward()` 时梯度为什么会累积？**
   - PyTorch 设计用于 RNN 等需要累积梯度的场景
   - 每次 backward 前需 `optimizer.zero_grad()`

3. **DataParallel 和 DDP 的区别**
   - DP：单进程，主 GPU 聚合梯度，有瓶颈
   - DDP：多进程，每个 GPU 独立计算，Ring-AllReduce 同步

4. **混合精度训练为什么能加速？**
   - Tensor Core 对 FP16/BF16 有硬件加速
   - 显存占用减半，可增大 batch size

5. **`with torch.no_grad()` 的作用**
   - 禁用梯度追踪，节省内存和计算
   - 推理时必须使用

6. **如何处理变长序列？**
   - `pad_sequence` 填充 + `pack_padded_sequence` 压缩
   - 或使用 Attention Mask

---

## 学习路径建议

1. **熟悉 Tensor 操作**：这是 PyTorch 的基础，每天写一点
2. **手写一个完整训练循环**：理解每个组件的作用
3. **用 PyTorch 复现一个简单模型**：如 LeNet、小型 Transformer
4. **在 GPU 上跑通**：理解设备管理
5. **尝试混合精度训练**：体验现代训练方式
6. **阅读 HuggingFace 源码**：学习工程最佳实践
