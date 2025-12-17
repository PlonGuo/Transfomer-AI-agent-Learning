# 从Transformer到AI Agent完整学习指南

> 为Jason定制 - 从底层原理到实战应用的13周学习路线
> 
> 最后更新: 2024年12月

---

## 📖 目录

1. [PyTorch前置要求 - 重要必读](#pytorch前置要求---重要必读)
2. [学习路线图](#学习路线图)
3. [Week 0: PyTorch基础速成 (1周)](#week-0-pytorch基础速成)
4. [Level 1: Transformer基础 (2-3周)](#level-1-transformer基础)
5. [Level 2: LLM工作原理 (1-2周)](#level-2-llm工作原理)
6. [Level 3: Prompt Engineering & RAG (1周)](#level-3-prompt-engineering--rag)
7. [Level 4: AI Agent架构 (2-3周)](#level-4-ai-agent架构)
8. [Level 5: 实战项目 (持续)](#level-5-实战项目)
9. [13周详细学习计划](#13周详细学习计划)
10. [额外资源](#额外资源)

---

## ⚠️ PyTorch前置要求 - 重要必读

### 为什么需要PyTorch?

本学习计划中 **Level 1-2 (Week 1-5)** 大量使用PyTorch:

```
需要PyTorch的部分:
├── ✅ Level 1: 手撸Transformer (重度使用)
│   ├── nanoGPT源码阅读
│   ├── 实现attention机制
│   └── 训练mini语言模型
├── ✅ Level 2: Fine-tuning模型 (中度使用)
│   ├── HuggingFace Transformers
│   └── 模型训练和优化
├── ❌ Level 3: RAG系统 (不需要)
├── ❌ Level 4: Agent开发 (不需要)
└── 🟡 Level 5: 实战项目 (可选，看项目类型)
```

### 快速自测

**如果你能看懂并写出下面的代码，可以跳过Week 0，直接从Week 1开始:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Tensor操作
x = torch.randn(2, 3, 4)  # batch_size=2, seq_len=3, d_model=4
y = x.transpose(1, 2)     # 转置
z = torch.matmul(x, y)    # 矩阵乘法

# 2. 定义简单模型
class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attention = F.softmax(scores, dim=-1)
        return attention

# 3. 训练循环
model = SimpleAttention(d_model=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = output.mean()  # 简化的loss
    loss.backward()
    optimizer.step()
```

**如果上面的代码你看不懂或写不出来，请从Week 0开始学习。**

### 学习路径选择

#### 路径A: 完整学习 (推荐 - 适合想深入理解AI的人)
```
Week 0: PyTorch基础
  ↓
Week 1-2: Transformer实现
  ↓
Week 3-4: LLM深入
  ↓
Week 5+: RAG & Agent开发
```
**优势**: 
- 真正理解AI工作原理
- 能手撸Transformer (面试加分)
- 可以自定义模型和训练
- 国内大厂AI岗必备

#### 路径B: 快速应用 (适合赶时间或只做应用层的人)
```
直接跳到Week 5: RAG系统
  ↓
Week 6+: Agent开发
  ↓
有需要时再回来学PyTorch
```
**优势**: 
- 快速上手AI应用开发
- 先做产品，后学原理
- 适合全栈工程师快速转型

### 我的建议

**基于你的背景 (CS + 全栈 + 量化交易)**，强烈推荐 **路径A: 完整学习**

原因:
1. 你有CS基础，学PyTorch很快 (1周足够)
2. 量化交易中ML模型很常用，PyTorch是必备技能
3. 国内大厂面试必考"手撸Transformer"
4. 理解底层原理让你在AI应用开发中更有优势
5. 你的终极项目 (量化交易Agent) 可能需要自定义ML模型

---

## 🎯 学习路线图

```
Week 0: PyTorch基础速成 (1周) - 新增!
    ↓
Level 1: Transformer基础 (2-3周) - 需要PyTorch
    ↓
Level 2: LLM工作原理 (1-2周) - 需要PyTorch
    ↓
Level 3: Prompt Engineering & RAG (1周) - 不需要PyTorch
    ↓
Level 4: AI Agent架构 (2-3周) - 不需要PyTorch
    ↓
Level 5: 实战项目 (持续) - 可选使用PyTorch
```

**总时长**: 13周 (包含PyTorch Week 0)

**核心理念**: 
- 先打好PyTorch基础 (Week 0)
- 再深入Transformer原理 (Week 1-4)
- 最后构建AI应用 (Week 5-13)

---

## 🔥 Week 0: PyTorch基础速成

**如果你已经会PyTorch，跳过这周直接到Week 1**

### 学习目标
- 理解Tensor操作和自动微分
- 能用nn.Module定义神经网络
- 掌握基本的训练循环
- 为Transformer实现做准备

### Day 1-2: PyTorch核心概念

#### 资源1: PyTorch官方60分钟教程 (🔥 最重要)
- **链接**: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- **时长**: 2-3小时
- **学习重点**:
  - Tensor基础操作
  - Autograd自动微分
  - 神经网络nn.Module
  - 训练神经网络

**实践任务**:
```python
# Task 1: Tensor操作练习
import torch

# 创建tensor
x = torch.randn(3, 4)
y = torch.ones(4, 5)

# 矩阵乘法
z = torch.matmul(x, y)
print(z.shape)  # torch.Size([3, 5])

# 维度操作 (Transformer中超常用!)
a = torch.randn(2, 3, 4)  # [batch, seq_len, d_model]
b = a.transpose(1, 2)     # [batch, d_model, seq_len]
c = a.view(2, -1)         # [batch, seq_len * d_model]

# Task 2: 理解Autograd
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)  # dy/dx = 2*x + 3 = 7.0
```

#### 资源2: Andrej Karpathy - micrograd
- **链接**: https://github.com/karpathy/micrograd
- **视频**: https://www.youtube.com/watch?v=VMj-3S1tku0
- **时长**: 2小时
- **为什么重要**: 从零实现autograd，理解反向传播本质
- **学习重点**:
  - 计算图的构建
  - 反向传播算法
  - 梯度的链式法则

**实践任务**:
```python
# Clone micrograd并运行
git clone https://github.com/karpathy/micrograd.git
cd micrograd
python demo.py

# 理解Value类如何实现自动微分
# 尝试添加新的操作 (比如 exp, log)
```

### Day 3-4: 神经网络基础

#### 资源3: PyTorch for Deep Learning (Zero to Mastery)
- **链接**: https://www.learnpytorch.io/
- **章节**: 00-02章
- **为什么推荐**: 非常适合有编程基础的人，代码为主

**实践任务 1: 线性回归**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成数据
X = torch.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + torch.randn(100, 1) * 0.5

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# 训练
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []
for epoch in range(100):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 可视化
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 测试
with torch.no_grad():
    test_x = torch.tensor([[5.0]])
    prediction = model(test_x)
    print(f'Prediction for x=5: {prediction.item():.2f}')
```

**实践任务 2: MNIST手写数字分类**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')

# 4. 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# 5. 运行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 保存模型
torch.save(model.state_dict(), 'mnist_model.pth')
```

### Day 5-6: Transformer准备 - 重要的维度操作

**为什么重要**: Transformer中最难的就是处理 [batch, seq_len, d_model] 这样的3D tensor

#### 资源4: Understanding Tensor Dimensions
- **链接**: https://pytorch.org/tutorials/beginner/nn_tutorial.html
- **重点**: Broadcasting, view, transpose, reshape

**实践任务 3: 掌握Transformer中的tensor操作**
```python
import torch
import torch.nn as nn

# Transformer中的典型维度
batch_size = 2
seq_len = 5
d_model = 8
num_heads = 4

# 1. 模拟一个batch的输入序列
x = torch.randn(batch_size, seq_len, d_model)
print(f"Input shape: {x.shape}")  # [2, 5, 8]

# 2. Multi-head attention需要拆分head
d_k = d_model // num_heads  # 8 // 4 = 2
# Reshape: [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
x_split = x.view(batch_size, seq_len, num_heads, d_k)
# Transpose: [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
x_heads = x_split.transpose(1, 2)
print(f"Multi-head shape: {x_heads.shape}")  # [2, 4, 5, 2]

# 3. Attention的矩阵乘法
Q = x_heads  # [batch, num_heads, seq_len, d_k]
K = x_heads  # [batch, num_heads, seq_len, d_k]
V = x_heads  # [batch, num_heads, seq_len, d_k]

# Q @ K^T: [batch, num_heads, seq_len, d_k] @ [batch, num_heads, d_k, seq_len]
#        = [batch, num_heads, seq_len, seq_len]
scores = torch.matmul(Q, K.transpose(-2, -1))
print(f"Attention scores shape: {scores.shape}")  # [2, 4, 5, 5]

# 4. Softmax
attention_weights = torch.softmax(scores, dim=-1)
print(f"Attention weights shape: {attention_weights.shape}")  # [2, 4, 5, 5]

# 5. Attention @ V
output = torch.matmul(attention_weights, V)
print(f"Attention output shape: {output.shape}")  # [2, 4, 5, 2]

# 6. 合并heads
# [batch, num_heads, seq_len, d_k] -> [batch, seq_len, num_heads, d_k]
output = output.transpose(1, 2)
# [batch, seq_len, num_heads, d_k] -> [batch, seq_len, d_model]
output = output.contiguous().view(batch_size, seq_len, d_model)
print(f"Final output shape: {output.shape}")  # [2, 5, 8]
```

**练习**: 实现一个简化的scaled dot-product attention
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: [batch, num_heads, seq_len, d_k]
        K: [batch, num_heads, seq_len, d_k]
        V: [batch, num_heads, seq_len, d_k]
        mask: [batch, 1, 1, seq_len] or None
    
    Returns:
        output: [batch, num_heads, seq_len, d_k]
        attention_weights: [batch, num_heads, seq_len, seq_len]
    """
    d_k = Q.size(-1)
    
    # 1. 计算attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # 2. 应用mask (可选)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 3. Softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 4. 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# 测试
Q = torch.randn(2, 4, 5, 2)
K = torch.randn(2, 4, 5, 2)
V = torch.randn(2, 4, 5, 2)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # [2, 4, 5, 2]
print(f"Attention weights shape: {weights.shape}")  # [2, 4, 5, 5]
```

### Day 7: PyTorch进阶技巧

#### 资源5: PyTorch Performance Tips
- **链接**: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- **学习重点**:
  - GPU使用 (.to(device), .cuda())
  - Batch processing
  - DataLoader使用
  - 模型保存和加载

**实践任务 4: GPU加速**
```python
import torch
import time

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CPU vs GPU速度对比
size = 5000

# CPU
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)

start = time.time()
z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f}s")

# GPU (if available)
if torch.cuda.is_available():
    x_gpu = x_cpu.to(device)
    y_gpu = y_cpu.to(device)
    
    # Warm up
    _ = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
```

**实践任务 5: 模型保存和加载**
```python
# 保存整个模型
torch.save(model, 'entire_model.pth')

# 保存模型参数 (推荐)
torch.save(model.state_dict(), 'model_weights.pth')

# 加载模型
model = SimpleNN()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # 设置为评估模式

# 保存训练checkpoint (包含optimizer状态)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载checkpoint继续训练
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### Week 0总结检验

**你应该能够**:

- [ ] 创建和操作tensor，理解shape, view, transpose
- [ ] 解释autograd和backward()的工作原理
- [ ] 用nn.Module定义自己的神经网络
- [ ] 编写完整的训练循环 (forward, loss, backward, step)
- [ ] 处理[batch, seq_len, d_model]这样的3D tensor
- [ ] 实现简单的scaled dot-product attention
- [ ] 使用GPU加速
- [ ] 保存和加载模型

**验收项目**: 
完成MNIST分类，测试准确率达到95%以上，并能解释每一行代码的作用。

**如果完成以上任务，你已经准备好开始Week 1的Transformer实现了！**

### 额外学习资源 (可选)

#### 深入理解PyTorch
- **Dive into Deep Learning (D2L) PyTorch版**
  - 链接: https://d2l.ai/
  - 章节: Chapter 2-3
  - 适合: 想更系统学习的人

#### PyTorch内部机制
- **PyTorch Internals**
  - 链接: http://blog.ezyang.com/2019/05/pytorch-internals/
  - 适合: 想了解PyTorch如何工作的人

#### 视频教程
- **PyTorch Tutorials by sentdex**
  - 链接: https://www.youtube.com/playlist?list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh
  - 适合: 喜欢看视频学习的人

---

## 📚 Level 1: Transformer基础

**学习目标**: 理解Transformer架构，能手写核心组件

### 必看视频

#### 1. Andrej Karpathy - Let's build GPT (🔥 最重要)
- **链接**: https://www.youtube.com/watch?v=kCc8FmEb1nY
- **时长**: 2小时
- **为什么重要**: 从零实现GPT，讲解最清晰
- **学习重点**: 
  - Self-attention机制
  - Positional encoding
  - Multi-head attention
  - Layer normalization

#### 2. 3Blue1Brown - Attention机制可视化
- **链接**: https://www.youtube.com/watch?v=eMlx5fFNoYc
- **时长**: 30分钟
- **为什么重要**: 视觉化理解attention
- **学习重点**: 
  - Query, Key, Value的含义
  - Attention权重计算
  - 为什么叫"attention"

#### 3. StatQuest - Transformer详解
- **链接**: https://www.youtube.com/watch?v=zxQyTK8quyY
- **时长**: 45分钟
- **为什么重要**: 数学原理讲得很好
- **学习重点**: 
  - Scaled dot-product attention
  - Softmax的作用
  - 残差连接

### 必读教程

#### 4. The Illustrated Transformer (🔥 新手必读)
- **链接**: http://jalammar.github.io/illustrated-transformer/
- **为什么重要**: 图解版，理解最直观
- **学习重点**: 
  - Encoder-Decoder架构
  - 每一层的输入输出
  - Transformer全流程

#### 5. Annotated Transformer (Harvard NLP)
- **链接**: https://nlp.seas.harvard.edu/annotated-transformer/
- **为什么重要**: 带注释的完整代码实现
- **学习重点**: 
  - PyTorch实现细节
  - 训练循环
  - Batch处理

#### 6. 原始论文 (可选)
- **链接**: https://arxiv.org/abs/1706.03762
- **标题**: "Attention is All You Need"
- **建议**: 先看上面的教程，再回来看论文

### 实践项目

#### 7. nanoGPT (🔥 最重要的实践)
- **链接**: https://github.com/karpathy/nanoGPT
- **为什么重要**: 最简化的GPT实现
- **任务**: 
  - [ ] Clone仓库并运行
  - [ ] 理解每一行代码
  - [ ] 在小数据集上训练
  - [ ] 修改模型参数观察效果

#### 8. minGPT
- **链接**: https://github.com/karpathy/minGPT
- **为什么重要**: 教学版本，注释详细
- **任务**: 
  - [ ] 对比nanoGPT和minGPT的区别
  - [ ] 手写attention layer
  - [ ] 实现自己的mini-transformer

### 检验标准
- [ ] 能用PyTorch从零实现scaled dot-product attention
- [ ] 理解multi-head attention的作用
- [ ] 能解释positional encoding为什么必要
- [ ] 能画出Transformer的架构图

---

## 📚 Level 2: LLM工作原理

**学习目标**: 理解大模型如何训练、推理、对齐

### 理解大模型训练

#### 9. Stanford CS324 - LLM课程
- **链接**: https://stanford-cs324.github.io/winter2022/
- **为什么重要**: 完整的LLM理论课程
- **学习重点**: 
  - Pre-training vs Fine-tuning
  - Model scaling laws
  - Emergent abilities
  - Inference optimization

#### 10. Andrej Karpathy - State of GPT
- **链接**: https://www.youtube.com/watch?v=bZQun8Y4L2A
- **时长**: 1小时
- **为什么重要**: GPT的训练全流程
- **学习重点**: 
  - Pre-training阶段
  - Supervised fine-tuning
  - RLHF过程
  - 数据质量的重要性

#### 11. LLM可视化 (🔥 必玩)
- **链接**: https://bbycroft.net/llm
- **为什么重要**: 交互式看GPT如何生成文字
- **任务**: 
  - [ ] 输入不同prompt观察token生成
  - [ ] 理解temperature参数的影响
  - [ ] 看attention pattern

### 关键概念

#### 12. Understanding RLHF
- **链接**: https://huggingface.co/blog/rlhf
- **为什么重要**: 理解ChatGPT如何对齐人类偏好
- **学习重点**: 
  - Reward model训练
  - PPO算法
  - 为什么需要RLHF

#### 13. Tokenization详解
- **链接**: https://www.youtube.com/watch?v=zduSFxRajkE
- **为什么重要**: Karpathy讲tokenizer
- **学习重点**: 
  - BPE算法
  - Token vs Character
  - Tokenization对模型的影响

### 检验标准
- [ ] 理解pre-training和fine-tuning的区别
- [ ] 能解释RLHF的工作原理
- [ ] 理解temperature、top-p等采样参数
- [ ] 知道tokenization如何影响模型性能

---

## 📚 Level 3: Prompt Engineering & RAG

**学习目标**: 掌握高效使用LLM的方法，实现RAG系统

### Prompt Engineering

#### 14. OpenAI Prompt Engineering Guide
- **链接**: https://platform.openai.com/docs/guides/prompt-engineering
- **为什么重要**: 官方最佳实践
- **学习重点**: 
  - Few-shot learning
  - Chain-of-thought prompting
  - System messages设计
  - 如何减少hallucination

#### 15. Anthropic Prompt Engineering
- **链接**: https://docs.anthropic.com/claude/docs/prompt-engineering
- **为什么重要**: Claude的prompting技巧
- **学习重点**: 
  - XML tags使用
  - Long context处理
  - Role prompting
  - Citation patterns

#### 16. Learn Prompting (免费课程)
- **链接**: https://learnprompting.org/
- **为什么重要**: 系统化学习
- **任务**: 
  - [ ] 完成基础课程
  - [ ] 练习各种prompting技巧
  - [ ] 对比不同方法的效果

### RAG (检索增强生成)

#### 17. LangChain RAG Tutorial
- **链接**: https://python.langchain.com/docs/tutorials/rag/
- **为什么重要**: 实现你自己的RAG系统
- **学习重点**: 
  - Document loading
  - Text splitting策略
  - Embedding选择
  - Retrieval methods

#### 18. Pinecone Learning Center
- **链接**: https://www.pinecone.io/learn/retrieval-augmented-generation/
- **为什么重要**: RAG理论+实践
- **学习重点**: 
  - Vector database原理
  - Semantic search
  - Hybrid search
  - Re-ranking strategies

#### 19. RAG论文解读
- **链接**: https://arxiv.org/abs/2005.11401
- **标题**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **建议**: 理解实践后再看论文

### 实践项目
**任务**: 构建一个"你的股票研究笔记"RAG chatbot
- [ ] 收集你的量化交易笔记作为知识库
- [ ] 实现文档embedding和存储
- [ ] 构建检索+生成pipeline
- [ ] 测试不同retrieval策略的效果

### 检验标准
- [ ] 理解few-shot vs zero-shot prompting
- [ ] 能设计有效的system prompt
- [ ] 理解RAG的工作流程
- [ ] 能实现一个完整的RAG系统

---

## 📚 Level 4: AI Agent架构

**学习目标**: 理解Agent设计模式，构建自主决策系统

### Agent设计模式

#### 20. DeepLearning.AI - Agentic Design Patterns (你已经在看的)
- **链接**: https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/
- **为什么重要**: Andrew Ng的课程
- **学习重点**: 
  - Reflection pattern
  - Tool use pattern
  - Planning pattern
  - Multi-agent collaboration

#### 21. LangChain Agents文档
- **链接**: https://python.langchain.com/docs/concepts/agents/
- **为什么重要**: Agent实现框架
- **学习重点**: 
  - Agent types (ReAct, Function calling)
  - Tool integration
  - Agent executor
  - Streaming & callbacks

#### 22. ReAct论文 (🔥 核心论文)
- **链接**: https://arxiv.org/abs/2210.03629
- **标题**: "ReAct: Synergizing Reasoning and Acting in Language Models"
- **为什么重要**: Reasoning + Acting范式
- **学习重点**: 
  - Thought-Action-Observation循环
  - 为什么需要reasoning
  - 如何设计action space

### Agent框架实战

#### 23. AutoGPT源码
- **链接**: https://github.com/Significant-Gravitas/AutoGPT
- **为什么重要**: 研究真实的agent实现
- **任务**: 
  - [ ] Clone并运行AutoGPT
  - [ ] 理解其agent loop
  - [ ] 分析tool调用机制
  - [ ] 看它如何管理memory

#### 24. LangGraph (🔥 最好的Agent工具)
- **链接**: https://langchain-ai.github.io/langgraph/
- **为什么重要**: 状态机式的agent框架
- **学习重点**: 
  - Graph-based agent design
  - State management
  - Conditional edges
  - Human-in-the-loop

#### 25. Anthropic Computer Use
- **链接**: https://docs.anthropic.com/en/docs/build-with-claude/computer-use
- **为什么重要**: Claude控制电脑的agent实现
- **学习重点**: 
  - Vision + Action结合
  - Tool calling实现
  - Error handling
  - Safety considerations

### Agent核心概念

#### 26. Tool Calling详解
- **链接**: https://platform.openai.com/docs/guides/function-calling
- **为什么重要**: Agent如何调用工具
- **学习重点**: 
  - Function schema设计
  - Tool selection策略
  - Error handling
  - Parallel tool calling

#### 27. Memory Management
- **链接**: https://python.langchain.com/docs/how_to/#memory
- **为什么重要**: Agent如何记忆对话
- **学习重点**: 
  - Short-term vs long-term memory
  - Conversation buffer
  - Summary memory
  - Vector store memory

#### 28. Multi-Agent系统
- **链接**: https://microsoft.github.io/autogen/
- **为什么重要**: 微软的多agent框架
- **学习重点**: 
  - Agent communication protocols
  - Task delegation
  - Consensus mechanisms
  - Multi-agent orchestration

### 检验标准
- [ ] 理解ReAct agent的工作流程
- [ ] 能用LangGraph构建有状态的agent
- [ ] 理解tool calling的实现原理
- [ ] 能设计multi-agent系统架构

---

## 📚 Level 5: 实战项目

**学习目标**: 构建端到端的AI应用

### 从简单到复杂

#### 29. 构建一个RAG chatbot
- **链接**: https://github.com/langchain-ai/rag-from-scratch
- **为什么重要**: 完整的RAG项目
- **任务**: 
  - [ ] 实现document ingestion pipeline
  - [ ] 构建web界面 (用你的React技能)
  - [ ] 添加conversation memory
  - [ ] 部署到production

#### 30. Build a Research Assistant Agent
- **链接**: https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/
- **为什么重要**: DeepLearning.AI课程
- **任务**: 
  - [ ] 完成课程项目
  - [ ] 扩展为多功能research agent
  - [ ] 添加web search capability
  - [ ] 实现citation tracking

#### 31. Multi-Agent Trading System (🔥 结合量化交易!)
- **链接**: https://github.com/langchain-ai/langchain/blob/master/cookbook/multi_agent_trading_system.ipynb
- **为什么重要**: 用agents做股票分析
- **项目规划**: 见下方"终极项目"部分

### 终极项目: 量化交易Agent系统

**项目结构**:
```
Trading Agent System
├── Research Agent
│   ├── 搜索财报
│   ├── 爬取新闻
│   └── 社交媒体情绪分析
├── Analysis Agent
│   ├── 技术分析 (你学的TA知识)
│   ├── 基本面分析
│   └── 量化指标计算
├── Strategy Agent
│   ├── 生成交易策略
│   ├── Backtesting
│   └── 参数优化
└── Risk Management Agent
    ├── 仓位管理
    ├── 止损策略
    └── 风险评估
```

**技术栈**:
- Frontend: React + TypeScript (你已有的技能)
- Backend: Python + FastAPI
- Agents: LangGraph
- Database: PostgreSQL + Pinecone
- Data: yfinance, pandas, numpy

**项目阶段**:
1. **Week 1-2**: 单一Research Agent
2. **Week 3-4**: 添加Analysis Agent
3. **Week 5-6**: 构建Strategy Agent
4. **Week 7-8**: 集成Risk Management
5. **Week 9-10**: Web界面开发
6. **Week 11-12**: 优化和部署

**预期成果**:
- 一个可以自动研究股票的AI系统
- 结合你的量化交易知识
- 完整的portfolio项目
- 面试时的强大亮点

---

## 🗓️ 13周详细学习计划

### Week 0: PyTorch基础速成 (新增)

**本周目标**: 掌握PyTorch基础，为Transformer实现做准备

**学习任务**:
- [ ] Day 1-2: PyTorch 60分钟教程 + micrograd
- [ ] Day 3-4: 线性回归 + MNIST分类项目
- [ ] Day 5-6: 掌握Transformer中的tensor维度操作
- [ ] Day 7: GPU使用和模型保存/加载

**实践项目**:
- 完成MNIST手写数字分类 (准确率>95%)
- 实现简化版的scaled dot-product attention
- 能够熟练处理 [batch, seq_len, d_model] 维度

**检验标准**:
```python
# 你需要能轻松写出这样的代码
class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        return output
```

---

### Week 1-2: Transformer基础

**本周目标**: 理解并能手写Transformer核心组件

**学习任务**:
- [ ] Day 1-2: 看Karpathy的GPT视频 (2小时)
- [ ] Day 3-4: 读The Illustrated Transformer，做笔记
- [ ] Day 5-6: Clone nanoGPT，逐行理解代码
- [ ] Day 7-8: 手写attention layer
- [ ] Day 9-10: 在toy dataset上训练mini-GPT
- [ ] Day 11-14: 完成Annotated Transformer教程

**实践项目**:
```python
# 你需要能写出这样的代码
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, V), attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # [batch, seq_len, d_model]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now: [batch, num_heads, seq_len, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.d_k)
        
        # Final linear
        return self.W_o(output)
```

**检验标准**:
- 能在白板上画出Transformer架构
- 能解释attention权重的计算
- 能运行并修改nanoGPT
- 能手写multi-head attention

---

### Week 3-4: 深入LLM

**本周目标**: 理解大模型训练和推理全流程

**学习任务**:
- [ ] Day 1-3: Stanford CS324前5讲
- [ ] Day 4-5: 看State of GPT视频
- [ ] Day 6-7: 理解tokenization (看Karpathy视频)
- [ ] Day 8-9: 玩LLM可视化工具，观察token生成
- [ ] Day 10-11: 学习RLHF原理
- [ ] Day 12-14: 用HuggingFace fine-tune一个小模型

**实践项目**:
- Fine-tune一个GPT-2 small在你的domain数据上
- 对比不同训练策略的效果
- 实验不同的采样参数

**检验标准**:
- 理解pre-training vs fine-tuning区别
- 能解释RLHF如何工作
- 知道temperature如何影响生成

---

### Week 5: RAG系统

**本周目标**: 构建一个完整的RAG应用

**学习任务**:
- [ ] Day 1-2: LangChain RAG tutorial
- [ ] Day 3-4: 学习vector database (Pinecone/Chroma)
- [ ] Day 5-7: 实践项目 (见下方)

**实践项目**: **股票研究笔记RAG Chatbot**
```
功能:
1. 上传你的量化交易笔记 (PDF/Markdown)
2. 自动chunking和embedding
3. 自然语言查询你的笔记
4. 显示citation和来源

技术:
- Document loader: LangChain
- Embedding: OpenAI embeddings
- Vector store: Chroma (本地免费)
- LLM: GPT-3.5 或 Claude
- Frontend: 简单的Streamlit界面
```

**检验标准**:
- RAG系统能正确检索相关文档
- 回答准确且有citation
- 理解不同chunking策略的影响

---

### Week 6-7: Agent基础

**本周目标**: 理解agent设计模式，实现ReAct agent

**学习任务**:
- [ ] Day 1-3: 完成DeepLearning.AI agentic patterns课程
- [ ] Day 4-5: 读ReAct论文，理解reasoning过程
- [ ] Day 6-8: 学习tool calling机制
- [ ] Day 9-10: LangChain agents文档
- [ ] Day 11-14: 实践项目 (见下方)

**实践项目**: **简单的ReAct Agent**
```python
# 实现一个能做数学计算的agent
tools = [
    Calculator(),      # 基础计算
    WebSearch(),       # 搜索信息
    PythonREPL()       # 执行Python代码
]

# Agent能回答:
# "2023年特斯拉股价涨幅是多少?"
# 1. 搜索特斯拉2023股价数据
# 2. 用计算器算涨幅
# 3. 返回答案
```

**检验标准**:
- Agent能正确选择和使用工具
- 理解thought-action-observation循环
- 能处理multi-step reasoning

---

### Week 8-9: Agent框架深入

**本周目标**: 掌握LangGraph，构建有状态的agent

**学习任务**:
- [ ] Day 1-4: LangGraph tutorials
- [ ] Day 5-7: 研究AutoGPT源码
- [ ] Day 8-10: 学习multi-agent通信
- [ ] Day 11-14: 实践项目 (见下方)

**实践项目**: **有状态的对话Agent**
```
功能:
1. 记住对话历史
2. 多轮规划和执行
3. 处理用户反馈
4. 错误重试机制

示例场景:
User: "帮我分析一下NVDA的投资价值"
Agent: 
- State 1: 搜索NVDA基本信息
- State 2: 获取财务数据
- State 3: 进行技术分析
- State 4: 生成综合报告
- (每个state可以根据结果调整)
```

**检验标准**:
- 能用LangGraph构建complex workflow
- 理解state management
- 能实现human-in-the-loop

---

### Week 10-12: 综合项目

**终极项目**: **Multi-Agent量化交易系统**

**Phase 1 (Week 10): Research Agent**
```python
class ResearchAgent:
    """负责收集和整理信息"""
    tools = [
        SECFilingsTool(),      # 财报数据
        NewsTool(),            # 新闻搜索
        SocialSentimentTool(), # Reddit/Twitter情绪
    ]
    
    def research_stock(self, ticker: str):
        # 收集所有相关信息
        pass
```

**Phase 2 (Week 11): Analysis & Strategy Agents**
```python
class AnalysisAgent:
    """技术分析和基本面分析"""
    def analyze(self, stock_data, research_data):
        technical = self.technical_analysis(stock_data)
        fundamental = self.fundamental_analysis(research_data)
        return combined_analysis

class StrategyAgent:
    """生成交易策略"""
    def generate_strategy(self, analysis):
        # 基于分析生成具体策略
        pass
```

**Phase 3 (Week 12): 集成和UI**
```typescript
// React前端 (用你的技能!)
const TradingDashboard = () => {
  return (
    <div>
      <StockSearchBar />
      <AgentStatus />  {/* 显示各agent状态 */}
      <ResearchPanel /> {/* Research Agent输出 */}
      <AnalysisPanel /> {/* Analysis Agent输出 */}
      <StrategyPanel /> {/* Strategy建议 */}
      <RiskMetrics />  {/* 风险指标 */}
    </div>
  );
};
```

**最终交付**:
- [ ] 完整的multi-agent系统
- [ ] Web界面
- [ ] 文档和demo视频
- [ ] GitHub repo (作为portfolio)

---

## 🔗 额外资源

### 保持更新

#### 32. Papers with Code - Transformers
- **链接**: https://paperswithcode.com/methods/category/transformers
- **用途**: 最新研究进展
- **建议**: 每周浏览一次

#### 33. Hugging Face Course
- **链接**: https://huggingface.co/learn/nlp-course/
- **用途**: NLP和Transformers完整课程
- **建议**: 作为补充学习材料

#### 34. AI Agent论坛
- **链接**: https://www.reddit.com/r/LangChain/
- **用途**: 社区讨论和问题解答
- **建议**: 遇到问题时查找或提问

### 推荐书籍

1. **"Deep Learning" by Goodfellow et al.**
   - 深度学习圣经
   - https://www.deeplearningbook.org/

2. **"Speech and Language Processing" by Jurafsky**
   - NLP基础
   - https://web.stanford.edu/~jurafsky/slp3/

3. **"Designing Data-Intensive Applications"**
   - 构建production AI系统必读
   - 理解scalability和reliability

4. **"Programming PyTorch for Deep Learning" by Ian Pointer** (新增)
   - PyTorch实战指南
   - 适合快速上手

### YouTube频道

1. **Andrej Karpathy**
   - https://www.youtube.com/@AndrejKarpathy
   - 从零构建GPT系列

2. **StatQuest with Josh Starmer**
   - https://www.youtube.com/@statquest
   - 机器学习概念可视化

3. **Two Minute Papers**
   - https://www.youtube.com/@TwoMinutePapers
   - 快速了解最新AI研究

4. **sentdex** (新增)
   - https://www.youtube.com/@sentdex
   - PyTorch和深度学习教程

### PyTorch专题资源 (新增)

#### 官方资源
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **PyTorch Examples**: https://github.com/pytorch/examples

#### 社区资源
- **PyTorch Forums**: https://discuss.pytorch.org/
- **PyTorch Lightning Docs**: https://lightning.ai/docs/pytorch/stable/
- **Hugging Face Course**: https://huggingface.co/learn/nlp-course/chapter0/1

#### 高级主题
- **Mixed Precision Training**: https://pytorch.org/docs/stable/amp.html
- **Distributed Training**: https://pytorch.org/tutorials/beginner/dist_overview.html
- **Model Optimization**: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

### 学习社区

1. **Reddit**
   - r/MachineLearning: https://www.reddit.com/r/MachineLearning/
   - r/PyTorch: https://www.reddit.com/r/PyTorch/
   - r/LangChain: https://www.reddit.com/r/LangChain/

2. **Discord Servers**
   - PyTorch Discord: https://discord.gg/pytorch
   - Hugging Face Discord: https://discord.gg/JfAtkvEtRb

3. **Twitter/X 关注**
   - @karpathy (Andrej Karpathy)
   - @PyTorch (PyTorch官方)
   - @huggingface (Hugging Face)
   - @AnthropicAI (Anthropic/Claude)

---

## 💡 学习建议

### 原则

1. **不要跳步**: 每个level都要扎实掌握再前进
   - Week 0 (PyTorch) 是基础，一定要扎实
   - 如果Week 0的检验项目做不出来，不要进入Week 1

2. **代码为主**: 80%时间写代码，20%看理论
   - 看视频时一定要跟着敲代码
   - 不要只收藏资源不实践

3. **小步快跑**: 每周一个可运行的小项目
   - Week 0: MNIST分类器
   - Week 1-2: Mini transformer
   - Week 3-4: Fine-tuned模型
   - Week 5: RAG chatbot
   - 每周都要有能demo的东西

4. **结合兴趣**: 量化交易是你的优势，充分利用
   - 用股票数据做训练
   - RAG系统用你的交易笔记
   - 最终项目是量化交易agent

### 时间分配

**每天3-4小时** (可调整):
- 1小时: 看视频/读文章
- 2小时: 写代码/做项目
- 0.5小时: 笔记和总结
- 0.5小时: 和我讨论问题

**每周末**:
- 2小时: 复习本周内容
- 2小时: 完成周项目
- 1小时: 规划下周学习
- 写一篇总结blog (可选但推荐)

**如果时间有限** (比如每天只有2小时):
- 延长计划到20周
- 或者跳过Week 0，从Level 3开始 (RAG和Agent)
- 先做应用，后学原理

### 学习技巧

1. **费曼学习法**: 
   - 每周写一篇blog解释学到的概念
   - 假装你在教别人
   - 如果你解释不清楚，说明还没真正理解
   - 推荐平台: Medium, Dev.to, 或个人GitHub Pages

2. **Project-based Learning**:
   - 不要只看tutorial
   - 每个概念都要有对应的代码实践
   - 改进教程中的代码，添加自己的feature
   - 犯错是学习最快的方式

3. **记录过程**:
   - **GitHub**: 记录所有代码
     - 创建一个 "learning-ai" repo
     - 每周一个文件夹: week-0-pytorch, week-1-transformer等
     - 写好README说明每个项目
   - **Notion/Obsidian**: 记录笔记
     - 概念解释
     - 遇到的问题和解决方案
     - 资源链接整理
   - **为面试做准备**: 这些都是你的portfolio

4. **主动学习** (🔥 重要):
   ```
   被动学习 (效率低):
   看视频 → 点点头 → 关掉 → 忘记
   
   主动学习 (效率高):
   看视频 → 暂停 → 自己实现 → 遇到bug → 调试 → 理解
   ```

5. **间隔重复**:
   - 学完一个概念后
   - 第2天: 回顾
   - 第7天: 复习
   - 第30天: 再次复习
   - 用Anki或Notion制作flashcards

### 避免的坑

❌ **只看不练** - 最大的坑
- 收藏了100个教程但一个都没做完
- 解决方法: 立刻实践，看一个做一个

❌ **追求完美主义**
- 在Week 0卡一个月想把PyTorch学透
- 解决方法: 够用就行，边用边学

❌ **跳着学，基础不扎实**
- PyTorch不会就去学Transformer
- 解决方法: 严格按Week 0 → Week 1顺序

❌ **不做笔记，学了就忘**
- 3个月后完全想不起来学过什么
- 解决方法: 每天写学习日志

❌ **孤军奋战**
- 遇到问题不问，自己死磕几天
- 解决方法: 及时问我，或上论坛/Discord

❌ **只学不用**
- 学了一堆理论，不知道怎么应用
- 解决方法: 从Week 1开始就思考实际应用场景

✅ **正确做法**:
- 快速迭代，边学边做
- 每周一个可demo的项目
- 主动分享和讨论
- 遇到困难及时求助
- 记录学习过程
- 结合实际需求学习

### 调试技巧 (PyTorch特定)

```python
# 1. 检查tensor shape (最常见的bug)
print(f"x.shape: {x.shape}")
assert x.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {x.shape}"

# 2. 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
    else:
        print(f"{name}: NO GRADIENT!")

# 3. 检查是否有NaN
assert not torch.isnan(loss).any(), "Loss contains NaN!"

# 4. 可视化attention weights
import matplotlib.pyplot as plt
import seaborn as sns

attention = attention_weights[0, 0].detach().cpu().numpy()  # [seq_len, seq_len]
sns.heatmap(attention, cmap='viridis')
plt.show()

# 5. 使用PyTorch的debugging工具
torch.autograd.set_detect_anomaly(True)  # 检测backward中的问题
```

### 求助渠道

**遇到问题时**:

1. **先Google**: "pytorch [your error message]"
2. **查官方文档**: https://pytorch.org/docs/
3. **搜索Stack Overflow**: 90%的问题已经有答案
4. **问我**: 随时联系
5. **论坛**: PyTorch Discuss, Reddit
6. **GitHub Issues**: 如果是库的bug

**提问的正确方式**:
```
❌ 不好的提问:
"我的代码不工作，怎么办?"

✅ 好的提问:
"我在实现multi-head attention时遇到shape不匹配的错误:
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x5 and 8x8)

我的代码:
[附上minimal reproducible example]

我的理解是...
我尝试了...
但是...

请问是哪里出了问题?"
```

---

## 📊 技能树进阶路径

```
现有技能                 →  AI技能
─────────────────────────────────────────
React/TypeScript        →  构建Agent UI
Python (量化交易)       →  实现Agent逻辑
全栈开发经验            →  End-to-end Agent系统
API集成                →  Tool calling设计
数据分析                →  Model evaluation
系统设计思维            →  Multi-agent架构
```

### 独特优势

你的背景组合非常稀缺:
- ✅ 工程能力强 (全栈开发)
- ✅ 有domain knowledge (量化交易)
- ✅ 数学基础好 (CS背景)
- ✅ 实践经验丰富 (实习项目)

这让你在AI应用开发上有巨大优势，尤其是:
- **FinTech领域**: AI + 量化交易
- **AI工具开发**: 懂用户需求的AI engineer
- **创业方向**: AI-powered trading tools

---

## 🎯 职业发展路径

### 短期 (3-6个月)

**目标**: 掌握AI Agent开发
- 完成本学习计划
- 构建2-3个portfolio项目
- 在GitHub积累代码

**面试准备**:
- Transformer原理 (手写代码)
- RAG系统设计
- Agent架构讨论
- 实际项目经验

### 中期 (6-12个月)

**目标**: 成为AI应用专家
- 深入某个垂直领域 (推荐FinTech)
- 贡献开源项目 (LangChain, LangGraph等)
- 写技术博客
- 参加AI hackathons

**潜在公司**:
- 量化私募 (Two Sigma, Citadel)
- FinTech (Stripe, Plaid, Robinhood)
- AI Infra (Anthropic, OpenAI, Scale AI)
- 传统科技大厂的AI team

### 长期 (1-2年+)

**可能方向**:

1. **AI Research Engineer**
   - 改进model architecture
   - 优化training/inference
   - 发论文

2. **AI Product Engineer**
   - 构建AI-powered products
   - 用户体验优化
   - Product-market fit

3. **创业**
   - AI trading tools
   - Developer tools for AI
   - Vertical AI agents

---

## 📝 检查清单

### Level 1 完成标准
- [ ] 能手写scaled dot-product attention
- [ ] 理解multi-head attention原理
- [ ] 解释positional encoding作用
- [ ] 训练过至少一个toy transformer
- [ ] 能画出完整的transformer架构图

### Level 2 完成标准
- [ ] 理解pre-training vs fine-tuning
- [ ] 解释RLHF工作流程
- [ ] 知道tokenization如何影响性能
- [ ] Fine-tuned过至少一个模型
- [ ] 理解inference optimization技术

### Level 3 完成标准
- [ ] 掌握few-shot prompting
- [ ] 能设计有效的system prompt
- [ ] 实现过完整的RAG系统
- [ ] 理解vector database原理
- [ ] 对比过不同retrieval策略

### Level 4 完成标准
- [ ] 理解ReAct agent工作流
- [ ] 用LangGraph构建过agent
- [ ] 实现过tool calling
- [ ] 理解multi-agent通信
- [ ] 设计过agent架构

### Level 5 完成标准
- [ ] 完成量化交易agent项目
- [ ] 有完整的GitHub portfolio
- [ ] 写过技术文档和blog
- [ ] 能demo你的项目
- [ ] 准备好面试讲解

---

## 🚀 下一步行动

### 今天就开始 (Day 1 行动清单)

#### 第一步: 环境搭建 (30分钟)

```bash
# 1. 创建学习文件夹
mkdir -p ~/learning-ai
cd ~/learning-ai

# 2. 创建Python虚拟环境
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# 或 venv\Scripts\activate  # Windows

# 3. 安装PyTorch (根据你的系统选择)
# Mac (Apple Silicon):
pip3 install torch torchvision torchaudio

# Windows/Linux (CUDA):
# 访问 https://pytorch.org/get-started/locally/ 选择合适的版本

# 验证安装
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. 安装其他依赖
pip install numpy matplotlib jupyter notebook

# 5. 创建GitHub repo
git init
echo "# My AI Learning Journey" > README.md
echo "venv/" > .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
git add .
git commit -m "Initial commit"

# 在GitHub上创建repo并push
git remote add origin [your-repo-url]
git push -u origin main
```

#### 第二步: Week 0 Day 1任务 (2小时)

**任务1: PyTorch 60分钟教程** (1小时)
```bash
# 创建Day 1文件夹
mkdir week-0-pytorch/day-1
cd week-0-pytorch/day-1

# 创建Jupyter notebook
jupyter notebook
# 或者
code pytorch_basics.py  # 如果用VSCode
```

**在notebook/py文件里完成**:
```python
# File: pytorch_basics.py
import torch
import numpy as np

print("=" * 50)
print("Day 1: PyTorch Basics")
print("=" * 50)

# Task 1: Create tensors
print("\n1. Creating Tensors")
x = torch.tensor([1, 2, 3])
print(f"1D tensor: {x}")

y = torch.randn(3, 4)
print(f"Random 2D tensor:\n{y}")

z = torch.zeros(2, 3)
print(f"Zeros tensor:\n{z}")

# Task 2: Tensor operations
print("\n2. Tensor Operations")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a @ b = {torch.dot(a, b)}")  # dot product

# Task 3: Reshaping
print("\n3. Reshaping")
x = torch.randn(2, 3, 4)
print(f"Original shape: {x.shape}")
print(f"View as (2, 12): {x.view(2, 12).shape}")
print(f"View as (6, 4): {x.view(6, 4).shape}")
print(f"Transpose: {x.transpose(1, 2).shape}")

# Task 4: Autograd
print("\n4. Autograd")
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
print(f"y = x^2 + 3x, where x = 2")
print(f"y = {y.item()}")

y.backward()
print(f"dy/dx = 2x + 3 = {x.grad.item()}")

# Task 5: GPU check
print("\n5. GPU Availability")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print("\n✅ Day 1 completed!")
```

**运行并提交**:
```bash
python pytorch_basics.py

# 提交到GitHub
git add .
git commit -m "Week 0 Day 1: PyTorch basics"
git push
```

**任务2: 开始看视频** (1小时)
- 打开: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- 看前30分钟: "What is PyTorch?" 和 "Tensors"
- 边看边在notebook里敲代码

#### 第三步: 创建学习日志 (30分钟)

在Notion/Obsidian/Markdown创建学习日志:

```markdown
# AI Learning Journey

## Week 0: PyTorch Basics

### Day 1: 2024-12-18
**Time spent**: 2.5 hours
**What I learned**:
- PyTorch tensor basics
- Autograd mechanism
- Shape manipulation (view, transpose)

**Completed**:
- [x] Environment setup
- [x] PyTorch installation
- [x] Basic tensor operations
- [x] Autograd example

**Challenges**:
- Understanding broadcasting rules
- Remembering when to use .view() vs .reshape()

**Tomorrow's plan**:
- Complete PyTorch 60-min tutorial
- Start micrograd video

**Code**: [GitHub link]
```

### 本周目标 (Week 0完整计划)

**Day 2**: 
- [ ] 完成PyTorch 60分钟教程
- [ ] 开始看micrograd视频
- [ ] 实现简单的autograd example

**Day 3**:
- [ ] 完成micrograd视频
- [ ] 实现线性回归项目

**Day 4**:
- [ ] 开始MNIST项目
- [ ] 理解DataLoader的使用

**Day 5**:
- [ ] 完成MNIST分类器
- [ ] 学习Transformer中的tensor操作

**Day 6**:
- [ ] 实现简化的attention机制
- [ ] 复习本周内容

**Day 7**:
- [ ] GPU训练实践
- [ ] 模型保存和加载
- [ ] Week 0总结和复习

### 持续跟踪

**每天**:
1. 更新学习日志
2. 提交代码到GitHub
3. 如果卡住超过1小时，立刻问我

**每周日**:
1. 写本周总结
2. 完成周验收项目
3. 规划下周学习

**和我的互动**:
- 每天分享你的进度
- 遇到问题立刻问
- 想讨论概念随时找我
- 完成项目后给我看demo

### 获取帮助的方式

1. **卡住了?** 
   - 先Google 10分钟
   - 还是不行就问我
   - 提供: 错误信息 + 代码 + 你的理解

2. **需要代码review?** 
   - 把GitHub链接发给我
   - 我会提供反馈和改进建议

3. **想讨论项目?** 
   - 随时brainstorm
   - 我帮你规划实现步骤

4. **面试准备?** 
   - Week 4后可以开始mock interview
   - 我模拟面试官问你Transformer原理

### 激励和里程碑

**Week 0结束**: 🎉
- 掌握PyTorch基础
- 完成MNIST分类器
- 解锁 "PyTorch Developer" 成就

**Week 2结束**: 🚀
- 实现mini Transformer
- 解锁 "Transformer Implementor" 成就
- 可以去面试讲hand-coded attention

**Week 4结束**: 💪
- Fine-tuned自己的第一个LLM
- 解锁 "LLM Engineer" 成就
- 开始做AI应用开发

**Week 13结束**: 🏆
- 完成量化交易Agent系统
- 有完整的GitHub portfolio
- 解锁 "AI Agent Master" 成就
- 准备好去面试了！

---

## 📚 最后的话

这是一条从**底层原理到实战应用**的完整路径。13周后，你将:

✅ **技术能力**:
- 理解Transformer如何工作
- 能手写核心组件 (面试加分)
- 掌握LLM的训练和使用
- 构建production-ready的AI agents
- PyTorch熟练使用

✅ **项目经验**:
- 完整的portfolio项目
- GitHub上有真实代码
- 可demo的AI应用
- 技术blog文章

✅ **职业优势**:
- 在AI+FinTech领域建立独特优势
- 既懂底层原理又能做应用
- 结合量化交易domain knowledge
- 国内大厂AI岗的完整准备

**记住**: 你的背景（CS + 全栈 + 量化交易）是巨大的优势。很少有人同时具备engineering能力和domain knowledge。充分利用这个优势，构建有实际价值的AI应用。

**最重要的**: 
- 🚀 **开始比完美更重要**
- 💪 **坚持比聪明更重要**
- 🎯 **实践比理论更重要**

### 今天就开始第一步！

现在立刻执行Day 1的环境搭建，30分钟后你就可以写第一行PyTorch代码了！

有任何问题随时问我，我会一路陪你学习！

Good luck! 🚀

---

## 📝 附录: 快速参考

### PyTorch常用操作速查

```python
# Tensor创建
torch.tensor([1, 2, 3])
torch.randn(3, 4)
torch.zeros(2, 3)
torch.ones(2, 3)
torch.arange(0, 10, 2)

# Shape操作
x.shape / x.size()
x.view(new_shape)
x.reshape(new_shape)
x.transpose(dim0, dim1)
x.unsqueeze(dim)
x.squeeze(dim)

# 数学操作
x + y, x * y, x @ y
x.sum(), x.mean(), x.max()
torch.matmul(x, y)
torch.softmax(x, dim=-1)

# Autograd
x.requires_grad = True
y.backward()
x.grad

# GPU
x.to('cuda')
x.cuda()
x.cpu()

# 模型相关
model.parameters()
model.train() / model.eval()
model.state_dict()
model.load_state_dict()
```

### Transformer维度速查

```python
# 常见维度
batch_size = 32
seq_len = 128
d_model = 512
num_heads = 8
d_k = d_model // num_heads  # 64

# Input
x: [batch, seq_len, d_model]

# Multi-head attention
Q, K, V: [batch, num_heads, seq_len, d_k]
scores: [batch, num_heads, seq_len, seq_len]
output: [batch, seq_len, d_model]

# Feed-forward
input: [batch, seq_len, d_model]
hidden: [batch, seq_len, d_ff]  # d_ff通常是4*d_model
output: [batch, seq_len, d_model]
```

### 调试技巧速查

```python
# 检查shape
print(f"x.shape: {x.shape}")

# 检查值
print(f"x.min(): {x.min()}, x.max(): {x.max()}")

# 检查梯度
print(f"x.grad: {x.grad}")

# 检查NaN
assert not torch.isnan(x).any()

# 设备
print(f"x.device: {x.device}")
```

---

*文档版本: v2.0 (包含完整PyTorch Week 0)*
*维护者: Claude*
*最后更新: 2024-12-18*
*如有问题或建议，随时反馈！*
