# 6. 训练技巧：让网络学得更好

> 🎯 **本课目标**：学会识别**过拟合**与**欠拟合**；理解学习率、**预热 + 余弦衰减**调度；看懂 SGD、Momentum、Adam、**AdamW** 的区别；知道为什么初始化、**LayerNorm/RMSNorm**、**残差连接**能让深层网络训练起来；掌握权重衰减、Dropout、梯度裁剪和混合精度（bf16）；拿到一份训练排错清单。这些就是训练大模型时每天都在用的「工程手艺」。
>
> ⏱ **预计用时**：90 分钟　｜　📋 **前置知识**：本部分第 1～5 课

上一课的例子都很「乖」：网络小、数据干净，随便设个学习率都能学会。真实的训练往往会遇到：损失不降、损失变成 `nan`、训练集上很好但新数据上很差、网络一深就学不动……本课介绍的每一个技巧都对应一个具体问题。本课大部分代码是纯 Python，不需要安装 PyTorch 也能运行。

## 6.1 过拟合与欠拟合

我们训练模型，真正关心的是它在**没见过的新数据**上表现如何，这叫**泛化**（generalization）。

- **欠拟合**（underfitting）：模型太简单或训练不够，连训练集都学不好。
- **过拟合**（overfitting）：模型把训练集「背」了下来，连其中的噪声和偶然也记住了，换成新数据就不行。

下面这个例子里，真正的规律是「x > 5 就是类别 1」，但 20% 的标签是错的（现实数据总有噪声）。分类方法是「看离 x 最近的 k 个训练样本，少数服从多数」：

```python
import random
random.seed(1)

def make(n):
    data = []
    for _ in range(n):
        x = random.uniform(0, 10)
        y = 1 if x > 5 else 0          # 真正的规律：x > 5 就是类别 1
        if random.random() < 0.2:      # 但有 20% 的标签是错的（噪声）
            y = 1 - y
        data.append((x, y))
    return data

train, test = make(100), make(1000)

def predict(x, k):                     # 看离 x 最近的 k 个训练样本，少数服从多数
    near = sorted(train, key=lambda p: abs(p[0] - x))[:k]
    return 1 if sum(y for _, y in near) * 2 > k else 0

def accuracy(data, k):
    return sum(predict(x, k) == y for x, y in data) / len(data)

for k in (1, 5, 25):
    print(f"k={k:2d}  训练准确率 {accuracy(train, k):.0%}   测试准确率 {accuracy(test, k):.0%}")
```

`k=1` 在训练集上 100% 正确（每个点最近的就是它自己），但测试准确率最低——它把错误标签也背下来了。**训练集上的成绩会骗人**，所以数据要分成三份：

| 数据集 | 用途 |
|--------|------|
| **训练集**（train） | 用来计算梯度、更新参数 |
| **验证集**（validation） | 训练过程中检查泛化效果，用来挑选超参数（学习率、模型大小……）和决定何时停止 |
| **测试集**（test） | 最后只用一次，报告最终成绩 |

训练时同时画出两条损失曲线，就能诊断问题：

```text
损失                                      损失
 │╲                                         │╲
 │ ╲___ 验证                                │ ╲      ___── 验证（开始上升 = 过拟合）
 │  ╲  ‾‾‾‾‾‾‾‾‾‾                           │  ╲___/
 │   ╲___ 训练                              │   ╲___
 │       ‾‾‾‾‾‾‾‾‾  两条都高 = 欠拟合        │       ‾‾‾‾‾‾─── 训练（一直下降）
 └──────────────── 步数                     └──────────────── 步数
```

> [!NOTE]
> 大语言模型预训练时数据极多（几万亿 token），每条数据通常只看一到几遍，所以**很少过拟合**，主要问题是欠拟合——这也是为什么模型和数据越大越好（第四部分会讲「缩放定律」）。但在**微调**时，数据少，过拟合非常常见。

## 6.2 学习率：最重要的超参数

用梯度下降最小化 `f(x) = x²`（最低点在 0），从 `x = 5` 出发，看看不同学习率会怎样：

```python
def descend(lr, steps=10):
    x = 5.0                       # 最小化 f(x) = x²，最低点在 x = 0
    path = []
    for _ in range(steps):
        x -= lr * 2 * x           # f'(x) = 2x
        path.append(x)
    return path

for lr in (0.01, 0.1, 0.9, 1.1):
    xs = descend(lr)
    print(f"lr={lr:<5} 10 步后 x = {xs[-1]:10.4f}   前 5 步：", " ".join(f"{v:.1f}" for v in xs[:5]))
```

| 学习率 | 现象 |
|--------|------|
| 太小（0.01） | 稳定但非常慢，10 步才走了一点点 |
| 合适（0.1） | 稳步接近 0 |
| 偏大（0.9） | 在两边来回跳（震荡），但还在收敛 |
| 太大（1.1） | 越跳越远，**发散**——真实训练中表现为损失变成 `inf` / `nan` |

> [!TIP]
> 训练出问题时，**第一个要怀疑的就是学习率**。常见的起点：Adam/AdamW 用 `1e-3`～`3e-4`；大模型预训练用 `1e-4`～`6e-4`，模型越大学习率通常越小；微调用 `1e-5`～`2e-4`。

## 6.3 学习率调度：预热 + 余弦衰减

学习率不必一成不变。几乎所有大模型训练都用这样的**调度**（schedule）：

1. **预热**（warmup）：最开始的几百到几千步，学习率从接近 0 线性升到最大值。刚开始参数是随机的，梯度很「乱」，一上来就迈大步容易把训练搞崩。
2. **余弦衰减**（cosine decay）：之后按余弦曲线慢慢降到一个很小的值。训练后期步子小一点，才能更精细地停在低谷里。

```python
import math

def lr_at(step, max_lr=3e-4, warmup=100, total=1000, min_lr=3e-5):
    if step < warmup:                                   # 线性预热
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / (total - warmup)       # 从 0 走到 1
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

for step in [0, 25, 50, 75] + list(range(100, 1001, 100)):
    lr = lr_at(step)
    print(f"step {step:4d}  lr {lr:.6f}  " + "█" * round(lr / 3e-4 * 40))
```

> [!NOTE]
> 一些新的训练方案用 **WSD**（Warmup-Stable-Decay）调度：预热后长时间保持最大学习率，最后一小段再快速衰减。好处是训练可以随时延长，不用事先定好总步数。

## 6.4 优化器：SGD、Momentum、Adam 与 AdamW

普通梯度下降（SGD）在「狭长山谷」形状的损失上会很吃力：陡的方向来回震荡，平的方向又走得很慢。下面的损失函数 `x² + 100y²` 就是这样一个山谷：

```python
import math

def loss_grad(x, y):                  # 狭长山谷：y 方向很陡，x 方向很平
    return x * x + 100 * y * y, (2 * x, 200 * y)

def run(update, steps=1000):
    p, state = [-4.0, 1.0], {}
    for t in range(1, steps + 1):
        loss, g = loss_grad(*p)
        if loss < 1e-4:
            return t
        update(p, g, state, t)
    return None

def sgd(p, g, s, t, lr=0.009):        # 学习率再大一点（0.01 以上）y 方向就会发散
    for i in range(2):
        p[i] -= lr * g[i]

def momentum(p, g, s, t, lr=0.005, beta=0.9):
    v = s.setdefault("v", [0.0, 0.0])
    for i in range(2):
        v[i] = beta * v[i] + g[i]                # 「速度」：过去梯度的累积，来回震荡的部分互相抵消
        p[i] -= lr * v[i]

def adam(p, g, s, t, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    m = s.setdefault("m", [0.0, 0.0])
    v = s.setdefault("v", [0.0, 0.0])
    for i in range(2):
        m[i] = b1 * m[i] + (1 - b1) * g[i]       # 一阶矩：梯度的移动平均（方向）
        v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2  # 二阶矩：梯度平方的移动平均（大小）
        m_hat = m[i] / (1 - b1 ** t)             # 偏差修正（开始时 m、v 偏向 0）
        v_hat = v[i] / (1 - b2 ** t)
        p[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)   # 每个参数有自己的步长

for name, fn in [("SGD", sgd), ("Momentum", momentum), ("Adam", adam)]:
    t = run(fn)
    print(f"{name:9s}", f"{t} 步到达最低点附近" if t else "1000 步内没有到达")
```

| 优化器 | 思路 | 说明 |
|--------|------|------|
| **SGD** | 沿负梯度走一步 | 最简单；对学习率很敏感 |
| **Momentum** | 加一个「惯性」：保留之前的速度 | 震荡相互抵消，平坦方向越走越快 |
| **Adam** | 动量 + 按每个参数梯度的大小**自动调整步长** | 梯度大的参数步子小，梯度小的步子大；对超参数不敏感 |
| **AdamW** | Adam + **解耦的权重衰减** | 大模型训练的默认选择；常用 `betas=(0.9, 0.95)`、`weight_decay=0.1` |

> [!NOTE]
> 优化器也在不断进步。**Muon** 等新优化器已经在一些大模型（如 Kimi K2）的训练中使用，号称比 AdamW 更省算力。不过 AdamW 仍然是最常见、最稳妥的选择。注意 Adam 要为每个参数额外保存 `m` 和 `v` 两个值，所以**优化器状态要占两倍于参数的内存**——训练比推理吃内存多得多的原因之一。

## 6.5 初始化：起点很重要

参数的初始值不能随便设：

- **全部设成 0？** 不行。同一层的所有神经元会算出一样的输出、得到一样的梯度，永远一模一样——相当于只有一个神经元。必须用**随机数**打破对称。
- **随机数的大小也很关键**。一层层传下去，太大的值会越乘越大（爆炸），太小的值会越乘越小（消失）。

```python
import math
import random
random.seed(0)

n, depth = 64, 10

def std(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / len(v))

def forward(scale):
    x = [random.gauss(0, 1) for _ in range(n)]
    stds = []
    for _ in range(depth):
        W = [[random.gauss(0, scale) for _ in range(n)] for _ in range(n)]
        x = [max(0.0, sum(w * a for w, a in zip(row, x))) for row in W]   # 线性层 + ReLU
        stds.append(std(x))
    return stds

print("初始化方式        第1层   第4层   第7层   第10层（每层输出的标准差）")
for name, scale in [("std=1", 1.0), ("std=0.01", 0.01), ("Kaiming √(2/n)", math.sqrt(2 / n))]:
    s = forward(scale)
    print(f"{name:15s}", "  ".join(f"{v:7.2g}" for v in (s[0], s[3], s[6], s[9])))
```

**Kaiming（He）初始化**让每层权重的标准差为 `√(2/n)`（`n` 是输入维度），信号一层层传下去大小基本不变。PyTorch 的 `nn.Linear` 默认就用了类似的缩放，所以平时不用自己操心；GPT 类模型常用均值 0、标准差 0.02 的正态分布初始化，并对残差分支的输出层再按层数缩小。

## 6.6 归一化：LayerNorm 与 RMSNorm

即使初始化得好，训练过程中各层数值的大小还是会漂移。**归一化**层把每个向量重新拉回「标准大小」：

- **LayerNorm**：减去均值，再除以标准差 → 均值 0、方差 1。原始 Transformer、GPT-2 使用。
- **RMSNorm**：不减均值，只除以均方根 `√(mean(x²))`。计算更省，效果相当，**Llama、Qwen、DeepSeek 等现代大模型几乎都用它**。

```python
import math

def layer_norm(x, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((a - mean) ** 2 for a in x) / len(x)
    return [(a - mean) / math.sqrt(var + eps) for a in x]

def rms_norm(x, eps=1e-5):
    rms = math.sqrt(sum(a * a for a in x) / len(x) + eps)
    return [a / rms for a in x]

x = [2.0, 4.0, 6.0, 8.0]
print("LayerNorm:", [round(v, 3) for v in layer_norm(x)])
print("RMSNorm:  ", [round(v, 3) for v in rms_norm(x)])
big = [v * 1000 for v in x]
print("放大 1000 倍后 RMSNorm：", [round(v, 3) for v in rms_norm(big)])   # 结果不变
```

实际的归一化层后面还会乘一个**可学习**的缩放向量 `γ`（LayerNorm 还加一个偏移 `β`），让网络自己决定需要多大的数值。`eps` 是一个很小的数，防止除以 0。

> [!NOTE]
> 你可能还听说过 **BatchNorm**：它在一个 batch 的样本之间做归一化，在图像网络（CNN）里常用。语言模型里句子长短不一、推理时常常一次只处理一句，所以用的是对**每个向量自己**做归一化的 LayerNorm / RMSNorm。

## 6.7 残差连接：给梯度修一条高速公路

网络越深，反向传播时梯度要连乘越多的局部导数。如果每层导数都小于 1，乘几十次就几乎为 0——**梯度消失**，前面的层根本学不动。

**残差连接**（residual connection，ResNet 2015 年提出）的办法极其简单：每层的输出 = **输入 + 这一层算出的变化**，即 `y = x + f(x)`。它的导数是 `1 + f'(x)`，那个「1」保证梯度至少能原样穿过每一层。

```python
import math

def grad_through(depth, residual, w=0.5):
    x, grad = 1.0, 1.0
    for _ in range(depth):
        f = math.tanh(w * x)
        local = w * (1 - f * f)                   # f(x) = tanh(w·x) 的导数
        if residual:
            x, grad = x + f, grad * (1 + local)   # y = x + f(x)，导数 = 1 + f'(x)
        else:
            x, grad = f, grad * local
    return grad

for depth in (5, 20, 50):
    print(f"{depth:2d} 层  普通：{grad_through(depth, False):.2e}   残差：{grad_through(depth, True):.2f}")
```

Transformer 的每一层都是两个残差块，结构正是：

```formula
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

这种「先归一化再计算，再加回去」的写法叫 **Pre-Norm**，是现在大模型的标准做法，能稳定地训练上百层的网络。第三部分会详细讲。

## 6.8 正则化：防止过拟合

| 方法 | 做法 | 直觉 |
|------|------|------|
| **权重衰减**（weight decay） | 每步把参数往 0 拉一点：`w -= lr * λ * w` | 不让权重变得太大，模型更「平滑」 |
| **Dropout** | 训练时随机把一部分神经元输出置 0 | 不能依赖任何单个神经元，迫使网络学到更稳健的特征 |
| **早停**（early stopping） | 验证损失不再下降就停止训练，保留最好的那次 | 在开始「背题」之前停下 |
| **更多数据 / 数据增强** | 收集更多数据，或对已有数据做变换 | 最根本的办法 |

Dropout 的实现很简单。注意训练时没被置 0 的值要**除以 `1 − p`**，让输出的平均值保持不变；评估时关闭 Dropout（这就是 `model.eval()` 的作用之一）：

```python
import random
random.seed(0)

def dropout(x, p=0.5, training=True):
    if not training:
        return x
    return [0.0 if random.random() < p else a / (1 - p) for a in x]

x = [1.0] * 8
print("训练时：", dropout(x))
print("评估时：", dropout(x, training=False))
avg = sum(sum(dropout(x)) for _ in range(10000)) / 10000 / len(x)
print("训练时的平均值：", round(avg, 3))     # 约等于 1，和评估时一致
```

> [!NOTE]
> 大模型**预训练**通常不用 Dropout（数据够多，不怎么过拟合），但会用 AdamW 的权重衰减。微调小数据集时，Dropout 和早停就很有用了。

## 6.9 梯度裁剪

偶尔遇到一批特别「怪」的数据，梯度会突然变得极大，一步就把参数推到很远的地方，损失跳成 `nan`（叫「损失尖峰」）。**梯度裁剪**：如果所有梯度合在一起的长度（**全局范数**）超过阈值，就按比例缩小，方向不变。

```python
import math

def clip_by_norm(grads, max_norm):
    norm = math.sqrt(sum(g * g for g in grads))      # 全局范数：所有梯度合成一个向量的长度
    if norm <= max_norm:
        return grads
    return [g * max_norm / norm for g in grads]       # 方向不变，长度缩到 max_norm

print(clip_by_norm([0.3, -0.4], 1.0))     # 长度 0.5，不需要裁剪
print(clip_by_norm([30.0, -40.0], 1.0))   # 长度 50，缩到 1
```

大模型训练几乎都设 `max_norm=1.0`。PyTorch 里一行：`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`，放在 `backward()` 之后、`step()` 之前。

## 6.10 混合精度：fp32、fp16、bf16

默认的 `float32`（fp32）每个数占 4 字节。用 16 位（2 字节）的格式，内存减半，GPU 上的计算也快得多：

| 格式 | 字节 | 范围 | 精度 | 说明 |
|------|------|------|------|------|
| fp32 | 4 | 约 ±3.4×10³⁸ | 约 7 位有效数字 | 默认 |
| fp16 | 2 | 约 ±65504 | 约 3～4 位 | 范围小，容易溢出，需要「损失缩放」 |
| **bf16** | 2 | 和 fp32 一样 | 约 2～3 位 | 范围大、不易溢出，**大模型训练的主流** |
| fp8 / fp4 | 1 / 0.5 | 更小 | 更低 | 最新的 GPU 支持，DeepSeek-V3 等用 fp8 训练 |

用 Python 自带的 `struct` 模块就能亲眼看到精度损失：

```python
import struct

def to_fp16(x):
    return struct.unpack("e", struct.pack("e", x))[0]       # "e" 表示 16 位半精度

def to_bf16(x):
    bits = struct.unpack("I", struct.pack("f", x))[0]       # 先拿到 32 位浮点数的二进制
    return struct.unpack("f", struct.pack("I", bits & 0xFFFF0000))[0]   # bf16 = 只保留高 16 位

for x in (0.1, 3.14159, 1000.7):
    print(f"{x:<8}  fp16 → {to_fp16(x):<12.7g} bf16 → {to_bf16(x):.7g}")

try:
    to_fp16(70000.0)
except OverflowError:
    print("70000 超出了 fp16 的范围（最大约 65504）")
print("bf16 能表示 70000 吗？", to_bf16(70000.0))           # 能，只是不太精确
```

**混合精度训练**：参数的「主副本」和优化器状态保留 fp32，前向和反向的大部分计算用 bf16。在 PyTorch 里（有 GPU 时）：

```python norun
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(xb), yb)
loss.backward()
```

> [!TIP]
> 估算内存的经验法则：**推理**时 bf16 模型每个参数 2 字节，70 亿（7B）参数 ≈ 14 GB；**训练**（AdamW + 混合精度）每个参数约 16 字节，7B 模型 ≈ 112 GB，还不算中间激活值。这就是为什么微调大模型常用 LoRA 等省内存的方法（第五部分会讲）。

## 6.11 放在一起：一个现代的训练脚本

下面用 PyTorch 把本课的技巧组合起来：残差块 + RMSNorm + GELU + Dropout + AdamW（权重衰减）+ 预热余弦调度 + 梯度裁剪。这几乎就是一个迷你 Transformer 的训练骨架（只差注意力层）：

```python requires=torch
import math
import torch
from torch import nn

torch.manual_seed(0)

class Block(nn.Module):                      # 现代的残差块：Norm → MLP → Dropout，再加回输入
    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return x + self.drop(self.mlp(self.norm(x)))      # 残差连接

model = nn.Sequential(nn.Linear(8, 32), *[Block(32) for _ in range(4)], nn.Linear(32, 1))
X = torch.randn(512, 8)
y = (X[:, :1] * X[:, 1:2]).sin()                         # 随便造一个非线性的目标

total_steps, warmup = 300, 30
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
def schedule(step):                                       # 返回「学习率倍数」
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

for step in range(total_steps):
    idx = torch.randint(0, len(X), (64,))                 # 随机取一个小批量
    loss = nn.functional.mse_loss(model(X[idx]), y[idx])
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # 梯度裁剪
    optimizer.step()
    scheduler.step()                                      # 更新学习率
    if step % 100 == 0 or step == total_steps - 1:
        print(f"step {step:3d}  loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.5f}  grad_norm {norm:.2f}")

model.eval()                                              # 评估模式：关闭 Dropout
with torch.no_grad():
    print("最终 MSE：", round(nn.functional.mse_loss(model(X), y).item(), 4))
```

## 6.12 训练排错清单

参考 Andrej Karpathy 的经典文章《A Recipe for Training Neural Networks》：

1. **先看数据**。打印几条样本和标签，确认没有读错、没有错位。
2. **从简单开始**。先用最小的模型、最小的数据跑通整个流程。
3. **先在一个小批量上过拟合**。用同一个 batch 反复训练，损失应该能降到接近 0；降不下去说明代码有 bug。
4. **检查初始损失**。比如 10 分类时，随机猜的交叉熵应该约为 `ln 10 ≈ 2.30`；差太多说明初始化或损失函数有问题。
5. **损失变 `nan`**：降低学习率、加梯度裁剪、检查数据里有没有 `inf` / `nan`、检查有没有 `log(0)`。
6. **损失完全不动**：检查是否忘了 `zero_grad()` / `step()`、学习率是否太小、梯度是否被 `no_grad()` / `.detach()` 切断。
7. **一次只改一个地方**，并把每次实验的设置和结果记下来。

## 6.13 小结

- 关心的是**泛化**：用验证集监控，训练损失降、验证损失升就是**过拟合**，两者都高是**欠拟合**。
- **学习率**最重要：太大发散、太小太慢；大模型用**预热 + 余弦衰减**。
- **AdamW** = 动量 + 自适应步长 + 解耦权重衰减，是默认的优化器；它的状态占参数两倍的内存。
- 好的**初始化**（随机、尺度合适）、**归一化**（RMSNorm/LayerNorm）和**残差连接**（`x + f(x)`）让深层网络能训练起来。
- 正则化：权重衰减、Dropout、早停、更多数据；**梯度裁剪**（全局范数 1.0）防止损失尖峰。
- **bf16** 混合精度：内存减半、速度更快，范围和 fp32 一样。

## 6.14 练习

<!-- exercise: lr-schedule -->
### 练习 1：预热 + 余弦调度

输入一行三个整数 `step warmup total`，输出这一步的学习率**倍数**（最大学习率记为 1，最小为 0），保留 4 位小数：

- `step < warmup` 时：`(step + 1) / warmup`
- 否则：`0.5 × (1 + cos(π × (step − warmup) / (total − warmup)))`

```text
样例输入：0 10 100
样例输出：0.1000
```

<details>
<summary>提示</summary>

用 `math.cos` 和 `math.pi`。注意 `step` 从 0 开始计数。
</details>

<!-- exercise: rmsnorm -->
### 练习 2：实现 RMSNorm

输入一行若干个数，输出 RMSNorm 的结果：每个数除以 `√(平方的平均值 + 1e-5)`，保留 3 位小数，空格隔开。

```text
样例输入：2 4 6 8
样例输出：0.365 0.730 1.095 1.461
```

<details>
<summary>提示</summary>

`rms = math.sqrt(sum(a * a for a in x) / len(x) + 1e-5)`，然后 `print(" ".join(f"{a / rms:.3f}" for a in x))`。
</details>

<!-- exercise: clip-norm -->
### 练习 3：梯度裁剪

第一行是若干个梯度值，第二行是 `max_norm`。如果梯度的全局范数（所有值平方和的平方根）大于 `max_norm`，就把每个值乘以 `max_norm / 范数`；否则保持不变。输出结果，保留 4 位小数，空格隔开。

```text
样例输入：
30 -40
1
样例输出：0.6000 -0.8000
```

<details>
<summary>提示</summary>

先算 `norm = math.sqrt(sum(g * g for g in grads))`，再判断要不要缩放。
</details>

<!-- exercise: diagnose -->
### 练习 4：诊断训练曲线

训练损失一直在下降，但验证损失从第 3 个 epoch 开始持续上升。最可能的情况是？

- 欠拟合，应该把模型变大
- 过拟合，可以早停、加正则化或增加数据
- 学习率太小
- 一切正常，继续训练就好

<details>
<summary>答案与解析</summary>

训练集越来越好、验证集越来越差，说明模型在「背」训练数据——典型的过拟合。可以在验证损失最低时停止（早停），或使用权重衰减、Dropout、更多数据。
</details>

<!-- exercise: zero-init -->
### 练习 5：全部初始化为 0

如果把一个 MLP 的所有权重都初始化为 0，会怎样？

- 训练会更快、更稳定
- 同一层的所有神经元始终完全相同，网络相当于每层只有一个神经元
- 没有任何影响
- 程序会报错

<details>
<summary>答案与解析</summary>

所有神经元的输入、权重都一样，算出的输出和梯度也一样，更新后依然一样——这种「对称性」永远无法打破。所以必须用随机数初始化。
</details>

<!-- exercise: bf16-memory -->
### 练习 6：估算内存

一个 70 亿参数（7B）的模型，用 bf16 格式加载做推理，仅模型参数大约需要多少内存？

- 7 GB
- 14 GB
- 28 GB
- 56 GB

<details>
<summary>答案与解析</summary>

bf16 每个参数 2 字节：7×10⁹ × 2 字节 = 14×10⁹ 字节 ≈ 14 GB。fp32 则需要 28 GB；训练时加上梯度和 AdamW 状态，需要的更多。
</details>

## 6.15 延伸阅读

- Andrej Karpathy：[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)（经典的训练经验总结）
- 《动手学深度学习》：[优化算法](https://zh.d2l.ai/chapter_optimization/index.html) 一章
- Google Research：[Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)（调参手册）
- 下一部分：[第三部分 · 语言模型](../../lm/01-language-models/zh-CN.md)——终于要开始做语言模型了！
