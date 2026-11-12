# 5. PyTorch 入门：张量、自动求导与训练循环

> 🎯 **本课目标**：认识 PyTorch 的核心——**张量**；掌握张量的创建、运算、广播和变形；看到 PyTorch 的自动求导和上一课的 micrograd 完全一致；用 `nn.Module` 搭模型；写出标准的五步训练循环；学会小批量训练、使用 GPU、保存模型，并认识最常见的报错。
>
> ⏱ **预计用时**：90 分钟　｜　📋 **前置知识**：本部分第 1～4 课；需要安装 PyTorch（见第一部分第 2 课）

## 5.1 为什么用 PyTorch？

上一课我们用纯 Python 写出了 micrograd：它原理完整，但一次只能处理**一个数字**，训练一个 17 个参数的网络都要好几秒。真正的大模型有几十亿个参数，需要：

1. 一次处理**整块数据**（向量、矩阵乃至更高维的数组）；
2. 在 **GPU** 上并行计算，速度快成千上万倍；
3. 现成的网络层、损失函数、优化器。

**PyTorch** 就是这样一个框架，由 Meta 开源，现在由 Linux 基金会下的 PyTorch 基金会维护。它是目前研究和工业界最主流的深度学习框架：Hugging Face Transformers、vLLM、Llama、DeepSeek、Qwen 等的训练和推理代码，绝大多数都基于 PyTorch。（另一个常见的选择是 Google 的 JAX。）

> [!TIP]
> **安装**：在你的项目目录运行 `uv add torch`（或 `pip install torch`），详见第一部分第 2 课。中国大陆用户可以加上清华镜像：`pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple`。
>
> 如果学习程序提示找不到 `torch`，说明它用的 Python 不是你装了 torch 的那个。打开**设置**，把 **Python 路径**改成项目虚拟环境里的 Python：macOS/Linux 是 `项目目录/.venv/bin/python`，Windows 是 `项目目录\.venv\Scripts\python.exe`。

## 5.2 张量（Tensor）

**张量**就是「多维数组」：0 维是一个数（标量），1 维是向量，2 维是矩阵，3 维以上统称张量。它和 NumPy 的数组几乎一样，但多了两项本领：**能在 GPU 上运行**、**能自动求导**。

```python requires=torch
import torch

print(torch.__version__)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(x)
print("shape:", x.shape, " dtype:", x.dtype, " device:", x.device)
print(torch.zeros(2, 3))                 # 全 0
print(torch.ones(3))                     # 全 1
print(torch.arange(6).reshape(2, 3))     # 0..5，再排成 2 行 3 列
torch.manual_seed(0)                     # 固定随机种子，让结果可以复现
print(torch.randn(2, 2))                 # 标准正态分布的随机数
```

每个张量都有三个最重要的属性，**出错时先看它们**：

| 属性 | 含义 | 例子 |
|------|------|------|
| `shape` | 形状：每一维有多大 | `torch.Size([2, 3])`：2 行 3 列 |
| `dtype` | 数据类型 | `torch.float32`（默认小数）、`torch.int64`（默认整数）、`torch.bfloat16`（大模型常用） |
| `device` | 存放在哪里 | `cpu`、`cuda`（NVIDIA 显卡）、`mps`（苹果芯片） |

## 5.3 张量运算

```python requires=torch
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
print(a + b)          # 逐元素相加
print(a * b)          # 逐元素相乘（不是矩阵乘法！）
print(a @ b)          # 矩阵乘法
print(a.sum(), a.mean(), a.max())
print(a.sum(dim=0))   # 沿第 0 维求和（把各行加起来）→ 每列的和
print(a.sum(dim=1))   # 沿第 1 维求和 → 每行的和
print(a.T)            # 转置
print(a[0], a[:, 1], a[1, 0].item())    # 取第 0 行、第 1 列、单个元素
```

> [!WARNING]
> `*` 是**逐元素**相乘，`@` 才是**矩阵乘法**。这是新手最常混淆的地方。另外 `.item()` 把只有一个元素的张量变成普通的 Python 数字，打印损失时经常用到。

`dim` 参数（维度）很重要：对形状 `(2, 3)` 的张量，`sum(dim=0)` 会把第 0 维「压扁」，结果形状是 `(3,)`；`sum(dim=1)` 结果形状是 `(2,)`。可以记成：**`dim` 指定哪一维消失。**

## 5.4 广播与变形

**广播**（broadcasting）：形状不同的张量做逐元素运算时，PyTorch 会自动把较小的那个「复制扩展」成相同形状。规则是**从最右边一维开始对齐**，每一维要么相等，要么其中一个是 1（或不存在）。

```python requires=torch
import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])     # 形状 (2, 3)
bias = torch.tensor([10.0, 20.0, 30.0])  # 形状 (3,)
print(x + bias)                          # bias 被广播到每一行
col = torch.tensor([[100.0], [200.0]])   # 形状 (2, 1)
print(x + col)                           # col 被广播到每一列
print(x.reshape(3, 2))                   # 改变形状（元素总数不变）
print(x.reshape(-1))                     # -1 表示「自动推算」→ 展平成 6 个元素
print(x.unsqueeze(0).shape)              # 在最前面加一维 → (1, 2, 3)
```

```text
x     (2, 3)          x     (2, 3)
bias     (3,)  ✓      col   (2, 1)  ✓      (1 会被扩展成 3)
结果  (2, 3)          结果  (2, 3)

(2, 3) 和 (2,) ✗：最右边一维 3 ≠ 2，而且都不是 1 → 报错
```

神经网络里到处都是广播：线性层 `x @ W.T + b` 中，偏置 `b` 的形状是 `(out,)`，会被广播到 batch 里的每个样本上。

## 5.5 自动求导：就是 micrograd

把上一课的例子 `L = (a × b + c)²` 用 PyTorch 再算一遍：

```python requires=torch
import torch

a = torch.tensor(2.0, requires_grad=True)    # requires_grad=True：需要对它求梯度
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(10.0, requires_grad=True)
L = (a * b + c) ** 2
L.backward()
print(L.item(), a.grad, b.grad, c.grad)      # 16.0 -24 16 8，和手算、micrograd 都一样
```

完全一样！PyTorch 在每次运算时也在后台记录计算图（每个结果张量都有一个 `grad_fn`），`backward()` 时按拓扑逆序把梯度传回去——这正是你上一课亲手实现的东西。

## 5.6 用 nn.Module 搭模型

`torch.nn` 里有现成的网络层。最常用的是 **`nn.Linear(in_features, out_features)`**：它内部有权重 `weight`（形状 `(out, in)`）和偏置 `bias`（形状 `(out,)`），计算 `x @ weight.T + bias`。

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
layer = nn.Linear(3, 2)          # 3 维输入 → 2 维输出
print(layer.weight.shape, layer.bias.shape)
x = torch.randn(4, 3)            # 4 个样本（一个 batch），每个 3 维
print(layer(x).shape)            # (4, 2)：每个样本得到 2 个输出

model = nn.Sequential(           # 按顺序把层串起来
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
print(model)
print("参数量：", sum(p.numel() for p in model.parameters()))   # 3×16+16 + 16×2+2 = 98
```

> [!NOTE]
> 第一维几乎总是 **batch**（一批样本的数量）。模型一次处理一整批数据，GPU 才能发挥并行的优势。`nn.Linear(3, 2)` 只关心**最后一维**是 3，前面可以有任意多维。

更灵活的写法是**继承 `nn.Module`**，自己写 `forward`（前向计算）。所有大模型（包括 GPT、Llama）都是这样写的：

```python requires=torch
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                    # 必须先调用父类的初始化
        self.fc1 = nn.Linear(d_in, d_hidden)  # 作为属性的层会被自动登记为参数
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):                     # 只需写前向；反向由 autograd 自动完成
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

model = MLP(2, 8, 1)
print(model(torch.zeros(5, 2)).shape)         # 调用 model(x) 会自动执行 forward
for name, p in model.named_parameters():
    print(name, tuple(p.shape))
```

## 5.7 标准训练循环

用 PyTorch 训练 XOR。注意循环里的五行，**几乎所有 PyTorch 训练代码都长这样**：

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
loss_fn = nn.BCEWithLogitsLoss()                  # 二分类损失（内部自带 sigmoid）
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for step in range(1, 501):
    logits = model(X)                             # ① 前向
    loss = loss_fn(logits, y)                     # ② 计算损失
    optimizer.zero_grad()                         # ③ 清零梯度
    loss.backward()                               # ④ 反向传播
    optimizer.step()                              # ⑤ 更新参数
    if step % 100 == 0:
        print(f"第 {step} 步：损失 {loss.item():.4f}")

with torch.no_grad():
    print(torch.sigmoid(model(X)).round().squeeze().tolist())   # [0.0, 1.0, 1.0, 0.0]
```

对照上一课：

| micrograd | PyTorch |
|-----------|---------|
| 自己写 `(p - y) ** 2` | `nn.MSELoss()`、`nn.BCEWithLogitsLoss()`、`nn.CrossEntropyLoss()` 等 |
| `p.grad = 0.0` | `optimizer.zero_grad()` |
| `p.data -= 0.1 * p.grad` | `optimizer.step()`（SGD）；更常用的是 `torch.optim.AdamW` |

> [!TIP]
> 分类任务的模型最后一层**不要**自己加 sigmoid/softmax：`BCEWithLogitsLoss` 和 `CrossEntropyLoss` 接收的是原始分数 **logits**，内部会用数值更稳定的方式处理。需要概率时（比如预测时）再手动 `torch.sigmoid` / `torch.softmax`。

## 5.8 完整例子：小批量、GPU 与评估

下面是一个更接近真实项目的例子：分类两个同心圆环上的点。新概念：

- **Dataset / DataLoader**：把数据切成小批量（mini-batch），并在每轮打乱顺序。用小批量的梯度下降叫**随机梯度下降**（SGD），比每次用全部数据省内存，而且往往学得更好。
- **epoch**：把全部训练数据过一遍叫一个 epoch。
- **device**：有 GPU 就用 GPU，模型和数据都要 `.to(device)`。
- **训练集 / 测试集**：在模型**没见过**的数据上评估，才能知道它是否真的学会了（而不是背下来了）。
- **`model.eval()` 与 `torch.no_grad()`**：评估时切换到评估模式，并且不记录计算图。

```python requires=torch
import math
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

# 1. 数据：两个同心圆环。内圈是类别 0，外圈是类别 1
def make_rings(n):
    r = torch.cat([torch.rand(n) * 1.0, 1.5 + torch.rand(n) * 1.0])   # 半径
    theta = torch.rand(2 * n) * 2 * math.pi                          # 角度
    X = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()            # 类别标签必须是整数
    return X, y

X_train, y_train = make_rings(500)
X_test, y_test = make_rings(100)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# 2. 设备：有 NVIDIA 显卡用 cuda，苹果芯片用 mps，否则用 cpu
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print("设备：", device)

# 3. 模型、损失、优化器
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()                     # 多分类交叉熵（内部自带 softmax）
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

def accuracy(X, y):
    model.eval()                                    # 切换到评估模式
    with torch.no_grad():                           # 评估时不需要梯度，更快、更省内存
        pred = model(X.to(device)).argmax(dim=1)    # 分数最高的类别就是预测
    model.train()                                   # 切回训练模式
    return (pred == y.to(device)).float().mean().item()

# 4. 训练：每个 epoch 把全部数据按小批量过一遍
for epoch in range(1, 21):
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  损失 {total / len(X_train):.4f}  "
              f"测试准确率 {accuracy(X_test, y_test):.1%}")
```

线性模型永远分不开两个同心圆，而这个小小的 MLP 很快就在**测试集**上达到接近 100% 的准确率。

## 5.9 保存与加载模型

```python requires=torch
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
torch.save(model.state_dict(), "rings.pt")        # 只保存参数（推荐做法）

model2 = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
model2.load_state_dict(torch.load("rings.pt"))    # 先建好同样结构的模型，再装入参数
print(torch.equal(model[0].weight, model2[0].weight))   # True
```

`state_dict()` 是一个「参数名 → 张量」的字典。Hugging Face 上下载的模型文件（如 `model.safetensors`）本质上也是这样一个字典，只是换了一种更安全、更快的存储格式。

## 5.10 最常见的报错

| 报错信息（节选） | 原因 | 解决办法 |
|------------------|------|----------|
| `mat1 and mat2 shapes cannot be multiplied (4x5 and 3x2)` | 输入最后一维（5）≠ `nn.Linear` 的 `in_features`（3） | `print(x.shape)`，改模型或数据 |
| `The size of tensor a (3) must match the size of tensor b (2)` | 两个张量无法广播 | 检查两边的 `shape` |
| `Expected all tensors to be on the same device` | 模型在 GPU、数据在 CPU（或反过来） | 两者都 `.to(device)` |
| `expected target dtype to be Long` | `CrossEntropyLoss` 的标签不是整数 | `y = y.long()` |
| `mat1 and mat2 must have the same dtype` | 一个 `float64`（常来自 NumPy）、一个 `float32` | `.float()` |
| 损失越来越大、变成 `nan` | 学习率太大，或忘了 `zero_grad()` | 减小 `lr`，检查训练循环 |

学习程序会识别这些报错，并用你的语言解释原因。

```python requires=torch
import torch
from torch import nn

layer = nn.Linear(3, 2)
x = torch.randn(4, 5)             # 故意给 5 维输入
try:
    layer(x)
except RuntimeError as e:
    print("出错了：", e)
```

## 5.11 小结

- **张量**是 PyTorch 的基本数据：多维数组，能放在 GPU 上，能自动求导；出错时先看 `shape`、`dtype`、`device`。
- `*` 是逐元素乘，`@` 是矩阵乘；`dim` 指定沿哪一维运算（那一维会消失）；**广播**从右往左对齐，每维相等或为 1。
- `requires_grad=True` + `backward()` 就是 micrograd 的张量版。
- 用 `nn.Linear`、`nn.Sequential` 或继承 `nn.Module` 搭模型；第一维通常是 batch。
- 训练五步：**前向 → 损失 → `zero_grad()` → `backward()` → `step()`**；用 `DataLoader` 小批量训练，在测试集上评估时用 `model.eval()` 和 `torch.no_grad()`。
- `torch.save(model.state_dict(), ...)` 保存，`load_state_dict` 加载。

## 5.12 练习

<!-- exercise: linear-shape -->
### 练习 1：输出形状

`x` 的形状是 `(32, 10)`，经过 `nn.Linear(10, 5)` 之后，输出的形状是？

- `(10, 5)`
- `(32, 5)`
- `(32, 10)`
- `(5, 32)`

<details>
<summary>答案与解析</summary>

`nn.Linear` 只改变**最后一维**：10 → 5；第一维 32 是 batch 大小，保持不变。
</details>

<!-- exercise: broadcast-shape -->
### 练习 2：广播规则

不用 PyTorch，自己实现广播规则。输入两行，每行是一个形状（若干个用空格隔开的正整数）。如果两个形状可以广播，输出结果形状（用空格隔开）；否则输出 `error`。

```text
样例输入：
2 3
3
样例输出：2 3
```

<details>
<summary>提示</summary>

先在较短的形状前面补 1，让两者一样长；然后逐维比较：相等就取这个值，有一个是 1 就取另一个，否则报错。

```python norun
a = list(map(int, input().split()))
b = list(map(int, input().split()))
n = max(len(a), len(b))
a = [1] * (n - len(a)) + a
b = [1] * (n - len(b)) + b
```
</details>

<!-- exercise: star-vs-at -->
### 练习 3：`*` 还是 `@`？

下面的程序会输出什么？

```python requires=torch
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 0], [0, 1]])
print((a * b).tolist())
print((a @ b).tolist())
```

<details>
<summary>答案与解析</summary>

`b` 是单位矩阵。逐元素乘 `a * b` 只保留对角线：`[[1, 0], [0, 4]]`；矩阵乘 `a @ b` 等于 `a` 本身：`[[1, 2], [3, 4]]`。
</details>

<!-- exercise: torch-grad -->
### 练习 4：用 PyTorch 求导

输入一个数 `x`，用 PyTorch 的自动求导计算 `y = x³ + 2x` 及其导数 `dy/dx`，输出 `y` 和 `dy/dx`（保留 2 位小数，空格隔开）。（需要安装 `torch`）

```text
样例输入：2
样例输出：12.00 14.00
```

<details>
<summary>提示</summary>

`x = torch.tensor(float(input()), requires_grad=True)`，算出 `y` 后调用 `y.backward()`，导数就在 `x.grad` 里。用 `.item()` 取出数字。
</details>

<!-- exercise: loop-order -->
### 练习 5：训练循环的顺序

下面哪一个是正确的训练循环顺序？

- `backward()` → `zero_grad()` → `step()` → 前向
- 前向 → 计算损失 → `zero_grad()` → `backward()` → `step()`
- 前向 → `step()` → `backward()` → `zero_grad()`
- `zero_grad()` → `step()` → 前向 → `backward()`

<details>
<summary>答案与解析</summary>

先前向算出损失，再清零旧梯度、反向传播算出新梯度，最后用梯度更新参数。`zero_grad()` 放在 `backward()` 之前的任何位置都可以（有人习惯放在循环最开头）。
</details>

<!-- exercise: no-grad -->
### 练习 6：为什么评估时用 `torch.no_grad()`？

- 能让模型预测得更准
- 不记录计算图，节省内存和时间
- 不加的话一定会报错
- 会把所有梯度变成 0

<details>
<summary>答案与解析</summary>

评估时不需要反向传播，所以没必要记录计算图。`no_grad()` 不会改变预测结果，但能明显节省显存和时间——对大模型推理尤其重要。
</details>

## 5.13 延伸阅读

- [PyTorch 官方教程](https://docs.pytorch.org/tutorials/)：从 “Learn the Basics” 开始
- 《动手学深度学习》（李沐等）：[zh.d2l.ai](https://zh.d2l.ai/)，有完整的 PyTorch 代码和中文讲解视频
- Andrej Karpathy：[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列视频
- 下一课：[训练技巧：让网络学得更好](../06-training-tricks/zh-CN.md)
