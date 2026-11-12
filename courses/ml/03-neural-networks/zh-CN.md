# 3. 神经网络：把简单的东西叠起来

> 🎯 **本课目标**：明白线性模型的局限（XOR 问题）；理解一个「神经元」在算什么；认识 ReLU、GELU、SiLU 等激活函数并知道为什么必须要有它们；搭出一个多层感知机（MLP）并手算前向传播；学会数参数量。
>
> ⏱ **预计用时**：60 分钟　｜　📋 **前置知识**：本部分第 1、2 课；第一部分第 4 课（矩阵乘法）

## 3.1 一条直线不够用：XOR 问题

线性模型（不管是回归还是 softmax 分类）只能画**直线**（高维时是平面）来划分类别。来看一个经典难题——**异或**（XOR）：两个输入相同时输出 0，不同时输出 1。

```text
 x2
 1 ┤  ●(0,1)→1        ○(1,1)→0
   │
 0 ┤  ○(0,0)→0        ●(1,0)→1
   └──┴─────────────────┴── x1
      0                 1
```

无论怎么画一条直线，都无法把两个 ● 和两个 ○ 分开。1969 年，这个问题曾让很多人对神经网络失去信心。解决办法是：**加一层「隐藏层」，并且在中间放一个非线性的函数。**

## 3.2 神经元：加权求和 + 激活

一个**神经元**（neuron）做两件事：

```formula
输出 = 激活函数( w₁x₁ + w₂x₂ + … + b ) = f(w · x + b)
```

1. **加权求和**：和线性模型一样（点积 + 偏置）；
2. **激活函数**（activation function）：一个**非线性**的函数 `f`，决定这个神经元「激活」的程度。

许多神经元并排在一起组成一**层**（layer）；一层的输出作为下一层的输入，就组成了**神经网络**。

## 3.3 激活函数

```python
import math

def relu(x):                 # 负数变 0，正数不变
    return max(0.0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):                 # 输出在 -1 到 1 之间
    return math.tanh(x)

def gelu(x):                 # GPT 系列使用；ReLU 的「平滑版」
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

def silu(x):                 # 也叫 Swish；Llama、Qwen 等模型使用（SwiGLU 的一部分）
    return x * sigmoid(x)

print(" x     ReLU    GELU    SiLU    tanh")
for x in [-3, -1, -0.5, 0, 0.5, 1, 3]:
    print(f"{x:>4}  {relu(x):6.3f}  {gelu(x):6.3f}  {silu(x):6.3f}  {tanh(x):6.3f}")
```

| 激活函数 | 特点 | 常见于 |
|----------|------|--------|
| **ReLU** `max(0, x)` | 简单、计算快、不容易梯度消失 | 早期深度网络、CNN |
| **sigmoid** | 输出 0～1；两端梯度接近 0（梯度消失） | 二分类的输出层、门控 |
| **tanh** | 输出 -1～1 | RNN/LSTM |
| **GELU** | 平滑版 ReLU，负数处有一点点输出 | BERT、GPT-2/3 |
| **SiLU / Swish** | `x · σ(x)`，平滑 | Llama、Qwen、DeepSeek 等（SwiGLU） |

### 为什么一定要非线性？

如果没有激活函数，两层线性变换叠在一起：`W₂(W₁x) = (W₂W₁)x`——**还是一个线性变换**！叠一百层也和一层一样，依然画不出曲线。激活函数打破了这一点，让网络能表示弯曲、复杂的边界。

```python
def matvec(W, x):
    return [sum(w * v for w, v in zip(row, x)) for row in W]

W1 = [[1, 2], [3, 4]]
W2 = [[1, -1], [0, 2]]
x = [1, 1]
two_layers = matvec(W2, matvec(W1, x))
W21 = [[sum(W2[i][k] * W1[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
print(two_layers, matvec(W21, x))       # 完全一样：两层线性 = 一层线性
```

## 3.4 用两层网络解决 XOR

下面这个网络有 2 个隐藏神经元，用 ReLU 激活。权重是手工设计的（下一课我们会让它自己学）：

```text
            h1 = ReLU(x1 + x2)
x1, x2 ──►                         ──► y = h1 − 2·h2
            h2 = ReLU(x1 + x2 − 1)
```

```python
def relu(v):
    return max(0, v)

def xor_net(x1, x2):
    h1 = relu(1 * x1 + 1 * x2 + 0)
    h2 = relu(1 * x1 + 1 * x2 - 1)
    return 1 * h1 - 2 * h2

for x1 in (0, 1):
    for x2 in (0, 1):
        print(x1, x2, "→", xor_net(x1, x2))
```

隐藏层把输入「变换」到一个新的空间，在那里问题变成线性可分的了——这就是隐藏层的作用：**学习更好的表示**（representation）。

## 3.5 多层感知机（MLP）

把多个全连接层（每个输入都连到每个神经元）叠在一起，中间夹着激活函数，就是**多层感知机**（Multi-Layer Perceptron，MLP），也叫前馈网络（feed-forward network）：

```text
输入 x (d_in)
   │  线性层：h = W₁x + b₁       W₁ 形状 d_hidden × d_in
   ▼  激活：  h = ReLU(h)
隐藏层 h (d_hidden)
   │  线性层：z = W₂h + b₂       W₂ 形状 d_out × d_hidden
   ▼
输出 z (d_out) ── 分类时接 softmax
```

```python
import math
import random
random.seed(0)

def linear(W, b, x):
    return [sum(w * v for w, v in zip(row, x)) + bi for row, bi in zip(W, b)]

def relu_vec(v):
    return [max(0.0, a) for a in v]

def softmax(z):
    m = max(z)
    e = [math.exp(a - m) for a in z]
    s = sum(e)
    return [a / s for a in e]

d_in, d_hidden, d_out = 3, 4, 2
W1 = [[random.gauss(0, 0.5) for _ in range(d_in)] for _ in range(d_hidden)]
b1 = [0.0] * d_hidden
W2 = [[random.gauss(0, 0.5) for _ in range(d_hidden)] for _ in range(d_out)]
b2 = [0.0] * d_out

x = [1.0, 0.5, -1.0]
h = relu_vec(linear(W1, b1, x))      # 4 个隐藏值
z = linear(W2, b2, h)                # 2 个 logits
print("隐藏层：", [round(v, 3) for v in h])
print("概率：  ", [round(v, 3) for v in softmax(z)])
```

参数是随机初始化的，所以输出的概率还没有意义——**训练**（下一课：反向传播）会把它们调整到有用的值。

### 万能近似

理论上可以证明：只要隐藏层足够宽，一个两层 MLP 就能以任意精度逼近几乎任何连续函数（**万能近似定理**）。但实践中，**更深**（层数多）的网络往往比只是更宽的网络更高效——这就是「深度学习」中「深度」的来历。

## 3.6 数参数

一个从 `d_in` 维到 `d_out` 维的线性层有：

```formula
参数量 = d_in × d_out （权重） + d_out （偏置）
```

上面的 MLP：`3×4 + 4 = 16` 加上 `4×2 + 2 = 10`，共 26 个参数。

在 Transformer 里，每一层都有一个这样的 MLP（叫 FFN，前馈网络），通常是 `d → 4d → d`。以 GPT-2 small 为例，`d = 768`：

```python
def linear_params(d_in, d_out, bias=True):
    return d_in * d_out + (d_out if bias else 0)

d = 768
ffn = linear_params(d, 4 * d) + linear_params(4 * d, d)
print(f"GPT-2 small 每层 FFN 参数：{ffn:,}")            # 约 470 万
print(f"12 层合计：{12 * ffn:,}")                        # 约 5700 万，接近全模型 1.24 亿的一半
```

大模型参数的很大一部分都在这些 MLP 里。研究认为，MLP 层在很大程度上负责「存储知识」。

## 3.7 小结

- 线性模型只能画直线，无法解决 **XOR** 这样的问题。
- **神经元** = 加权求和 + **激活函数**；许多神经元组成**层**，层叠起来是**神经网络**。
- 常用激活：ReLU、GELU（GPT）、SiLU/Swish（Llama 等）；没有激活函数，多层线性等于一层线性。
- **MLP**：线性 → 激活 → 线性……隐藏层学习更好的**表示**；更深的网络往往更高效。
- 线性层参数量 = `d_in × d_out + d_out`；Transformer 的 FFN 通常是 `d → 4d → d`，占了大量参数。

## 3.8 练习

<!-- exercise: activations -->
### 练习 1：激活函数

输入一个数 `x`，依次输出 ReLU、sigmoid、GELU（用 `0.5·x·(1 + erf(x/√2))`）的值，每行一个，保留 4 位小数。

```text
样例输入：1
样例输出：
1.0000
0.7311
0.8413
```

<details>
<summary>提示</summary>

`math.erf` 是误差函数，`math.sqrt(2)` 是 √2。
</details>

<!-- exercise: mlp-params -->
### 练习 2：数参数

输入一行若干个整数，表示 MLP 每一层的宽度（第一个是输入维度，最后一个是输出维度）。每相邻两层之间是一个带偏置的线性层。输出总参数量。

```text
样例输入：3 4 2
样例输出：26
```

<details>
<summary>提示</summary>

`for a, b in zip(sizes, sizes[1:]): total += a * b + b`
</details>

<!-- exercise: forward -->
### 练习 3：前向传播

起始代码里给出了一个 2 → 2 → 1 的网络的权重（隐藏层用 ReLU，输出层没有激活）。输入一行两个数 `x1 x2`，输出网络的输出值，保留 2 位小数。

```text
样例输入：1 2
样例输出：1.50
```

<details>
<summary>提示</summary>

先算隐藏层：`h = [relu(W1[i][0]*x1 + W1[i][1]*x2 + b1[i]) for i in range(2)]`，再算输出：`W2[0]*h[0] + W2[1]*h[1] + b2`。
</details>

<!-- exercise: no-activation -->
### 练习 4：去掉激活函数

把一个 10 层神经网络里的所有激活函数都去掉（只剩线性层），它的表达能力相当于？

- 一个 10 倍强大的网络
- 一个单层线性模型
- 一个只有 5 层的网络
- 没有变化

<details>
<summary>答案与解析</summary>

线性变换的组合还是线性变换：`W₁₀⋯W₂W₁x = (W₁₀⋯W₁)x`。去掉非线性后，再多层也只等于一层。
</details>

<!-- exercise: xor-predict -->
### 练习 5：读代码，写输出

```python
def relu(v):
    return max(0, v)

def net(x1, x2):
    h1 = relu(x1 + x2)
    h2 = relu(x1 + x2 - 1)
    return h1 - 2 * h2

print(net(1, 1), net(0, 1), net(2, 0))
```

<details>
<summary>答案与解析</summary>

`net(1,1)`：h1 = 2，h2 = 1，输出 0；`net(0,1)`：h1 = 1，h2 = 0，输出 1；`net(2,0)`：h1 = 2，h2 = 1，输出 0。
</details>

## 3.9 延伸阅读

- 3Blue1Brown：[深度学习系列第 1 集：神经网络是什么](https://www.bilibili.com/video/BV1bx411M7Zx)（官方中文版）
- [TensorFlow Playground](https://playground.tensorflow.org/)：在浏览器里调整层数和神经元，实时看网络怎么划分数据——非常直观。
- 下一课：[反向传播与自动求导](../04-backprop-autograd/zh-CN.md)
