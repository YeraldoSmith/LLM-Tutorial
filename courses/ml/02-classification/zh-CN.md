# 2. 分类与 softmax 回归

> 🎯 **本课目标**：从「预测数字」进入「预测类别」；理解 sigmoid 和逻辑回归；把二分类推广到多分类（softmax 回归）；记住一个非常重要的结论：softmax + 交叉熵的梯度就是「预测概率 − 正确答案」；用纯 Python 训练一个分类器并计算准确率。
>
> ⏱ **预计用时**：60 分钟　｜　📋 **前置知识**：第一部分第 6 课（softmax、交叉熵），本部分第 1 课

上一课的模型输出一个**数**（房价）。但很多问题要输出的是**类别**：这封邮件是不是垃圾邮件？这张图是猫、狗还是鸟？以及最重要的——**下一个 token 是词表里的哪一个？** 语言模型本质上就是一个有几万个类别的分类器。

## 2.1 二分类：输出一个概率

判断一封邮件是否是垃圾邮件，我们希望模型输出「是垃圾邮件的**概率**」——一个 0 到 1 之间的数。但线性模型 `z = w·x + b` 的输出可以是任何实数。需要一个函数把它「压」到 0～1 之间，这就是 **sigmoid** 函数：

```formula
σ(z) = 1 / (1 + e^(−z))
```

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

for z in [-6, -2, 0, 2, 6]:
    print(f"σ({z:>2}) = {sigmoid(z):.4f}")
```

```text
       1 ┤                 ●●●●●
         │              ●●
     0.5 ┤ - - - - - - ●- - - - -     z 很大 → 接近 1
         │          ●●                 z = 0  → 0.5
       0 ┤ ●●●●●●●●                    z 很小 → 接近 0
         └────────────┬───────────
                      0
```

「线性模型 + sigmoid」就是**逻辑回归**（logistic regression）——名字里有「回归」，做的却是分类。

### 二元交叉熵

对二分类，正确答案 `y` 是 0 或 1，模型给出 `p = σ(z)`。损失是第一部分第 6 课的交叉熵：

```formula
L = −[ y·log p + (1 − y)·log(1 − p) ]
```

`y = 1` 时就是 `−log p`，`y = 0` 时就是 `−log(1 − p)`——依然是「正确答案的惊讶度」。

## 2.2 多分类：softmax 回归

如果有 `K` 个类别（比如手写数字 0～9 共 10 类，或者语言模型的 5 万个 token），就让模型输出 `K` 个分数（logits），再用 **softmax** 变成概率分布：

```text
输入 x ──► 线性层：K 个分数 z = W·x + b ──► softmax ──► K 个概率 ──► 交叉熵损失
           W 的形状是 K × d                     （加起来是 1）     −log p[正确类别]
```

这叫 **softmax 回归**（也叫多项逻辑回归）。语言模型的**最后一层**正是这个结构：把 Transformer 输出的向量乘以一个 `V × d` 的矩阵，得到 `V` 个 logits，再做 softmax。

## 2.3 最重要的梯度：p − y

softmax + 交叉熵的组合，对 logits 的梯度**简单得惊人**：

```formula
∂L/∂zₖ = pₖ − yₖ
```

其中 `p` 是 softmax 输出的概率，`y` 是正确答案的 **one-hot 向量**（正确类别处为 1，其余为 0）。

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

z = [2.0, 1.0, 0.1]          # 3 个类别的分数
target = 0                    # 正确答案是第 0 类
p = softmax(z)
y = [1 if k == target else 0 for k in range(len(z))]     # one-hot：[1, 0, 0]
grad = [pk - yk for pk, yk in zip(p, y)]
print("概率：", [round(v, 4) for v in p])
print("梯度：", [round(v, 4) for v in grad])
```

直觉：

- 对**正确类别**，梯度 `p − 1` 是负数 → 梯度下降会**增大**它的分数；
- 对**错误类别**，梯度 `p` 是正数 → 梯度下降会**减小**它的分数；
- 预测越准（正确类别的 p 越接近 1），梯度越小，调整就越少。

> [!IMPORTANT]
> 记住 **p − y**。训练语言模型时，每个位置的输出层梯度都是这样算的：把模型给出的概率分布，减去「正确 token 处为 1」的向量。

用数值导数验证一下这个公式（第一部分第 5 课的方法）：

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def loss(z, target):
    return -math.log(softmax(z)[target])

z, target, h = [2.0, 1.0, 0.1], 0, 1e-5
numeric = []
for k in range(len(z)):
    zp = z[:]; zp[k] += h
    zm = z[:]; zm[k] -= h
    numeric.append((loss(zp, target) - loss(zm, target)) / (2 * h))
print("数值梯度：", [round(v, 4) for v in numeric])
```

两种方法结果一致。

## 2.4 动手：训练一个二维分类器

我们生成两团点：类别 0 在左下，类别 1 在右上；训练一个 softmax 回归来区分它们。

```python
import math
import random
random.seed(0)

# 数据：每个样本是 (x1, x2)，标签 0 或 1
data = [((random.gauss(-1, 0.7), random.gauss(-1, 0.7)), 0) for _ in range(50)]
data += [((random.gauss(1, 0.7), random.gauss(1, 0.7)), 1) for _ in range(50)]

K, D = 2, 2
W = [[0.0] * D for _ in range(K)]      # K × D
b = [0.0] * K
lr = 0.5

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def forward(x):
    z = [sum(W[k][d] * x[d] for d in range(D)) + b[k] for k in range(K)]
    return softmax(z)

for epoch in range(1, 31):
    gW = [[0.0] * D for _ in range(K)]
    gb = [0.0] * K
    total_loss = 0.0
    for x, label in data:
        p = forward(x)
        total_loss += -math.log(p[label])
        for k in range(K):
            g = p[k] - (1 if k == label else 0)        # p − y
            gb[k] += g
            for d in range(D):
                gW[k][d] += g * x[d]                   # 链式法则：∂z/∂W = x
    n = len(data)
    for k in range(K):
        b[k] -= lr * gb[k] / n
        for d in range(D):
            W[k][d] -= lr * gW[k][d] / n
    if epoch % 10 == 0:
        acc = sum(max(range(K), key=lambda k: forward(x)[k]) == label for x, label in data) / n
        print(f"第 {epoch} 轮：平均损失 {total_loss / n:.4f}，准确率 {acc:.0%}")
```

几十轮之后，准确率接近 100%。**epoch**（轮）的意思是把全部训练数据完整地过一遍。

### 准确率

分类问题除了看损失，还常看**准确率**（accuracy）：预测类别（概率最大的那个，即 argmax）等于真实类别的比例。损失是训练时用来求梯度的，准确率则更直观，便于人来判断。

## 2.5 从分类到语言模型

把上面的 `K` 换成词表大小（比如 5 万），把输入 `x` 换成「前文的向量表示」，这就是语言模型最后一步在做的事：

| | 二维点分类 | 语言模型 |
|---|---|---|
| 类别数 K | 2 | 词表大小（几万～十几万） |
| 输入 x | 坐标 (x1, x2) | Transformer 对前文的理解（几千维向量） |
| 输出 | 2 个概率 | 每个 token 是下一个 token 的概率 |
| 损失 | 交叉熵 | 交叉熵（对每个位置求平均） |
| 梯度 | p − y | p − y |

## 2.6 小结

- **sigmoid** `1/(1+e^(−z))` 把任意实数压到 0～1；线性模型 + sigmoid = **逻辑回归**（二分类），损失为二元交叉熵。
- 多分类：线性层输出 K 个 logits → **softmax** → 交叉熵，叫 **softmax 回归**；语言模型的输出层就是它。
- softmax + 交叉熵对 logits 的梯度 = **p − y**（y 是 one-hot）。
- 训练时把每个样本的梯度累加求平均再更新；**epoch** 是完整过一遍数据。
- 分类常用**准确率**评估：argmax 预测与真实标签一致的比例。

## 2.7 练习

<!-- exercise: sigmoid -->
### 练习 1：sigmoid

输入一行若干个数 z，输出每个数的 sigmoid 值（保留 4 位小数，空格隔开）。

```text
样例输入：0 2 -2
样例输出：0.5000 0.8808 0.1192
```

<details>
<summary>提示</summary>

`1 / (1 + math.exp(-z))`
</details>

<!-- exercise: softmax-grad -->
### 练习 2：softmax 交叉熵的梯度

第一行是若干个 logits，第二行是正确类别的下标。输出交叉熵损失对每个 logit 的梯度（`p − y`），保留 4 位小数，空格隔开。

```text
样例输入：
2 1 0.1
0
样例输出：
-0.3410 0.2424 0.0986
```

<details>
<summary>提示</summary>

先用稳定的 softmax 求出 p，再对正确类别减 1。
</details>

<!-- exercise: accuracy -->
### 练习 3：准确率

第一行是整数 `n` 和 `K`（样本数和类别数）。接下来 `n` 行，每行先是 `K` 个模型输出的概率，最后一个数是真实类别下标。输出准确率（预测类别取概率最大的那个；若有并列，取下标较小的），用百分数表示，保留 1 位小数，例如 `75.0%`。

```text
样例输入：
4 3
0.7 0.2 0.1 0
0.1 0.3 0.6 2
0.5 0.4 0.1 1
0.2 0.5 0.3 1
样例输出：
75.0%
```

<details>
<summary>提示</summary>

每行 `vals = input().split()`，前 `K` 个是概率，最后一个是标签。预测类别：`max(range(K), key=lambda k: probs[k])`——并列时 `max` 返回第一个，正好是下标较小的。
</details>

<!-- exercise: grad-sign -->
### 练习 4：梯度的方向

用 softmax + 交叉熵训练时，**正确类别**那个 logit 的梯度 `p − 1` 一定是？

- 正数，所以它的分数会被减小
- 负数（或 0），所以梯度下降会增大它的分数
- 总是 0
- 取决于学习率

<details>
<summary>答案与解析</summary>

概率 p 不超过 1，所以 `p − 1 ≤ 0`。梯度下降减去「学习率 × 负数」，等于增大该类的分数——模型会越来越偏向正确答案。
</details>

<!-- exercise: lm-classes -->
### 练习 5：语言模型的类别

把语言模型看作一个分类器，它的「类别」是什么？

- 0 和 1
- 词表里的所有 token
- 句子的情感（正面 / 负面）
- 文章的主题

<details>
<summary>答案与解析</summary>

每一步都是在词表的所有 token 中选出「下一个」，类别数等于词表大小。
</details>

## 2.8 延伸阅读

- [《动手学深度学习》3.4 softmax 回归](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)
- CS231n：[Linear classification 讲义](https://cs231n.github.io/linear-classify/)（英文）
- 下一课：[神经网络](../03-neural-networks/zh-CN.md)
