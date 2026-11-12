# 4. 注意力机制：让每个 token「看到」上下文

> 🎯 **本课目标**：明白为什么需要注意力；从「求平均」一步步推导出**自注意力**；理解 **Query、Key、Value** 的直觉（一次「模糊的查字典」）；掌握缩放点积注意力公式 `softmax(QKᵀ/√d)·V`、**因果掩码**和**多头注意力**；用纯 Python 和 PyTorch 各实现一遍；理解注意力的 O(n²) 代价，以及 **RoPE** 旋转位置编码为什么能表示相对位置。
>
> ⏱ **预计用时**：100 分钟　｜　📋 **前置知识**：本部分第 1～3 课；第一部分第 4 课（矩阵乘法）

这是整门课最核心的一课。2017 年的论文《Attention Is All You Need》提出的 **Transformer**，几乎完全由注意力构成；今天所有的大模型，GPT、Claude、Llama、Qwen、DeepSeek，都建立在它之上。慢慢读，每段代码都运行一下。

> [!NOTE]
> 本课的推导顺序参考了 Andrej Karpathy 的视频 [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)，强烈推荐配合观看。

## 4.1 为什么需要注意力？

上一课的神经语言模型只看**固定的前 3 个**字符，然后把它们拼起来。这有两个问题：

1. **窗口固定**：想看前 1000 个 token，输入就要拼成 1000 倍长，参数也跟着暴增；
2. **一视同仁**：每个位置都按固定方式处理，但真正有用的信息可能在任何位置。

看这句话：「小明把书还给了小红，因为**她**明天要考试。」要理解「她」指谁，模型需要**回头去找**前面相关的词（小红），而且这个词可能隔得很远。我们希望：**每个 token 都能根据需要，从前面所有 token 中「挑出」相关的信息，汇总到自己身上。** 这就是**注意力**（attention）。

## 4.2 第一步：把前面的信息平均起来

最简单的「汇总前文」方法：第 t 个位置的输出 = 前 t 个位置（包括自己）向量的**平均值**。

```python
# 4 个 token，每个是 2 维向量（假装是嵌入）
X = [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [1.0, 1.0]]

# 第 t 个位置的输出 = 前 t+1 个向量（包括自己）的平均
out = []
for t in range(len(X)):
    seen = X[:t + 1]
    out.append([sum(v[i] for v in seen) / len(seen) for i in range(2)])
for t, v in enumerate(out):
    print(f"位置 {t}: 平均了 {t + 1} 个 token →", [round(a, 3) for a in v])
```

注意：每个位置**只能看自己和之前的位置**，不能看后面的——因为语言模型要预测下一个 token，如果能「偷看」后面的答案，就学不到东西了。这叫**因果**（causal）约束。

平均有个明显的缺点：所有前文的**权重都一样**。我们想让相关的 token 权重大、无关的权重小，而且权重**取决于内容**。

## 4.3 第二步：按相关程度加权——Query、Key、Value

把注意力想象成一次**模糊的查字典**（或者搜索引擎）：

- **Query（查询）**：「我想找什么？」——当前 token 发出的问题；
- **Key（键）**：「我这里有什么？」——每个 token 贴出的标签；
- **Value（值）**：「如果你选中我，我给你什么」——每个 token 提供的内容。

用 query 和每个 key 做**点积**得到匹配分数，用 **softmax** 把分数变成加起来等于 1 的权重，再用权重对 value 做**加权平均**：

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# 一个「迷你搜索引擎」：每条资料有一个 key（标签）和一个 value（内容）
keys   = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
values = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
query  = [1.0, 0.2]                          # 我想找的东西

scores = [dot(query, k) for k in keys]       # 和每个 key 有多匹配
weights = softmax(scores)                    # 变成加起来等于 1 的权重
output = [sum(w * v[i] for w, v in zip(weights, values)) for i in range(2)]
print("匹配分数：", [round(s, 3) for s in scores])
print("注意力权重：", [round(w, 3) for w in weights])
print("输出（各 value 的加权平均）：", [round(o, 3) for o in output])
```

普通的字典要求「完全匹配」才取出一项；注意力是「**按匹配程度取出所有项的混合**」，而且这个过程处处可导，可以用梯度下降训练。

**自注意力**（self-attention）中，query、key、value **都来自同一串 token 本身**：每个 token 的嵌入 `x` 分别乘以三个可学习的矩阵，得到它的 q、k、v：

```formula
q = x·W_Q　　k = x·W_K　　v = x·W_V
```

模型通过训练自己学会：该发出什么样的「问题」，该贴什么样的「标签」，被选中时该提供什么「内容」。

## 4.4 缩放点积注意力

把所有 token 的 q、k、v 叠成矩阵 Q、K、V（每行一个 token），所有位置的注意力可以用矩阵一次算完：

```formula
Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```

- `Q·Kᵀ` 是一个 n×n 的矩阵：第 i 行第 j 列 = 第 i 个 token 的 query 与第 j 个 token 的 key 的点积；
- 除以 `√d`（d 是 q、k 的维度）是**缩放**，下面解释；
- softmax 按**每一行**做，所以每个 token 的权重加起来等于 1；
- 再乘以 V：每个 token 的输出 = 所有 value 的加权平均。

### 为什么要除以 √d？

d 维随机向量的点积，大小大约随 `√d` 增长。维度一大，分数的差距就大，softmax 会变得非常「尖」：几乎所有权重都压到一个 token 上，其他 token 的梯度接近 0，模型很难学。除以 `√d` 把分数拉回合适的范围：

```python
import math
import random

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

random.seed(0)
for d in (4, 64, 1024):
    q = [random.gauss(0, 1) for _ in range(d)]
    keys = [[random.gauss(0, 1) for _ in range(d)] for _ in range(5)]
    raw = [sum(a * b for a, b in zip(q, k)) for k in keys]
    scaled = [s / math.sqrt(d) for s in raw]
    print(f"d={d:4d}  不缩放时最大的权重 {max(softmax(raw)):.3f}   除以 √d 后 {max(softmax(scaled)):.3f}")
```

### 因果掩码

语言模型里，第 i 个 token 只能看位置 ≤ i 的 token。做法是在 softmax **之前**把「未来」位置的分数设成 **−∞**：e^(−∞) = 0，softmax 之后这些位置的权重就是 0。权重矩阵因此是**下三角**的。

## 4.5 纯 Python 实现完整的因果自注意力

```python
import math
import random

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def matmul(A, B):                       # (n×k) @ (k×m)
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

random.seed(0)
tokens = ["我", "爱", "大", "模型"]
n, d = len(tokens), 4
X = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]          # 4 个 token 的嵌入（n×d）
Wq = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]       # 三个可学习的矩阵（d×d）
Wk = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]
Wv = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]

Q, K, V = matmul(X, Wq), matmul(X, Wk), matmul(X, Wv)                   # 每个 token 的 query、key、value
scores = matmul(Q, [list(c) for c in zip(*K)])                          # Q @ Kᵀ：n×n 的匹配分数
scale = 1 / math.sqrt(d)
weights = []
for i in range(n):
    row = [scores[i][j] * scale if j <= i else float("-inf") for j in range(n)]   # 因果掩码：看不到未来
    weights.append(softmax(row))
out = matmul(weights, V)                                                # 加权平均 value

print("注意力权重（第 i 行 = 第 i 个 token 在看谁）：")
print("        " + "".join(f"{t:>8}" for t in tokens))
for t, row in zip(tokens, weights):
    print(f"{t:>6}  " + "".join(f"{w:8.3f}" for w in row))
print("输出形状：", len(out), "×", len(out[0]))
```

权重矩阵是下三角的，每一行加起来等于 1。现在的权重是随机矩阵算出来的，没有意义；**训练之后**，模型会学到有用的模式，例如代词去「看」它指代的名词，动词去「看」它的主语。

> [!TIP]
> 注意力的输出形状和输入一样（n 个 token，每个 d 维），所以可以一层接一层地叠起来。每个 token 的向量经过一层注意力后，就**融入了上下文的信息**：同一个「苹果」，在「我吃了一个苹果」和「苹果发布了新手机」里，经过注意力后得到的向量是不同的。

## 4.6 多头注意力

一个 token 可能同时需要好几种信息：它的主语是谁、前一个词是什么、这句话在讲什么主题……一组 Q、K、V 只能算出**一种**权重分布。**多头注意力**（multi-head attention）就是并行做 h 次注意力：

1. 把 d 维切成 h 份，每个「头」用自己的 Q、K、V（每份 d/h 维）独立地做注意力；
2. 把 h 个头的输出**拼接**回 d 维；
3. 再乘一个输出矩阵 `W_O` 混合各头的信息。

计算量和单头差不多，但模型可以同时关注不同的关系。研究发现，训练好的模型里确实有专门的头：有的只看前一个 token，有的负责「复制」前文出现过的模式（**归纳头**，induction head）……

下面用 PyTorch 实现一个多头因果自注意力，结构和 GPT-2 的代码几乎一样：

```python requires=torch
import math
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads          # 头数、每个头的维度
        self.qkv = nn.Linear(d_model, 3 * d_model)             # 一次算出 Q、K、V
        self.proj = nn.Linear(d_model, d_model)                # 输出投影 W_O

    def forward(self, x):
        B, T, C = x.shape                                      # batch、序列长度、维度
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, h, T, dh)：把 C 维切成 h 个头，每个头单独做注意力
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.dh)    # (B, h, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # 下三角：只能看自己和之前
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                            # (B, h, T, dh)
        y = y.transpose(1, 2).reshape(B, T, C)                 # 把各个头拼回来
        return self.proj(y), att

torch.manual_seed(0)
attn = CausalSelfAttention(d_model=32, n_heads=4)
x = torch.randn(2, 5, 32)                                      # 2 个序列，每个 5 个 token，每个 32 维
y, att = attn(x)
print("输出：", tuple(y.shape), " 注意力权重：", tuple(att.shape))
print("第 1 个序列、第 1 个头的注意力权重：")
print(att[0, 0].detach().round(decimals=2))
print("参数量：", sum(p.numel() for p in attn.parameters()))   # 4 × 32 × 32 + 4 × 32

# PyTorch 自带的高效实现（会自动选用 FlashAttention 等算法），结果应该一样
q, k, v = attn.qkv(x).split(32, dim=2)
q, k, v = (t.view(2, 5, 4, 8).transpose(1, 2) for t in (q, k, v))
fast = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 5, 32)
print("和 F.scaled_dot_product_attention 一致：", torch.allclose(attn.proj(fast), y, atol=1e-6))
```

多头注意力的参数：`W_Q`、`W_K`、`W_V`、`W_O` 四个 d×d 矩阵，共 **4d²** 个（加上偏置 4d 个）。GPT-2 small（d = 768）每层注意力约 236 万个参数。

## 4.7 注意力的代价：O(n²)

注意力分数矩阵是 **n×n** 的：序列长度翻倍，计算量和内存翻**四**倍。100 万个 token 的上下文就有 10¹² 个分数——这就是「长上下文」昂贵的原因。研究者想了很多办法：

| 技术 | 思路 | 使用者 |
|------|------|--------|
| **FlashAttention** | 不把 n×n 矩阵完整存下来，分块在 GPU 高速缓存里算完，结果完全相同但快得多、省内存 | 几乎所有现代训练和推理框架 |
| **KV 缓存** | 生成时把之前算过的 K、V 存起来，每生成一个新 token 只算新的那一行 | 所有大模型推理（第六部分详细讲） |
| **GQA / MQA** | 多个 query 头**共用**一组 K、V 头，KV 缓存小好几倍 | Llama 3、Qwen 等 |
| **MLA**（多头潜在注意力） | 把 K、V 压缩成一个低维「潜在向量」再缓存 | DeepSeek-V2/V3 |
| **滑动窗口 / 稀疏注意力** | 每个 token 只看附近的一段，或者只挑选部分 token 去看 | Mistral、Gemma、DeepSeek 的稀疏注意力等 |
| **线性注意力 / 状态空间模型** | 用别的结构替代 softmax 注意力，代价随长度线性增长 | Mamba，以及一些混合架构 |

## 4.8 位置信息：RoPE 旋转位置编码

上一课说过：注意力本身不知道顺序——把 token 打乱，每个 token 的权重分布不变。GPT-2 的做法是给输入**加上**位置嵌入。现代模型（Llama、Qwen、DeepSeek、Mistral……）几乎都用 **RoPE**（Rotary Position Embedding，旋转位置编码，苏剑林等 2021 年提出）：

- 把 q、k 向量的维度两两分组，每组看成平面上的一个点；
- 位置为 m 的 token，把它的每组都**旋转** `m × θ` 角度（不同的组转速 θ 不同）；
- 两个向量都旋转后再做点积，结果只和**角度差**——也就是**相对位置 m − n**——有关。

```python
import math

def rotate(v, angle):                      # 把二维向量逆时针旋转 angle 弧度
    x, y = v
    return [x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)]

theta = 0.5                                 # 每往后一个位置，多转 0.5 弧度
q, k = [1.0, 0.5], [0.3, 1.0]
for m, n in [(5, 3), (10, 8), (102, 100), (3, 5), (7, 7)]:
    qm, kn = rotate(q, m * theta), rotate(k, n * theta)   # q 在位置 m，k 在位置 n
    score = qm[0] * kn[0] + qm[1] * kn[1]
    print(f"q 在位置 {m:3d}，k 在位置 {n:3d}，相距 {m - n:2d}：q·k = {score:.4f}")
```

只要相距一样（都是 2），不管在第 5 位还是第 102 位，注意力分数都相同。这正是我们想要的：「前一个词」「三个词之前」这样的**相对关系**才是语言中重要的。真实的 RoPE 中，第 i 组的转速是 `θᵢ = base^(−2i/d)`（base 通常是 10000，Llama 3 用 500000），既有转得快的组（区分近处的位置），也有转得很慢的组（区分远处的位置）。调整 base 等参数（如 YaRN 方法）还能让模型处理比训练时更长的上下文。

## 4.9 小结

- 注意力让每个 token 根据内容，从（之前的）所有 token 中**加权汇总**信息；权重由 query 与 key 的匹配程度决定。
- 自注意力：`q = xW_Q, k = xW_K, v = xW_V`；`Attention = softmax(QKᵀ/√d)·V`；除以 `√d` 防止 softmax 过尖。
- **因果掩码**：softmax 前把未来位置设为 −∞，权重矩阵是下三角的。
- **多头注意力**：h 个头并行关注不同关系，拼接后乘 `W_O`；参数约 4d²。
- 代价随序列长度 **O(n²)** 增长；FlashAttention、KV 缓存、GQA/MLA、稀疏注意力等都是为了降低代价。
- **RoPE** 按位置旋转 q、k，使注意力分数只依赖相对位置，是现代大模型的标准做法。

## 4.10 练习

<!-- exercise: attn-weights -->
### 练习 1：注意力权重

第一行是 query 向量，第二行是整数 `n`，接下来 `n` 行是 `n` 个 key 向量（和 query 一样长）。输出缩放点积注意力的权重 `softmax(q·kⱼ / √d)`（d 是向量长度），保留 4 位小数，空格隔开。

```text
样例输入：
1 0
2
1 0
0 1
样例输出：0.6698 0.3302
```

<details>
<summary>提示</summary>

分数 `s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q)) for k in keys]`，再做 softmax（记得先减去最大值，防止溢出）。
</details>

<!-- exercise: causal-mask -->
### 练习 2：因果掩码

输入整数 `n`，输出 `n×n` 的因果掩码：第 i 行第 j 列，如果第 i 个 token 能看到第 j 个 token（j ≤ i）就是 `1`，否则是 `0`。每行的数字用空格隔开。

```text
样例输入：3
样例输出：
1 0 0
1 1 0
1 1 1
```

<details>
<summary>提示</summary>

两层循环，`1 if j <= i else 0`；或者 `" ".join("1" if j <= i else "0" for j in range(n))`。
</details>

<!-- exercise: attn-output -->
### 练习 3：注意力的输出

第一行是 query 向量，第二行是整数 `n`，接下来 `n` 行是 key 向量，再接下来 `n` 行是 value 向量。计算缩放点积注意力的输出（各 value 按权重的加权平均），保留 4 位小数，空格隔开。

```text
样例输入：
1 0
2
1 0
0 1
10 0
0 10
样例输出：6.6976 3.3024
```

<details>
<summary>提示</summary>

先像练习 1 一样算出权重 `w`，然后输出向量的第 i 个分量是 `sum(wj * v[i] for wj, v in zip(w, values))`。
</details>

<!-- exercise: mha-params -->
### 练习 4：多头注意力的参数量

输入一行两个整数 `d bias`：模型维度，以及是否有偏置（1 有，0 没有）。输出一层多头注意力（`W_Q`、`W_K`、`W_V`、`W_O` 四个 d×d 矩阵，有偏置时每个矩阵再加 d 个偏置）的参数总数。注意：参数量和头数无关。

```text
样例输入：768 1
样例输出：2362368
```

<details>
<summary>提示</summary>

`4 * d * d + (4 * d if bias else 0)`
</details>

<!-- exercise: why-scale -->
### 练习 5：为什么除以 √d？

- 让计算更快
- 维度越大点积的数值越大，softmax 会变得很「尖」、梯度接近 0；除以 √d 把分数拉回合适范围
- 为了让权重加起来等于 1
- 为了实现因果掩码

<details>
<summary>答案与解析</summary>

d 维随机向量点积的标准差约为 √d。不缩放的话，高维时 softmax 几乎变成 one-hot，大多数位置得不到梯度。「加起来等于 1」是 softmax 本身保证的。
</details>

<!-- exercise: quadratic -->
### 练习 6：上下文翻倍

其他条件不变，把序列长度从 4000 个 token 增加到 8000 个，注意力分数矩阵的元素个数会变成原来的几倍？

- 1 倍（不变）
- 2 倍
- 4 倍
- 8 倍

<details>
<summary>答案与解析</summary>

分数矩阵是 n×n 的：8000² / 4000² = 4。这就是注意力代价随长度「平方增长」的含义，也是长上下文昂贵的原因。
</details>

## 4.11 延伸阅读

- Vaswani 等：《Attention Is All You Need》（2017）——Transformer 的原始论文
- Jay Alammar：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（经典图解，网上有中文翻译）
- 3Blue1Brown：[Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention)（深度学习系列第 6 集；在 B 站搜索「3Blue1Brown 注意力机制」可以找到官方中文版）
- 苏剑林：《Transformer 升级之路：博采众长的旋转式位置编码》（RoPE 作者的中文博客，科学空间 kexue.fm）
- 下一课：[Transformer：把积木搭起来](../05-transformer/zh-CN.md)
