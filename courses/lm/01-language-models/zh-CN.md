# 1. 语言模型：预测下一个字

> 🎯 **本课目标**：理解「语言模型」到底在做什么——**预测下一个 token 的概率**；用概率的链式法则给一句话打分；亲手做一个**字符级二元（bigram）模型**：数数、采样生成新名字、用**困惑度**评估；再用一个最简单的神经网络学到同样的东西，为后面的 GPT 铺路。
>
> ⏱ **预计用时**：75 分钟　｜　📋 **前置知识**：第一部分第 6 课（概率与信息）；第二部分第 2 课（softmax 与交叉熵）

第一部分我们说过：ChatGPT、DeepSeek、Qwen 这些大模型，本质上都在反复做同一件事——**根据前面的内容，预测下一个 token**。从这一课开始，我们将一步步亲手造出一个这样的模型。先从最简单、最古老的方法开始。

> [!NOTE]
> 本部分的思路参考了 Andrej Karpathy 的 [makemore](https://github.com/karpathy/makemore) 和 [nanoGPT](https://github.com/karpathy/nanoGPT) 项目（MIT 许可证）及其配套视频，强烈推荐。

## 1.1 什么是语言模型？

**语言模型**（language model）是一个给文本分配概率的模型。最常用的形式是：给定前面的内容，输出**下一个 token 是各个可能值的概率**。

```text
输入：「今天天气真」
输出：好 62%　　热 11%　　冷 8%　　不 5%　　……（所有可能的字的概率加起来 = 1）
```

手机输入法的联想、搜索框的自动补全，都是小型的语言模型。大语言模型（LLM）只是**大得多**（几十亿到上万亿个参数）、**看得远得多**（能参考前面几十万个 token），并且用海量文本训练出来的。

**token**（词元）是模型处理文本的基本单位，可以是一个字、一个词或词的一部分。下一课会详细讲怎么切分。这一课为了简单，我们**以单个字符为 token**。

## 1.2 给一句话打分：概率的链式法则

一句话出现的概率，可以拆成「每一步预测下一个 token」的概率相乘：

```formula
P(w₁ w₂ … wₙ) = P(w₁) × P(w₂ | w₁) × P(w₃ | w₁ w₂) × … × P(wₙ | w₁ … wₙ₋₁)
```

`P(w₂ | w₁)` 读作「已知 w₁ 时 w₂ 的概率」（条件概率）。所以，**只要能预测下一个 token，就能给任何文本打分，也能一个接一个地生成文本**——这就是 GPT 的全部工作方式。

```python
import math

# 假设模型对「我 爱 你」每一步的预测概率如下
steps = [("我", 0.05), ("爱", 0.10), ("你", 0.60)]
p = 1.0
for token, prob in steps:
    p *= prob
    print(f"P({token} | 前文) = {prob}")
print("整句概率：", round(p, 6))
print("负对数似然（NLL）：", round(-math.log(p), 3))   # 概率越小，NLL 越大
```

长句子的概率会小到几乎为 0（很多个小于 1 的数相乘），所以实际中总是用**对数**：概率相乘变成对数相加，用**负对数似然**（negative log-likelihood，NLL）当作损失——这正是第二部分学过的交叉熵。

## 1.3 最简单的语言模型：只看前一个字符

完整的条件概率 `P(wₙ | w₁ … wₙ₋₁)` 太复杂了。最简单的近似是：**只看前一个 token**。

```formula
P(wₙ | w₁ … wₙ₋₁) ≈ P(wₙ | wₙ₋₁)
```

这叫**二元模型**（bigram）。只看前 2 个叫三元模型（trigram），一般地叫 **n-gram 模型**。n-gram 是统计语言模型的经典方法，香农（Claude Shannon）在 1948 年就用它生成过「像英语」的文字。

我们来训练一个**字符级 bigram 模型**，任务是学习英文名字的样子，然后生成新名字。数据是一百多个常见英文名。每个名字前后加一个特殊字符 `.`，表示「开头」和「结尾」，例如 `emma` 变成 `.emma.`，从中得到 5 对相邻字符：`.e`、`em`、`mm`、`ma`、`a.`。

「训练」就是**数数**：统计每个字符后面跟着各个字符的次数，再除以总数变成概率。

```python
import math
import random

NAMES = """emma olivia ava isabella sophia mia charlotte amelia harper evelyn abigail emily
elizabeth sofia avery ella madison scarlett victoria aria grace chloe camila penelope riley
layla lillian nora zoey mila aubrey hannah lily addison eleanor natalie luna savannah leah
stella hazel ellie audrey violet claire bella aurora lucy anna samantha caroline sarah maya
liam noah william james oliver benjamin elijah lucas mason logan alexander ethan jacob
michael daniel henry jackson sebastian aiden matthew samuel david joseph carter owen wyatt
john jack luke dylan levi isaac gabriel julian mateo anthony lincoln joshua andrew caleb
ryan nathan thomas leo isaiah charles hudson hunter connor eli ezra aaron adrian nolan""".split()

chars = list(".abcdefghijklmnopqrstuvwxyz")    # "." 表示名字的开头和结尾
print(len(NAMES), "个名字，", len(chars), "种字符：", "".join(chars))

# 1. 数一数：每个字符后面跟着各个字符的次数
counts = {a: {b: 0 for b in chars} for a in chars}
for name in NAMES:
    seq = ["."] + list(name) + ["."]
    for a, b in zip(seq, seq[1:]):
        counts[a][b] += 1

# 2. 变成概率：P(b | a) = count(a, b) / count(a, 任意)；每个计数先加 0.1（平滑）
probs = {}
for a in chars:
    total = sum(counts[a][b] + 0.1 for b in chars)
    probs[a] = {b: (counts[a][b] + 0.1) / total for b in chars}

top = sorted(probs["."].items(), key=lambda kv: -kv[1])[:5]
print("名字最常见的开头：", [(b, round(p, 3)) for b, p in top])

# 3. 采样：从 "." 出发，按概率一个字符一个字符地「接龙」，直到又抽到 "."
random.seed(7)
def sample():
    out, cur = "", "."
    while True:
        cur = random.choices(chars, weights=[probs[cur][b] for b in chars])[0]
        if cur == ".":
            return out
        out += cur
print("生成的名字：", [sample() for _ in range(10)])

# 4. 评估：平均负对数似然（NLL）与困惑度（perplexity）
def avg_nll(words):
    nll, n = 0.0, 0
    for w in words:
        seq = ["."] + list(w) + ["."]
        for a, b in zip(seq, seq[1:]):
            nll += -math.log(probs[a][b])
            n += 1
    return nll / n

loss = avg_nll(NAMES)
print(f"训练集：平均 NLL = {loss:.3f}，困惑度 = {math.exp(loss):.2f}")
print(f"瞎猜（均匀分布）：平均 NLL = {math.log(len(chars)):.3f}，困惑度 = {len(chars)}")
print(f"新名字 'sophie'：{avg_nll(['sophie']):.3f}   乱码 'zqxj'：{avg_nll(['zqxj']):.3f}")
```

逐段看看这个程序做了什么。

### 采样（生成）

生成文本的方法叫**自回归**（autoregressive）：从开头符号出发，按模型给出的概率**随机抽**一个字符，把它接上，再以它为条件抽下一个……直到抽到结尾符号。`random.choices(候选, weights=概率)` 就是「按概率抽签」。GPT 生成回答时也是这样，一个 token 接一个 token。

生成的名字有些像模像样，有些很奇怪——因为模型**只记得前一个字符**，不知道整个名字已经多长、前面用过什么音节。要做得更好，就需要看更长的上下文。

### 评估：困惑度

怎么衡量一个语言模型好不好？看它给**真实文本**分配的概率高不高：用平均 NLL（越低越好），或者它的指数——**困惑度**（perplexity）：

```formula
困惑度 = e^(平均 NLL)
```

困惑度可以直观地理解为：**模型在每一步「平均在几个选项之间犹豫」**。完全瞎猜（27 个字符等概率）的困惑度就是 27；我们的 bigram 模型约为 9，说明它确实学到了一些规律。乱码 `zqxj` 的 NLL 远高于像名字的 `sophie`。

> [!NOTE]
> 困惑度是语言模型最经典的评估指标，至今仍用于比较预训练模型（不同的分词方式会影响它，所以只能在相同分词下比较）。

### 平滑：为什么要加 0.1？

如果某对字符在训练数据里从没出现过（比如 `qx`），它的计数是 0，概率就是 0，`−log(0)` 是**无穷大**——只要遇到一次，整个评估就爆掉了。给每个计数加一个小数（这叫**平滑**，smoothing），就能让「没见过」的组合也有一点点概率。

## 1.4 用神经网络做同样的事

数数很简单，但它没法扩展：如果要看前 10 个字符，可能的组合有 27¹⁰ ≈ 2×10¹⁴ 种，绝大多数从没出现过，根本数不过来。**神经网络**的办法是：不去数，而是**学一个函数**，输入上下文，输出下一个字符的概率。

先用神经网络重做 bigram，看它能否学到和数数一样的结果：

- 输入：当前字符（编号 0～26），表示成 **one-hot 向量**（第 i 位是 1，其余是 0）；
- 模型：一个 27×27 的权重矩阵 `W`；one-hot 向量乘以 `W`，结果正好是 `W` 的第 i 行，作为 **logits**；
- 输出：对 logits 做 softmax，得到下一个字符的概率；
- 损失：交叉熵（就是上面的平均 NLL）；用梯度下降训练。

```python
import math
import random

NAMES = """emma olivia ava isabella sophia mia charlotte amelia harper evelyn abigail emily
elizabeth sofia avery ella madison scarlett victoria aria grace chloe camila penelope riley
layla lillian nora zoey mila aubrey hannah lily addison eleanor natalie luna savannah leah
stella hazel ellie audrey violet claire bella aurora lucy anna samantha caroline sarah maya
liam noah william james oliver benjamin elijah lucas mason logan alexander ethan jacob
michael daniel henry jackson sebastian aiden matthew samuel david joseph carter owen wyatt
john jack luke dylan levi isaac gabriel julian mateo anthony lincoln joshua andrew caleb
ryan nathan thomas leo isaiah charles hudson hunter connor eli ezra aaron adrian nolan""".split()

chars = list(".abcdefghijklmnopqrstuvwxyz")
idx = {c: i for i, c in enumerate(chars)}          # 字符 → 编号
V = len(chars)

pairs = []                                          # 训练数据：(当前字符编号, 下一个字符编号)
for name in NAMES:
    seq = ["."] + list(name) + ["."]
    pairs += [(idx[a], idx[b]) for a, b in zip(seq, seq[1:])]
print("训练样本数：", len(pairs))

# 模型：一个 V×V 的权重矩阵。输入字符 a 的 one-hot 向量乘以 W，正好取出第 a 行——这一行就是 logits
random.seed(0)
W = [[random.gauss(0, 0.1) for _ in range(V)] for _ in range(V)]

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

lr = 20.0
for epoch in range(1, 201):
    grad = [[0.0] * V for _ in range(V)]
    loss = 0.0
    for a, b in pairs:
        p = softmax(W[a])                          # 前向：下一个字符的概率分布
        loss += -math.log(p[b])                    # 交叉熵
        for j in range(V):                         # 反向：softmax + 交叉熵的梯度 = p − one_hot(b)
            grad[a][j] += p[j] - (1.0 if j == b else 0.0)
    for i in range(V):                             # 梯度下降（梯度取平均）
        for j in range(V):
            W[i][j] -= lr * grad[i][j] / len(pairs)
    if epoch in (1, 10, 50, 100, 200):
        print(f"epoch {epoch:3d}  平均 NLL {loss / len(pairs):.3f}")

p = softmax(W[idx["a"]])
best = sorted(range(V), key=lambda j: -p[j])[:3]
print("网络学到的 P(下一个 | 'a')：", [(chars[j], round(p[j], 3)) for j in best])
```

损失从约 3.3（瞎猜）降到约 2.14，和数数法几乎一样；网络学到的概率表也和计数得到的一致。**两种方法学到的是同一个东西**，区别在于：

| | 数数（n-gram） | 神经网络 |
|---|---|---|
| 「训练」 | 统计次数 | 梯度下降 |
| 上下文变长 | 组合数指数爆炸，数据不够用 | 参数量可控，可以接收任意长的上下文 |
| 泛化 | 没见过的组合只能靠平滑 | 相似的上下文会得到相似的预测 |

> [!TIP]
> 有趣的对应：神经网络版本里给权重加 L2 正则化（权重衰减），效果就相当于计数版本里的平滑——都是让概率分布不要太「极端」。

## 1.5 从 n-gram 到 GPT：一段简史

| 年份 | 里程碑 | 关键思想 |
|------|--------|----------|
| 1948 | 香农的 n-gram 实验 | 用统计频率预测下一个字母 / 单词 |
| 1980s–2000s | 统计 n-gram 模型 | 语音识别、机器翻译的核心组件 |
| 2003 | Bengio 的神经概率语言模型 | 用**词向量** + 神经网络代替计数（下一节课会讲词向量） |
| 2010s | RNN / LSTM | 用循环结构「记住」很长的上下文 |
| 2017 | **Transformer**（《Attention Is All You Need》） | 用**注意力**一次看到所有上下文，可以高度并行训练 |
| 2018– | GPT、BERT 系列 | 在海量文本上预训练巨大的 Transformer |
| 2022– | ChatGPT 之后的 LLM 时代 | 指令微调、对齐、推理模型…… |

接下来几课，我们会沿着这条路线一步步走：**分词**（第 2 课）→ **词向量**（第 3 课）→ **注意力**（第 4 课）→ **Transformer**（第 5 课）→ 亲手训练一个**小 GPT**（第 6 课）→ **解码策略**（第 7 课）。

## 1.6 小结

- 语言模型 = 预测**下一个 token 的概率分布**；用链式法则可以给整段文本打分，用**自回归采样**可以生成文本。
- 实际计算用**负对数似然**（NLL，即交叉熵）；**困惑度** = e^(平均 NLL)，可以理解为「每一步在几个选项之间犹豫」，越低越好。
- **n-gram** 模型只看前 n−1 个 token，靠计数训练；没见过的组合需要**平滑**；上下文一长就数不过来。
- 一个 one-hot 输入 + 线性层 + softmax 的神经网络，用梯度下降能学到与计数完全相同的 bigram 模型——而神经网络可以扩展到长得多的上下文。

## 1.7 练习

<!-- exercise: perplexity -->
### 练习 1：计算困惑度

输入一行若干个概率，表示模型在一句话每一步给正确 token 的概率。输出三个数（空格隔开）：整句概率（保留 6 位小数）、平均 NLL（保留 4 位小数）、困惑度（保留 4 位小数）。

```text
样例输入：0.5 0.25 0.5
样例输出：0.062500 0.9242 2.5198
```

<details>
<summary>提示</summary>

平均 NLL = `-sum(math.log(p) for p in ps) / len(ps)`，困惑度 = `math.exp(平均 NLL)`。
</details>

<!-- exercise: bigram-next -->
### 练习 2：最可能的下一个字符

第一行是若干个单词（空格隔开），第二行是一个字符 `c`。像本课一样，在每个单词前后加上 `.`，统计相邻字符对。输出**跟在 `c` 后面次数最多**的字符；如果有并列，输出其中按字符顺序最小的那个（`.` 排在所有字母前面）。

```text
样例输入：
emma ella eva
e
样例输出：l
```

<details>
<summary>提示</summary>

`e` 后面分别跟着 `m`、`l`、`v` 各一次，并列时取最小的 `l`。可以用 `sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]`。
</details>

<!-- exercise: why-weird -->
### 练习 3：为什么生成的名字很奇怪？

bigram 模型生成的名字常常很长或者读不通，最主要的原因是？

- 训练数据里有错别字
- 模型只看前一个字符，不知道更早的内容（比如名字已经多长）
- 随机种子选得不好
- 字符表里少了大写字母

<details>
<summary>答案与解析</summary>

bigram 只根据前一个字符预测下一个，完全「忘记」了更早的内容，所以无法保证整体像一个名字。看更长的上下文是改进的关键——这正是后面的神经网络和 Transformer 要做的。
</details>

<!-- exercise: zero-prob -->
### 练习 4：没有平滑会怎样？

如果不做平滑，而测试文本里出现了一个训练集中从未出现过的字符对，那么测试集的平均 NLL 会是？

- 0
- 无穷大
- 不受影响
- 变成负数

<details>
<summary>答案与解析</summary>

没出现过的组合概率为 0，`−log(0) = ∞`，平均值也就是无穷大（困惑度无穷大）。这就是需要平滑的原因。
</details>

<!-- exercise: uniform-ppl -->
### 练习 5：瞎猜的困惑度

一个「什么都没学」的模型，对词表里 50000 个 token 都给出相同的概率。它的困惑度是多少？

- 1
- 约 10.8（即 ln 50000）
- 50000
- 25000

<details>
<summary>答案与解析</summary>

每个 token 的概率都是 1/50000，NLL = ln 50000，困惑度 = e^(ln 50000) = 50000。均匀分布的困惑度就等于选项个数——这也是「困惑度 = 在几个选项之间犹豫」的由来。
</details>

## 1.8 延伸阅读

- Andrej Karpathy：[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)（本课的灵感来源）
- Claude Shannon：《A Mathematical Theory of Communication》（1948），第 3 节用 n-gram 生成英文
- 《动手学深度学习》：[语言模型和数据集](https://zh.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html)
- 下一课：[分词：文本怎样变成 token](../02-tokenization/zh-CN.md)
