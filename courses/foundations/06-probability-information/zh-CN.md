# 6. 概率与信息量：模型怎么「打分」

> 🎯 **本课目标**：理解概率分布；用 softmax 把任意分数变成概率；理解为什么到处都是对数 `log`；学会交叉熵损失——几乎所有语言模型训练时最小化的那个数；用困惑度（perplexity）直观衡量模型好坏。
>
> ⏱ **预计用时**：60 分钟　｜　📋 **前置知识**：第 3～5 课

第 1 课说过：语言模型输出的是**下一个 token 的概率分布**。这一课把这句话里的每个词都弄明白，并学会给模型的预测「打分」。

## 6.1 概率分布

**概率**是 0 到 1 之间的数，表示一件事有多可能发生。把所有可能结果的概率列出来，就是一个**概率分布**，它们**加起来等于 1**。

```text
掷骰子：   1      2      3      4      5      6
概率：    1/6    1/6    1/6    1/6    1/6    1/6     （均匀分布）

模型预测「今天天气很___」的下一个字：
         好     热     冷     差     怪
概率：   0.55   0.25   0.12   0.05   0.03            （加起来是 1）
```

### 按概率抽样

有了分布，就可以**随机抽一个**——概率越大越容易被抽中。语言模型生成文字时就是这样选词的：

```python
import random
random.seed(0)

words = ["好", "热", "冷", "差", "怪"]
probs = [0.55, 0.25, 0.12, 0.05, 0.03]
samples = random.choices(words, weights=probs, k=1000)
for w in words:
    print(w, samples.count(w) / 1000)       # 抽样频率接近原来的概率
```

如果每次都选概率最大的（「好」），叫**贪心解码**（greedy decoding）；按概率随机抽，生成的文字更多样。第三部分第 7 课会专门讲各种采样策略。

## 6.2 Softmax：把分数变成概率

神经网络的最后一层输出的是一组任意的实数，叫 **logits**（可以是负数，加起来也不等于 1）。要把它们变成概率，用 **softmax**：

```formula
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

两步：① 每个数取指数 `e^z`（变成正数，并且放大差距）；② 除以总和（让它们加起来等于 1）。

```python
import math

def softmax(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

logits = [2.0, 1.0, 0.1]
probs = softmax(logits)
print([round(p, 4) for p in probs])     # [0.659, 0.2424, 0.0986]
print(sum(probs))                        # 1.0（可能有极小的浮点误差）
```

分数最高的项概率最大，但其他项也分到了一些——这比直接选最大值「温和」，所以叫 soft（软的）max。

### 数值稳定的写法

`math.exp(1000)` 会溢出（结果太大，电脑存不下）。真实的 logits 可能很大，所以实际中会**先减去最大值**——结果完全一样（分子分母同乘了 `e^(−max)`），但不会溢出：

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

print([round(p, 4) for p in softmax([1000.0, 999.0, 998.1])])   # 不会报错
```

### 温度：控制「确定程度」

把 logits 除以一个**温度**（temperature）`T` 再做 softmax：

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]

logits = [2.0, 1.0, 0.1]
for T in [0.5, 1.0, 2.0]:
    print(f"T={T}:", [round(p, 3) for p in softmax([z / T for z in logits])])
```

- `T < 1`：分布更「尖」，几乎总选最高分的，生成的内容更保守、稳定；
- `T > 1`：分布更「平」，更随机、更有创意，也更容易胡说。

你在聊天机器人的 API 里看到的 `temperature` 参数，就是这个。

## 6.3 为什么到处都是 log？

**对数**（logarithm）`log` 是指数的反运算：`e^x = y` ⇔ `x = ln y`（本课程的 `log` 都指以 e 为底的自然对数）。在概率计算中它无处不在，原因有二：

**① 把乘法变成加法。** 一句话的概率 = 每个词概率的**乘积**。100 个词，每个概率 0.1，乘起来是 10⁻¹⁰⁰——小到电脑无法精确表示（下溢）。取对数后变成**加法**：`log(a × b) = log a + log b`，数值稳定得多。

```python
import math
probs = [0.1] * 100
product = 1.0
for p in probs:
    product *= p
print(product)                              # 1e-100，再小一些就会变成 0.0
print(sum(math.log(p) for p in probs))      # -230.26：用 log 很安全
```

**② 衡量「意外程度」。** `−log p` 叫一件事的**信息量**或**惊讶度**（surprisal）：

| 概率 p | −log p | 直觉 |
|--------|--------|------|
| 1.0 | 0 | 完全意料之中，毫不惊讶 |
| 0.5 | 0.69 | 抛硬币的结果 |
| 0.1 | 2.30 | 有点意外 |
| 0.01 | 4.61 | 很意外 |
| 0.0001 | 9.21 | 非常震惊 |

## 6.4 交叉熵：给预测打分

训练时，我们知道**正确的下一个 token** 是什么。模型给它分配的概率越高，说明预测得越好。**交叉熵损失**（cross-entropy loss）就是**正确答案的惊讶度**：

```formula
loss = −log p(正确的 token)
```

```python
import math

def cross_entropy(probs, target):
    return -math.log(probs[target])

probs = [0.7, 0.2, 0.1]          # 模型对三个候选的预测
print(round(cross_entropy(probs, 0), 4))    # 正确答案是第 0 个：0.3567（预测得不错）
print(round(cross_entropy(probs, 2), 4))    # 正确答案是第 2 个：2.3026（预测得很差）
```

- 正确答案概率为 1 → 损失为 0（完美）；
- 正确答案概率趋近 0 → 损失趋向无穷大（大错特错）。

训练语言模型时，把一段文字里**每个位置**的交叉熵**取平均**，就是这一批数据的损失。梯度下降（第 5 课）做的就是不断调整参数，让这个平均损失变小——**也就是让模型给正确的下一个词分配更高的概率**。

> [!IMPORTANT]
> 这就是大语言模型预训练的完整目标：**最小化下一个 token 的平均交叉熵**。几十亿个参数、几千张 GPU、几个月的训练，都在优化这一个数字。

一个有用的基准：如果词表有 `V` 个 token，一个什么都没学会、对所有 token 一视同仁的模型，每个 token 的概率是 `1/V`，损失是 `log V`。GPT-2 的词表约有 5 万个 token，`log 50257 ≈ 10.8`——训练刚开始时的损失大约就是这个数。

## 6.5 困惑度：模型在「几个选项里犹豫」

交叉熵不太直观。把平均交叉熵取指数，得到**困惑度**（perplexity）：

```formula
perplexity = e^(平均交叉熵)
```

直觉：**模型平均在几个选项之间犹豫不决**。困惑度为 1 表示每次都确信无疑（且正确）；困惑度为 50 表示好像在 50 个词里随机猜。

```python
import math

def perplexity(target_probs):
    """target_probs：模型在每个位置给正确 token 的概率"""
    avg_ce = sum(-math.log(p) for p in target_probs) / len(target_probs)
    return math.exp(avg_ce)

print(round(perplexity([0.25, 0.25, 0.25, 0.25]), 2))    # 4.0：相当于在 4 个里瞎猜
print(round(perplexity([0.9, 0.8, 0.95, 0.7]), 2))       # 约 1.2：很有把握
```

## 6.6 熵（了解即可）

一个分布本身的「不确定程度」叫**熵**（entropy）：`H = −Σ pᵢ log pᵢ`，即「平均惊讶度」。均匀分布的熵最大（最不确定），只有一个结果概率为 1 时熵为 0。之所以叫「**交叉**熵」，是因为它是用**模型的分布**去衡量**真实数据**的平均惊讶度。

## 6.7 小结

- **概率分布**：所有结果的概率，加起来为 1；`random.choices` 可以按概率抽样。
- **Softmax**：`e^z / Σ e^z`，把任意 logits 变成概率；实现时先减最大值防溢出。
- **温度**：logits 除以 T，T 小更确定、T 大更随机。
- **log** 把乘法变加法、避免下溢；`−log p` 是惊讶度。
- **交叉熵损失** = `−log p(正确答案)`；语言模型训练就是最小化每个位置的平均交叉熵。
- **困惑度** = `e^(平均交叉熵)`，表示模型平均在几个选项间犹豫；随机猜 V 个时困惑度为 V。

## 6.8 练习

<!-- exercise: softmax -->
### 练习 1：实现 softmax

输入一行若干个数（logits），输出 softmax 后的概率，每个保留 4 位小数，空格隔开。请使用「先减最大值」的稳定写法。

```text
样例输入：2 1 0.1
样例输出：0.6590 0.2424 0.0986
```

<details>
<summary>提示</summary>

`m = max(z)`；`exps = [math.exp(v - m) for v in z]`；再除以 `sum(exps)`。格式化用 `f"{p:.4f}"`。
</details>

<!-- exercise: cross-entropy -->
### 练习 2：交叉熵

第一行是模型给出的概率分布（若干个数，空格隔开），第二行是正确答案的下标（从 0 开始）。输出交叉熵损失，保留 4 位小数。

```text
样例输入：
0.7 0.2 0.1
2
样例输出：
2.3026
```

<details>
<summary>提示</summary>

`-math.log(probs[target])`
</details>

<!-- exercise: perplexity -->
### 练习 3：困惑度

输入一行若干个数：模型在每个位置分配给正确 token 的概率。输出困惑度，保留 2 位小数。

```text
样例输入：0.5 0.25 0.125
样例输出：4.00
```

<details>
<summary>提示</summary>

先求平均交叉熵 `sum(-math.log(p) for p in ps) / len(ps)`，再 `math.exp(...)`。
</details>

<!-- exercise: temperature -->
### 练习 4：温度

想让聊天机器人的回答**更稳定、更少随机性**（比如写代码、回答事实问题），应该把温度调成？

- 更高，例如 1.5
- 更低，例如 0.2
- 温度对结果没有影响
- 设置为负数

<details>
<summary>答案与解析</summary>

温度越低，softmax 分布越「尖」，越倾向于选择最可能的 token，输出更稳定。写诗、头脑风暴时可以调高一些。
</details>

<!-- exercise: uniform-loss -->
### 练习 5：随机猜的损失

一个模型对词表里的 4 个 token 总是给出均匀的概率（各 0.25）。下面的程序输出什么？（保留 2 位小数）

```python
import math
print(f"{-math.log(0.25):.2f}")
print(f"{math.exp(-math.log(0.25)):.2f}")
```

<details>
<summary>答案与解析</summary>

交叉熵 `−log 0.25 = log 4 ≈ 1.39`；困惑度 `e^1.39 = 4.00`——正好是「在 4 个里瞎猜」。
</details>

## 6.9 延伸阅读

- [《动手学深度学习》3.4 softmax 回归](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)：从另一个角度讲 softmax 和交叉熵。
- Chris Olah：[Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)（英文）：用图解释熵和交叉熵，非常漂亮。
- 第一部分完成！下一步：[第二部分 · 用梯度下降拟合直线](../../ml/01-linear-regression/zh-CN.md)
