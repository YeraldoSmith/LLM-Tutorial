# 7. 解码：模型怎样选出下一个词

> 🎯 **本课目标**：明白模型只给出「下一个 token 的概率」，**怎么选**是另一回事（解码）；理解贪心解码为什么会重复、无聊；掌握**温度**、**top-k**、**top-p**、**min-p** 采样和**重复惩罚**；了解束搜索、停止条件、**约束解码**（结构化输出）；知道调用大模型 API 时这些参数该怎么设。
>
> ⏱ **预计用时**：60 分钟　｜　📋 **前置知识**：本部分第 1、6 课

训练好的语言模型，每一步只输出一个**概率分布**：下一个 token 是「好」的概率 63%，是「热」的概率 14%……真正生成文字时，还要决定**选哪一个**。这一步叫**解码**（decoding）或**采样**（sampling）。同一个模型，解码方式不同，回答的风格、质量和多样性可以差别很大。你在 ChatGPT 等产品的 API 里看到的 `temperature`、`top_p` 等参数，控制的就是这一步。

## 7.1 贪心解码：每次都选最可能的

最简单的方法：每一步都选**概率最大**的 token（上一课的加法 GPT 就是这样）。它确定、可复现，适合「答案唯一」的任务（算术、分类、提取信息）。但用来写文章，问题很明显：

- **无聊**：永远选最常见的说法，没有变化；
- **重复**：一旦进入循环就出不来。

看一个手工做的「词级」小模型。它在 `cat` 之后最可能接 `the`，在 `the` 之后最可能接 `cat`……

```python
# 一个手工做的「词级」模型：每个词后面接各个词的 logits
LM = {
    "<s>": {"the": 2.0, "a": 1.0},
    "the": {"cat": 2.0, "dog": 1.5, "mat": 0.5},
    "a":   {"cat": 1.0, "dog": 1.0},
    "cat": {"the": 1.6, "sat": 1.4, ".": 0.5},
    "dog": {"the": 1.6, "ran": 1.2, ".": 0.5},
    "sat": {"on": 2.0, ".": 0.5},
    "ran": {"away": 2.0, ".": 0.5},
    "on":  {"the": 2.0, "a": 1.0},
    "away": {".": 2.0},
    "mat": {".": 2.0, "the": 0.5},
}

def greedy(penalty=1.0, max_len=12):
    out, cur = [], "<s>"
    while len(out) < max_len:
        scores = dict(LM[cur])
        for w in set(out):                        # 重复惩罚：已经出现过的词，分数打折扣
            if w in scores:
                scores[w] = scores[w] / penalty if scores[w] > 0 else scores[w] * penalty
        cur = max(scores, key=scores.get)
        out.append(cur)
        if cur == ".":
            break
    return " ".join(out)

print("贪心：          ", greedy())
print("重复惩罚 1.5：  ", greedy(penalty=1.5))
```

贪心解码陷入了 `the cat the cat …` 的死循环，直到长度上限。**重复惩罚**（repetition penalty）把已经出现过的词的分数打个折扣，循环就被打破了。（惩罚太重也不好：模型会被迫避开本该重复的词，比如「的」「the」。）

## 7.2 温度：调节「大胆」的程度

想要多样性，就按概率**随机抽**（采样）。**温度**（temperature，T）在 softmax 之前把 logits 除以 T：

```formula
P(token i) = softmax(zᵢ / T)
```

```python
import math

def softmax(z, T=1.0):
    z = [v / T for v in z]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

words = ["好", "热", "冷", "不错", "糟糕", "紫色"]
logits = [3.0, 1.5, 1.2, 1.0, 0.2, -2.0]          # 模型对「今天天气真___」给出的分数

for T in (0.2, 0.7, 1.0, 1.5):
    probs = softmax(logits, T)
    print(f"T={T:<4}", "  ".join(f"{w}:{p:5.1%}" for w, p in zip(words, probs)))
```

- **T < 1**：分布变「尖」，高概率的更高——更保守、更稳定。T → 0 时就等于贪心。
- **T = 1**：模型原本的分布。
- **T > 1**：分布变「平」，低概率的选项（甚至「紫色」）机会变大——更有创意，也更容易胡说。

用第 1 课的 bigram 名字模型直观地感受一下：

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
counts = {a: {b: 0.1 for b in chars} for a in chars}      # 第 1 课的 bigram 模型（带平滑）
for name in NAMES:
    seq = ["."] + list(name) + ["."]
    for a, b in zip(seq, seq[1:]):
        counts[a][b] += 1
logits = {a: [math.log(counts[a][b]) for b in chars] for a in chars}   # 把概率变回「logits」

def generate(T=1.0, top_k=None, greedy=False, max_len=15):
    out, cur = "", "."
    while len(out) < max_len:
        z = logits[cur]
        if greedy:
            cur = chars[max(range(len(z)), key=lambda i: z[i])]
        else:
            z = [v / T for v in z]                         # 温度
            if top_k:                                      # top-k：只在最可能的 k 个里抽
                kth = sorted(z, reverse=True)[top_k - 1]
                z = [v if v >= kth else float("-inf") for v in z]
            m = max(z)
            w = [math.exp(v - m) for v in z]
            cur = random.choices(chars, weights=w)[0]
        if cur == ".":
            break
        out += cur
    return out

random.seed(0)
print("贪心          ", [generate(greedy=True) for _ in range(5)])
for T in (0.5, 1.0, 1.5):
    print(f"温度 {T:<4}     ", [generate(T=T) for _ in range(5)])
print("温度 1 + top-5", [generate(T=1.0, top_k=5) for _ in range(5)])
```

贪心每次都生成同一个名字 `a`（最常见的开头是 `a`，`a` 后面最常见的是结尾）；温度低时名字短而保守，温度高时变成一长串乱码；加上 top-k 之后，乱码明显少了。

## 7.3 截掉「长尾」：top-k、top-p、min-p

即使是好模型，词表里也有几万个概率很小但不为零的 token。随机采样时，「长尾」里的怪词加起来也有不小的机会被抽中，一旦抽中一个，后面可能就全乱了。所以通常在采样前**只保留一部分候选**：

- **top-k**：只保留概率最高的 k 个；
- **top-p**（核采样，nucleus sampling）：从最可能的开始累加，保留**累计概率刚好达到 p** 的最小集合。模型很确定时只留一两个，不确定时留很多——比固定的 k 更灵活；
- **min-p**（较新的方法）：只保留概率**不低于「最大概率 × min_p」**的 token。

保留下来的候选重新归一化（加起来等于 1），再按概率抽：

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

def top_k(probs, k):
    keep = sorted(range(len(probs)), key=lambda i: -probs[i])[:k]
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]

def top_p(probs, p):
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    keep, cum = [], 0.0
    for i in order:                      # 从最可能的开始累加，直到累计概率达到 p
        keep.append(i)
        cum += probs[i]
        if cum >= p:
            break
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]

def min_p(probs, ratio):
    limit = ratio * max(probs)           # 只保留概率不低于「最大概率 × ratio」的 token
    total = sum(q for q in probs if q >= limit)
    return [q / total if q >= limit else 0.0 for q in probs]

words = ["好", "热", "冷", "不错", "糟糕", "紫色"]
probs = softmax([3.0, 1.5, 1.2, 1.0, 0.2, -2.0])
show = lambda ps: "  ".join(f"{w}:{q:5.1%}" for w, q in zip(words, ps) if q > 0)
print("原始     ", show(probs))
print("top-k=3  ", show(top_k(probs, 3)))
print("top-p=0.9", show(top_p(probs, 0.9)))
print("min-p=0.1", show(min_p(probs, 0.1)))
```

实际中常把**温度和 top-p（或 top-k、min-p）组合使用**：先用温度调整分布，再截掉长尾，最后抽样。

## 7.4 惩罚重复：重复惩罚、频率惩罚、存在惩罚

7.1 节已经见过**重复惩罚**。OpenAI 等的 API 还提供两个类似的参数：

| 参数 | 做法 | 效果 |
|------|------|------|
| `frequency_penalty` | 某个 token 已经出现了几次，就从它的 logit 里减去「次数 × 惩罚值」 | 出现越多，越不想再用；减少啰嗦的重复 |
| `presence_penalty` | 只要出现过（不管几次），就减去一个固定值 | 鼓励谈论新的话题 |
| `repetition_penalty`（Hugging Face、vLLM 等） | 出现过的 token，正的 logit 除以惩罚值、负的乘以惩罚值 | 打破循环 |

这些参数一般保持默认（0 或 1.0）即可，出现明显的重复时再适当调大。

## 7.5 束搜索

贪心每一步只看眼前最好的选择，但「每一步都最好」不等于「整句最好」。**束搜索**（beam search）每一步都保留 **b 个**累计概率最高的候选序列（b 叫束宽），最后选出整体概率最高的一个：

```text
束宽 b = 2
第 1 步：the (0.7)         a (0.3)
第 2 步：the cat (0.42)    the dog (0.21)     ← 从所有扩展里只留下最好的 2 个
第 3 步：the cat sat (…)   the dog ran (…)
……
```

束搜索在**机器翻译、语音识别**这类「答案比较确定」的任务里很常用。但在开放式写作和对话里，它容易生成平淡、重复的文字，所以聊天机器人基本都用**采样**。

## 7.6 什么时候停下来？

生成会一直继续，直到满足某个**停止条件**：

- 生成了**结束 token**（如 `<|endoftext|>`、`<|im_end|>`，第 2 课讲过的特殊 token）——模型自己「决定」说完了；
- 达到**最大长度**（API 里的 `max_tokens` / `max_output_tokens`）；
- 出现了你指定的**停止序列**（`stop` 参数），比如遇到 `"\n\n"` 就停。

> [!WARNING]
> 如果 `max_tokens` 设得太小，回答会在半句话处被截断。推理模型（第五部分）在回答前会先生成很长的「思考」内容，这些也算在 token 数里，需要给足长度。

## 7.7 约束解码：保证输出符合格式

很多应用需要模型输出**严格的格式**，比如合法的 JSON、只能是「是 / 否」、必须是数字。**约束解码**（constrained decoding）在每一步采样之前，把**不符合格式的 token 的 logit 设成 −∞**，这样模型根本不可能输出不合法的内容：

```python
import math

vocab = ["1", "2", "3", "七", "abc", "。", "<end>"]
logits = [1.0, 0.5, 0.2, 2.5, 1.8, 0.3, 0.1]       # 模型最想说「七」，但我们只要阿拉伯数字

allowed = {"1", "2", "3", "<end>"}                   # 格式规则：只允许数字和结束
masked = [z if tok in allowed else float("-inf") for tok, z in zip(vocab, logits)]
m = max(masked)
e = [math.exp(z - m) for z in masked]
probs = [v / sum(e) for v in e]
for tok, p in zip(vocab, probs):
    print(f"{tok:6s}{p:6.1%}")
```

真实系统会根据 JSON Schema 或语法规则，动态算出每一步允许哪些 token。OpenAI、Anthropic 等 API 的「结构化输出」功能，以及 vLLM、llama.cpp 等推理框架的语法约束，背后都是这个原理。第七部分讲工具调用时会再用到。

## 7.8 实用建议

| 场景 | 建议 |
|------|------|
| 代码、数学、提取信息、分类 | 低温度（0～0.3），或直接贪心 |
| 一般问答、对话 | 温度 0.7 左右，top-p 0.9～0.95 |
| 写故事、头脑风暴 | 温度 0.8～1.2 |
| 推理模型 | **按官方建议设置**：例如 DeepSeek-R1 官方建议温度 0.5～0.7（推荐 0.6）；有的推理模型 API 甚至不允许修改温度 |

> [!NOTE]
> **温度为 0 也不一定完全可复现**：GPU 上的浮点运算顺序、服务器把多个请求合成一批处理等，都可能让结果有微小差别，而一个 token 不同，后面就可能完全不同。需要可复现时，可以固定随机种子（`seed` 参数，如果 API 支持），并记录模型版本。

## 7.9 小结

- 模型只给出下一个 token 的**概率分布**；**解码**决定选哪个，同一个模型的回答质量因此差别很大。
- **贪心**：确定、适合答案唯一的任务，但无聊、易重复。**采样**：按概率随机抽，多样但可能胡说。
- **温度** T：logits 除以 T；T < 1 更保守（T → 0 即贪心），T > 1 更大胆。
- **top-k / top-p / min-p**：只在高概率的候选里采样，截掉「长尾」；常与温度组合使用。
- **重复、频率、存在惩罚**减少重复；**束搜索**适合翻译等任务；**停止条件**有结束 token、最大长度和停止序列。
- **约束解码**把不合格式的 token 屏蔽掉，保证输出合法（结构化输出、JSON）。

## 7.10 练习

<!-- exercise: temperature -->
### 练习 1：带温度的 softmax

第一行是若干个 logits，第二行是温度 `T`。输出 `softmax(z / T)`，保留 4 位小数，空格隔开。

```text
样例输入：
2 1 0
0.5
样例输出：0.8668 0.1173 0.0159
```

<details>
<summary>提示</summary>

先 `z = [v / T for v in logits]`，再做 softmax（先减去最大值）。
</details>

<!-- exercise: top-k -->
### 练习 2：top-k

第一行是若干个 logits，第二行是整数 `k`。先做 softmax（温度 1），只保留概率最高的 `k` 个 token 并重新归一化，其余的概率设为 0。按原来的顺序输出所有 token 的概率，保留 4 位小数，空格隔开。

```text
样例输入：
2 1 0 -1
2
样例输出：0.7311 0.2689 0.0000 0.0000
```

<details>
<summary>提示</summary>

`keep = sorted(range(n), key=lambda i: -p[i])[:k]`，然后用保留下来的概率之和去除。
</details>

<!-- exercise: top-p -->
### 练习 3：top-p 保留了谁？

第一行是若干个已经归一化的概率，第二行是 `p`。按概率从大到小（概率相同时编号小的在前）依次加入候选，直到累计概率**大于或等于** `p` 为止。输出被保留的 token 的编号（从 0 开始），按加入的顺序，空格隔开。

```text
样例输入：
0.5 0.3 0.15 0.05
0.75
样例输出：0 1
```

<details>
<summary>提示</summary>

`sorted(range(n), key=lambda i: -probs[i])` 是稳定排序，概率相同时会保持编号从小到大的顺序。
</details>

<!-- exercise: greedy-repeat -->
### 练习 4：贪心解码的问题

用贪心解码写长文章时，最常见的问题是？

- 速度太慢
- 文字平淡、容易陷入重复的循环
- 每次运行结果都不一样
- 无法停止生成

<details>
<summary>答案与解析</summary>

贪心每次都选最常见的说法，一旦进入「A 之后最可能是 B，B 之后最可能是 A」的循环就出不来。它的结果是确定的（每次一样），这对唯一答案的任务反而是优点。
</details>

<!-- exercise: low-temp -->
### 练习 5：温度趋近于 0

把温度设得非常接近 0，采样的效果会接近什么？

- 均匀随机地选一个 token
- 贪心解码（总是选概率最大的 token）
- 总是选概率最小的 token
- 束搜索

<details>
<summary>答案与解析</summary>

logits 除以一个很小的 T 后，最大值和其他值的差距被放得极大，softmax 几乎把全部概率给了最大的那个——相当于贪心。温度很大时才接近均匀分布。
</details>

## 7.11 延伸阅读

- Holtzman 等：《The Curious Case of Neural Text Degeneration》（2019，提出 top-p 核采样）
- Hugging Face 博客：[How to generate text: using different decoding methods](https://huggingface.co/blog/how-to-generate)
- Nguyen 等：《Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs》（2024）
- 第三部分到此结束！下一部分：[第四部分 · 大规模训练](../../training/01-data/zh-CN.md)——数据、算力、分布式训练与评测。
