# 1. 预训练数据：模型读了什么

> 🎯 **本课目标**：知道大模型预训练用了多少数据、数据从哪里来；理解一条完整的数据处理流水线——提取文本、语言识别、启发式过滤、**去重**（精确去重和 MinHash 近似去重）、质量分类、去除隐私信息、**去污染**；明白数据配比、重复轮数、退火阶段和合成数据；知道训练前怎样把文档**打包**成固定长度的序列。
>
> ⏱ **预计用时**：80 分钟　｜　📋 **前置知识**：第三部分（特别是第 1、2、6 课）

第三部分里，我们用几千个字符训练了一个小 GPT。真正的大模型做的是同一件事——预测下一个 token——只是读的文字多了几十亿倍。业内有一句话：**模型的能力，上限由数据决定**。很多团队花在数据上的人力，比花在模型结构上的还多。

## 1.1 有多少数据？

| 模型 | 年份 | 预训练 token 数 |
|------|------|-----------------|
| GPT-3 | 2020 | 约 3000 亿（0.3T） |
| Llama 2 | 2023 | 约 2 万亿（2T） |
| Llama 3 | 2024 | 超过 15 万亿（15T） |
| DeepSeek-V3 | 2024 | 约 14.8 万亿 |
| Qwen3 | 2025 | 约 36 万亿 |

15 万亿个 token 是什么概念？按英文平均每个 token 约 4 个字符、一本书约 50 万个字符来算，大约相当于 **1.2 亿本书**。一个人每天读一本书，要读 30 多万年。

## 1.2 数据从哪里来

- **网页**：绝大部分数据来自网页，最重要的公开来源是 **Common Crawl**——一个非营利组织，从 2008 年起每隔一两个月抓取一次互联网，每次几十亿个网页，免费开放。
- **代码**：GitHub 等开源仓库。代码数据不仅让模型会写程序，也被认为能提升推理能力。
- **书籍、论文、百科**：维基百科、arXiv 论文、公版书等，质量高但数量少。
- **数学、科学**：数学网页、教材、题解，近年越来越受重视。
- **多语言数据**：中文、俄语等语言的网页比英文少得多，通常要专门收集和增加比例。
- **合成数据**：用已有的大模型生成或改写的文本（1.6 节）。

数据的来源也涉及**版权和许可**：哪些数据可以用来训练、是否需要授权，是 2023 年以来各国诉讼和立法的热点。开源社区因此整理了不少来源清楚、许可明确的数据集，比如 Hugging Face 的 **FineWeb**（2024 年发布，从 96 次 Common Crawl 抓取中处理出约 15 万亿 token 的英文网页文本，处理代码全部公开）。

## 1.3 数据处理流水线

原始网页里有大量导航栏、广告、乱码、重复内容和低质量文本。一条典型的流水线是这样的：

```text
原始网页（HTML）
   │ ① 提取正文：去掉 HTML 标签、导航栏、页脚
   ▼
纯文本 ── ② 语言识别：这是哪种语言？
   │ ③ 启发式过滤：太短、乱码、符号太多、重复太多的丢掉
   │ ④ 去重：完全相同的、几乎相同的只留一份
   │ ⑤ 质量分类：用模型给「教育价值」「写作质量」打分
   │ ⑥ 去除隐私信息：邮箱、电话、身份证号……
   │ ⑦ 去污染：删掉和评测题目重合的内容
   ▼
可以训练的文本（往往只剩原始数据的百分之几）
```

### 启发式过滤

先用一些简单的规则快速去掉明显的垃圾。DeepMind 训练 Gopher（2021）时总结了一组这样的规则（文档长度、平均词长、符号比例、重复程度等），后来被广泛使用。下面是一个简化版：

```python
def check(doc):
    words = doc.split()
    if not 5 <= len(words) <= 100_000:          # Gopher 原本要求至少 50 个词，这里为了演示放宽到 5
        return "太短或太长"
    mean_len = sum(len(w) for w in words) / len(words)
    if not 3 <= mean_len <= 10:
        return "平均词长异常"
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8:
        return "含字母的词太少"
    if len({w.lower() for w in words}) / len(words) < 0.3:
        return "重复太多"
    return "保留"

docs = [
    "Photosynthesis is the process by which green plants turn sunlight, water and carbon dioxide into sugar.",
    "Home | About | Login",
    "$$$ 100% !!! ### 2026 *** >>> <<< ### $$$ 100%",
    "click here click here click here click here click here click here",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb c d e",
]
for d in docs:
    print(f"{d[:40]:<42}→ {check(d)}")
```

这些规则粗糙但很快，每秒能处理大量文档。要注意它们是**针对某种语言**设计的：中文没有空格分词，`doc.split()` 对中文几乎没用，处理中文时要按字符统计，或者先分词。

### 精确去重

同一篇文章会被转载到成百上千个网站上。重复的数据会浪费算力，还会让模型**背诵**这些文本。最简单的去重是：先把文本规范化（统一大小写、合并空白），再算一个**哈希值**，哈希相同的只留第一份：

```python
import hashlib
import re

def key(doc):
    normalized = re.sub(r"\s+", " ", doc.lower()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()

docs = [
    "Water boils at 100 degrees Celsius at sea level.",
    "Paris is the capital of France.",
    "water  boils at 100 degrees celsius at   sea level.",
    "Water boils at 100 degrees Celsius at sea level!",
]
seen, unique = set(), []
for d in docs:
    k = key(d)
    if k not in seen:
        seen.add(k)
        unique.append(d)
print(f"{len(docs)} 篇 → {len(unique)} 篇")
for d in unique:
    print(" ", d)
```

第三篇只是大小写和空格不同，被去掉了；但第四篇只多了一个感叹号，哈希就完全不同了。网上大量的「几乎重复」——改了几个字、加了个日期、换了个页脚——需要**近似去重**。

### 近似去重：MinHash

衡量两篇文档有多像，常用 **Jaccard 相似度**：把文档切成一个个小片段（**shingle**，比如每 5 个连续字符），两篇文档的片段集合 A、B 的交集大小除以并集大小：

```formula
J(A, B) = |A ∩ B| / |A ∪ B|
```

但要在几十亿篇文档里两两比较，是不可能的。**MinHash** 的妙处是：用一个哈希函数给所有片段打分，取最小的那个；两篇文档「最小片段相同」的概率，**恰好等于**它们的 Jaccard 相似度。用 128 个不同的哈希函数，就得到 128 个数组成的「签名」，签名里相同位置相等的比例，就是 Jaccard 的估计值：

```python
import hashlib

def shingles(text, k=5):
    t = " ".join(text.lower().split())
    return {t[i:i + k] for i in range(len(t) - k + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b)

def h(s, seed):                       # 第 seed 个哈希函数
    digest = hashlib.md5(f"{seed}:{s}".encode()).digest()
    return int.from_bytes(digest[:8], "big")

def minhash(sh, num=128):             # 签名：每个哈希函数下的最小值
    return [min(h(s, seed) for s in sh) for seed in range(num)]

def estimate(sig1, sig2):
    return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)

docs = {
    "A": "The quick brown fox jumps over the lazy dog near the old river bank.",
    "B": "The quick brown fox jumped over the lazy dog near the old river bank!",
    "C": "Large language models are trained on trillions of tokens of text.",
}
sh = {k: shingles(v) for k, v in docs.items()}
sig = {k: minhash(v) for k, v in sh.items()}
for x, y in [("A", "B"), ("A", "C")]:
    print(f"{x}-{y}: 真实 Jaccard {jaccard(sh[x], sh[y]):.2f}，MinHash 估计 {estimate(sig[x], sig[y]):.2f}")
```

签名只有 128 个数，比原文短得多。再配合**局部敏感哈希**（LSH）：把签名切成 b 段、每段 r 个数，只要有一段完全相同，就把两篇文档列为「候选」，再仔细比较。相似度为 s 的两篇文档成为候选的概率是 `1 - (1 - s^r)^b`。取 b = 16、r = 8：

| 相似度 s | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
|----------|-----|-----|-----|-----|-----|
| 成为候选的概率 | 0.001 | 0.06 | 0.61 | 0.95 | 1.00 |

相似的几乎都能被找出来，不相似的几乎都不会被比较——这样几十亿篇文档的去重也只需要线性的时间。

### 质量分类器

规则只能去掉明显的垃圾。更进一步的做法是**训练一个分类器给文档打分**。FineWeb-Edu（2024）的做法很有代表性：先让一个大模型（Llama 3 70B）给几十万篇网页的「教育价值」打 0～5 分，再用这些打分训练一个小而快的分类器，给全部网页打分，只保留高分的。只用这 1.3 万亿 token 的「教育类」网页训练的模型，在知识和推理测试上明显好于用全部网页训练的模型。

### 去除隐私信息和去污染

- **隐私信息**：用规则和模型找出邮箱、电话号码、IP 地址、身份证号等，替换掉或删除。
- **去污染**（decontamination）：如果评测题目（以及答案）出现在训练数据里，模型在评测中的高分可能只是「背出来的」。所以要把和评测集有大段重合（比如连续 13 个词相同）的文档删掉。第 5 课会再谈这个问题。

## 1.4 数据配比、重复和退火

不同来源的数据怎样混合，叫做**数据配比**（data mixture）。它对模型的影响非常大，通常要先用小模型做大量实验来确定。一个关键的问题是：高质量的数据往往不多，按配比算下来可能要**重复使用好几遍**：

```python
budget = 15_000                # 总共要训练 15 万亿 token（单位：十亿）
sources = {                    # 名称：(这类数据有多少，配比)  —— 数字只是示意
    "网页":  (30_000, 0.70),
    "代码":  (2_000, 0.15),
    "数学":  (500, 0.07),
    "书籍与论文": (400, 0.05),
    "多语言": (3_000, 0.03),
}
for name, (size, share) in sources.items():
    used = budget * share
    print(f"读 {used / size:4.2f} 遍（使用 {used:>6,.0f}B token）  {name}")
```

研究表明（Muennighoff 等，2023），同样的数据重复读 **4 遍以内**，效果和读全新的数据差不多；再多，收益就迅速下降。

很多模型在预训练的**最后阶段**，会把学习率降到很低，同时提高高质量数据（数学、代码、教材、精选网页）的比例，这叫**退火**（annealing）。用很少的 token 就能明显提升模型的能力。

## 1.5 打包成训练序列

模型每次处理固定长度的序列（比如 4096 或 8192 个 token），而文档有长有短。常见的做法是把所有文档首尾相接，每篇文档后面加一个**结束符**（EOS），再切成等长的块：

```python
docs = [[5, 8, 2], [7, 7], [1, 2, 3, 4, 5, 6], [9]]   # 已经分好词的文档（token id）
EOS = 0
L = 4                                                 # 每个训练序列的长度

stream = []
for d in docs:
    stream += d + [EOS]                               # 文档之间用 EOS 隔开
print("拼接后：", stream)

for i in range(0, len(stream) - L, L):
    x = stream[i:i + L]                               # 输入
    y = stream[i + 1:i + L + 1]                       # 目标：每个位置的「下一个 token」
    print("输入", x, "→ 目标", y)
```

- 目标序列就是输入序列向后错开一位：每个位置都要预测下一个 token，这和第三部分第 6 课一模一样。
- 打包后，一个序列里可能有好几篇不相关的文档。简单的实现让它们互相「看得见」；更讲究的实现会修改注意力掩码，让每个 token 只能看到**同一篇文档**里前面的内容。
- 长文档会被切到不同的序列里。专门的「长上下文」训练阶段会用更长的序列和更多的长文档（第 4 课）。

## 1.6 合成数据

高质量的人类文本快要被用完了，于是**合成数据**越来越重要：

- **改写**：让模型把质量一般的网页改写成清楚的问答、教材风格的文本；
- **生成**：微软的 Phi 系列模型（2023 年起）大量使用模型生成的「教科书式」文本，小模型也能有不错的推理能力；
- **可验证的数据**：数学题和代码可以自动检查对错，可以大量生成再筛选（第五部分会讲推理模型如何利用这一点）。

但合成数据也有风险：如果一代又一代的模型只用前一代模型生成的数据训练，输出会越来越单一，罕见的知识逐渐丢失，这被称为**模型崩溃**（model collapse）。所以合成数据一般和真实数据混合使用，并且仔细筛选。

## 1.7 小结

- 预训练数据以万亿 token 计：Llama 3 超过 15T，Qwen3 约 36T。
- 主要来源是网页（Common Crawl），加上代码、书籍、论文、数学和多语言数据。
- 流水线：提取正文 → 语言识别 → 启发式过滤 → 去重 → 质量分类 → 去隐私 → 去污染，最后往往只剩百分之几。
- 精确去重用规范化 + 哈希；近似去重用 shingle + MinHash（相同位置相等的比例 ≈ Jaccard）+ LSH 分段。
- 数据配比决定各类数据的比例；高质量数据可以重复几遍；最后的退火阶段多用高质量数据。
- 文档首尾相接、用 EOS 隔开，切成等长序列；目标是输入错开一位。
- 合成数据越来越重要，但要防止模型崩溃。

## 1.8 练习

<!-- exercise: jaccard -->
### 练习 1：Jaccard 相似度

输入两行英文。把每行转成小写、按空白切成单词，取所有**相邻两个词**组成的片段（比如 `the cat sat` 的片段是 `the cat` 和 `cat sat`），输出两个片段集合的 Jaccard 相似度，保留 3 位小数。（每行至少有 2 个词。）

```text
样例输入：
the cat sat on the mat
the cat sat on a mat
样例输出：0.429
```

<details>
<summary>提示</summary>

`words = line.lower().split()`，片段集合 `{(words[i], words[i + 1]) for i in range(len(words) - 1)}`。Jaccard = 交集大小 / 并集大小。
</details>

<!-- exercise: gopher-filter -->
### 练习 2：启发式过滤

第一行输入文档数 `n`，接下来 `n` 行，每行是一篇文档。按下面四条规则判断每篇文档，全部满足就输出 `keep`，否则输出 `drop`：

1. 词数（按空白切分）在 5 到 1000 之间；
2. 平均词长在 3 到 10 之间（包含两端）；
3. 至少 80% 的词含有字母；
4. 不同的词（转小写后）占全部词的比例至少是 0.3。

```text
样例输入：
2
Photosynthesis turns sunlight water and carbon dioxide into sugar.
click here click here click here click here
样例输出：
keep
drop
```

<details>
<summary>提示</summary>

照着 1.3 节的 `check` 函数写，把词数上限改成 1000。含有字母：`any(c.isalpha() for c in w)`。
</details>

<!-- exercise: exact-dedup -->
### 练习 3：精确去重

第一行输入文档数 `n`，接下来 `n` 行是文档。两篇文档在「转小写、把连续的空白合并成一个空格、去掉首尾空白」之后完全相同，就算重复。只保留每组重复中第一次出现的那篇，先输出保留的篇数，再按原来的顺序输出保留的文档（输出**原文**）。

```text
样例输入：
3
Hello World
hello   world
Goodbye
样例输出：
2
Hello World
Goodbye
```

<details>
<summary>提示</summary>

规范化：`" ".join(doc.lower().split())`。用一个集合记住见过的规范化结果。
</details>

<!-- exercise: mixture-epochs -->
### 练习 4：每类数据读几遍

第一行是训练的总 token 数 `T`（单位：十亿）。之后每行描述一类数据：`名称 数据量 配比`（数据量单位也是十亿 token），一直到输入结束。按 1.4 节的方法，输出每类数据被读了几遍，格式 `名称 遍数`，遍数保留 2 位小数。

```text
样例输入：
1000
web 2000 0.8
code 100 0.15
math 20 0.05
样例输出：
web 0.40
code 1.50
math 2.50
```

<details>
<summary>提示</summary>

遍数 = `T * 配比 / 数据量`。读到输入结束：`import sys`，`for line in sys.stdin:`。
</details>

<!-- exercise: decontam -->
### 练习 5：为什么要去污染？

预训练数据处理中的「去污染」（decontamination）是为了什么？

- 删除含有病毒和恶意代码的网页
- 防止评测题目出现在训练数据里，使评测分数虚高
- 删除含有个人隐私的文本
- 去除重复的网页

<details>
<summary>答案与解析</summary>

如果评测题和答案在训练时就见过，模型的高分可能只是记住了答案，不能反映真实能力。所以要删掉与评测集大段重合的训练文档。隐私信息和去重是流水线里的其他步骤。
</details>

<!-- exercise: minhash-why -->
### 练习 6：MinHash 有什么用？

在数据处理中，MinHash 主要用来做什么？

- 给文档的质量打分
- 识别文档的语言
- 高效地找出几乎相同的文档
- 把文档切成 token

<details>
<summary>答案与解析</summary>

MinHash 用一个很短的签名来估计两篇文档的 Jaccard 相似度，配合 LSH 分段，可以在海量文档里快速找出近似重复的文档。质量打分用分类器，语言识别用专门的语言识别模型，切分 token 用分词器。
</details>

## 1.9 延伸阅读

- Penedo 等：《The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale》（2024）——详细介绍了数据处理的每一步，并公开了代码和数据
- Rae 等：《Scaling Language Models: Methods, Analysis & Insights from Training Gopher》（2021）附录中的质量过滤规则
- Muennighoff 等：《Scaling Data-Constrained Language Models》（2023）——数据重复几遍还有用
- Shumailov 等：《AI models collapse when trained on recursively generated data》（Nature，2024）
- 下一课：[规模定律与算力](../02-scaling-compute/zh-CN.md)
