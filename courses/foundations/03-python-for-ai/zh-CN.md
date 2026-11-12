# 3. AI 编程需要的 Python

> 🎯 **本课目标**：复习 AI 代码里最常用的 Python 写法（推导式、字典计数、排序、函数、类）；学会用 `random` 和 `math` 做实验；读写文本文件；初步认识 NumPy 和「向量化」为什么快。
>
> ⏱ **预计用时**：45 分钟　｜　📋 **前置知识**：Python 基础（变量、条件、循环、列表、字典、函数）

这一课不是从零教 Python（那是 [Our-tutorial 的 Python 课](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/python)），而是把**读懂和编写 AI 代码时最常用的那些写法**集中复习一遍。如果哪里看不懂，回到对应的 Python 课补一下就好。

## 3.1 列表推导式：一行处理一批数据

AI 代码里到处是「对一组数字做同样的处理」：

```python
scores = [2.0, -1.0, 0.5, 3.0]
doubled = [x * 2 for x in scores]
positive = [x for x in scores if x > 0]
relu = [max(0.0, x) for x in scores]        # ReLU：负数变 0（神经网络常用）
print(doubled)
print(positive)
print(relu)
```

嵌套推导式可以生成二维列表（矩阵）：

```python
rows, cols = 2, 3
zeros = [[0.0 for _ in range(cols)] for _ in range(rows)]
table = [[r * cols + c for c in range(cols)] for r in range(rows)]
print(zeros)
print(table)
```

> [!WARNING]
> 不要写成 `[[0.0] * cols] * rows`！外层的 `* rows` 复制的是**同一个**内层列表的引用，改一行，所有行都会变（Python 课第 8 课讲过）。

## 3.2 字典：计数与查表

分词、统计词频、建立「词 → 编号」的映射，全靠字典。

```python
text = "to be or not to be"
counts = {}
for w in text.split():
    counts[w] = counts.get(w, 0) + 1
print(counts)

from collections import Counter
print(Counter(text.split()).most_common(2))   # 标准库的计数器，更方便
```

**词表**（vocabulary）是语言模型的基础：给每个不同的词（或字）一个编号，把文字变成数字：

```python
text = "hello world hello"
vocab = sorted(set(text.split()))                  # 去重并排序
stoi = {w: i for i, w in enumerate(vocab)}        # string to index
itos = {i: w for w, i in stoi.items()}            # index to string
ids = [stoi[w] for w in text.split()]
print(stoi)
print(ids)
print(" ".join(itos[i] for i in ids))             # 再变回文字
```

`stoi` / `itos` 这两个名字你会在很多开源项目（如 Karpathy 的 nanoGPT）里见到。

## 3.3 排序与 `key`

```python
words = {"the": 7, "cat": 3, "sat": 3, "on": 5}
by_count = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
print(by_count)
top2 = [w for w, c in by_count[:2]]
print(top2)
```

`key=lambda kv: kv[1]` 表示「按每一项的第 2 个元素（次数）排序」。`lambda` 是一种一行的小函数：`lambda 参数: 返回值`。

如果想次数相同时按字母排序，可以让 `key` 返回一个元组：`key=lambda kv: (-kv[1], kv[0])`——先按次数从大到小（加负号），再按单词从小到大。

## 3.4 函数与类：把模型「装」起来

神经网络的每一层通常写成一个**类**：初始化时保存参数，调用时计算输出。

```python
class Linear:
    """一个最简单的「层」：y = w * x + b"""
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, x):          # 让对象可以像函数一样被调用
        return self.w * x + self.b

layer = Linear(w=2.0, b=1.0)
print(layer(3.0))                   # 7.0
print([layer(x) for x in [0, 1, 2]])
```

`__call__` 让 `layer(3.0)` 这种写法成为可能——PyTorch 里的模型就是这样使用的：`output = model(input)`。

## 3.5 random 与 math：做实验的工具

```python
import math
import random

random.seed(42)                         # 固定随机种子：每次运行结果一样，方便复现
print(random.random())                  # [0, 1) 之间的随机小数
print(random.randint(1, 6))             # 掷骰子
print(random.choice(["石头", "剪刀", "布"]))
print(random.choices(["a", "b"], weights=[0.9, 0.1], k=5))   # 按概率抽样（语言模型生成文字时就是这样选词的）
print(random.gauss(0, 1))               # 正态分布随机数（初始化神经网络参数时常用）

print(math.exp(1), math.log(math.e))    # 指数与自然对数，后面计算概率时天天用
print(math.sqrt(2), math.pi)
```

> [!TIP]
> **随机种子**（seed）让「随机」变得可重复：同样的种子，产生同样的一串随机数。做 AI 实验时一定要设种子，否则你无法确认结果的变化是改了代码造成的，还是运气。

## 3.6 读写文本文件

训练语言模型的第一步就是读入文本：

```python
from pathlib import Path

Path("tiny.txt").write_text("第一行\n第二行\n第三行\n", encoding="utf-8")

text = Path("tiny.txt").read_text(encoding="utf-8")
print(len(text), "个字符")
for i, line in enumerate(text.splitlines(), 1):
    print(i, line)
```

永远写上 `encoding="utf-8"`，否则在 Windows 上可能出现乱码（[编程起步第 6 课](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/06-numbers-and-text)）。

## 3.7 NumPy 初见：为什么要「向量化」

用 Python 列表做大量数值计算很慢，因为每个数字都要被 Python 解释器一个个处理。**NumPy** 把数据存在连续的内存里，用编译好的 C 代码**一次处理整个数组**，速度可以快几十到几百倍。几乎所有 AI 库（PyTorch 也一样）都采用这种「数组」思维。

```python requires=numpy
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])
print(a + b)            # 对应元素相加：[11. 22. 33.]
print(a * 2)            # 每个元素乘 2
print(a @ b)            # 点积（下一课的重点）：140.0
m = np.zeros((2, 3))    # 2 行 3 列的全 0 矩阵
print(m.shape)
print(np.exp(a))        # 对每个元素求 e 的次方
```

这段代码需要先安装 NumPy（`pip install numpy` 或 `uv add numpy`）。本课程前三部分的练习都用**纯 Python** 实现——这样每一步都看得见、改得动；理解原理之后，你会发现 NumPy/PyTorch 只是把同样的事情做得更快。

### 比一比速度

```python requires=numpy
import time
import numpy as np

n = 1_000_000
xs = list(range(n))
t = time.perf_counter()
total = sum(x * x for x in xs)
t_list = time.perf_counter() - t

arr = np.arange(n, dtype=np.float64)
t = time.perf_counter()
total_np = float((arr * arr).sum())
t_np = time.perf_counter() - t
print(f"Python 列表：{t_list * 1000:.1f} ms，NumPy：{t_np * 1000:.1f} ms")
```

## 3.8 小结

- 推导式 `[f(x) for x in xs if 条件]` 处理一批数据；二维列表用嵌套推导式，不要用 `[[0]*n]*m`。
- 字典用于计数和建立词表（`stoi`、`itos`）；`Counter.most_common` 更方便。
- `sorted(..., key=lambda ...)` 按指定规则排序；元组 key 实现多级排序。
- 用类和 `__call__` 封装「层」和「模型」。
- `random.seed` 让实验可复现；`random.choices` 按概率抽样；`math.exp`、`math.log` 天天用。
- 文件读写写上 `encoding="utf-8"`；NumPy 通过向量化让数值计算快很多。

## 3.9 练习

<!-- exercise: top-words -->
### 练习 1：词频前三名

读入一行英文文本（单词之间用空格隔开，全部小写），输出出现次数最多的 3 个单词及次数，每行一个，格式为 `单词 次数`。次数相同时按单词的字母顺序排列。

```text
样例输入：to be or not to be that is the question to be
样例输出：
be 3
to 3
is 1
```

<details>
<summary>提示 1</summary>

先用字典（或 `Counter`）计数。
</details>

<details>
<summary>提示 2</summary>

`sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]`：先按次数从大到小，再按单词从小到大。
</details>

<!-- exercise: vocab -->
### 练习 2：建立词表

读入一行文本，按**字符**建立词表：把出现过的所有不同字符**排序**后依次编号（从 0 开始），然后输出原文本每个字符对应的编号（空格隔开）。

```text
样例输入：hello
样例输出：1 0 2 2 3
```

（排序后的字符是 `e h l o`，编号分别为 0、1、2、3。）

<details>
<summary>提示</summary>

`chars = sorted(set(text))`，`stoi = {ch: i for i, ch in enumerate(chars)}`。
</details>

<!-- exercise: grid-trap -->
### 练习 3：推导式的陷阱

运行下面的代码后，`grid` 是什么？

```python
grid = [[0] * 2] * 2
grid[0][0] = 9
print(grid)
```

<details>
<summary>答案与解析</summary>

`[[9, 0], [9, 0]]`。`* 2` 复制的是同一个内层列表的引用，两行其实是同一个列表。正确写法：`[[0] * 2 for _ in range(2)]`。
</details>

<!-- exercise: seed -->
### 练习 4：随机种子

为什么做 AI 实验时要设置随机种子（如 `random.seed(42)`）？

- 让程序运行得更快
- 让结果可以复现，方便比较不同改动的效果
- 让随机数更随机
- 让模型更准确

<details>
<summary>答案与解析</summary>

同一个种子产生同一串随机数。固定种子后，结果的差别只来自你改的代码，而不是运气。
</details>

## 3.10 延伸阅读

- [NumPy 官方入门（中文）](https://numpy.org/zh/learn/)
- [《Python 数据科学手册》](https://jakevdp.github.io/PythonDataScienceHandbook/)：NumPy 章节写得很清楚（英文，免费在线）。
- 下一课：[向量与矩阵](../04-vectors-matrices/zh-CN.md)
