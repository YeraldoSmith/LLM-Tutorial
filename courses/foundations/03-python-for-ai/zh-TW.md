# 3. AI 程式設計需要的 Python

> 🎯 **本課目標**：複習 AI 程式碼裡最常用的 Python 寫法（推導式、字典計數、排序、函式、類）；學會用 `random` 和 `math` 做實驗；讀寫文字檔案；初步認識 NumPy 和「向量化」為什麼快。
>
> ⏱ **預計用時**：45 分鐘　｜　📋 **前置知識**：Python 基礎（變數、條件、迴圈、列表、字典、函式）

這一課不是從零教 Python（那是 [Our-tutorial 的 Python 課](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/python)），而是把**讀懂和編寫 AI 程式碼時最常用的那些寫法**集中複習一遍。如果哪裡看不懂，回到對應的 Python 課補一下就好。

## 3.1 列表推導式：一行處理一批資料

AI 程式碼裡到處是「對一組數字做同樣的處理」：

```python
scores = [2.0, -1.0, 0.5, 3.0]
doubled = [x * 2 for x in scores]
positive = [x for x in scores if x > 0]
relu = [max(0.0, x) for x in scores]        # ReLU：負數變 0（神經網路常用）
print(doubled)
print(positive)
print(relu)
```

巢狀推導式可以生成二維列表（矩陣）：

```python
rows, cols = 2, 3
zeros = [[0.0 for _ in range(cols)] for _ in range(rows)]
table = [[r * cols + c for c in range(cols)] for r in range(rows)]
print(zeros)
print(table)
```

> [!WARNING]
> 不要寫成 `[[0.0] * cols] * rows`！外層的 `* rows` 複製的是**同一個**內層列表的引用，改一行，所有行都會變（Python 課第 8 課講過）。

## 3.2 字典：計數與查表

分詞、統計詞頻、建立「詞 → 編號」的對映，全靠字典。

```python
text = "to be or not to be"
counts = {}
for w in text.split():
    counts[w] = counts.get(w, 0) + 1
print(counts)

from collections import Counter
print(Counter(text.split()).most_common(2))   # 標準庫的計數器，更方便
```

**詞表**（vocabulary）是語言模型的基礎：給每個不同的詞（或字）一個編號，把文字變成數字：

```python
text = "hello world hello"
vocab = sorted(set(text.split()))                  # 去重並排序
stoi = {w: i for i, w in enumerate(vocab)}        # string to index
itos = {i: w for w, i in stoi.items()}            # index to string
ids = [stoi[w] for w in text.split()]
print(stoi)
print(ids)
print(" ".join(itos[i] for i in ids))             # 再變回文字
```

`stoi` / `itos` 這兩個名字你會在很多開源專案（如 Karpathy 的 nanoGPT）裡見到。

## 3.3 排序與 `key`

```python
words = {"the": 7, "cat": 3, "sat": 3, "on": 5}
by_count = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
print(by_count)
top2 = [w for w, c in by_count[:2]]
print(top2)
```

`key=lambda kv: kv[1]` 表示「按每一項的第 2 個元素（次數）排序」。`lambda` 是一種一行的小函式：`lambda 引數: 返回值`。

如果想次數相同時按字母排序，可以讓 `key` 返回一個元組：`key=lambda kv: (-kv[1], kv[0])`——先按次數從大到小（加負號），再按單詞從小到大。

## 3.4 函式與類：把模型「裝」起來

神經網路的每一層通常寫成一個**類**：初始化時儲存引數，呼叫時計算輸出。

```python
class Linear:
    """一個最簡單的「層」：y = w * x + b"""
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, x):          # 讓物件可以像函式一樣被呼叫
        return self.w * x + self.b

layer = Linear(w=2.0, b=1.0)
print(layer(3.0))                   # 7.0
print([layer(x) for x in [0, 1, 2]])
```

`__call__` 讓 `layer(3.0)` 這種寫法成為可能——PyTorch 裡的模型就是這樣使用的：`output = model(input)`。

## 3.5 random 與 math：做實驗的工具

```python
import math
import random

random.seed(42)                         # 固定隨機種子：每次執行結果一樣，方便復現
print(random.random())                  # [0, 1) 之間的隨機小數
print(random.randint(1, 6))             # 擲骰子
print(random.choice(["石頭", "剪刀", "布"]))
print(random.choices(["a", "b"], weights=[0.9, 0.1], k=5))   # 按機率抽樣（語言模型生成文字時就是這樣選詞的）
print(random.gauss(0, 1))               # 正態分佈隨機數（初始化神經網路引數時常用）

print(math.exp(1), math.log(math.e))    # 指數與自然對數，後面計算機率時天天用
print(math.sqrt(2), math.pi)
```

> [!TIP]
> **隨機種子**（seed）讓「隨機」變得可重複：同樣的種子，產生同樣的一串隨機數。做 AI 實驗時一定要設種子，否則你無法確認結果的變化是改了程式碼造成的，還是運氣。

## 3.6 讀寫文字檔案

訓練語言模型的第一步就是讀入文字：

```python
from pathlib import Path

Path("tiny.txt").write_text("第一行\n第二行\n第三行\n", encoding="utf-8")

text = Path("tiny.txt").read_text(encoding="utf-8")
print(len(text), "個字元")
for i, line in enumerate(text.splitlines(), 1):
    print(i, line)
```

永遠寫上 `encoding="utf-8"`，否則在 Windows 上可能出現亂碼（[程式設計起步第 6 課](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/06-numbers-and-text)）。

## 3.7 NumPy 初見：為什麼要「向量化」

用 Python 列表做大量數值計算很慢，因為每個數字都要被 Python 直譯器一個個處理。**NumPy** 把資料存在連續的記憶體裡，用編譯好的 C 程式碼**一次處理整個陣列**，速度可以快幾十到幾百倍。幾乎所有 AI 庫（PyTorch 也一樣）都採用這種「陣列」思維。

```python requires=numpy
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])
print(a + b)            # 對應元素相加：[11. 22. 33.]
print(a * 2)            # 每個元素乘 2
print(a @ b)            # 點積（下一課的重點）：140.0
m = np.zeros((2, 3))    # 2 行 3 列的全 0 矩陣
print(m.shape)
print(np.exp(a))        # 對每個元素求 e 的次方
```

這段程式碼需要先安裝 NumPy（`pip install numpy` 或 `uv add numpy`）。本課程前三部分的練習都用**純 Python** 實現——這樣每一步都看得見、改得動；理解原理之後，你會發現 NumPy/PyTorch 只是把同樣的事情做得更快。

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

## 3.8 小結

- 推導式 `[f(x) for x in xs if 條件]` 處理一批資料；二維列表用巢狀推導式，不要用 `[[0]*n]*m`。
- 字典用於計數和建立詞表（`stoi`、`itos`）；`Counter.most_common` 更方便。
- `sorted(..., key=lambda ...)` 按指定規則排序；元組 key 實現多級排序。
- 用類和 `__call__` 封裝「層」和「模型」。
- `random.seed` 讓實驗可復現；`random.choices` 按機率抽樣；`math.exp`、`math.log` 天天用。
- 檔案讀寫寫上 `encoding="utf-8"`；NumPy 透過向量化讓數值計算快很多。

## 3.9 練習

<!-- exercise: top-words -->
### 練習 1：詞頻前三名

讀入一行英文文字（單詞之間用空格隔開，全部小寫），輸出出現次數最多的 3 個單詞及次數，每行一個，格式為 `單詞 次數`。次數相同時按單詞的字母順序排列。

```text
樣例輸入：to be or not to be that is the question to be
樣例輸出：
be 3
to 3
is 1
```

<details>
<summary>提示 1</summary>

先用字典（或 `Counter`）計數。
</details>

<details>
<summary>提示 2</summary>

`sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]`：先按次數從大到小，再按單詞從小到大。
</details>

<!-- exercise: vocab -->
### 練習 2：建立詞表

讀入一行文字，按**字元**建立詞表：把出現過的所有不同字元**排序**後依次編號（從 0 開始），然後輸出原文字每個字元對應的編號（空格隔開）。

```text
樣例輸入：hello
樣例輸出：1 0 2 2 3
```

（排序後的字元是 `e h l o`，編號分別為 0、1、2、3。）

<details>
<summary>提示</summary>

`chars = sorted(set(text))`，`stoi = {ch: i for i, ch in enumerate(chars)}`。
</details>

<!-- exercise: grid-trap -->
### 練習 3：推導式的陷阱

執行下面的程式碼後，`grid` 是什麼？

```python
grid = [[0] * 2] * 2
grid[0][0] = 9
print(grid)
```

<details>
<summary>答案與解析</summary>

`[[9, 0], [9, 0]]`。`* 2` 複製的是同一個內層列表的引用，兩行其實是同一個列表。正確寫法：`[[0] * 2 for _ in range(2)]`。
</details>

<!-- exercise: seed -->
### 練習 4：隨機種子

為什麼做 AI 實驗時要設定隨機種子（如 `random.seed(42)`）？

- 讓程式執行得更快
- 讓結果可以復現，方便比較不同改動的效果
- 讓隨機數更隨機
- 讓模型更準確

<details>
<summary>答案與解析</summary>

同一個種子產生同一串隨機數。固定種子後，結果的差別只來自你改的程式碼，而不是運氣。
</details>

## 3.10 延伸閱讀

- [NumPy 官方入門（中文）](https://numpy.org/zh/learn/)
- [《Python 資料科學手冊》](https://jakevdp.github.io/PythonDataScienceHandbook/)：NumPy 章節寫得很清楚（英文，免費線上）。
- 下一課：[向量與矩陣](../04-vectors-matrices/zh-CN.md)
