# 2. 分類與 softmax 迴歸

> 🎯 **本課目標**：從「預測數字」進入「預測類別」；理解 sigmoid 和邏輯迴歸；把二分類推廣到多分類（softmax 迴歸）；記住一個非常重要的結論：softmax + 交叉熵的梯度就是「預測機率 − 正確答案」；用純 Python 訓練一個分類器並計算準確率。
>
> ⏱ **預計用時**：60 分鐘　｜　📋 **前置知識**：第一部分第 6 課（softmax、交叉熵），本部分第 1 課

上一課的模型輸出一個**數**（房價）。但很多問題要輸出的是**類別**：這封郵件是不是垃圾郵件？這張圖是貓、狗還是鳥？以及最重要的——**下一個 token 是詞表裡的哪一個？** 語言模型本質上就是一個有幾萬個類別的分類器。

## 2.1 二分類：輸出一個機率

判斷一封郵件是否是垃圾郵件，我們希望模型輸出「是垃圾郵件的**機率**」——一個 0 到 1 之間的數。但線性模型 `z = w·x + b` 的輸出可以是任何實數。需要一個函式把它「壓」到 0～1 之間，這就是 **sigmoid** 函式：

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

「線性模型 + sigmoid」就是**邏輯迴歸**（logistic regression）——名字裡有「迴歸」，做的卻是分類。

### 二元交叉熵

對二分類，正確答案 `y` 是 0 或 1，模型給出 `p = σ(z)`。損失是第一部分第 6 課的交叉熵：

```formula
L = −[ y·log p + (1 − y)·log(1 − p) ]
```

`y = 1` 時就是 `−log p`，`y = 0` 時就是 `−log(1 − p)`——依然是「正確答案的驚訝度」。

## 2.2 多分類：softmax 迴歸

如果有 `K` 個類別（比如手寫數字 0～9 共 10 類，或者語言模型的 5 萬個 token），就讓模型輸出 `K` 個分數（logits），再用 **softmax** 變成機率分佈：

```text
輸入 x ──► 線性層：K 個分數 z = W·x + b ──► softmax ──► K 個機率 ──► 交叉熵損失
           W 的形狀是 K × d                     （加起來是 1）     −log p[正確類別]
```

這叫 **softmax 迴歸**（也叫多項邏輯迴歸）。語言模型的**最後一層**正是這個結構：把 Transformer 輸出的向量乘以一個 `V × d` 的矩陣，得到 `V` 個 logits，再做 softmax。

## 2.3 最重要的梯度：p − y

softmax + 交叉熵的組合，對 logits 的梯度**簡單得驚人**：

```formula
∂L/∂zₖ = pₖ − yₖ
```

其中 `p` 是 softmax 輸出的機率，`y` 是正確答案的 **one-hot 向量**（正確類別處為 1，其餘為 0）。

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

z = [2.0, 1.0, 0.1]          # 3 個類別的分數
target = 0                    # 正確答案是第 0 類
p = softmax(z)
y = [1 if k == target else 0 for k in range(len(z))]     # one-hot：[1, 0, 0]
grad = [pk - yk for pk, yk in zip(p, y)]
print("機率：", [round(v, 4) for v in p])
print("梯度：", [round(v, 4) for v in grad])
```

直覺：

- 對**正確類別**，梯度 `p − 1` 是負數 → 梯度下降會**增大**它的分數；
- 對**錯誤類別**，梯度 `p` 是正數 → 梯度下降會**減小**它的分數；
- 預測越準（正確類別的 p 越接近 1），梯度越小，調整就越少。

> [!IMPORTANT]
> 記住 **p − y**。訓練語言模型時，每個位置的輸出層梯度都是這樣算的：把模型給出的機率分佈，減去「正確 token 處為 1」的向量。

用數值導數驗證一下這個公式（第一部分第 5 課的方法）：

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
print("數值梯度：", [round(v, 4) for v in numeric])
```

兩種方法結果一致。

## 2.4 動手：訓練一個二維分類器

我們生成兩團點：類別 0 在左下，類別 1 在右上；訓練一個 softmax 迴歸來區分它們。

```python
import math
import random
random.seed(0)

# 資料：每個樣本是 (x1, x2)，標籤 0 或 1
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
                gW[k][d] += g * x[d]                   # 鏈式法則：∂z/∂W = x
    n = len(data)
    for k in range(K):
        b[k] -= lr * gb[k] / n
        for d in range(D):
            W[k][d] -= lr * gW[k][d] / n
    if epoch % 10 == 0:
        acc = sum(max(range(K), key=lambda k: forward(x)[k]) == label for x, label in data) / n
        print(f"第 {epoch} 輪：平均損失 {total_loss / n:.4f}，準確率 {acc:.0%}")
```

幾十輪之後，準確率接近 100%。**epoch**（輪）的意思是把全部訓練資料完整地過一遍。

### 準確率

分類問題除了看損失，還常看**準確率**（accuracy）：預測類別（機率最大的那個，即 argmax）等於真實類別的比例。損失是訓練時用來求梯度的，準確率則更直觀，便於人來判斷。

## 2.5 從分類到語言模型

把上面的 `K` 換成詞表大小（比如 5 萬），把輸入 `x` 換成「前文的向量表示」，這就是語言模型最後一步在做的事：

| | 二維點分類 | 語言模型 |
|---|---|---|
| 類別數 K | 2 | 詞表大小（幾萬～十幾萬） |
| 輸入 x | 座標 (x1, x2) | Transformer 對前文的理解（幾千維向量） |
| 輸出 | 2 個機率 | 每個 token 是下一個 token 的機率 |
| 損失 | 交叉熵 | 交叉熵（對每個位置求平均） |
| 梯度 | p − y | p − y |

## 2.6 小結

- **sigmoid** `1/(1+e^(−z))` 把任意實數壓到 0～1；線性模型 + sigmoid = **邏輯迴歸**（二分類），損失為二元交叉熵。
- 多分類：線性層輸出 K 個 logits → **softmax** → 交叉熵，叫 **softmax 迴歸**；語言模型的輸出層就是它。
- softmax + 交叉熵對 logits 的梯度 = **p − y**（y 是 one-hot）。
- 訓練時把每個樣本的梯度累加求平均再更新；**epoch** 是完整過一遍資料。
- 分類常用**準確率**評估：argmax 預測與真實標籤一致的比例。

## 2.7 練習

<!-- exercise: sigmoid -->
### 練習 1：sigmoid

輸入一行若干個數 z，輸出每個數的 sigmoid 值（保留 4 位小數，空格隔開）。

```text
樣例輸入：0 2 -2
樣例輸出：0.5000 0.8808 0.1192
```

<details>
<summary>提示</summary>

`1 / (1 + math.exp(-z))`
</details>

<!-- exercise: softmax-grad -->
### 練習 2：softmax 交叉熵的梯度

第一行是若干個 logits，第二行是正確類別的下標。輸出交叉熵損失對每個 logit 的梯度（`p − y`），保留 4 位小數，空格隔開。

```text
樣例輸入：
2 1 0.1
0
樣例輸出：
-0.3410 0.2424 0.0986
```

<details>
<summary>提示</summary>

先用穩定的 softmax 求出 p，再對正確類別減 1。
</details>

<!-- exercise: accuracy -->
### 練習 3：準確率

第一行是整數 `n` 和 `K`（樣本數和類別數）。接下來 `n` 行，每行先是 `K` 個模型輸出的機率，最後一個數是真實類別下標。輸出準確率（預測類別取機率最大的那個；若有並列，取下標較小的），用百分數表示，保留 1 位小數，例如 `75.0%`。

```text
樣例輸入：
4 3
0.7 0.2 0.1 0
0.1 0.3 0.6 2
0.5 0.4 0.1 1
0.2 0.5 0.3 1
樣例輸出：
75.0%
```

<details>
<summary>提示</summary>

每行 `vals = input().split()`，前 `K` 個是機率，最後一個是標籤。預測類別：`max(range(K), key=lambda k: probs[k])`——並列時 `max` 返回第一個，正好是下標較小的。
</details>

<!-- exercise: grad-sign -->
### 練習 4：梯度的方向

用 softmax + 交叉熵訓練時，**正確類別**那個 logit 的梯度 `p − 1` 一定是？

- 正數，所以它的分數會被減小
- 負數（或 0），所以梯度下降會增大它的分數
- 總是 0
- 取決於學習率

<details>
<summary>答案與解析</summary>

機率 p 不超過 1，所以 `p − 1 ≤ 0`。梯度下降減去「學習率 × 負數」，等於增大該類的分數——模型會越來越偏向正確答案。
</details>

<!-- exercise: lm-classes -->
### 練習 5：語言模型的類別

把語言模型看作一個分類器，它的「類別」是什麼？

- 0 和 1
- 詞表裡的所有 token
- 句子的情感（正面 / 負面）
- 文章的主題

<details>
<summary>答案與解析</summary>

每一步都是在詞表的所有 token 中選出「下一個」，類別數等於詞表大小。
</details>

## 2.8 延伸閱讀

- [《動手學深度學習》3.4 softmax 迴歸](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)
- CS231n：[Linear classification 講義](https://cs231n.github.io/linear-classify/)（英文）
- 下一課：[神經網路](../03-neural-networks/zh-CN.md)
