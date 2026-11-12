# 4. 注意力機制：讓每個 token「看到」上下文

> 🎯 **本課目標**：明白為什麼需要注意力；從「求平均」一步步推匯出**自注意力**；理解 **Query、Key、Value** 的直覺（一次「模糊的查字典」）；掌握縮放點積注意力公式 `softmax(QKᵀ/√d)·V`、**因果掩碼**和**多頭注意力**；用純 Python 和 PyTorch 各實現一遍；理解注意力的 O(n²) 代價，以及 **RoPE** 旋轉位置編碼為什麼能表示相對位置。
>
> ⏱ **預計用時**：100 分鐘　｜　📋 **前置知識**：本部分第 1～3 課；第一部分第 4 課（矩陣乘法）

這是整門課最核心的一課。2017 年的論文《Attention Is All You Need》提出的 **Transformer**，幾乎完全由注意力構成；今天所有的大模型，GPT、Claude、Llama、Qwen、DeepSeek，都建立在它之上。慢慢讀，每段程式碼都執行一下。

> [!NOTE]
> 本課的推導順序參考了 Andrej Karpathy 的影片 [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)，強烈推薦配合觀看。

## 4.1 為什麼需要注意力？

上一課的神經語言模型只看**固定的前 3 個**字元，然後把它們拼起來。這有兩個問題：

1. **視窗固定**：想看前 1000 個 token，輸入就要拼成 1000 倍長，引數也跟著暴增；
2. **一視同仁**：每個位置都按固定方式處理，但真正有用的資訊可能在任何位置。

看這句話：「小明把書還給了小紅，因為**她**明天要考試。」要理解「她」指誰，模型需要**回頭去找**前面相關的詞（小紅），而且這個詞可能隔得很遠。我們希望：**每個 token 都能根據需要，從前面所有 token 中「挑出」相關的資訊，彙總到自己身上。** 這就是**注意力**（attention）。

## 4.2 第一步：把前面的資訊平均起來

最簡單的「彙總前文」方法：第 t 個位置的輸出 = 前 t 個位置（包括自己）向量的**平均值**。

```python
# 4 個 token，每個是 2 維向量（假裝是嵌入）
X = [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [1.0, 1.0]]

# 第 t 個位置的輸出 = 前 t+1 個向量（包括自己）的平均
out = []
for t in range(len(X)):
    seen = X[:t + 1]
    out.append([sum(v[i] for v in seen) / len(seen) for i in range(2)])
for t, v in enumerate(out):
    print(f"位置 {t}: 平均了 {t + 1} 個 token →", [round(a, 3) for a in v])
```

注意：每個位置**只能看自己和之前的位置**，不能看後面的——因為語言模型要預測下一個 token，如果能「偷看」後面的答案，就學不到東西了。這叫**因果**（causal）約束。

平均有個明顯的缺點：所有前文的**權重都一樣**。我們想讓相關的 token 權重大、無關的權重小，而且權重**取決於內容**。

## 4.3 第二步：按相關程度加權——Query、Key、Value

把注意力想象成一次**模糊的查字典**（或者搜尋引擎）：

- **Query（查詢）**：「我想找什麼？」——當前 token 發出的問題；
- **Key（鍵）**：「我這裡有什麼？」——每個 token 貼出的標籤；
- **Value（值）**：「如果你選中我，我給你什麼」——每個 token 提供的內容。

用 query 和每個 key 做**點積**得到匹配分數，用 **softmax** 把分數變成加起來等於 1 的權重，再用權重對 value 做**加權平均**：

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# 一個「迷你搜尋引擎」：每條資料有一個 key（標籤）和一個 value（內容）
keys   = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
values = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
query  = [1.0, 0.2]                          # 我想找的東西

scores = [dot(query, k) for k in keys]       # 和每個 key 有多匹配
weights = softmax(scores)                    # 變成加起來等於 1 的權重
output = [sum(w * v[i] for w, v in zip(weights, values)) for i in range(2)]
print("匹配分數：", [round(s, 3) for s in scores])
print("注意力權重：", [round(w, 3) for w in weights])
print("輸出（各 value 的加權平均）：", [round(o, 3) for o in output])
```

普通的字典要求「完全匹配」才取出一項；注意力是「**按匹配程度取出所有項的混合**」，而且這個過程處處可導，可以用梯度下降訓練。

**自注意力**（self-attention）中，query、key、value **都來自同一串 token 本身**：每個 token 的嵌入 `x` 分別乘以三個可學習的矩陣，得到它的 q、k、v：

```formula
q = x·W_Q　　k = x·W_K　　v = x·W_V
```

模型透過訓練自己學會：該發出什麼樣的「問題」，該貼什麼樣的「標籤」，被選中時該提供什麼「內容」。

## 4.4 縮放點積注意力

把所有 token 的 q、k、v 疊成矩陣 Q、K、V（每行一個 token），所有位置的注意力可以用矩陣一次算完：

```formula
Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```

- `Q·Kᵀ` 是一個 n×n 的矩陣：第 i 行第 j 列 = 第 i 個 token 的 query 與第 j 個 token 的 key 的點積；
- 除以 `√d`（d 是 q、k 的維度）是**縮放**，下面解釋；
- softmax 按**每一行**做，所以每個 token 的權重加起來等於 1；
- 再乘以 V：每個 token 的輸出 = 所有 value 的加權平均。

### 為什麼要除以 √d？

d 維隨機向量的點積，大小大約隨 `√d` 增長。維度一大，分數的差距就大，softmax 會變得非常「尖」：幾乎所有權重都壓到一個 token 上，其他 token 的梯度接近 0，模型很難學。除以 `√d` 把分數拉回合適的範圍：

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
    print(f"d={d:4d}  不縮放時最大的權重 {max(softmax(raw)):.3f}   除以 √d 後 {max(softmax(scaled)):.3f}")
```

### 因果掩碼

語言模型裡，第 i 個 token 只能看位置 ≤ i 的 token。做法是在 softmax **之前**把「未來」位置的分數設成 **−∞**：e^(−∞) = 0，softmax 之後這些位置的權重就是 0。權重矩陣因此是**下三角**的。

## 4.5 純 Python 實現完整的因果自注意力

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
tokens = ["我", "愛", "大", "模型"]
n, d = len(tokens), 4
X = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]          # 4 個 token 的嵌入（n×d）
Wq = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]       # 三個可學習的矩陣（d×d）
Wk = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]
Wv = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]

Q, K, V = matmul(X, Wq), matmul(X, Wk), matmul(X, Wv)                   # 每個 token 的 query、key、value
scores = matmul(Q, [list(c) for c in zip(*K)])                          # Q @ Kᵀ：n×n 的匹配分數
scale = 1 / math.sqrt(d)
weights = []
for i in range(n):
    row = [scores[i][j] * scale if j <= i else float("-inf") for j in range(n)]   # 因果掩碼：看不到未來
    weights.append(softmax(row))
out = matmul(weights, V)                                                # 加權平均 value

print("注意力權重（第 i 行 = 第 i 個 token 在看誰）：")
print("        " + "".join(f"{t:>8}" for t in tokens))
for t, row in zip(tokens, weights):
    print(f"{t:>6}  " + "".join(f"{w:8.3f}" for w in row))
print("輸出形狀：", len(out), "×", len(out[0]))
```

權重矩陣是下三角的，每一行加起來等於 1。現在的權重是隨機矩陣算出來的，沒有意義；**訓練之後**，模型會學到有用的模式，例如代詞去「看」它指代的名詞，動詞去「看」它的主語。

> [!TIP]
> 注意力的輸出形狀和輸入一樣（n 個 token，每個 d 維），所以可以一層接一層地疊起來。每個 token 的向量經過一層注意力後，就**融入了上下文的資訊**：同一個「蘋果」，在「我吃了一個蘋果」和「蘋果釋出了新手機」裡，經過注意力後得到的向量是不同的。

## 4.6 多頭注意力

一個 token 可能同時需要好幾種資訊：它的主語是誰、前一個詞是什麼、這句話在講什麼主題……一組 Q、K、V 只能算出**一種**權重分佈。**多頭注意力**（multi-head attention）就是並行做 h 次注意力：

1. 把 d 維切成 h 份，每個「頭」用自己的 Q、K、V（每份 d/h 維）獨立地做注意力；
2. 把 h 個頭的輸出**拼接**回 d 維；
3. 再乘一個輸出矩陣 `W_O` 混合各頭的資訊。

計算量和單頭差不多，但模型可以同時關注不同的關係。研究發現，訓練好的模型裡確實有專門的頭：有的只看前一個 token，有的負責「複製」前文出現過的模式（**歸納頭**，induction head）……

下面用 PyTorch 實現一個多頭因果自注意力，結構和 GPT-2 的程式碼幾乎一樣：

```python requires=torch
import math
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads          # 頭數、每個頭的維度
        self.qkv = nn.Linear(d_model, 3 * d_model)             # 一次算出 Q、K、V
        self.proj = nn.Linear(d_model, d_model)                # 輸出投影 W_O

    def forward(self, x):
        B, T, C = x.shape                                      # batch、序列長度、維度
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, h, T, dh)：把 C 維切成 h 個頭，每個頭單獨做注意力
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.dh)    # (B, h, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # 下三角：只能看自己和之前
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                            # (B, h, T, dh)
        y = y.transpose(1, 2).reshape(B, T, C)                 # 把各個頭拼回來
        return self.proj(y), att

torch.manual_seed(0)
attn = CausalSelfAttention(d_model=32, n_heads=4)
x = torch.randn(2, 5, 32)                                      # 2 個序列，每個 5 個 token，每個 32 維
y, att = attn(x)
print("輸出：", tuple(y.shape), " 注意力權重：", tuple(att.shape))
print("第 1 個序列、第 1 個頭的注意力權重：")
print(att[0, 0].detach().round(decimals=2))
print("引數量：", sum(p.numel() for p in attn.parameters()))   # 4 × 32 × 32 + 4 × 32

# PyTorch 自帶的高效實現（會自動選用 FlashAttention 等演算法），結果應該一樣
q, k, v = attn.qkv(x).split(32, dim=2)
q, k, v = (t.view(2, 5, 4, 8).transpose(1, 2) for t in (q, k, v))
fast = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 5, 32)
print("和 F.scaled_dot_product_attention 一致：", torch.allclose(attn.proj(fast), y, atol=1e-6))
```

多頭注意力的引數：`W_Q`、`W_K`、`W_V`、`W_O` 四個 d×d 矩陣，共 **4d²** 個（加上偏置 4d 個）。GPT-2 small（d = 768）每層注意力約 236 萬個引數。

## 4.7 注意力的代價：O(n²)

注意力分數矩陣是 **n×n** 的：序列長度翻倍，計算量和記憶體翻**四**倍。100 萬個 token 的上下文就有 10¹² 個分數——這就是「長上下文」昂貴的原因。研究者想了很多辦法：

| 技術 | 思路 | 使用者 |
|------|------|--------|
| **FlashAttention** | 不把 n×n 矩陣完整存下來，分塊在 GPU 快取記憶體裡算完，結果完全相同但快得多、省記憶體 | 幾乎所有現代訓練和推理框架 |
| **KV 快取** | 生成時把之前算過的 K、V 存起來，每生成一個新 token 只算新的那一行 | 所有大模型推理（第六部分詳細講） |
| **GQA / MQA** | 多個 query 頭**共用**一組 K、V 頭，KV 快取小好幾倍 | Llama 3、Qwen 等 |
| **MLA**（多頭潛在注意力） | 把 K、V 壓縮成一個低維「潛在向量」再快取 | DeepSeek-V2/V3 |
| **滑動視窗 / 稀疏注意力** | 每個 token 只看附近的一段，或者只挑選部分 token 去看 | Mistral、Gemma、DeepSeek 的稀疏注意力等 |
| **線性注意力 / 狀態空間模型** | 用別的結構替代 softmax 注意力，代價隨長度線性增長 | Mamba，以及一些混合架構 |

## 4.8 位置資訊：RoPE 旋轉位置編碼

上一課說過：注意力本身不知道順序——把 token 打亂，每個 token 的權重分佈不變。GPT-2 的做法是給輸入**加上**位置嵌入。現代模型（Llama、Qwen、DeepSeek、Mistral……）幾乎都用 **RoPE**（Rotary Position Embedding，旋轉位置編碼，蘇劍林等 2021 年提出）：

- 把 q、k 向量的維度兩兩分組，每組看成平面上的一個點；
- 位置為 m 的 token，把它的每組都**旋轉** `m × θ` 角度（不同的組轉速 θ 不同）；
- 兩個向量都旋轉後再做點積，結果只和**角度差**——也就是**相對位置 m − n**——有關。

```python
import math

def rotate(v, angle):                      # 把二維向量逆時針旋轉 angle 弧度
    x, y = v
    return [x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)]

theta = 0.5                                 # 每往後一個位置，多轉 0.5 弧度
q, k = [1.0, 0.5], [0.3, 1.0]
for m, n in [(5, 3), (10, 8), (102, 100), (3, 5), (7, 7)]:
    qm, kn = rotate(q, m * theta), rotate(k, n * theta)   # q 在位置 m，k 在位置 n
    score = qm[0] * kn[0] + qm[1] * kn[1]
    print(f"q 在位置 {m:3d}，k 在位置 {n:3d}，相距 {m - n:2d}：q·k = {score:.4f}")
```

只要相距一樣（都是 2），不管在第 5 位還是第 102 位，注意力分數都相同。這正是我們想要的：「前一個詞」「三個詞之前」這樣的**相對關係**才是語言中重要的。真實的 RoPE 中，第 i 組的轉速是 `θᵢ = base^(−2i/d)`（base 通常是 10000，Llama 3 用 500000），既有轉得快的組（區分近處的位置），也有轉得很慢的組（區分遠處的位置）。調整 base 等引數（如 YaRN 方法）還能讓模型處理比訓練時更長的上下文。

## 4.9 小結

- 注意力讓每個 token 根據內容，從（之前的）所有 token 中**加權彙總**資訊；權重由 query 與 key 的匹配程度決定。
- 自注意力：`q = xW_Q, k = xW_K, v = xW_V`；`Attention = softmax(QKᵀ/√d)·V`；除以 `√d` 防止 softmax 過尖。
- **因果掩碼**：softmax 前把未來位置設為 −∞，權重矩陣是下三角的。
- **多頭注意力**：h 個頭並行關注不同關係，拼接後乘 `W_O`；引數約 4d²。
- 代價隨序列長度 **O(n²)** 增長；FlashAttention、KV 快取、GQA/MLA、稀疏注意力等都是為了降低代價。
- **RoPE** 按位置旋轉 q、k，使注意力分數只依賴相對位置，是現代大模型的標準做法。

## 4.10 練習

<!-- exercise: attn-weights -->
### 練習 1：注意力權重

第一行是 query 向量，第二行是整數 `n`，接下來 `n` 行是 `n` 個 key 向量（和 query 一樣長）。輸出縮放點積注意力的權重 `softmax(q·kⱼ / √d)`（d 是向量長度），保留 4 位小數，空格隔開。

```text
樣例輸入：
1 0
2
1 0
0 1
樣例輸出：0.6698 0.3302
```

<details>
<summary>提示</summary>

分數 `s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q)) for k in keys]`，再做 softmax（記得先減去最大值，防止溢位）。
</details>

<!-- exercise: causal-mask -->
### 練習 2：因果掩碼

輸入整數 `n`，輸出 `n×n` 的因果掩碼：第 i 行第 j 列，如果第 i 個 token 能看到第 j 個 token（j ≤ i）就是 `1`，否則是 `0`。每行的數字用空格隔開。

```text
樣例輸入：3
樣例輸出：
1 0 0
1 1 0
1 1 1
```

<details>
<summary>提示</summary>

兩層迴圈，`1 if j <= i else 0`；或者 `" ".join("1" if j <= i else "0" for j in range(n))`。
</details>

<!-- exercise: attn-output -->
### 練習 3：注意力的輸出

第一行是 query 向量，第二行是整數 `n`，接下來 `n` 行是 key 向量，再接下來 `n` 行是 value 向量。計算縮放點積注意力的輸出（各 value 按權重的加權平均），保留 4 位小數，空格隔開。

```text
樣例輸入：
1 0
2
1 0
0 1
10 0
0 10
樣例輸出：6.6976 3.3024
```

<details>
<summary>提示</summary>

先像練習 1 一樣算出權重 `w`，然後輸出向量的第 i 個分量是 `sum(wj * v[i] for wj, v in zip(w, values))`。
</details>

<!-- exercise: mha-params -->
### 練習 4：多頭注意力的引數量

輸入一行兩個整數 `d bias`：模型維度，以及是否有偏置（1 有，0 沒有）。輸出一層多頭注意力（`W_Q`、`W_K`、`W_V`、`W_O` 四個 d×d 矩陣，有偏置時每個矩陣再加 d 個偏置）的引數總數。注意：引數量和頭數無關。

```text
樣例輸入：768 1
樣例輸出：2362368
```

<details>
<summary>提示</summary>

`4 * d * d + (4 * d if bias else 0)`
</details>

<!-- exercise: why-scale -->
### 練習 5：為什麼除以 √d？

- 讓計算更快
- 維度越大點積的數值越大，softmax 會變得很「尖」、梯度接近 0；除以 √d 把分數拉回合適範圍
- 為了讓權重加起來等於 1
- 為了實現因果掩碼

<details>
<summary>答案與解析</summary>

d 維隨機向量點積的標準差約為 √d。不縮放的話，高維時 softmax 幾乎變成 one-hot，大多數位置得不到梯度。「加起來等於 1」是 softmax 本身保證的。
</details>

<!-- exercise: quadratic -->
### 練習 6：上下文翻倍

其他條件不變，把序列長度從 4000 個 token 增加到 8000 個，注意力分數矩陣的元素個數會變成原來的幾倍？

- 1 倍（不變）
- 2 倍
- 4 倍
- 8 倍

<details>
<summary>答案與解析</summary>

分數矩陣是 n×n 的：8000² / 4000² = 4。這就是注意力代價隨長度「平方增長」的含義，也是長上下文昂貴的原因。
</details>

## 4.11 延伸閱讀

- Vaswani 等：《Attention Is All You Need》（2017）——Transformer 的原始論文
- Jay Alammar：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（經典圖解，網上有中文翻譯）
- 3Blue1Brown：[Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention)（深度學習系列第 6 集；在 B 站搜尋「3Blue1Brown 注意力機制」可以找到官方中文版）
- 蘇劍林：《Transformer 升級之路：博採眾長的旋轉式位置編碼》（RoPE 作者的中文部落格，科學空間 kexue.fm）
- 下一課：[Transformer：把積木搭起來](../05-transformer/zh-CN.md)
