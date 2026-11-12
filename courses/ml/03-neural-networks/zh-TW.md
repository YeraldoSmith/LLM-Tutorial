# 3. 神經網路：把簡單的東西疊起來

> 🎯 **本課目標**：明白線性模型的侷限（XOR 問題）；理解一個「神經元」在算什麼；認識 ReLU、GELU、SiLU 等啟用函式並知道為什麼必須要有它們；搭出一個多層感知機（MLP）並手算前向傳播；學會數引數量。
>
> ⏱ **預計用時**：60 分鐘　｜　📋 **前置知識**：本部分第 1、2 課；第一部分第 4 課（矩陣乘法）

## 3.1 一條直線不夠用：XOR 問題

線性模型（不管是迴歸還是 softmax 分類）只能畫**直線**（高維時是平面）來劃分類別。來看一個經典難題——**異或**（XOR）：兩個輸入相同時輸出 0，不同時輸出 1。

```text
 x2
 1 ┤  ●(0,1)→1        ○(1,1)→0
   │
 0 ┤  ○(0,0)→0        ●(1,0)→1
   └──┴─────────────────┴── x1
      0                 1
```

無論怎麼畫一條直線，都無法把兩個 ● 和兩個 ○ 分開。1969 年，這個問題曾讓很多人對神經網路失去信心。解決辦法是：**加一層「隱藏層」，並且在中間放一個非線性的函式。**

## 3.2 神經元：加權求和 + 啟用

一個**神經元**（neuron）做兩件事：

```formula
輸出 = 啟用函式( w₁x₁ + w₂x₂ + … + b ) = f(w · x + b)
```

1. **加權求和**：和線性模型一樣（點積 + 偏置）；
2. **啟用函式**（activation function）：一個**非線性**的函式 `f`，決定這個神經元「啟用」的程度。

許多神經元並排在一起組成一**層**（layer）；一層的輸出作為下一層的輸入，就組成了**神經網路**。

## 3.3 啟用函式

```python
import math

def relu(x):                 # 負數變 0，正數不變
    return max(0.0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):                 # 輸出在 -1 到 1 之間
    return math.tanh(x)

def gelu(x):                 # GPT 系列使用；ReLU 的「平滑版」
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

def silu(x):                 # 也叫 Swish；Llama、Qwen 等模型使用（SwiGLU 的一部分）
    return x * sigmoid(x)

print(" x     ReLU    GELU    SiLU    tanh")
for x in [-3, -1, -0.5, 0, 0.5, 1, 3]:
    print(f"{x:>4}  {relu(x):6.3f}  {gelu(x):6.3f}  {silu(x):6.3f}  {tanh(x):6.3f}")
```

| 啟用函式 | 特點 | 常見於 |
|----------|------|--------|
| **ReLU** `max(0, x)` | 簡單、計算快、不容易梯度消失 | 早期深度網路、CNN |
| **sigmoid** | 輸出 0～1；兩端梯度接近 0（梯度消失） | 二分類的輸出層、門控 |
| **tanh** | 輸出 -1～1 | RNN/LSTM |
| **GELU** | 平滑版 ReLU，負數處有一點點輸出 | BERT、GPT-2/3 |
| **SiLU / Swish** | `x · σ(x)`，平滑 | Llama、Qwen、DeepSeek 等（SwiGLU） |

### 為什麼一定要非線性？

如果沒有啟用函式，兩層線性變換疊在一起：`W₂(W₁x) = (W₂W₁)x`——**還是一個線性變換**！疊一百層也和一層一樣，依然畫不出曲線。啟用函式打破了這一點，讓網路能表示彎曲、複雜的邊界。

```python
def matvec(W, x):
    return [sum(w * v for w, v in zip(row, x)) for row in W]

W1 = [[1, 2], [3, 4]]
W2 = [[1, -1], [0, 2]]
x = [1, 1]
two_layers = matvec(W2, matvec(W1, x))
W21 = [[sum(W2[i][k] * W1[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
print(two_layers, matvec(W21, x))       # 完全一樣：兩層線性 = 一層線性
```

## 3.4 用兩層網路解決 XOR

下面這個網路有 2 個隱藏神經元，用 ReLU 啟用。權重是手工設計的（下一課我們會讓它自己學）：

```text
            h1 = ReLU(x1 + x2)
x1, x2 ──►                         ──► y = h1 − 2·h2
            h2 = ReLU(x1 + x2 − 1)
```

```python
def relu(v):
    return max(0, v)

def xor_net(x1, x2):
    h1 = relu(1 * x1 + 1 * x2 + 0)
    h2 = relu(1 * x1 + 1 * x2 - 1)
    return 1 * h1 - 2 * h2

for x1 in (0, 1):
    for x2 in (0, 1):
        print(x1, x2, "→", xor_net(x1, x2))
```

隱藏層把輸入「變換」到一個新的空間，在那裡問題變成線性可分的了——這就是隱藏層的作用：**學習更好的表示**（representation）。

## 3.5 多層感知機（MLP）

把多個全連線層（每個輸入都連到每個神經元）疊在一起，中間夾著啟用函式，就是**多層感知機**（Multi-Layer Perceptron，MLP），也叫前饋網路（feed-forward network）：

```text
輸入 x (d_in)
   │  線性層：h = W₁x + b₁       W₁ 形狀 d_hidden × d_in
   ▼  啟用：  h = ReLU(h)
隱藏層 h (d_hidden)
   │  線性層：z = W₂h + b₂       W₂ 形狀 d_out × d_hidden
   ▼
輸出 z (d_out) ── 分類時接 softmax
```

```python
import math
import random
random.seed(0)

def linear(W, b, x):
    return [sum(w * v for w, v in zip(row, x)) + bi for row, bi in zip(W, b)]

def relu_vec(v):
    return [max(0.0, a) for a in v]

def softmax(z):
    m = max(z)
    e = [math.exp(a - m) for a in z]
    s = sum(e)
    return [a / s for a in e]

d_in, d_hidden, d_out = 3, 4, 2
W1 = [[random.gauss(0, 0.5) for _ in range(d_in)] for _ in range(d_hidden)]
b1 = [0.0] * d_hidden
W2 = [[random.gauss(0, 0.5) for _ in range(d_hidden)] for _ in range(d_out)]
b2 = [0.0] * d_out

x = [1.0, 0.5, -1.0]
h = relu_vec(linear(W1, b1, x))      # 4 個隱藏值
z = linear(W2, b2, h)                # 2 個 logits
print("隱藏層：", [round(v, 3) for v in h])
print("機率：  ", [round(v, 3) for v in softmax(z)])
```

引數是隨機初始化的，所以輸出的機率還沒有意義——**訓練**（下一課：反向傳播）會把它們調整到有用的值。

### 萬能近似

理論上可以證明：只要隱藏層足夠寬，一個兩層 MLP 就能以任意精度逼近幾乎任何連續函式（**萬能近似定理**）。但實踐中，**更深**（層數多）的網路往往比只是更寬的網路更高效——這就是「深度學習」中「深度」的來歷。

## 3.6 數引數

一個從 `d_in` 維到 `d_out` 維的線性層有：

```formula
引數量 = d_in × d_out （權重） + d_out （偏置）
```

上面的 MLP：`3×4 + 4 = 16` 加上 `4×2 + 2 = 10`，共 26 個引數。

在 Transformer 裡，每一層都有一個這樣的 MLP（叫 FFN，前饋網路），通常是 `d → 4d → d`。以 GPT-2 small 為例，`d = 768`：

```python
def linear_params(d_in, d_out, bias=True):
    return d_in * d_out + (d_out if bias else 0)

d = 768
ffn = linear_params(d, 4 * d) + linear_params(4 * d, d)
print(f"GPT-2 small 每層 FFN 引數：{ffn:,}")            # 約 470 萬
print(f"12 層合計：{12 * ffn:,}")                        # 約 5700 萬，接近全模型 1.24 億的一半
```

大模型引數的很大一部分都在這些 MLP 裡。研究認為，MLP 層在很大程度上負責「儲存知識」。

## 3.7 小結

- 線性模型只能畫直線，無法解決 **XOR** 這樣的問題。
- **神經元** = 加權求和 + **啟用函式**；許多神經元組成**層**，層疊起來是**神經網路**。
- 常用啟用：ReLU、GELU（GPT）、SiLU/Swish（Llama 等）；沒有啟用函式，多層線性等於一層線性。
- **MLP**：線性 → 啟用 → 線性……隱藏層學習更好的**表示**；更深的網路往往更高效。
- 線性層引數量 = `d_in × d_out + d_out`；Transformer 的 FFN 通常是 `d → 4d → d`，佔了大量引數。

## 3.8 練習

<!-- exercise: activations -->
### 練習 1：啟用函式

輸入一個數 `x`，依次輸出 ReLU、sigmoid、GELU（用 `0.5·x·(1 + erf(x/√2))`）的值，每行一個，保留 4 位小數。

```text
樣例輸入：1
樣例輸出：
1.0000
0.7311
0.8413
```

<details>
<summary>提示</summary>

`math.erf` 是誤差函式，`math.sqrt(2)` 是 √2。
</details>

<!-- exercise: mlp-params -->
### 練習 2：數引數

輸入一行若干個整數，表示 MLP 每一層的寬度（第一個是輸入維度，最後一個是輸出維度）。每相鄰兩層之間是一個帶偏置的線性層。輸出總引數量。

```text
樣例輸入：3 4 2
樣例輸出：26
```

<details>
<summary>提示</summary>

`for a, b in zip(sizes, sizes[1:]): total += a * b + b`
</details>

<!-- exercise: forward -->
### 練習 3：前向傳播

起始程式碼裡給出了一個 2 → 2 → 1 的網路的權重（隱藏層用 ReLU，輸出層沒有啟用）。輸入一行兩個數 `x1 x2`，輸出網路的輸出值，保留 2 位小數。

```text
樣例輸入：1 2
樣例輸出：1.50
```

<details>
<summary>提示</summary>

先算隱藏層：`h = [relu(W1[i][0]*x1 + W1[i][1]*x2 + b1[i]) for i in range(2)]`，再算輸出：`W2[0]*h[0] + W2[1]*h[1] + b2`。
</details>

<!-- exercise: no-activation -->
### 練習 4：去掉啟用函式

把一個 10 層神經網路裡的所有啟用函式都去掉（只剩線性層），它的表達能力相當於？

- 一個 10 倍強大的網路
- 一個單層線性模型
- 一個只有 5 層的網路
- 沒有變化

<details>
<summary>答案與解析</summary>

線性變換的組合還是線性變換：`W₁₀⋯W₂W₁x = (W₁₀⋯W₁)x`。去掉非線性後，再多層也只等於一層。
</details>

<!-- exercise: xor-predict -->
### 練習 5：讀程式碼，寫輸出

```python
def relu(v):
    return max(0, v)

def net(x1, x2):
    h1 = relu(x1 + x2)
    h2 = relu(x1 + x2 - 1)
    return h1 - 2 * h2

print(net(1, 1), net(0, 1), net(2, 0))
```

<details>
<summary>答案與解析</summary>

`net(1,1)`：h1 = 2，h2 = 1，輸出 0；`net(0,1)`：h1 = 1，h2 = 0，輸出 1；`net(2,0)`：h1 = 2，h2 = 1，輸出 0。
</details>

## 3.9 延伸閱讀

- 3Blue1Brown：[深度學習系列第 1 集：神經網路是什麼](https://www.bilibili.com/video/BV1bx411M7Zx)（官方中文版）
- [TensorFlow Playground](https://playground.tensorflow.org/)：在瀏覽器裡調整層數和神經元，實時看網路怎麼劃分資料——非常直觀。
- 下一課：[反向傳播與自動求導](../04-backprop-autograd/zh-CN.md)
