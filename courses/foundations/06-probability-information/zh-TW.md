# 6. 機率與資訊量：模型怎麼「打分」

> 🎯 **本課目標**：理解機率分佈；用 softmax 把任意分數變成機率；理解為什麼到處都是對數 `log`；學會交叉熵損失——幾乎所有語言模型訓練時最小化的那個數；用困惑度（perplexity）直觀衡量模型好壞。
>
> ⏱ **預計用時**：60 分鐘　｜　📋 **前置知識**：第 3～5 課

第 1 課說過：語言模型輸出的是**下一個 token 的機率分佈**。這一課把這句話裡的每個詞都弄明白，並學會給模型的預測「打分」。

## 6.1 機率分佈

**機率**是 0 到 1 之間的數，表示一件事有多可能發生。把所有可能結果的機率列出來，就是一個**機率分佈**，它們**加起來等於 1**。

```text
擲骰子：   1      2      3      4      5      6
機率：    1/6    1/6    1/6    1/6    1/6    1/6     （均勻分佈）

模型預測「今天天氣很___」的下一個字：
         好     熱     冷     差     怪
機率：   0.55   0.25   0.12   0.05   0.03            （加起來是 1）
```

### 按機率抽樣

有了分佈，就可以**隨機抽一個**——機率越大越容易被抽中。語言模型生成文字時就是這樣選詞的：

```python
import random
random.seed(0)

words = ["好", "熱", "冷", "差", "怪"]
probs = [0.55, 0.25, 0.12, 0.05, 0.03]
samples = random.choices(words, weights=probs, k=1000)
for w in words:
    print(w, samples.count(w) / 1000)       # 抽樣頻率接近原來的機率
```

如果每次都選機率最大的（「好」），叫**貪心解碼**（greedy decoding）；按機率隨機抽，生成的文字更多樣。第三部分第 7 課會專門講各種取樣策略。

## 6.2 Softmax：把分數變成機率

神經網路的最後一層輸出的是一組任意的實數，叫 **logits**（可以是負數，加起來也不等於 1）。要把它們變成機率，用 **softmax**：

```formula
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

兩步：① 每個數取指數 `e^z`（變成正數，並且放大差距）；② 除以總和（讓它們加起來等於 1）。

```python
import math

def softmax(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

logits = [2.0, 1.0, 0.1]
probs = softmax(logits)
print([round(p, 4) for p in probs])     # [0.659, 0.2424, 0.0986]
print(sum(probs))                        # 1.0（可能有極小的浮點誤差）
```

分數最高的項機率最大，但其他項也分到了一些——這比直接選最大值「溫和」，所以叫 soft（軟的）max。

### 數值穩定的寫法

`math.exp(1000)` 會溢位（結果太大，電腦存不下）。真實的 logits 可能很大，所以實際中會**先減去最大值**——結果完全一樣（分子分母同乘了 `e^(−max)`），但不會溢位：

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

print([round(p, 4) for p in softmax([1000.0, 999.0, 998.1])])   # 不會報錯
```

### 溫度：控制「確定程度」

把 logits 除以一個**溫度**（temperature）`T` 再做 softmax：

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

- `T < 1`：分佈更「尖」，幾乎總選最高分的，生成的內容更保守、穩定；
- `T > 1`：分佈更「平」，更隨機、更有創意，也更容易胡說。

你在聊天機器人的 API 裡看到的 `temperature` 引數，就是這個。

## 6.3 為什麼到處都是 log？

**對數**（logarithm）`log` 是指數的反運算：`e^x = y` ⇔ `x = ln y`（本課程的 `log` 都指以 e 為底的自然對數）。在機率計算中它無處不在，原因有二：

**① 把乘法變成加法。** 一句話的機率 = 每個詞機率的**乘積**。100 個詞，每個機率 0.1，乘起來是 10⁻¹⁰⁰——小到電腦無法精確表示（下溢）。取對數後變成**加法**：`log(a × b) = log a + log b`，數值穩定得多。

```python
import math
probs = [0.1] * 100
product = 1.0
for p in probs:
    product *= p
print(product)                              # 1e-100，再小一些就會變成 0.0
print(sum(math.log(p) for p in probs))      # -230.26：用 log 很安全
```

**② 衡量「意外程度」。** `−log p` 叫一件事的**資訊量**或**驚訝度**（surprisal）：

| 機率 p | −log p | 直覺 |
|--------|--------|------|
| 1.0 | 0 | 完全意料之中，毫不驚訝 |
| 0.5 | 0.69 | 拋硬幣的結果 |
| 0.1 | 2.30 | 有點意外 |
| 0.01 | 4.61 | 很意外 |
| 0.0001 | 9.21 | 非常震驚 |

## 6.4 交叉熵：給預測打分

訓練時，我們知道**正確的下一個 token** 是什麼。模型給它分配的機率越高，說明預測得越好。**交叉熵損失**（cross-entropy loss）就是**正確答案的驚訝度**：

```formula
loss = −log p(正確的 token)
```

```python
import math

def cross_entropy(probs, target):
    return -math.log(probs[target])

probs = [0.7, 0.2, 0.1]          # 模型對三個候選的預測
print(round(cross_entropy(probs, 0), 4))    # 正確答案是第 0 個：0.3567（預測得不錯）
print(round(cross_entropy(probs, 2), 4))    # 正確答案是第 2 個：2.3026（預測得很差）
```

- 正確答案機率為 1 → 損失為 0（完美）；
- 正確答案機率趨近 0 → 損失趨向無窮大（大錯特錯）。

訓練語言模型時，把一段文字裡**每個位置**的交叉熵**取平均**，就是這一批資料的損失。梯度下降（第 5 課）做的就是不斷調整引數，讓這個平均損失變小——**也就是讓模型給正確的下一個詞分配更高的機率**。

> [!IMPORTANT]
> 這就是大語言模型預訓練的完整目標：**最小化下一個 token 的平均交叉熵**。幾十億個引數、幾千張 GPU、幾個月的訓練，都在最佳化這一個數字。

一個有用的基準：如果詞表有 `V` 個 token，一個什麼都沒學會、對所有 token 一視同仁的模型，每個 token 的機率是 `1/V`，損失是 `log V`。GPT-2 的詞表約有 5 萬個 token，`log 50257 ≈ 10.8`——訓練剛開始時的損失大約就是這個數。

## 6.5 困惑度：模型在「幾個選項裡猶豫」

交叉熵不太直觀。把平均交叉熵取指數，得到**困惑度**（perplexity）：

```formula
perplexity = e^(平均交叉熵)
```

直覺：**模型平均在幾個選項之間猶豫不決**。困惑度為 1 表示每次都確信無疑（且正確）；困惑度為 50 表示好像在 50 個詞裡隨機猜。

```python
import math

def perplexity(target_probs):
    """target_probs：模型在每個位置給正確 token 的機率"""
    avg_ce = sum(-math.log(p) for p in target_probs) / len(target_probs)
    return math.exp(avg_ce)

print(round(perplexity([0.25, 0.25, 0.25, 0.25]), 2))    # 4.0：相當於在 4 個裡瞎猜
print(round(perplexity([0.9, 0.8, 0.95, 0.7]), 2))       # 約 1.2：很有把握
```

## 6.6 熵（瞭解即可）

一個分佈本身的「不確定程度」叫**熵**（entropy）：`H = −Σ pᵢ log pᵢ`，即「平均驚訝度」。均勻分佈的熵最大（最不確定），只有一個結果機率為 1 時熵為 0。之所以叫「**交叉**熵」，是因為它是用**模型的分佈**去衡量**真實資料**的平均驚訝度。

## 6.7 小結

- **機率分佈**：所有結果的機率，加起來為 1；`random.choices` 可以按機率抽樣。
- **Softmax**：`e^z / Σ e^z`，把任意 logits 變成機率；實現時先減最大值防溢位。
- **溫度**：logits 除以 T，T 小更確定、T 大更隨機。
- **log** 把乘法變加法、避免下溢；`−log p` 是驚訝度。
- **交叉熵損失** = `−log p(正確答案)`；語言模型訓練就是最小化每個位置的平均交叉熵。
- **困惑度** = `e^(平均交叉熵)`，表示模型平均在幾個選項間猶豫；隨機猜 V 個時困惑度為 V。

## 6.8 練習

<!-- exercise: softmax -->
### 練習 1：實現 softmax

輸入一行若干個數（logits），輸出 softmax 後的機率，每個保留 4 位小數，空格隔開。請使用「先減最大值」的穩定寫法。

```text
樣例輸入：2 1 0.1
樣例輸出：0.6590 0.2424 0.0986
```

<details>
<summary>提示</summary>

`m = max(z)`；`exps = [math.exp(v - m) for v in z]`；再除以 `sum(exps)`。格式化用 `f"{p:.4f}"`。
</details>

<!-- exercise: cross-entropy -->
### 練習 2：交叉熵

第一行是模型給出的機率分佈（若干個數，空格隔開），第二行是正確答案的下標（從 0 開始）。輸出交叉熵損失，保留 4 位小數。

```text
樣例輸入：
0.7 0.2 0.1
2
樣例輸出：
2.3026
```

<details>
<summary>提示</summary>

`-math.log(probs[target])`
</details>

<!-- exercise: perplexity -->
### 練習 3：困惑度

輸入一行若干個數：模型在每個位置分配給正確 token 的機率。輸出困惑度，保留 2 位小數。

```text
樣例輸入：0.5 0.25 0.125
樣例輸出：4.00
```

<details>
<summary>提示</summary>

先求平均交叉熵 `sum(-math.log(p) for p in ps) / len(ps)`，再 `math.exp(...)`。
</details>

<!-- exercise: temperature -->
### 練習 4：溫度

想讓聊天機器人的回答**更穩定、更少隨機性**（比如寫程式碼、回答事實問題），應該把溫度調成？

- 更高，例如 1.5
- 更低，例如 0.2
- 溫度對結果沒有影響
- 設定為負數

<details>
<summary>答案與解析</summary>

溫度越低，softmax 分佈越「尖」，越傾向於選擇最可能的 token，輸出更穩定。寫詩、頭腦風暴時可以調高一些。
</details>

<!-- exercise: uniform-loss -->
### 練習 5：隨機猜的損失

一個模型對詞表裡的 4 個 token 總是給出均勻的機率（各 0.25）。下面的程式輸出什麼？（保留 2 位小數）

```python
import math
print(f"{-math.log(0.25):.2f}")
print(f"{math.exp(-math.log(0.25)):.2f}")
```

<details>
<summary>答案與解析</summary>

交叉熵 `−log 0.25 = log 4 ≈ 1.39`；困惑度 `e^1.39 = 4.00`——正好是「在 4 個裡瞎猜」。
</details>

## 6.9 延伸閱讀

- [《動手學深度學習》3.4 softmax 迴歸](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)：從另一個角度講 softmax 和交叉熵。
- Chris Olah：[Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)（英文）：用圖解釋熵和交叉熵，非常漂亮。
- 第一部分完成！下一步：[第二部分 · 用梯度下降擬合直線](../../ml/01-linear-regression/zh-CN.md)
