# 1. 語言模型：預測下一個字

> 🎯 **本課目標**：理解「語言模型」到底在做什麼——**預測下一個 token 的機率**；用機率的鏈式法則給一句話打分；親手做一個**字元級二元（bigram）模型**：數數、取樣生成新名字、用**困惑度**評估；再用一個最簡單的神經網路學到同樣的東西，為後面的 GPT 鋪路。
>
> ⏱ **預計用時**：75 分鐘　｜　📋 **前置知識**：第一部分第 6 課（機率與資訊）；第二部分第 2 課（softmax 與交叉熵）

第一部分我們說過：ChatGPT、DeepSeek、Qwen 這些大模型，本質上都在反覆做同一件事——**根據前面的內容，預測下一個 token**。從這一課開始，我們將一步步親手造出一個這樣的模型。先從最簡單、最古老的方法開始。

> [!NOTE]
> 本部分的思路參考了 Andrej Karpathy 的 [makemore](https://github.com/karpathy/makemore) 和 [nanoGPT](https://github.com/karpathy/nanoGPT) 專案（MIT 許可證）及其配套影片，強烈推薦。

## 1.1 什麼是語言模型？

**語言模型**（language model）是一個給文字分配機率的模型。最常用的形式是：給定前面的內容，輸出**下一個 token 是各個可能值的機率**。

```text
輸入：「今天天氣真」
輸出：好 62%　　熱 11%　　冷 8%　　不 5%　　……（所有可能的字的機率加起來 = 1）
```

手機輸入法的聯想、搜尋框的自動補全，都是小型的語言模型。大語言模型（LLM）只是**大得多**（幾十億到上萬億個引數）、**看得遠得多**（能參考前面幾十萬個 token），並且用海量文字訓練出來的。

**token**（詞元）是模型處理文字的基本單位，可以是一個字、一個詞或詞的一部分。下一課會詳細講怎麼切分。這一課為了簡單，我們**以單個字元為 token**。

## 1.2 給一句話打分：機率的鏈式法則

一句話出現的機率，可以拆成「每一步預測下一個 token」的機率相乘：

```formula
P(w₁ w₂ … wₙ) = P(w₁) × P(w₂ | w₁) × P(w₃ | w₁ w₂) × … × P(wₙ | w₁ … wₙ₋₁)
```

`P(w₂ | w₁)` 讀作「已知 w₁ 時 w₂ 的機率」（條件機率）。所以，**只要能預測下一個 token，就能給任何文字打分，也能一個接一個地生成文字**——這就是 GPT 的全部工作方式。

```python
import math

# 假設模型對「我 愛 你」每一步的預測機率如下
steps = [("我", 0.05), ("愛", 0.10), ("你", 0.60)]
p = 1.0
for token, prob in steps:
    p *= prob
    print(f"P({token} | 前文) = {prob}")
print("整句機率：", round(p, 6))
print("負對數似然（NLL）：", round(-math.log(p), 3))   # 機率越小，NLL 越大
```

長句子的機率會小到幾乎為 0（很多個小於 1 的數相乘），所以實際中總是用**對數**：機率相乘變成對數相加，用**負對數似然**（negative log-likelihood，NLL）當作損失——這正是第二部分學過的交叉熵。

## 1.3 最簡單的語言模型：只看前一個字元

完整的條件機率 `P(wₙ | w₁ … wₙ₋₁)` 太複雜了。最簡單的近似是：**只看前一個 token**。

```formula
P(wₙ | w₁ … wₙ₋₁) ≈ P(wₙ | wₙ₋₁)
```

這叫**二元模型**（bigram）。只看前 2 個叫三元模型（trigram），一般地叫 **n-gram 模型**。n-gram 是統計語言模型的經典方法，夏農（Claude Shannon）在 1948 年就用它生成過「像英語」的文字。

我們來訓練一個**字元級 bigram 模型**，任務是學習英文名字的樣子，然後生成新名字。資料是一百多個常見英文名。每個名字前後加一個特殊字元 `.`，表示「開頭」和「結尾」，例如 `emma` 變成 `.emma.`，從中得到 5 對相鄰字元：`.e`、`em`、`mm`、`ma`、`a.`。

「訓練」就是**數數**：統計每個字元後面跟著各個字元的次數，再除以總數變成機率。

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

chars = list(".abcdefghijklmnopqrstuvwxyz")    # "." 表示名字的開頭和結尾
print(len(NAMES), "個名字，", len(chars), "種字元：", "".join(chars))

# 1. 數一數：每個字元後面跟著各個字元的次數
counts = {a: {b: 0 for b in chars} for a in chars}
for name in NAMES:
    seq = ["."] + list(name) + ["."]
    for a, b in zip(seq, seq[1:]):
        counts[a][b] += 1

# 2. 變成機率：P(b | a) = count(a, b) / count(a, 任意)；每個計數先加 0.1（平滑）
probs = {}
for a in chars:
    total = sum(counts[a][b] + 0.1 for b in chars)
    probs[a] = {b: (counts[a][b] + 0.1) / total for b in chars}

top = sorted(probs["."].items(), key=lambda kv: -kv[1])[:5]
print("名字最常見的開頭：", [(b, round(p, 3)) for b, p in top])

# 3. 取樣：從 "." 出發，按機率一個字元一個字元地「接龍」，直到又抽到 "."
random.seed(7)
def sample():
    out, cur = "", "."
    while True:
        cur = random.choices(chars, weights=[probs[cur][b] for b in chars])[0]
        if cur == ".":
            return out
        out += cur
print("生成的名字：", [sample() for _ in range(10)])

# 4. 評估：平均負對數似然（NLL）與困惑度（perplexity）
def avg_nll(words):
    nll, n = 0.0, 0
    for w in words:
        seq = ["."] + list(w) + ["."]
        for a, b in zip(seq, seq[1:]):
            nll += -math.log(probs[a][b])
            n += 1
    return nll / n

loss = avg_nll(NAMES)
print(f"訓練集：平均 NLL = {loss:.3f}，困惑度 = {math.exp(loss):.2f}")
print(f"瞎猜（均勻分佈）：平均 NLL = {math.log(len(chars)):.3f}，困惑度 = {len(chars)}")
print(f"新名字 'sophie'：{avg_nll(['sophie']):.3f}   亂碼 'zqxj'：{avg_nll(['zqxj']):.3f}")
```

逐段看看這個程式做了什麼。

### 取樣（生成）

生成文字的方法叫**自迴歸**（autoregressive）：從開頭符號出發，按模型給出的機率**隨機抽**一個字元，把它接上，再以它為條件抽下一個……直到抽到結尾符號。`random.choices(候選, weights=機率)` 就是「按機率抽籤」。GPT 生成回答時也是這樣，一個 token 接一個 token。

生成的名字有些像模像樣，有些很奇怪——因為模型**只記得前一個字元**，不知道整個名字已經多長、前面用過什麼音節。要做得更好，就需要看更長的上下文。

### 評估：困惑度

怎麼衡量一個語言模型好不好？看它給**真實文字**分配的機率高不高：用平均 NLL（越低越好），或者它的指數——**困惑度**（perplexity）：

```formula
困惑度 = e^(平均 NLL)
```

困惑度可以直觀地理解為：**模型在每一步「平均在幾個選項之間猶豫」**。完全瞎猜（27 個字元等機率）的困惑度就是 27；我們的 bigram 模型約為 9，說明它確實學到了一些規律。亂碼 `zqxj` 的 NLL 遠高於像名字的 `sophie`。

> [!NOTE]
> 困惑度是語言模型最經典的評估指標，至今仍用於比較預訓練模型（不同的分詞方式會影響它，所以只能在相同分詞下比較）。

### 平滑：為什麼要加 0.1？

如果某對字元在訓練資料裡從沒出現過（比如 `qx`），它的計數是 0，機率就是 0，`−log(0)` 是**無窮大**——只要遇到一次，整個評估就爆掉了。給每個計數加一個小數（這叫**平滑**，smoothing），就能讓「沒見過」的組合也有一點點機率。

## 1.4 用神經網路做同樣的事

數數很簡單，但它沒法擴充套件：如果要看前 10 個字元，可能的組合有 27¹⁰ ≈ 2×10¹⁴ 種，絕大多數從沒出現過，根本數不過來。**神經網路**的辦法是：不去數，而是**學一個函式**，輸入上下文，輸出下一個字元的機率。

先用神經網路重做 bigram，看它能否學到和數數一樣的結果：

- 輸入：當前字元（編號 0～26），表示成 **one-hot 向量**（第 i 位是 1，其餘是 0）；
- 模型：一個 27×27 的權重矩陣 `W`；one-hot 向量乘以 `W`，結果正好是 `W` 的第 i 行，作為 **logits**；
- 輸出：對 logits 做 softmax，得到下一個字元的機率；
- 損失：交叉熵（就是上面的平均 NLL）；用梯度下降訓練。

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
idx = {c: i for i, c in enumerate(chars)}          # 字元 → 編號
V = len(chars)

pairs = []                                          # 訓練資料：(當前字元編號, 下一個字元編號)
for name in NAMES:
    seq = ["."] + list(name) + ["."]
    pairs += [(idx[a], idx[b]) for a, b in zip(seq, seq[1:])]
print("訓練樣本數：", len(pairs))

# 模型：一個 V×V 的權重矩陣。輸入字元 a 的 one-hot 向量乘以 W，正好取出第 a 行——這一行就是 logits
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
        p = softmax(W[a])                          # 前向：下一個字元的機率分佈
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
print("網路學到的 P(下一個 | 'a')：", [(chars[j], round(p[j], 3)) for j in best])
```

損失從約 3.3（瞎猜）降到約 2.14，和數數法幾乎一樣；網路學到的機率表也和計數得到的一致。**兩種方法學到的是同一個東西**，區別在於：

| | 數數（n-gram） | 神經網路 |
|---|---|---|
| 「訓練」 | 統計次數 | 梯度下降 |
| 上下文變長 | 組合數指數爆炸，資料不夠用 | 引數量可控，可以接收任意長的上下文 |
| 泛化 | 沒見過的組合只能靠平滑 | 相似的上下文會得到相似的預測 |

> [!TIP]
> 有趣的對應：神經網路版本里給權重加 L2 正則化（權重衰減），效果就相當於計數版本里的平滑——都是讓機率分佈不要太「極端」。

## 1.5 從 n-gram 到 GPT：一段簡史

| 年份 | 里程碑 | 關鍵思想 |
|------|--------|----------|
| 1948 | 夏農的 n-gram 實驗 | 用統計頻率預測下一個字母 / 單詞 |
| 1980s–2000s | 統計 n-gram 模型 | 語音識別、機器翻譯的核心元件 |
| 2003 | Bengio 的神經機率語言模型 | 用**詞向量** + 神經網路代替計數（下一節課會講詞向量） |
| 2010s | RNN / LSTM | 用迴圈結構「記住」很長的上下文 |
| 2017 | **Transformer**（《Attention Is All You Need》） | 用**注意力**一次看到所有上下文，可以高度並行訓練 |
| 2018– | GPT、BERT 系列 | 在海量文字上預訓練巨大的 Transformer |
| 2022– | ChatGPT 之後的 LLM 時代 | 指令微調、對齊、推理模型…… |

接下來幾課，我們會沿著這條路線一步步走：**分詞**（第 2 課）→ **詞向量**（第 3 課）→ **注意力**（第 4 課）→ **Transformer**（第 5 課）→ 親手訓練一個**小 GPT**（第 6 課）→ **解碼策略**（第 7 課）。

## 1.6 小結

- 語言模型 = 預測**下一個 token 的機率分佈**；用鏈式法則可以給整段文字打分，用**自迴歸取樣**可以生成文字。
- 實際計算用**負對數似然**（NLL，即交叉熵）；**困惑度** = e^(平均 NLL)，可以理解為「每一步在幾個選項之間猶豫」，越低越好。
- **n-gram** 模型只看前 n−1 個 token，靠計數訓練；沒見過的組合需要**平滑**；上下文一長就數不過來。
- 一個 one-hot 輸入 + 線性層 + softmax 的神經網路，用梯度下降能學到與計數完全相同的 bigram 模型——而神經網路可以擴充套件到長得多的上下文。

## 1.7 練習

<!-- exercise: perplexity -->
### 練習 1：計算困惑度

輸入一行若干個機率，表示模型在一句話每一步給正確 token 的機率。輸出三個數（空格隔開）：整句機率（保留 6 位小數）、平均 NLL（保留 4 位小數）、困惑度（保留 4 位小數）。

```text
樣例輸入：0.5 0.25 0.5
樣例輸出：0.062500 0.9242 2.5198
```

<details>
<summary>提示</summary>

平均 NLL = `-sum(math.log(p) for p in ps) / len(ps)`，困惑度 = `math.exp(平均 NLL)`。
</details>

<!-- exercise: bigram-next -->
### 練習 2：最可能的下一個字元

第一行是若干個單詞（空格隔開），第二行是一個字元 `c`。像本課一樣，在每個單詞前後加上 `.`，統計相鄰字元對。輸出**跟在 `c` 後面次數最多**的字元；如果有並列，輸出其中按字元順序最小的那個（`.` 排在所有字母前面）。

```text
樣例輸入：
emma ella eva
e
樣例輸出：l
```

<details>
<summary>提示</summary>

`e` 後面分別跟著 `m`、`l`、`v` 各一次，並列時取最小的 `l`。可以用 `sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]`。
</details>

<!-- exercise: why-weird -->
### 練習 3：為什麼生成的名字很奇怪？

bigram 模型生成的名字常常很長或者讀不通，最主要的原因是？

- 訓練資料裡有錯別字
- 模型只看前一個字元，不知道更早的內容（比如名字已經多長）
- 隨機種子選得不好
- 字元表裡少了大寫字母

<details>
<summary>答案與解析</summary>

bigram 只根據前一個字元預測下一個，完全「忘記」了更早的內容，所以無法保證整體像一個名字。看更長的上下文是改進的關鍵——這正是後面的神經網路和 Transformer 要做的。
</details>

<!-- exercise: zero-prob -->
### 練習 4：沒有平滑會怎樣？

如果不做平滑，而測試文字里出現了一個訓練集中從未出現過的字元對，那麼測試集的平均 NLL 會是？

- 0
- 無窮大
- 不受影響
- 變成負數

<details>
<summary>答案與解析</summary>

沒出現過的組合機率為 0，`−log(0) = ∞`，平均值也就是無窮大（困惑度無窮大）。這就是需要平滑的原因。
</details>

<!-- exercise: uniform-ppl -->
### 練習 5：瞎猜的困惑度

一個「什麼都沒學」的模型，對詞表裡 50000 個 token 都給出相同的機率。它的困惑度是多少？

- 1
- 約 10.8（即 ln 50000）
- 50000
- 25000

<details>
<summary>答案與解析</summary>

每個 token 的機率都是 1/50000，NLL = ln 50000，困惑度 = e^(ln 50000) = 50000。均勻分佈的困惑度就等於選項個數——這也是「困惑度 = 在幾個選項之間猶豫」的由來。
</details>

## 1.8 延伸閱讀

- Andrej Karpathy：[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)（本課的靈感來源）
- Claude Shannon：《A Mathematical Theory of Communication》（1948），第 3 節用 n-gram 生成英文
- 《動手學深度學習》：[語言模型和資料集](https://zh.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html)
- 下一課：[分詞：文字怎樣變成 token](../02-tokenization/zh-CN.md)
