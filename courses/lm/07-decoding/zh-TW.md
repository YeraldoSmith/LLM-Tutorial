# 7. 解碼：模型怎樣選出下一個詞

> 🎯 **本課目標**：明白模型只給出「下一個 token 的機率」，**怎麼選**是另一回事（解碼）；理解貪心解碼為什麼會重複、無聊；掌握**溫度**、**top-k**、**top-p**、**min-p** 取樣和**重複懲罰**；瞭解束搜尋、停止條件、**約束解碼**（結構化輸出）；知道呼叫大模型 API 時這些引數該怎麼設。
>
> ⏱ **預計用時**：60 分鐘　｜　📋 **前置知識**：本部分第 1、6 課

訓練好的語言模型，每一步只輸出一個**機率分佈**：下一個 token 是「好」的機率 63%，是「熱」的機率 14%……真正生成文字時，還要決定**選哪一個**。這一步叫**解碼**（decoding）或**取樣**（sampling）。同一個模型，解碼方式不同，回答的風格、品質和多樣性可以差別很大。你在 ChatGPT 等產品的 API 裡看到的 `temperature`、`top_p` 等引數，控制的就是這一步。

## 7.1 貪心解碼：每次都選最可能的

最簡單的方法：每一步都選**機率最大**的 token（上一課的加法 GPT 就是這樣）。它確定、可復現，適合「答案唯一」的任務（算術、分類、提取資訊）。但用來寫文章，問題很明顯：

- **無聊**：永遠選最常見的說法，沒有變化；
- **重複**：一旦進入迴圈就出不來。

看一個手工做的「詞級」小模型。它在 `cat` 之後最可能接 `the`，在 `the` 之後最可能接 `cat`……

```python
# 一個手工做的「詞級」模型：每個詞後面接各個詞的 logits
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
        for w in set(out):                        # 重複懲罰：已經出現過的詞，分數打折扣
            if w in scores:
                scores[w] = scores[w] / penalty if scores[w] > 0 else scores[w] * penalty
        cur = max(scores, key=scores.get)
        out.append(cur)
        if cur == ".":
            break
    return " ".join(out)

print("貪心：          ", greedy())
print("重複懲罰 1.5：  ", greedy(penalty=1.5))
```

貪心解碼陷入了 `the cat the cat …` 的死迴圈，直到長度上限。**重複懲罰**（repetition penalty）把已經出現過的詞的分數打個折扣，迴圈就被打破了。（懲罰太重也不好：模型會被迫避開本該重複的詞，比如「的」「the」。）

## 7.2 溫度：調節「大膽」的程度

想要多樣性，就按機率**隨機抽**（取樣）。**溫度**（temperature，T）在 softmax 之前把 logits 除以 T：

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

words = ["好", "熱", "冷", "不錯", "糟糕", "紫色"]
logits = [3.0, 1.5, 1.2, 1.0, 0.2, -2.0]          # 模型對「今天天氣真___」給出的分數

for T in (0.2, 0.7, 1.0, 1.5):
    probs = softmax(logits, T)
    print(f"T={T:<4}", "  ".join(f"{w}:{p:5.1%}" for w, p in zip(words, probs)))
```

- **T < 1**：分佈變「尖」，高機率的更高——更保守、更穩定。T → 0 時就等於貪心。
- **T = 1**：模型原本的分佈。
- **T > 1**：分佈變「平」，低機率的選項（甚至「紫色」）機會變大——更有創意，也更容易胡說。

用第 1 課的 bigram 名字模型直觀地感受一下：

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
counts = {a: {b: 0.1 for b in chars} for a in chars}      # 第 1 課的 bigram 模型（帶平滑）
for name in NAMES:
    seq = ["."] + list(name) + ["."]
    for a, b in zip(seq, seq[1:]):
        counts[a][b] += 1
logits = {a: [math.log(counts[a][b]) for b in chars] for a in chars}   # 把機率變回「logits」

def generate(T=1.0, top_k=None, greedy=False, max_len=15):
    out, cur = "", "."
    while len(out) < max_len:
        z = logits[cur]
        if greedy:
            cur = chars[max(range(len(z)), key=lambda i: z[i])]
        else:
            z = [v / T for v in z]                         # 溫度
            if top_k:                                      # top-k：只在最可能的 k 個裡抽
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
print("貪心          ", [generate(greedy=True) for _ in range(5)])
for T in (0.5, 1.0, 1.5):
    print(f"溫度 {T:<4}     ", [generate(T=T) for _ in range(5)])
print("溫度 1 + top-5", [generate(T=1.0, top_k=5) for _ in range(5)])
```

貪心每次都生成同一個名字 `a`（最常見的開頭是 `a`，`a` 後面最常見的是結尾）；溫度低時名字短而保守，溫度高時變成一長串亂碼；加上 top-k 之後，亂碼明顯少了。

## 7.3 截掉「長尾」：top-k、top-p、min-p

即使是好模型，詞表裡也有幾萬個機率很小但不為零的 token。隨機取樣時，「長尾」裡的怪詞加起來也有不小的機會被抽中，一旦抽中一個，後面可能就全亂了。所以通常在取樣前**只保留一部分候選**：

- **top-k**：只保留機率最高的 k 個；
- **top-p**（核取樣，nucleus sampling）：從最可能的開始累加，保留**累計機率剛好達到 p** 的最小集合。模型很確定時只留一兩個，不確定時留很多——比固定的 k 更靈活；
- **min-p**（較新的方法）：只保留機率**不低於「最大機率 × min_p」**的 token。

保留下來的候選重新歸一化（加起來等於 1），再按機率抽：

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
    for i in order:                      # 從最可能的開始累加，直到累計機率達到 p
        keep.append(i)
        cum += probs[i]
        if cum >= p:
            break
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]

def min_p(probs, ratio):
    limit = ratio * max(probs)           # 只保留機率不低於「最大機率 × ratio」的 token
    total = sum(q for q in probs if q >= limit)
    return [q / total if q >= limit else 0.0 for q in probs]

words = ["好", "熱", "冷", "不錯", "糟糕", "紫色"]
probs = softmax([3.0, 1.5, 1.2, 1.0, 0.2, -2.0])
show = lambda ps: "  ".join(f"{w}:{q:5.1%}" for w, q in zip(words, ps) if q > 0)
print("原始     ", show(probs))
print("top-k=3  ", show(top_k(probs, 3)))
print("top-p=0.9", show(top_p(probs, 0.9)))
print("min-p=0.1", show(min_p(probs, 0.1)))
```

實際中常把**溫度和 top-p（或 top-k、min-p）組合使用**：先用溫度調整分佈，再截掉長尾，最後抽樣。

## 7.4 懲罰重複：重複懲罰、頻率懲罰、存在懲罰

7.1 節已經見過**重複懲罰**。OpenAI 等的 API 還提供兩個類似的引數：

| 引數 | 做法 | 效果 |
|------|------|------|
| `frequency_penalty` | 某個 token 已經出現了幾次，就從它的 logit 裡減去「次數 × 懲罰值」 | 出現越多，越不想再用；減少囉嗦的重複 |
| `presence_penalty` | 只要出現過（不管幾次），就減去一個固定值 | 鼓勵談論新的話題 |
| `repetition_penalty`（Hugging Face、vLLM 等） | 出現過的 token，正的 logit 除以懲罰值、負的乘以懲罰值 | 打破迴圈 |

這些引數一般保持預設（0 或 1.0）即可，出現明顯的重複時再適當調大。

## 7.5 束搜尋

貪心每一步只看眼前最好的選擇，但「每一步都最好」不等於「整句最好」。**束搜尋**（beam search）每一步都保留 **b 個**累計機率最高的候選序列（b 叫束寬），最後選出整體機率最高的一個：

```text
束寬 b = 2
第 1 步：the (0.7)         a (0.3)
第 2 步：the cat (0.42)    the dog (0.21)     ← 從所有擴充套件裡只留下最好的 2 個
第 3 步：the cat sat (…)   the dog ran (…)
……
```

束搜尋在**機器翻譯、語音識別**這類「答案比較確定」的任務裡很常用。但在開放式寫作和對話裡，它容易生成平淡、重複的文字，所以聊天機器人基本都用**取樣**。

## 7.6 什麼時候停下來？

生成會一直繼續，直到滿足某個**停止條件**：

- 生成了**結束 token**（如 `<|endoftext|>`、`<|im_end|>`，第 2 課講過的特殊 token）——模型自己「決定」說完了；
- 達到**最大長度**（API 裡的 `max_tokens` / `max_output_tokens`）；
- 出現了你指定的**停止序列**（`stop` 引數），比如遇到 `"\n\n"` 就停。

> [!WARNING]
> 如果 `max_tokens` 設得太小，回答會在半句話處被截斷。推理模型（第五部分）在回答前會先生成很長的「思考」內容，這些也算在 token 數里，需要給足長度。

## 7.7 約束解碼：保證輸出符合格式

很多應用需要模型輸出**嚴格的格式**，比如合法的 JSON、只能是「是 / 否」、必須是數字。**約束解碼**（constrained decoding）在每一步取樣之前，把**不符合格式的 token 的 logit 設成 −∞**，這樣模型根本不可能輸出不合法的內容：

```python
import math

vocab = ["1", "2", "3", "七", "abc", "。", "<end>"]
logits = [1.0, 0.5, 0.2, 2.5, 1.8, 0.3, 0.1]       # 模型最想說「七」，但我們只要阿拉伯數字

allowed = {"1", "2", "3", "<end>"}                   # 格式規則：只允許數字和結束
masked = [z if tok in allowed else float("-inf") for tok, z in zip(vocab, logits)]
m = max(masked)
e = [math.exp(z - m) for z in masked]
probs = [v / sum(e) for v in e]
for tok, p in zip(vocab, probs):
    print(f"{tok:6s}{p:6.1%}")
```

真實系統會根據 JSON Schema 或語法規則，動態算出每一步允許哪些 token。OpenAI、Anthropic 等 API 的「結構化輸出」功能，以及 vLLM、llama.cpp 等推理框架的語法約束，背後都是這個原理。第七部分講工具呼叫時會再用到。

## 7.8 實用建議

| 場景 | 建議 |
|------|------|
| 程式碼、數學、提取資訊、分類 | 低溫度（0～0.3），或直接貪心 |
| 一般問答、對話 | 溫度 0.7 左右，top-p 0.9～0.95 |
| 寫故事、頭腦風暴 | 溫度 0.8～1.2 |
| 推理模型 | **按官方建議設定**：例如 DeepSeek-R1 官方建議溫度 0.5～0.7（推薦 0.6）；有的推理模型 API 甚至不允許修改溫度 |

> [!NOTE]
> **溫度為 0 也不一定完全可復現**：GPU 上的浮點運算順序、伺服器把多個請求合成一批處理等，都可能讓結果有微小差別，而一個 token 不同，後面就可能完全不同。需要可復現時，可以固定隨機種子（`seed` 引數，如果 API 支援），並記錄模型版本。

## 7.9 小結

- 模型只給出下一個 token 的**機率分佈**；**解碼**決定選哪個，同一個模型的回答品質因此差別很大。
- **貪心**：確定、適合答案唯一的任務，但無聊、易重複。**取樣**：按機率隨機抽，多樣但可能胡說。
- **溫度** T：logits 除以 T；T < 1 更保守（T → 0 即貪心），T > 1 更大膽。
- **top-k / top-p / min-p**：只在高機率的候選裡取樣，截掉「長尾」；常與溫度組合使用。
- **重複、頻率、存在懲罰**減少重複；**束搜尋**適合翻譯等任務；**停止條件**有結束 token、最大長度和停止序列。
- **約束解碼**把不合格式的 token 遮蔽掉，保證輸出合法（結構化輸出、JSON）。

## 7.10 練習

<!-- exercise: temperature -->
### 練習 1：帶溫度的 softmax

第一行是若干個 logits，第二行是溫度 `T`。輸出 `softmax(z / T)`，保留 4 位小數，空格隔開。

```text
樣例輸入：
2 1 0
0.5
樣例輸出：0.8668 0.1173 0.0159
```

<details>
<summary>提示</summary>

先 `z = [v / T for v in logits]`，再做 softmax（先減去最大值）。
</details>

<!-- exercise: top-k -->
### 練習 2：top-k

第一行是若干個 logits，第二行是整數 `k`。先做 softmax（溫度 1），只保留機率最高的 `k` 個 token 並重新歸一化，其餘的機率設為 0。按原來的順序輸出所有 token 的機率，保留 4 位小數，空格隔開。

```text
樣例輸入：
2 1 0 -1
2
樣例輸出：0.7311 0.2689 0.0000 0.0000
```

<details>
<summary>提示</summary>

`keep = sorted(range(n), key=lambda i: -p[i])[:k]`，然後用保留下來的機率之和去除。
</details>

<!-- exercise: top-p -->
### 練習 3：top-p 保留了誰？

第一行是若干個已經歸一化的機率，第二行是 `p`。按機率從大到小（機率相同時編號小的在前）依次加入候選，直到累計機率**大於或等於** `p` 為止。輸出被保留的 token 的編號（從 0 開始），按加入的順序，空格隔開。

```text
樣例輸入：
0.5 0.3 0.15 0.05
0.75
樣例輸出：0 1
```

<details>
<summary>提示</summary>

`sorted(range(n), key=lambda i: -probs[i])` 是穩定排序，機率相同時會保持編號從小到大的順序。
</details>

<!-- exercise: greedy-repeat -->
### 練習 4：貪心解碼的問題

用貪心解碼寫長文章時，最常見的問題是？

- 速度太慢
- 文字平淡、容易陷入重複的迴圈
- 每次執行結果都不一樣
- 無法停止生成

<details>
<summary>答案與解析</summary>

貪心每次都選最常見的說法，一旦進入「A 之後最可能是 B，B 之後最可能是 A」的迴圈就出不來。它的結果是確定的（每次一樣），這對唯一答案的任務反而是優點。
</details>

<!-- exercise: low-temp -->
### 練習 5：溫度趨近於 0

把溫度設得非常接近 0，取樣的效果會接近什麼？

- 均勻隨機地選一個 token
- 貪心解碼（總是選機率最大的 token）
- 總是選機率最小的 token
- 束搜尋

<details>
<summary>答案與解析</summary>

logits 除以一個很小的 T 後，最大值和其他值的差距被放得極大，softmax 幾乎把全部機率給了最大的那個——相當於貪心。溫度很大時才接近均勻分佈。
</details>

## 7.11 延伸閱讀

- Holtzman 等：《The Curious Case of Neural Text Degeneration》（2019，提出 top-p 核取樣）
- Hugging Face 部落格：[How to generate text: using different decoding methods](https://huggingface.co/blog/how-to-generate)
- Nguyen 等：《Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs》（2024）
- 第三部分到此結束！下一部分：[第四部分 · 大規模訓練](../../training/01-data/zh-CN.md)——資料、算力、分散式訓練與評測。
