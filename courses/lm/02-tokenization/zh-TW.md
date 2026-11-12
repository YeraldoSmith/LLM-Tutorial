# 2. 分詞：文字怎樣變成 token

> 🎯 **本課目標**：明白模型為什麼需要「分詞」，以及按字元、按詞、按子詞切分各有什麼利弊；理解 Unicode 與 **UTF-8 位元組**；親手實現 GPT 系列使用的 **BPE（位元組對編碼）** 分詞器：訓練、編碼、解碼；看到分詞器的訓練資料如何影響中文、俄文等非英語使用者；認識特殊 token 和對話模板；理解大模型「數不清字母」等怪現象的原因。
>
> ⏱ **預計用時**：80 分鐘　｜　📋 **前置知識**：本部分第 1 課

上一課我們把每個字元當成一個 token。真正的大模型並不這樣做：它們用一個**分詞器**（tokenizer）把文字切成**子詞**（subword）片段，再把每個片段換成一個編號。分詞器看起來只是個「預處理」步驟，但它深刻地影響著模型的能力、速度和使用成本——尤其是對不說英語的使用者。

> [!NOTE]
> 本課的 BPE 實現參考了 Andrej Karpathy 的 [minbpe](https://github.com/karpathy/minbpe)（MIT 許可證）和影片 [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)。

## 2.1 為什麼要分詞？三種切法

神經網路只認數字，所以必須先把文字切成小塊、給每塊一個編號。怎麼切是一個權衡：

| 切法 | 詞表大小 | 序列長度 | 問題 |
|------|----------|----------|------|
| **按字元** | 小（英文幾十個；加上漢字幾萬個） | 很長 | 序列太長，計算量大；單個字元幾乎沒有意義 |
| **按詞** | 巨大（幾十萬以上） | 短 | 新詞、拼錯的詞、人名無法表示（「未登入詞」）；中文還要先分詞 |
| **按子詞**（BPE 等） | 適中（3 萬～25 萬） | 適中 | 常見詞是一個 token，罕見詞拆成幾塊——兩全其美 |

現代大模型幾乎都用**子詞**分詞。例如 `unbelievable` 可能被切成 `un` + `believ` + `able`，模型從沒見過的新詞也總能用已有的小塊拼出來。

## 2.2 從字元到位元組：Unicode 與 UTF-8

計算機裡每個字元都有一個 **Unicode 編號**（碼位），例如 `A` 是 65，`你` 是 20320。儲存和傳輸時最常用 **UTF-8** 編碼把碼位變成 1～4 個**位元組**（0～255 的數）：英文字母 1 個位元組，俄文字母 2 個，漢字 3 個，表情符號 4 個。

```python
for ch in ["A", "é", "Ж", "你", "😀"]:
    b = ch.encode("utf-8")
    print(f"{ch!r}: 碼位 {ord(ch):6d}  UTF-8 位元組 {list(b)}（{len(b)} 個）")

text = "Hi 你好 Привет"
print(len(text), "個字元，", len(text.encode("utf-8")), "個位元組")
print(bytes([228, 189, 160]).decode("utf-8"))     # 位元組還原成文字
```

GPT-2 以來的大多數分詞器都以**位元組**為起點：任何文字（任何語言、表情、甚至亂碼）都能表示成位元組序列，所以**永遠不會遇到「無法表示」的字元**。詞表的前 256 個 token 就是這 256 種位元組，後面的 token 都是由它們**合併**出來的。

## 2.3 BPE：位元組對編碼

**BPE**（Byte Pair Encoding，位元組對編碼）的訓練過程極其簡單：

1. 把訓練文字變成位元組序列（每個位元組是一個 token，編號 0～255）；
2. 統計所有**相鄰 token 對**出現的次數；
3. 把**出現最多的一對**合併成一個新 token（編號 256、257……），並記下這條「合併規則」；
4. 重複第 2～3 步，直到詞表達到想要的大小。

**編碼**新文字時，按學到的規則**按先後順序**合併；**解碼**時把每個 token 對應的位元組拼起來再按 UTF-8 還原。

```python
EN = """A language model reads text as a sequence of tokens and learns to predict the next token.
The model never sees letters or words directly: it only sees numbers. Before training, we must
decide how to cut the text into pieces. If the pieces are single characters, the sequences become
very long. If the pieces are whole words, the vocabulary becomes huge and new words cannot be
represented. Byte pair encoding finds a middle way: it starts from bytes and repeatedly merges the
most frequent pair of neighbouring tokens into a new token. Frequent words become single tokens,
while rare words are split into smaller pieces. The same tokenizer is used for training the model
and for every request the model receives later, so the tokenizer is part of the model."""

def get_stats(ids):                       # 統計相鄰 token 對出現的次數
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):             # 把 ids 裡所有的 pair 替換成 new_id
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text, vocab_size):
    ids = list(text.encode("utf-8"))      # 從位元組開始：0～255
    merges = {}                           # (a, b) → 新 token 的編號
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)  # 出現次數最多的一對
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))   # 先用最早學到的規則
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

merges, vocab = train(EN, 256 + 100)      # 學 100 條合併規則
print("最先學到的 10 個 token：", [vocab[i].decode("utf-8", "replace") for i in range(256, 266)])
sentence = "the model predicts the next token"
ids = encode(sentence, merges)
print("編碼：", ids)
print("切分：", [vocab[i].decode("utf-8", "replace") for i in ids])
print(f"{len(sentence)} 個字元 → {len(ids)} 個 token；解碼還原：{decode(ids, vocab) == sentence}")
```

可以看到，最先合併出來的是 `e `（詞尾的 e 加空格）、` t`、`he ` 這樣極其常見的片段，很快就出現了 ` the `、`model `、` token` 這樣的整詞。常見的詞變成 1 個 token，不常見的（如 `predicts`）被拆成幾塊。

> [!NOTE]
> 真實的分詞器在 BPE 之前還有一步**預分詞**：先用正規表示式把文字按單詞、數字、標點、空格切開，BPE 只在每一塊內部合併。這樣可以避免 `dog.`、`dog!`、`dog?` 被學成三個不同的 token。

## 2.4 分詞器偏向誰？非英語使用者的「token 稅」

分詞器是在**訓練資料**上學出來的。如果訓練資料主要是英文，那麼英文會被壓縮得很好，而其他語言只能拆成零碎的位元組。下面分別用「只有英文」和「英文 + 中文 + 俄文」的資料訓練分詞器，比較同樣意思的句子需要多少個 token：

```python
EN = """A language model reads text as a sequence of tokens and learns to predict the next token.
The model never sees letters or words directly: it only sees numbers. Before training, we must
decide how to cut the text into pieces. If the pieces are single characters, the sequences become
very long. If the pieces are whole words, the vocabulary becomes huge and new words cannot be
represented. Byte pair encoding finds a middle way: it starts from bytes and repeatedly merges the
most frequent pair of neighbouring tokens into a new token. Frequent words become single tokens,
while rare words are split into smaller pieces. The same tokenizer is used for training the model
and for every request the model receives later, so the tokenizer is part of the model."""
ZH = """語言模型把文字看成一串詞元，學習預測下一個詞元。模型看不到漢字本身，只能看到數字。
訓練之前，我們必須決定怎樣把文字切成小塊。如果每個漢字都是一塊，序列就會很長；
如果每個詞都是一塊，詞表就會非常大，新詞也無法表示。位元組對編碼從位元組出發，
不斷把最常見的相鄰兩塊合併成新的一塊，於是常見的詞變成一個詞元，少見的詞被拆成更小的部分。
同一個分詞器既用於訓練模型，也用於模型以後收到的每一個請求，所以分詞器也是模型的一部分。"""
RU = """Языковая модель читает текст как последовательность токенов и учится предсказывать
следующий токен. Модель не видит букв и слов, она видит только числа. Перед обучением нужно
решить, как разрезать текст на кусочки. Если кусочки — отдельные символы, последовательности
становятся очень длинными. Если кусочки — целые слова, словарь становится огромным, а новые
слова нельзя представить. Кодирование пар байтов находит середину: оно начинает с байтов и
снова и снова объединяет самую частую пару соседних токенов в новый токен."""

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text, vocab_size):
    ids, merges = list(text.encode("utf-8")), {}
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

for name, data in [("只用英文訓練", EN), ("用英文 + 中文 + 俄文訓練", EN + ZH + RU)]:
    merges, vocab = train(data, 256 + 300)
    print(name)
    for s in ["the model predicts the next token", "模型預測下一個詞元", "модель предсказывает токен"]:
        ids = encode(s, merges)
        pieces = [vocab[i].decode("utf-8", "replace") for i in ids]
        print(f"   {len(s):2d} 個字元 → {len(ids):2d} 個 token  ", pieces[:6])
```

只用英文訓練時，中文句子 9 個字被拆成 27 個 token（每個漢字 3 個位元組，一個都沒合併），俄文也幾乎是一個位元組一個 token；加入中文和俄文資料後，同樣的句子只需要原來一半甚至更少的 token。`�` 表示這個 token 只是某個字元的**一部分位元組**，單獨顯示不出來。

這在現實中很重要，因為：

- **按 token 計費**：API 按 token 數收費，同樣的意思用 token 多的語言就更貴；
- **上下文更短**：模型一次能讀的 token 數有限，token 多就意味著能放進去的文字少；
- **生成更慢**：模型一個 token 一個 token 地生成，token 越多越慢。

早期的 GPT-2 分詞器對中文很不友好。如今的分詞器（GPT-4o 的 `o200k_base`、Llama 3、Qwen、DeepSeek 等）都用了大量多語言資料訓練，詞表也更大，對中文、俄文的效率已經好得多。**Qwen、DeepSeek 等國產模型的分詞器對中文尤其高效。**

| 模型 / 分詞器 | 詞表大小 |
|---------------|----------|
| GPT-2 | 50,257 |
| GPT-4（`cl100k_base`） | 約 10 萬 |
| GPT-4o（`o200k_base`） | 約 20 萬 |
| Llama 3 | 約 12.8 萬 |
| Qwen 系列 | 約 15 萬 |
| DeepSeek-V3 | 約 12.9 萬 |

## 2.5 使用真實的分詞器

真實專案裡不用自己寫 BPE，直接用現成的庫。例如 OpenAI 的 `tiktoken`，或 Hugging Face 的 `transformers`（任何開源模型的分詞器都能這樣載入）：

```python norun
# pip install tiktoken
import tiktoken
enc = tiktoken.get_encoding("o200k_base")          # GPT-4o 使用的分詞器
ids = enc.encode("大語言模型很有趣！")
print(len(ids), ids)
print([enc.decode([i]) for i in ids])               # 看看每個 token 是什麼

# pip install transformers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")   # 第一次執行會從 Hugging Face 下載
print(tok.tokenize("大語言模型很有趣！"))
```

> [!TIP]
> 想直觀地看一段文字怎麼被切分，可以用線上工具，例如 OpenAI 的 [Tokenizer](https://platform.openai.com/tokenizer) 頁面，或 Hugging Face 上的 [The Tokenizer Playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)。

## 2.6 特殊 token 與對話模板

除了普通文字片段，詞表裡還有一些**特殊 token**，它們不對應任何文字，而是起「標記」作用：

- `<|endoftext|>`：一篇文件結束（預訓練時把不同文件隔開）；
- 對話模型的**角色標記**：表示「下面是系統提示 / 使用者 / 助手說的話」。

你在聊天介面裡看到的一問一答，送進模型時其實被拼成了一整段帶特殊 token 的文字，這種格式叫**對話模板**（chat template）。例如 Qwen 等模型使用的格式大致是：

```text
<|im_start|>system
你是一個樂於助人的助手。<|im_end|>
<|im_start|>user
1 + 1 等於幾？<|im_end|>
<|im_start|>assistant
1 + 1 等於 2。<|im_end|>
```

模型學會了：看到 `<|im_start|>assistant` 之後就該生成回答，生成到 `<|im_end|>` 就停下。不同模型的模板不一樣，用錯模板會讓模型表現變差——Hugging Face 的 `tokenizer.apply_chat_template(...)` 會自動套用正確的模板。第五部分講微調時還會用到。

## 2.7 分詞帶來的怪現象

很多讓人困惑的大模型行為，根源都在分詞：

| 現象 | 原因 |
|------|------|
| 數不清 `strawberry` 裡有幾個 `r` | 模型看到的是 `str` + `aw` + `berry` 這樣的 token，而不是一個個字母 |
| 不擅長把單詞倒著拼、按字母操作 | 同上，字母藏在 token 內部 |
| 大數字的算術容易出錯 | 數字被切成不規則的塊（如 `12345` → `123` + `45`），位數對不齊 |
| 句尾多一個空格，回答就變差 | `" token"`（帶前導空格）和 `"token"` 是不同的 token，訓練中極少出現孤立的空格 token |
| 中文、小語種更「貴」 | 分詞器訓練資料裡這些語言少，壓縮率低（2.4 節） |

> [!NOTE]
> 新一代的推理模型會把問題「拆開來想」（比如先把單詞逐個字母寫出來再數），所以這些問題已經緩解了很多，但原因仍然是分詞。也有研究嘗試完全去掉分詞器，直接在位元組上建模。

## 2.8 小結

- 模型只認數字：**分詞器**把文字切成 token 並編號。子詞分詞在詞表大小和序列長度之間取得平衡。
- 文字 → Unicode 碼位 → **UTF-8 位元組**（英文 1 位元組、俄文 2 位元組、漢字 3 位元組）；以位元組為起點，任何文字都能表示。
- **BPE**：從位元組出發，反覆合併最常見的相鄰 token 對；編碼時按學到的順序合併，解碼時拼回位元組。
- 分詞器的訓練資料決定了各種語言的壓縮率：token 越多，就越貴、能讀的內容越少、生成越慢。現代多語言分詞器已大大改善。
- **特殊 token** 和**對話模板**把多輪對話拼成一段文字；數字母、算術等怪現象常常源於分詞。

## 2.9 練習

<!-- exercise: utf8-bytes -->
### 練習 1：字元與位元組

輸入一行文字，輸出它的**字元數**和 **UTF-8 位元組數**（空格隔開）。

```text
樣例輸入：Hi 你好
樣例輸出：5 9
```

<details>
<summary>提示</summary>

`len(s)` 是字元數，`len(s.encode("utf-8"))` 是位元組數。
</details>

<!-- exercise: bpe-step -->
### 練習 2：BPE 的一步合併

輸入一個不含空格的字串，把它看成字元序列。找出**出現次數最多的相鄰字元對**（統計時允許重疊，例如 `aaa` 裡 `aa` 算 2 次；次數相同時取**最先出現**的那一對），在第一行輸出這一對；然後**從左到右**把所有這一對合併成一個 token（不重疊），在第二行輸出合併後的 token 序列，用空格隔開。

```text
樣例輸入：aaabdaaabac
樣例輸出：
aa
aa a b d aa a b a c
```

<details>
<summary>提示</summary>

用字典按出現順序統計 `zip(seq, seq[1:])`；`max(counts, key=counts.get)` 在並列時會返回最先加入字典的那一個。合並可以照抄本課的 `merge` 函式，只是把新編號換成兩個字元拼起來的字串。
</details>

<!-- exercise: decode-bytes -->
### 練習 3：讀程式碼，寫輸出

```python
tokens = [72, 105, 32, 228, 189, 160]
print(bytes(tokens).decode("utf-8"))
print(len(tokens))
```

<details>
<summary>答案與解析</summary>

`72, 105, 32` 是 `H`、`i` 和空格；`228, 189, 160` 這 3 個位元組合起來是漢字「你」的 UTF-8 編碼。所以輸出 `Hi 你`，然後是 `6`（6 個位元組，但只有 4 個字元）。
</details>

<!-- exercise: strawberry -->
### 練習 4：數字母

早期的大模型經常答錯「strawberry 裡有幾個 r」。最根本的原因是？

- 模型沒學過英語
- 模型看到的是 token（若干個字母組成的塊），而不是一個個字母
- 模型的計算能力不夠
- 訓練資料裡沒有這個單詞

<details>
<summary>答案與解析</summary>

單詞被切成 `str`、`aw`、`berry` 這樣的塊，每個塊對模型來說只是一個編號，塊裡有哪些字母並不直接可見。
</details>

<!-- exercise: vocab-size -->
### 練習 5：詞表越大越好嗎？

把分詞器的詞表從 3 萬增加到 20 萬，下面哪種說法**正確**？

- 同樣的文字會變成更多 token
- 同樣的文字會變成更少 token，但嵌入層和輸出層的引數會變多
- 對模型沒有任何影響
- 模型將無法處理沒見過的詞

<details>
<summary>答案與解析</summary>

詞表大，能合併的片段多，序列就短（更快、能讀更多內容）；代價是每個 token 都要一行嵌入向量、輸出層也要為每個 token 打分，引數和計算量隨之增加。位元組級 BPE 無論詞表多大都能表示任何文字。
</details>

## 2.10 延伸閱讀

- Andrej Karpathy：[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)（影片）與 [minbpe](https://github.com/karpathy/minbpe)
- Sennrich 等：《Neural Machine Translation of Rare Words with Subword Units》（2016，把 BPE 引入 NLP 的論文）
- Hugging Face LLM 課程：[分詞器](https://huggingface.co/learn/llm-course/zh-CN/chapter2/4)（有中文版）
- 下一課：[詞向量：用一串數字表示意思](../03-embeddings/zh-CN.md)
