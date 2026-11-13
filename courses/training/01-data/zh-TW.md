# 1. 預訓練資料：模型讀了什麼

> 🎯 **本課目標**：知道大模型預訓練用了多少資料、資料從哪裡來；理解一條完整的資料處理流水線——提取文字、語言識別、啟發式過濾、**去重**（精確去重和 MinHash 近似去重）、品質分類、去除隱私資訊、**去汙染**；明白資料配比、重複輪數、退火階段和合成資料；知道訓練前怎樣把文件**打包**成固定長度的序列。
>
> ⏱ **預計用時**：80 分鐘　｜　📋 **前置知識**：第三部分（特別是第 1、2、6 課）

第三部分裡，我們用幾千個字元訓練了一個小 GPT。真正的大模型做的是同一件事——預測下一個 token——只是讀的文字多了幾十億倍。業內有一句話：**模型的能力，上限由資料決定**。很多團隊花在資料上的人力，比花在模型結構上的還多。

## 1.1 有多少資料？

| 模型 | 年份 | 預訓練 token 數 |
|------|------|-----------------|
| GPT-3 | 2020 | 約 3000 億（0.3T） |
| Llama 2 | 2023 | 約 2 萬億（2T） |
| Llama 3 | 2024 | 超過 15 萬億（15T） |
| DeepSeek-V3 | 2024 | 約 14.8 萬億 |
| Qwen3 | 2025 | 約 36 萬億 |

15 萬億個 token 是什麼概念？按英文平均每個 token 約 4 個字元、一本書約 50 萬個字元來算，大約相當於 **1.2 億本書**。一個人每天讀一本書，要讀 30 多萬年。

## 1.2 資料從哪裡來

- **網頁**：絕大部分資料來自網頁，最重要的公開來源是 **Common Crawl**——一個非營利組織，從 2008 年起每隔一兩個月抓取一次網際網路，每次幾十億個網頁，免費開放。
- **程式碼**：GitHub 等開源倉庫。程式碼資料不僅讓模型會寫程式，也被認為能提升推理能力。
- **書籍、論文、百科**：維基百科、arXiv 論文、公版書等，品質高但數量少。
- **數學、科學**：數學網頁、教材、題解，近年越來越受重視。
- **多語言資料**：中文、俄語等語言的網頁比英文少得多，通常要專門收集和增加比例。
- **合成資料**：用已有的大模型生成或改寫的文字（1.6 節）。

資料的來源也涉及**版權和許可**：哪些資料可以用來訓練、是否需要授權，是 2023 年以來各國訴訟和立法的熱點。開源社群因此整理了不少來源清楚、許可明確的資料集，比如 Hugging Face 的 **FineWeb**（2024 年釋出，從 96 次 Common Crawl 抓取中處理出約 15 萬億 token 的英文網頁文字，處理程式碼全部公開）。

## 1.3 資料處理流水線

原始網頁裡有大量導航欄、廣告、亂碼、重複內容和低品質文字。一條典型的流水線是這樣的：

```text
原始網頁（HTML）
   │ ① 提取正文：去掉 HTML 標籤、導航欄、頁尾
   ▼
純文字 ── ② 語言識別：這是哪種語言？
   │ ③ 啟發式過濾：太短、亂碼、符號太多、重複太多的丟掉
   │ ④ 去重：完全相同的、幾乎相同的只留一份
   │ ⑤ 品質分類：用模型給「教育價值」「寫作品質」打分
   │ ⑥ 去除隱私資訊：郵箱、電話、身份證號……
   │ ⑦ 去汙染：刪掉和評測題目重合的內容
   ▼
可以訓練的文字（往往只剩原始資料的百分之幾）
```

### 啟發式過濾

先用一些簡單的規則快速去掉明顯的垃圾。DeepMind 訓練 Gopher（2021）時總結了一組這樣的規則（文件長度、平均詞長、符號比例、重複程度等），後來被廣泛使用。下面是一個簡化版：

```python
def check(doc):
    words = doc.split()
    if not 5 <= len(words) <= 100_000:          # Gopher 原本要求至少 50 個詞，這裡為了演示放寬到 5
        return "太短或太長"
    mean_len = sum(len(w) for w in words) / len(words)
    if not 3 <= mean_len <= 10:
        return "平均詞長異常"
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8:
        return "含字母的詞太少"
    if len({w.lower() for w in words}) / len(words) < 0.3:
        return "重複太多"
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

這些規則粗糙但很快，每秒能處理大量文件。要注意它們是**針對某種語言**設計的：中文沒有空格分詞，`doc.split()` 對中文幾乎沒用，處理中文時要按字元統計，或者先分詞。

### 精確去重

同一篇文章會被轉載到成百上千個網站上。重複的資料會浪費算力，還會讓模型**背誦**這些文字。最簡單的去重是：先把文字規範化（統一大小寫、合併空白），再算一個**雜湊值**，雜湊相同的只留第一份：

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

第三篇只是大小寫和空格不同，被去掉了；但第四篇只多了一個感嘆號，雜湊就完全不同了。網上大量的「幾乎重複」——改了幾個字、加了個日期、換了個頁尾——需要**近似去重**。

### 近似去重：MinHash

衡量兩篇文件有多像，常用 **Jaccard 相似度**：把文件切成一個個小片段（**shingle**，比如每 5 個連續字元），兩篇文件的片段集合 A、B 的交集大小除以並集大小：

```formula
J(A, B) = |A ∩ B| / |A ∪ B|
```

但要在幾十億篇文件裡兩兩比較，是不可能的。**MinHash** 的妙處是：用一個雜湊函式給所有片段打分，取最小的那個；兩篇文件「最小片段相同」的機率，**恰好等於**它們的 Jaccard 相似度。用 128 個不同的雜湊函式，就得到 128 個陣列成的「簽名」，簽名裡相同位置相等的比例，就是 Jaccard 的估計值：

```python
import hashlib

def shingles(text, k=5):
    t = " ".join(text.lower().split())
    return {t[i:i + k] for i in range(len(t) - k + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b)

def h(s, seed):                       # 第 seed 個雜湊函式
    digest = hashlib.md5(f"{seed}:{s}".encode()).digest()
    return int.from_bytes(digest[:8], "big")

def minhash(sh, num=128):             # 簽名：每個雜湊函式下的最小值
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
    print(f"{x}-{y}: 真實 Jaccard {jaccard(sh[x], sh[y]):.2f}，MinHash 估計 {estimate(sig[x], sig[y]):.2f}")
```

簽名只有 128 個數，比原文短得多。再配合**區域性敏感雜湊**（LSH）：把簽名切成 b 段、每段 r 個數，只要有一段完全相同，就把兩篇文件列為「候選」，再仔細比較。相似度為 s 的兩篇文件成為候選的機率是 `1 - (1 - s^r)^b`。取 b = 16、r = 8：

| 相似度 s | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
|----------|-----|-----|-----|-----|-----|
| 成為候選的機率 | 0.001 | 0.06 | 0.61 | 0.95 | 1.00 |

相似的幾乎都能被找出來，不相似的幾乎都不會被比較——這樣幾十億篇文件的去重也只需要線性的時間。

### 品質分類器

規則只能去掉明顯的垃圾。更進一步的做法是**訓練一個分類器給文件打分**。FineWeb-Edu（2024）的做法很有代表性：先讓一個大模型（Llama 3 70B）給幾十萬篇網頁的「教育價值」打 0～5 分，再用這些打分訓練一個小而快的分類器，給全部網頁打分，只保留高分的。只用這 1.3 萬億 token 的「教育類」網頁訓練的模型，在知識和推理測試上明顯好於用全部網頁訓練的模型。

### 去除隱私資訊和去汙染

- **隱私資訊**：用規則和模型找出郵箱、電話號碼、IP 地址、身份證號等，替換掉或刪除。
- **去汙染**（decontamination）：如果評測題目（以及答案）出現在訓練資料裡，模型在評測中的高分可能只是「背出來的」。所以要把和評測集有大段重合（比如連續 13 個詞相同）的文件刪掉。第 5 課會再談這個問題。

## 1.4 資料配比、重複和退火

不同來源的資料怎樣混合，叫做**資料配比**（data mixture）。它對模型的影響非常大，通常要先用小模型做大量實驗來確定。一個關鍵的問題是：高品質的資料往往不多，按配比算下來可能要**重複使用好幾遍**：

```python
budget = 15_000                # 總共要訓練 15 萬億 token（單位：十億）
sources = {                    # 名稱：(這類資料有多少，配比)  —— 數字只是示意
    "網頁":  (30_000, 0.70),
    "程式碼":  (2_000, 0.15),
    "數學":  (500, 0.07),
    "書籍與論文": (400, 0.05),
    "多語言": (3_000, 0.03),
}
for name, (size, share) in sources.items():
    used = budget * share
    print(f"讀 {used / size:4.2f} 遍（使用 {used:>6,.0f}B token）  {name}")
```

研究表明（Muennighoff 等，2023），同樣的資料重複讀 **4 遍以內**，效果和讀全新的資料差不多；再多，收益就迅速下降。

很多模型在預訓練的**最後階段**，會把學習率降到很低，同時提高高品質資料（數學、程式碼、教材、精選網頁）的比例，這叫**退火**（annealing）。用很少的 token 就能明顯提升模型的能力。

## 1.5 打包成訓練序列

模型每次處理固定長度的序列（比如 4096 或 8192 個 token），而文件有長有短。常見的做法是把所有文件首尾相接，每篇文件後面加一個**結束符**（EOS），再切成等長的塊：

```python
docs = [[5, 8, 2], [7, 7], [1, 2, 3, 4, 5, 6], [9]]   # 已經分好詞的文件（token id）
EOS = 0
L = 4                                                 # 每個訓練序列的長度

stream = []
for d in docs:
    stream += d + [EOS]                               # 文件之間用 EOS 隔開
print("拼接後：", stream)

for i in range(0, len(stream) - L, L):
    x = stream[i:i + L]                               # 輸入
    y = stream[i + 1:i + L + 1]                       # 目標：每個位置的「下一個 token」
    print("輸入", x, "→ 目標", y)
```

- 目標序列就是輸入序列向後錯開一位：每個位置都要預測下一個 token，這和第三部分第 6 課一模一樣。
- 打包後，一個序列裡可能有好幾篇不相關的文件。簡單的實現讓它們互相「看得見」；更講究的實現會修改注意力掩碼，讓每個 token 只能看到**同一篇文件**裡前面的內容。
- 長文件會被切到不同的序列裡。專門的「長上下文」訓練階段會用更長的序列和更多的長文件（第 4 課）。

## 1.6 合成資料

高品質的人類文字快要被用完了，於是**合成資料**越來越重要：

- **改寫**：讓模型把品質一般的網頁改寫成清楚的問答、教材風格的文字；
- **生成**：微軟的 Phi 系列模型（2023 年起）大量使用模型生成的「教科書式」文字，小模型也能有不錯的推理能力；
- **可驗證的資料**：數學題和程式碼可以自動檢查對錯，可以大量生成再篩選（第五部分會講推理模型如何利用這一點）。

但合成資料也有風險：如果一代又一代的模型只用前一代模型生成的資料訓練，輸出會越來越單一，罕見的知識逐漸丟失，這被稱為**模型崩潰**（model collapse）。所以合成資料一般和真實資料混合使用，並且仔細篩選。

## 1.7 小結

- 預訓練資料以萬億 token 計：Llama 3 超過 15T，Qwen3 約 36T。
- 主要來源是網頁（Common Crawl），加上程式碼、書籍、論文、數學和多語言資料。
- 流水線：提取正文 → 語言識別 → 啟發式過濾 → 去重 → 品質分類 → 去隱私 → 去汙染，最後往往只剩百分之幾。
- 精確去重用規範化 + 雜湊；近似去重用 shingle + MinHash（相同位置相等的比例 ≈ Jaccard）+ LSH 分段。
- 資料配比決定各類資料的比例；高品質資料可以重複幾遍；最後的退火階段多用高品質資料。
- 文件首尾相接、用 EOS 隔開，切成等長序列；目標是輸入錯開一位。
- 合成資料越來越重要，但要防止模型崩潰。

## 1.8 練習

<!-- exercise: jaccard -->
### 練習 1：Jaccard 相似度

輸入兩行英文。把每行轉成小寫、按空白切成單詞，取所有**相鄰兩個詞**組成的片段（比如 `the cat sat` 的片段是 `the cat` 和 `cat sat`），輸出兩個片段集合的 Jaccard 相似度，保留 3 位小數。（每行至少有 2 個詞。）

```text
樣例輸入：
the cat sat on the mat
the cat sat on a mat
樣例輸出：0.429
```

<details>
<summary>提示</summary>

`words = line.lower().split()`，片段集合 `{(words[i], words[i + 1]) for i in range(len(words) - 1)}`。Jaccard = 交集大小 / 並集大小。
</details>

<!-- exercise: gopher-filter -->
### 練習 2：啟發式過濾

第一行輸入文件數 `n`，接下來 `n` 行，每行是一篇文件。按下面四條規則判斷每篇文件，全部滿足就輸出 `keep`，否則輸出 `drop`：

1. 詞數（按空白切分）在 5 到 1000 之間；
2. 平均詞長在 3 到 10 之間（包含兩端）；
3. 至少 80% 的詞含有字母；
4. 不同的詞（轉小寫後）佔全部詞的比例至少是 0.3。

```text
樣例輸入：
2
Photosynthesis turns sunlight water and carbon dioxide into sugar.
click here click here click here click here
樣例輸出：
keep
drop
```

<details>
<summary>提示</summary>

照著 1.3 節的 `check` 函式寫，把詞數上限改成 1000。含有字母：`any(c.isalpha() for c in w)`。
</details>

<!-- exercise: exact-dedup -->
### 練習 3：精確去重

第一行輸入文件數 `n`，接下來 `n` 行是文件。兩篇文件在「轉小寫、把連續的空白合併成一個空格、去掉首尾空白」之後完全相同，就算重複。只保留每組重複中第一次出現的那篇，先輸出保留的篇數，再按原來的順序輸出保留的文件（輸出**原文**）。

```text
樣例輸入：
3
Hello World
hello   world
Goodbye
樣例輸出：
2
Hello World
Goodbye
```

<details>
<summary>提示</summary>

規範化：`" ".join(doc.lower().split())`。用一個集合記住見過的規範化結果。
</details>

<!-- exercise: mixture-epochs -->
### 練習 4：每類資料讀幾遍

第一行是訓練的總 token 數 `T`（單位：十億）。之後每行描述一類資料：`名稱 資料量 配比`（資料量單位也是十億 token），一直到輸入結束。按 1.4 節的方法，輸出每類資料被讀了幾遍，格式 `名稱 遍數`，遍數保留 2 位小數。

```text
樣例輸入：
1000
web 2000 0.8
code 100 0.15
math 20 0.05
樣例輸出：
web 0.40
code 1.50
math 2.50
```

<details>
<summary>提示</summary>

遍數 = `T * 配比 / 資料量`。讀到輸入結束：`import sys`，`for line in sys.stdin:`。
</details>

<!-- exercise: decontam -->
### 練習 5：為什麼要去汙染？

預訓練資料處理中的「去汙染」（decontamination）是為了什麼？

- 刪除含有病毒和惡意程式碼的網頁
- 防止評測題目出現在訓練資料裡，使評測分數虛高
- 刪除含有個人隱私的文字
- 去除重複的網頁

<details>
<summary>答案與解析</summary>

如果評測題和答案在訓練時就見過，模型的高分可能只是記住了答案，不能反映真實能力。所以要刪掉與評測集大段重合的訓練文件。隱私資訊和去重是流水線裡的其他步驟。
</details>

<!-- exercise: minhash-why -->
### 練習 6：MinHash 有什麼用？

在資料處理中，MinHash 主要用來做什麼？

- 給文件的品質打分
- 識別文件的語言
- 高效地找出幾乎相同的文件
- 把文件切成 token

<details>
<summary>答案與解析</summary>

MinHash 用一個很短的簽名來估計兩篇文件的 Jaccard 相似度，配合 LSH 分段，可以在海量文件裡快速找出近似重複的文件。品質打分用分類器，語言識別用專門的語言識別模型，切分 token 用分詞器。
</details>

## 1.9 延伸閱讀

- Penedo 等：《The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale》（2024）——詳細介紹了資料處理的每一步，並公開了程式碼和資料
- Rae 等：《Scaling Language Models: Methods, Analysis & Insights from Training Gopher》（2021）附錄中的品質過濾規則
- Muennighoff 等：《Scaling Data-Constrained Language Models》（2023）——資料重複幾遍還有用
- Shumailov 等：《AI models collapse when trained on recursively generated data》（Nature，2024）
- 下一課：[規模定律與算力](../02-scaling-compute/zh-CN.md)
