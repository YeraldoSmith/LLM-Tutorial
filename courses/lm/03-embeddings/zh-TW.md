# 3. 詞向量：用一串數字表示意思

> 🎯 **本課目標**：明白為什麼不直接用 token 的編號或 one-hot 向量；理解**嵌入**（embedding）就是一張「查詢表」；用**餘弦相似度**衡量意思的遠近，用向量做「國王 − 男人 + 女人 ≈ 女王」的類比；從「詞的上下文」中自動學出詞向量；在 PyTorch 裡用 `nn.Embedding` 訓練一個 Bengio 式的神經語言模型，看它自己學會把母音放在一起；認識位置編碼和句向量。
>
> ⏱ **預計用時**：80 分鐘　｜　📋 **前置知識**：本部分第 1、2 課；第一部分第 4 課（向量、點積）

分詞之後，每個 token 都有一個編號，比如 `貓` 是 1234、`狗` 是 5678。但編號本身沒有意義：1234 和 1235 並不比 1234 和 5678 更「像」。這一課我們給每個 token 一個**向量**，讓意思相近的 token 在空間裡也靠得近——這就是**詞向量**（word vector），也叫**嵌入**（embedding），它是所有語言模型的第一層。

## 3.1 編號和 one-hot 的問題

上一課結尾，我們用 one-hot 向量表示字元：27 個字元就是 27 維，每個向量只有一個 1。這有兩個大問題：

1. **太大**：詞表有 15 萬個 token，每個 one-hot 向量就有 15 萬維，幾乎全是 0；
2. **沒有「相似」的概念**：任意兩個不同的 one-hot 向量都互相垂直（點積為 0），`貓` 和 `狗` 的距離與 `貓` 和 `飛機` 的距離完全一樣。

我們想要的是：每個 token 用一個**短得多的稠密向量**（比如 768 維，每一維都是一個小數）表示，並且意思相近的 token 向量也相近。

## 3.2 嵌入就是一張查詢表

**嵌入層**其實就是一個矩陣 `E`，形狀是「詞表大小 × 向量維度」。第 i 行就是第 i 個 token 的向量。「把 token 變成向量」就是**按編號取出那一行**：

```text
              維度 0   維度 1   維度 2   維度 3
token 0  [    0.12,   -0.40,    0.88,    0.05 ]
token 1  [   -0.71,    0.33,    0.10,   -0.26 ]
token 2  [    0.45,    0.91,   -0.37,    0.64 ]     ← 輸入編號 2，就取出這一行
…
```

數學上，「取第 i 行」等於「one-hot 向量乘以矩陣 E」——所以嵌入層就是一個線性層，只是用查表的方式算得更快。上一課神經網路版 bigram 中的 `W[a]` 正是這樣一張表。嵌入矩陣一開始是隨機的，和模型的其他引數一起**透過訓練學出來**。

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
emb = nn.Embedding(num_embeddings=10, embedding_dim=4)   # 詞表 10 個 token，每個 4 維
ids = torch.tensor([3, 1, 3])
print(emb(ids))                                          # 取出第 3、1、3 行（形狀 3×4）
print(torch.equal(emb(ids)[0], emb.weight[3]))            # True：就是查表
onehot = nn.functional.one_hot(ids, 10).float()          # 3×10 的 one-hot 矩陣
print(torch.allclose(onehot @ emb.weight, emb(ids)))      # True：等價於 one-hot 乘矩陣
```

## 3.3 用向量表示意思：相似度與類比

假設（為了演示，我們**手工**設計）每個詞的向量有 4 個維度，分別表示「皇室」「男性」「女性」「年輕」：

```python
import math

E = {                    # 皇室  男性  女性  年輕
    "king":     [1.0, 1.0, 0.0, 0.0],
    "queen":    [1.0, 0.0, 1.0, 0.0],
    "man":      [0.0, 1.0, 0.0, 0.0],
    "woman":    [0.0, 0.0, 1.0, 0.0],
    "prince":   [1.0, 1.0, 0.0, 1.0],
    "princess": [1.0, 0.0, 1.0, 1.0],
    "boy":      [0.0, 1.0, 0.0, 1.0],
    "girl":     [0.0, 0.0, 1.0, 1.0],
}

def cosine(a, b):         # 餘弦相似度：方向越一致越接近 1
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

print("king 和 queen：", round(cosine(E["king"], E["queen"]), 3))
print("king 和 girl： ", round(cosine(E["king"], E["girl"]), 3))

def analogy(a, b, c):     # a 之於 b，相當於 c 之於 ?   → 找最接近 b − a + c 的詞
    target = [vb - va + vc for va, vb, vc in zip(E[a], E[b], E[c])]
    return max((w for w in E if w not in (a, b, c)), key=lambda w: cosine(target, E[w]))

print("man : king = woman : ?   →", analogy("man", "king", "woman"))
print("man : boy  = woman : ?   →", analogy("man", "boy", "woman"))
print("king : prince = queen : ? →", analogy("king", "prince", "queen"))
```

`king − man + woman`：從「國王」裡減去「男性」、加上「女性」，正好落在「女王」上。**向量的方向編碼了意思上的關係**（性別、皇室、年齡……）。

**餘弦相似度**（cosine similarity）是比較兩個嵌入最常用的方法：兩個向量的夾角越小，餘弦值越接近 1；互相垂直是 0；方向相反是 −1。它只看方向、不看長度。

> [!NOTE]
> 真實模型的嵌入有幾百到幾千維，每一維**不是**人能直接看懂的「性別」「皇室」，意思分散在很多維度的組合裡。但 2013 年 Google 的 **word2vec** 確實發現，自動學出的詞向量裡存在 `king − man + woman ≈ queen` 這樣的規律，引起了轟動。

## 3.4 從上下文中學出詞向量

真正的詞向量當然不是手工設計的。語言學家 J. R. Firth 有一句名言：

> 「看一個詞和誰做伴，就知道它的意思。」（You shall know a word by the company it keeps.）

這叫**分佈假說**：出現在相似上下文中的詞，意思也相似。`貓` 和 `狗` 都常出現在「喂」「養」「可愛」附近，所以它們的意思接近。

最直接的做法：統計每個詞周圍出現了哪些詞（**共現矩陣**），每個詞的那一行就是它的向量。再用 **PPMI**（正點互資訊）降低「the」「a」這種到處都有的詞的影響：

```python
import math

CORPUS = """the cat drinks milk . the dog drinks water . a cat chases a mouse . a dog chases a cat .
the cat sleeps on the sofa . the dog sleeps on the floor . my cat is cute . my dog is cute .
i eat an apple . i eat a banana . she eats an orange . he eats a banana . an apple is sweet .
a banana is sweet . an orange is sweet . i drink milk . she drinks water . he drinks juice .
paris is a city in france . berlin is a city in germany . rome is a city in italy .
i live in paris . she lives in berlin . he lives in rome . we visit paris . they visit rome ."""
words = CORPUS.split()
vocab = sorted(set(words))
idx = {w: i for i, w in enumerate(vocab)}

# 共現矩陣：counts[w][c] = 詞 c 出現在詞 w 前後 2 個位置以內的次數
window = 2
counts = [[0] * len(vocab) for _ in vocab]
for i, w in enumerate(words):
    for j in range(max(0, i - window), min(len(words), i + window + 1)):
        if j != i:
            counts[idx[w]][idx[words[j]]] += 1

# PPMI：比「隨機碰到」更經常一起出現，才算有關係
total = sum(map(sum, counts))
row = [sum(r) for r in counts]
col = [sum(counts[i][j] for i in range(len(vocab))) for j in range(len(vocab))]
vec = [[max(0.0, math.log(counts[i][j] * total / (row[i] * col[j]))) if counts[i][j] else 0.0
        for j in range(len(vocab))] for i in range(len(vocab))]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)

def nearest(word, k=3):
    v = vec[idx[word]]
    skip = {".", "the", "a", "an", "is", "in"}
    scores = [(cosine(v, vec[idx[w]]), w) for w in vocab if w != word and w not in skip]
    return [(w, round(s, 2)) for s, w in sorted(scores, reverse=True)[:k]]

print("詞表大小：", len(vocab), "，每個詞的向量有", len(vocab), "維")
for w in ["cat", "apple", "paris", "mouse"]:
    print(f"{w:6s} 最相近的詞：", nearest(w))
```

即使只有 30 句話，程式也自己發現了 `cat` 和 `dog`、`apple` 和 `banana`、`paris` 和 `rome` 意思相近——我們從沒告訴它這些詞是什麼。

這種「數共現」的方法有兩個缺點：向量維度等於詞表大小（太大），而且很稀疏。**word2vec**（2013）、**GloVe**（2014）等方法改為**訓練**一個短的稠密向量，讓它能預測周圍的詞。而在現代大模型裡，詞向量不再單獨訓練，而是作為模型的第一層，**和整個模型一起**為「預測下一個 token」這個目標而學習。

## 3.5 動手：讓神經語言模型自己學出嵌入

2003 年，Yoshua Bengio 等人提出了第一個成功的神經語言模型：

1. 把前面 n 個 token 分別查表得到嵌入向量；
2. 把這些向量**拼接**起來，送進一個 MLP；
3. 輸出下一個 token 的機率（softmax）。

這比 bigram 強在：看的上下文更長（這裡是 3 個字元），而且嵌入讓相似的字元共享知識。我們把嵌入設成只有 **2 維**，這樣就能直接看看它學到了什麼：

```python requires=torch
import torch
from torch import nn

NAMES = """emma olivia ava isabella sophia mia charlotte amelia harper evelyn abigail emily
elizabeth sofia avery ella madison scarlett victoria aria grace chloe camila penelope riley
layla lillian nora zoey mila aubrey hannah lily addison eleanor natalie luna savannah leah
stella hazel ellie audrey violet claire bella aurora lucy anna samantha caroline sarah maya
liam noah william james oliver benjamin elijah lucas mason logan alexander ethan jacob
michael daniel henry jackson sebastian aiden matthew samuel david joseph carter owen wyatt
john jack luke dylan levi isaac gabriel julian mateo anthony lincoln joshua andrew caleb
ryan nathan thomas leo isaiah charles hudson hunter connor eli ezra aaron adrian nolan""".split()

chars = list(".abcdefghijklmnopqrstuvwxyz")
stoi = {c: i for i, c in enumerate(chars)}
block = 3                                   # 上下文長度：看前 3 個字元
X, Y = [], []
for name in NAMES:
    context = [0] * block                   # 開頭用 "." 填充
    for ch in name + ".":
        X.append(context)
        Y.append(stoi[ch])
        context = context[1:] + [stoi[ch]]
X, Y = torch.tensor(X), torch.tensor(Y)
print("訓練樣本：", tuple(X.shape), " 例如", [chars[i] for i in X[7]], "→", chars[Y[7]])

torch.manual_seed(0)
class NameMLP(nn.Module):                   # Bengio 等人 2003 年神經語言模型的縮小版
    def __init__(self, vocab=27, d=2, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)   # 每個字元一個 d 維向量
        self.fc1 = nn.Linear(block * d, hidden)
        self.fc2 = nn.Linear(hidden, vocab)
    def forward(self, x):                   # x：(batch, 3) 個字元編號
        e = self.emb(x)                     # (batch, 3, d)
        h = torch.tanh(self.fc1(e.flatten(1)))   # 把 3 個向量拼接成 (batch, 3d)
        return self.fc2(h)                  # (batch, 27) 個 logits

model = NameMLP()
opt = torch.optim.AdamW(model.parameters(), lr=0.01)
for step in range(2001):
    ix = torch.randint(0, len(X), (64,))
    loss = nn.functional.cross_entropy(model(X[ix]), Y[ix])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        with torch.no_grad():
            full = nn.functional.cross_entropy(model(X), Y).item()
        print(f"step {step:4d}  loss {full:.3f}")

E = model.emb.weight.detach()
vowels = [stoi[c] for c in "aeiou"]
others = [stoi[c] for c in "bcdfghjklmnprstvwz"]
def avg_dist(A, B):
    d = [torch.dist(E[a], E[b]).item() for a in A for b in B if a != b]
    return sum(d) / len(d)
print(f"母音之間的平均距離 {avg_dist(vowels, vowels):.2f}，母音與子音之間的平均距離 {avg_dist(vowels, others):.2f}")

torch.manual_seed(1)                        # 用訓練好的模型生成名字
names = []
for _ in range(8):
    context, out = [0] * block, ""
    while len(out) < 20:
        probs = torch.softmax(model(torch.tensor([context])), dim=1)
        ix = torch.multinomial(probs, 1).item()
        if ix == 0:
            break
        out += chars[ix]
        context = context[1:] + [ix]
    names.append(out)
print("生成的名字：", names)
```

幾點觀察：

- 損失比 bigram（約 2.1）低得多，生成的名字也像樣多了——因為模型能看到前 3 個字元。
- 我們從沒告訴模型什麼是母音，但學到的 2 維嵌入裡，**母音彼此靠得更近**：因為它們在名字裡起的作用相似（常常可以互相替換）。這就是嵌入學到「意思」的方式。
- 訓練資料只有一百多個名字，所以模型也會「背出」一些訓練集裡原有的名字（第二部分第 6 課講的過擬合）。真實模型用幾十億甚至幾萬億個 token 訓練。

## 3.6 位置也需要嵌入

上面的 MLP 把 3 個向量**按順序拼接**，所以知道誰在前誰在後。但下一課要學的**注意力機制**會把所有 token「一視同仁」地放在一起處理，本身**不知道順序**——而「狗咬人」和「人咬狗」意思完全不同。

所以 Transformer 還要給每個**位置**一個向量：

- **可學習的位置嵌入**（GPT-2）：再加一張「位置 × 維度」的表，第 0 個位置、第 1 個位置……各有一個向量，和 token 嵌入**相加**；
- **正弦位置編碼**（原始 Transformer）：用不同頻率的 sin/cos 函式生成，不需要學習；
- **RoPE 旋轉位置編碼**（Llama、Qwen、DeepSeek 等現代模型）：不加在輸入上，而是在注意力裡按位置「旋轉」向量。下一課再詳細講。

```formula
輸入向量 = token 嵌入[token 編號] + 位置嵌入[位置]
```

## 3.7 大模型裡的嵌入

| 模型 | 詞表大小 | 嵌入維度 | 嵌入層引數 |
|------|----------|----------|------------|
| GPT-2 small | 50,257 | 768 | 約 3,860 萬（佔全部 1.24 億的約 31%） |
| Llama 3 8B | 128,256 | 4,096 | 約 5.25 億 |
| Qwen3 系列（大模型） | 約 151,000 | 4,096～5,120 等 | 數億 |

模型的最後一層要把向量變回「每個 token 的分數」（logits），這一層的權重形狀和嵌入矩陣一樣。很多模型讓兩者**共用同一個矩陣**（叫**權重繫結**，weight tying），既省引數，也讓「輸入時的意思」和「輸出時的意思」保持一致。

## 3.8 句向量：給整段話一個向量

把整句話、整段文字變成一個向量，就能比較兩段話的意思是否相近。這正是**語義搜尋**和第七部分要講的 **RAG**（檢索增強生成）的基礎：把文件和問題都變成向量，找出餘弦相似度最高的文件。專門做這件事的模型叫**嵌入模型**，很多都支援多種語言，用中文提問也能找到英文資料：

```python norun
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")      # 支援 100 多種語言；也可以試試 Qwen/Qwen3-Embedding-0.6B
docs = ["貓喜歡喝牛奶", "The cat drinks milk", "巴黎是法國的首都", "Paris is the capital of France"]
vecs = model.encode(docs, normalize_embeddings=True)
query = model.encode(["小貓愛喝什麼？"], normalize_embeddings=True)
print(query @ vecs.T)                           # 和兩句關於貓的話相似度最高（不管中文還是英文）
```

## 3.9 小結

- token 編號和 one-hot 向量都不能表達「相似」；**嵌入**給每個 token 一個短的稠密向量，讓意思相近的 token 向量也相近。
- 嵌入層就是一張**查詢表**（詞表大小 × 維度），等價於 one-hot 乘矩陣；它和模型一起訓練。
- **餘弦相似度**衡量方向的接近程度；向量之差可以編碼關係（`king − man + woman ≈ queen`）。
- **分佈假說**：上下文相似的詞意思相似；從共現統計或「預測周圍的詞」中都能學出詞向量。
- 神經語言模型（Bengio 2003）：嵌入 → 拼接 → MLP → softmax；嵌入會自動把作用相似的 token 放在一起。
- 注意力本身不知道順序，需要**位置編碼**（可學習、正弦、RoPE）；**句向量**是語義搜尋和 RAG 的基礎。

## 3.10 練習

<!-- exercise: nearest-word -->
### 練習 1：最相近的詞

第一行是整數 `n`，接下來 `n` 行，每行是一個詞和它的向量（若干個數，空格隔開），最後一行是要查詢的詞。輸出與查詢詞**餘弦相似度最高**的**另一個**詞，以及相似度（保留 4 位小數），空格隔開。

```text
樣例輸入：
4
cat 0.9 0.8 0.1
dog 0.8 0.9 0.2
car 0.1 0.2 0.9
bus 0.2 0.1 0.8
cat
樣例輸出：dog 0.9899
```

<details>
<summary>提示</summary>

先把每行拆成 `parts = line.split()`，`parts[0]` 是詞，`parts[1:]` 轉成小數就是向量；用 `max(..., key=...)` 找相似度最高的詞。
</details>

<!-- exercise: analogy -->
### 練習 2：詞語類比

起始程式碼裡給出了 3.3 節那張手工設計的詞向量表。輸入一行三個詞 `a b c`，輸出最接近 `b − a + c` 的詞（不能是 `a`、`b`、`c` 本身）。

```text
樣例輸入：man king woman
樣例輸出：queen
```

<details>
<summary>提示</summary>

`target = [vb - va + vc for va, vb, vc in zip(E[a], E[b], E[c])]`，再在其他詞裡找餘弦相似度最高的那個。
</details>

<!-- exercise: embedding-params -->
### 練習 3：嵌入層有多少引數？

輸入一行三個整數 `V d tied`：詞表大小、嵌入維度，以及輸入嵌入和輸出層是否共用一個矩陣（1 表示共用，0 表示不共用）。輸出這兩部分（不算偏置）一共有多少個引數。

```text
樣例輸入：50257 768 1
樣例輸出：38597376
```

<details>
<summary>提示</summary>

一個矩陣是 `V × d` 個引數；不共用時有兩個。
</details>

<!-- exercise: why-not-onehot -->
### 練習 4：one-hot 的問題

用 one-hot 向量表示詞，最主要的問題是？

- 計算機無法儲存 0 和 1
- 任意兩個不同的詞都互相垂直、距離一樣，無法表達「意思相近」，而且維度等於詞表大小
- one-hot 向量不能輸入神經網路
- one-hot 只能表示英文

<details>
<summary>答案與解析</summary>

不同的 one-hot 向量點積都是 0，彼此「一樣遠」；維度還和詞表一樣大（十幾萬）。嵌入用短的稠密向量解決了這兩個問題。
</details>

<!-- exercise: why-position -->
### 練習 5：為什麼需要位置編碼？

- 為了讓模型知道詞表有多大
- 因為注意力機制本身不區分 token 的先後順序，而「狗咬人」和「人咬狗」意思不同
- 為了減少引數量
- 為了讓模型支援中文

<details>
<summary>答案與解析</summary>

注意力把所有 token 放在一起計算，如果不加位置資訊，打亂順序結果也不變。位置編碼（可學習的、正弦的或 RoPE）把「第幾個」的資訊告訴模型。
</details>

## 3.11 延伸閱讀

- Mikolov 等：《Efficient Estimation of Word Representations in Vector Space》（2013，word2vec）
- Bengio 等：《A Neural Probabilistic Language Model》（2003）
- Andrej Karpathy：[Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)（本課 3.5 節的靈感來源）
- Jay Alammar：[The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)（圖解，網上有中文翻譯）
- 下一課：[注意力機制：讓每個詞「看到」上下文](../04-attention/zh-CN.md)
