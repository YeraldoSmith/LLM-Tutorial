# 6. 親手訓練一個 GPT

> 🎯 **本課目標**：把前五課的零件組裝成一個**完整的 GPT**（約 100 行程式碼）；理解語言模型的訓練資料是怎樣構造的——**錯一位**的輸入和目標、**每個位置都在預測**；在普通電腦上用不到半分鐘訓練它學會兩位數加法，並在**從沒見過的題目**上驗證它真的學會了；再學會用同一份程式碼在你自己的文字（比如中文小說）上訓練；最後看清從「迷你 GPT」到真正的大模型還差什麼。
>
> ⏱ **預計用時**：90 分鐘　｜　📋 **前置知識**：本部分第 1～5 課；第二部分第 5 課（PyTorch）；需要安裝 PyTorch

這一課是第三部分的高潮。我們不再只看零件，而是從頭到尾訓練一個真正的 Transformer 語言模型，和 GPT 的結構完全相同，只是小得多。

> [!NOTE]
> 本課的程式碼結構參考了 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 [minGPT](https://github.com/karpathy/minGPT)（MIT 許可證，minGPT 裡也有一個加法的例子）。讀完本課再去讀 nanoGPT 的 `model.py`，你會發現幾乎每一行都看得懂。

## 6.1 任務：讓 GPT 學會做加法

為什麼選加法，而不是一上來就用小說或網頁？

- **不需要下載任何資料**：題目用程式碼生成，任何人、任何語言的讀者都能直接執行；
- **能檢驗「真的學會了」還是「背下來了」**：兩位數加法一共 100 × 100 = 10000 道題，我們**留出 1000 道**完全不給模型看，訓練後再拿它們來考；
- **它就是語言建模**：把每道題寫成一串字元，比如 `37+48=580`，模型只是在「預測下一個字元」——和 GPT 預測下一個 token 完全一樣。

你可能注意到了：37 + 48 = 85，可我們寫的是 `580`。這是一個小技巧：**答案補成 3 位後倒過來寫**（`085` → `580`）。我們做加法是從個位開始算、向高位進位的；而模型是從左到右一個一個地生成。倒過來寫，模型就能先生成個位、再生成十位和百位，和人做豎式的順序一致，學起來容易得多。

## 6.2 訓練資料：錯一位，每個位置都在預測

語言模型的訓練樣本構造方式非常簡單：**輸入是一段文字，目標是同一段文字往後錯一位**。

```text
完整序列：   3   7   +   4   8   =   5   8   0
輸入 x：     3   7   +   4   8   =   5   8
目標 y：     7   +   4   8   =   5   8   0
             ↑ 看到「3」時預測「7」；看到「37」時預測「+」；……看到「37+48=58」時預測「0」
```

因為有因果掩碼，位置 t 只能看到前 t 個字元，所以**一次前向傳播就同時訓練了每個位置的預測**——一個長度為 T 的序列提供 T 個訓練訊號。這正是 Transformer 訓練效率高的原因之一（RNN 必須一步一步地算）。

在這個任務裡，題目本身（兩個隨機數）是無法預測的，我們只想讓模型學會**根據題目給出答案**。所以把題目部分的目標設成 `-1`，再告訴損失函式 `ignore_index=-1`，這些位置就不計入損失。（對話模型的微調也是這樣做的：只在「助手的回答」上計算損失，第五部分會講。）

## 6.3 完整程式碼

下面就是全部程式碼：資料、模型、訓練、評估。模型的每個部分你都在前幾課見過：token 嵌入 + 位置嵌入（第 3 課）、多頭因果自注意力（第 4 課）、Block 與殘差（第 5 課）、權重繫結（第 3 課）、AdamW（第二部分第 6 課）。

```python requires=torch timeout=300
import random
import time
import torch
from torch import nn
import torch.nn.functional as F

# ---------------- 1. 資料：兩位數加法，例如 "37+48=" 後面跟著答案 85（補成 3 位並倒過來寫："580"）
random.seed(0)
torch.manual_seed(0)
pairs = [(a, b) for a in range(100) for b in range(100)]
random.shuffle(pairs)
test_pairs, train_pairs = pairs[:1000], pairs[1000:]       # 1000 道題留作測試，模型訓練時從沒見過

def to_text(a, b):
    return f"{a:02d}+{b:02d}={str(a + b).zfill(3)[::-1]}"    # 結果補成 3 位再倒過來："085" → "580"

chars = list("0123456789+=")                               # 詞表只有 12 個字元
stoi = {c: i for i, c in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]

block_size = 9                                             # "37+48=580" 正好 9 個字元
def get_batch(batch_size=128):
    batch = random.sample(train_pairs, batch_size)
    data = torch.tensor([encode(to_text(a, b)) for a, b in batch])
    x, y = data[:, :-1], data[:, 1:].clone()               # y 是 x 往後錯一位：每個位置預測下一個字元
    y[:, :5] = -1                                          # 題目部分不算損失，只學習「=」之後的答案
    return x, y

# ---------------- 2. 模型：一個迷你 GPT
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab, block_size, d=64, n_layer=2, n_heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)                  # token 嵌入
        self.pos = nn.Embedding(block_size, d)             # 位置嵌入
        self.blocks = nn.Sequential(*[Block(d, n_heads) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight                 # 權重繫結：輸出層與嵌入共用矩陣
    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        logits = self.head(self.ln(self.blocks(x)))        # (B, T, vocab)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new):                        # 貪心生成：每次取機率最大的 token
        for _ in range(n_new):
            logits = self(idx[:, -block_size:])
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

model = GPT(len(chars), block_size)
print(f"引數量：{sum(p.numel() for p in model.parameters()):,}")

def accuracy(pairs):
    model.eval()
    prompts = torch.tensor([encode(f"{a:02d}+{b:02d}=") for a, b in pairs])
    out = model.generate(prompts, 3)[:, 6:]                # 生成 3 個字元的（倒序）答案
    model.train()
    right = 0
    for (a, b), row in zip(pairs, out.tolist()):
        answer = int("".join(chars[i] for i in row)[::-1])
        right += answer == a + b
    return right / len(pairs)

# ---------------- 3. 訓練
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.1)
start = time.time()
for step in range(1, 1001):
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss.item():.3f}  測試集準確率 {accuracy(test_pairs):.1%}  ({time.time() - start:.0f}s)")

print("幾道模型從沒見過的題：")
for a, b in test_pairs[:6]:
    out = model.generate(torch.tensor([encode(f"{a:02d}+{b:02d}=")]), 3)[0, 6:].tolist()
    reply = int("".join(chars[i] for i in out)[::-1])
    print(f"  {a:2d} + {b:2d} = {reply:3d}   {'✓' if reply == a + b else '✗（正確答案 ' + str(a + b) + '）'}")
```

在普通膝上型電腦的 CPU 上，這段程式大約執行 10～30 秒。

## 6.4 讀懂結果

執行後你會看到類似這樣的輸出（具體數字因電腦而略有不同）：

```text
引數量：101,440
step  200  loss 1.515  測試集準確率 2.7%
step  400  loss 0.287  測試集準確率 66.4%
step  600  loss 0.072  測試集準確率 98.2%
step  800  loss 0.097  測試集準確率 93.4%
step 1000  loss 0.002  測試集準確率 100.0%
```

- **一開始完全不會**：準確率只有百分之幾，和瞎猜差不多。
- **然後突然「開竅」**：幾百步之間準確率從個位數跳到 90% 以上。神經網路學習規律時常常有這種「突變」。
- **測試題也全對**：這 1000 道題模型**一次都沒見過**，說明它學到的不是「37+48 的答案是 85」這樣的死記硬背，而是**加法和進位的規則**。
- 中間偶爾有起伏（比如 800 步時降到 93%），這是正常的訓練噪聲。

一個只有 10 萬個引數的模型，在不到一分鐘裡，僅僅透過「預測下一個字元」就學會了加法。把同樣的方法放大一百萬倍，就是今天的大語言模型。

> [!TIP]
> 動手試試：把 `n_layer` 改成 1、把 `d` 改成 32，或者去掉「倒著寫」的技巧（把 `[::-1]` 去掉，同時改掉 `accuracy` 裡的倒序），看看準確率和學習速度怎麼變化。這就是研究人員每天在做的**消融實驗**。

## 6.5 在你自己的文字上訓練

同一個模型，換一份資料，就能學習任何語言的文字。下面是一個**字元級**語言模型的訓練指令碼：準備一個文字檔案 `input.txt`（任何語言都行，比如一部中文小說、一本詩集），它會學習文字的風格，然後「續寫」。相比加法，主要的改動是：詞表來自文字本身、訓練樣本是文字中隨機擷取的片段、生成時按機率**取樣**而不是總取最大值（下一課詳細講取樣）。

```python norun
# 需要先準備好 input.txt（建議至少幾十萬字；文字越多效果越好），並把 6.3 節的 Block、GPT 類複製到這裡
import torch

text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))                            # 文字里出現過的所有字元就是詞表
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]            # 後 10% 做驗證集，監控是否過擬合

block_size, batch_size = 128, 32
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])       # 同樣是錯一位，這次每個位置都算損失
    return x, y

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT(len(chars), block_size, d=256, n_layer=6, n_heads=8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(5000):
    x, y = get_batch("train")
    _, loss = model(x.to(device), y.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(step, loss.item())

# 生成：從一個開頭出發，按機率取樣下一個字元
idx = torch.tensor([[stoi[c] for c in "從前"]], device=device)    # 開頭必須是 input.txt 裡出現過的字
for _ in range(200):
    logits = model(idx[:, -block_size:])[:, -1] / 0.8   # 0.8 是「溫度」，越低越保守（下一課講）
    nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
    idx = torch.cat([idx, nxt], dim=1)
print("".join(chars[i] for i in idx[0].tolist()))
```

> [!NOTE]
> 字元級模型最適合入門：不需要分詞器，而且對中文特別直觀（一個漢字就是一個 token）。中文常用字有幾千個，詞表比英文大，所以需要更多的資料和訓練步數。有 GPU 的話用 Google Colab 等免費的雲端 GPU 會快很多（第一部分第 2 課）。英文的經典練習資料是 Karpathy 整理的 [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)（約 1 MB 的莎士比亞劇本）。

## 6.6 從迷你 GPT 到真正的大模型

我們的模型和真正的大模型，**結構上沒有本質區別**。差別在於規模和工程：

| | 本課的加法 GPT | GPT-2 small（2019） | GPT-3（2020） | 今天的前沿模型 |
|---|---|---|---|---|
| 引數 | 10 萬 | 1.24 億 | 1750 億 | 數千億到上萬億（常用 MoE） |
| 層數 × 維度 | 2 × 64 | 12 × 768 | 96 × 12288 | 幾十到上百層 × 數千維 |
| 詞表 | 12 個字元 | 5 萬（BPE） | 5 萬（BPE） | 10 萬～25 萬 |
| 上下文 | 9 | 1024 | 2048 | 12.8 萬～數百萬 |
| 訓練資料 | 9000 道加法題 | 約 100 億 token | 約 3000 億 token | 十幾萬億到幾十萬億 token |
| 訓練硬體 | 一臺筆記本的 CPU，幾十秒 | 幾十塊 GPU | 上千塊 GPU | 上萬到十萬塊 GPU，數月 |

要從這裡走到那裡，需要：**更好的資料**（大規模清洗、去重、配比）、**更多的算力和分散式訓練**（把模型和資料切到成千上萬塊 GPU 上）、**縮放定律**（預先算出多大的模型配多少資料最划算）、以及上一部分講過的各種訓練技巧。這些是第四部分的內容。

而且，預訓練出來的模型只會「續寫」，還不會「聽指令、好好回答問題」。把它變成一個有用、安全的助手，還需要**微調和對齊**（第五部分）。

## 6.7 小結

- 一個完整的 GPT = token 嵌入 + 位置嵌入 → 若干個 Block（注意力 + MLP + 殘差 + 歸一化）→ 歸一化 → 輸出層；約 100 行 PyTorch 程式碼。
- 訓練資料：**目標 = 輸入錯一位**；因果掩碼讓一次前向傳播同時訓練序列中的每個位置；不想學的位置用 `ignore_index` 排除。
- 留出**測試集**才能判斷模型是學會了規律還是背下了答案；把答案倒著寫這樣的「資料表示」技巧會大大影響學習難度。
- 同一份程式碼換成任何文字就能訓練字元級語言模型；生成時按機率取樣。
- 迷你 GPT 和真正的大模型結構相同，區別在於資料、規模、算力和工程，以及之後的微調與對齊。

## 6.8 練習

<!-- exercise: shift-pairs -->
### 練習 1：錯一位

輸入一個字串，輸出語言模型用它做訓練時的所有「輸入 → 目標」對：對第 i 個字元（i 從 0 開始，到倒數第二個字元為止），輸出一行，包含這個字元和它後面的那個字元，中間用一個空格隔開。

```text
樣例輸入：hello
樣例輸出：
h e
e l
l l
l o
```

<details>
<summary>提示</summary>

`for a, b in zip(s, s[1:]): print(a, b)`
</details>

<!-- exercise: format-problem -->
### 練習 2：把題目寫成訓練文字

輸入兩個 0～99 之間的整數 `a b`，按本課的格式輸出訓練文字：兩個數都補成 2 位，答案補成 3 位後倒過來寫。

```text
樣例輸入：37 48
樣例輸出：37+48=580
```

<details>
<summary>提示</summary>

`f"{a:02d}+{b:02d}={str(a + b).zfill(3)[::-1]}"`
</details>

<!-- exercise: read-answer -->
### 練習 3：讀出模型的回答

模型生成的是本課格式的完整文字，例如 `37+48=580`。輸入這樣一行，輸出正常寫法的算式：答案正過來，所有數都去掉前面多餘的 0（例如 `04+21=520` 應輸出 `4+21=25`）。

```text
樣例輸入：37+48=580
樣例輸出：37+48=85
```

<details>
<summary>提示</summary>

用 `split("=")` 分開題目和答案；倒序答案再倒回來：`int(ans[::-1])`。題目部分裡的數可以用 `int()` 去掉前面的 0。
</details>

<!-- exercise: how-many-signals -->
### 練習 4：一次能學多少

一個 batch 裡有 32 個序列，每個序列長 128 個 token（每個位置都計算損失）。一次前向傳播一共提供了多少個「預測下一個 token」的訓練訊號？

- 32
- 128
- 4096
- 160

<details>
<summary>答案與解析</summary>

每個序列的每個位置都在預測下一個 token：32 × 128 = 4096。這就是 Transformer 訓練高效的原因之一。
</details>

<!-- exercise: why-holdout -->
### 練習 5：為什麼要留出 1000 道題？

- 為了讓訓練更快
- 為了檢驗模型學到的是加法規則，而不只是背下了見過的題
- 因為模型容量不夠，裝不下 10000 道題
- 為了讓損失降得更低

<details>
<summary>答案與解析</summary>

只在訓練過的題上測試，無法區分「學會了」和「背下來了」。在從沒見過的題上全對，才說明模型學到了可以推廣的規律——這就是**泛化**。
</details>

## 6.9 延伸閱讀

- Andrej Karpathy：[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)（影片，用 tinyshakespeare 訓練一個字元級 GPT）
- [nanoGPT](https://github.com/karpathy/nanoGPT)、[nanochat](https://github.com/karpathy/nanochat)（從分詞器到對話介面，約 100 美元訓練一個小型 ChatGPT）
- Sebastian Raschka：[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（第 4、5 章：實現並預訓練 GPT）
- 下一課：[解碼：模型怎樣選出下一個詞](../07-decoding/zh-CN.md)
