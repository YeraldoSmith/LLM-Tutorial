# 5. Transformer：把積木搭起來

> 🎯 **本課目標**：看懂一個 GPT 式 Transformer 的完整結構圖；理解一個 **Block** 裡的兩件事——注意力（token 之間交流）和 MLP（每個 token 各自計算），以及**殘差流**和 **Pre-Norm**；用 PyTorch 寫出 Block；親手算出 GPT-2 的 1.24 億個引數從哪裡來；分清編碼器（BERT）、解碼器（GPT）和編碼器-解碼器（T5）；認識 Llama 式的現代改進（RMSNorm、SwiGLU、RoPE、GQA）和**混合專家**（MoE）。
>
> ⏱ **預計用時**：90 分鐘　｜　📋 **前置知識**：本部分第 3、4 課；第二部分第 3、6 課

我們已經有了所有零件：嵌入（第 3 課）、注意力（第 4 課）、MLP（第二部分第 3 課）、歸一化和殘差連線（第二部分第 6 課）。這一課把它們組裝成 **Transformer**——更準確地說，是 GPT 系列使用的**僅解碼器**（decoder-only）Transformer。下一課我們就用它訓練一個真正的小 GPT。

## 5.1 全景圖

```text
輸入文字  "今天天氣"
   │  分詞器
   ▼
token 編號  [1123, 452, 452, 9981]                    （T 個 token）
   │  token 嵌入 + 位置資訊
   ▼
向量  (T × d)  ─────────────────────────┐
   │                                     │  殘差流（residual stream）
   │   ┌──────────── Block 1 ──────────┐ │  每個 Block 讀取它、
   │   │ x = x + Attention(Norm(x))    │ │  再把自己的結果「加」回去
   │   │ x = x + MLP(Norm(x))          │ │
   │   └───────────────────────────────┘ │
   │            …… 重複 N 次 ……           │
   │   ┌──────────── Block N ──────────┐ │
   │   └───────────────────────────────┘ │
   ▼                                     │
最後的 Norm  ◄───────────────────────────┘
   │  輸出層（d → 詞表大小）
   ▼
logits  (T × V)  →  softmax  →  每個位置上「下一個 token」的機率
```

幾個關鍵點：

- 整個過程中，每個 token 始終對應一個 d 維向量，形狀一直是 `T × d`，直到最後才變成 `T × V` 的 logits。
- 第 t 個位置的輸出預測的是**第 t+1 個** token。生成時只用最後一個位置的預測；**訓練時每個位置都在預測**，一次前向傳播就得到 T 個訓練訊號（下一課細講）。
- 模型的「深度」就是 Block 的層數 N：GPT-2 small 有 12 層，Llama 3 70B 有 80 層。

## 5.2 Block：先交流，再思考

每個 Block 由兩個子層組成，每個子層都是「**歸一化 → 計算 → 加回殘差**」：

```formula
x = x + Attention(Norm(x))     ← token 之間交流：從上下文收集資訊
x = x + MLP(Norm(x))           ← 每個 token 各自思考：處理收集到的資訊
```

1. **注意力**是唯一讓不同位置的 token **互相交換資訊**的地方（上一課）。
2. **MLP**（也叫 FFN，前饋網路）對**每個 token 獨立地**做同樣的計算：`d → 4d → d`，中間用 GELU 等啟用函式。研究認為大量「知識」（比如「巴黎是法國的首都」）儲存在 MLP 的權重裡。
3. **殘差連線**：每個子層只計算一個「修改量」加到 x 上，而不是替換 x。可以把 x 想象成一條貫穿全部層的**殘差流**，每一層都從中讀取、再往裡寫入。這讓梯度能暢通無阻地傳到最前面（第二部分第 6 課），上百層的網路也能訓練。
4. **Pre-Norm**：歸一化放在每個子層的**輸入**處（而不是原始 Transformer 那樣放在相加之後），訓練更穩定，是 GPT-2 以來的標準做法。

## 5.3 用 PyTorch 寫一個 Block

```python requires=torch
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):           # 上一課的多頭因果自注意力（用 PyTorch 自帶的高效實現）
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

class MLP(nn.Module):                           # d → 4d → d
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 4 * d)
        self.proj = nn.Linear(4 * d, d)
    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # ① token 之間交流資訊
        x = x + self.mlp(self.ln2(x))           # ② 每個 token 各自「思考」
        return x

torch.manual_seed(0)
block = Block(d=64, n_heads=4)
x = torch.randn(2, 10, 64)                      # 2 個序列，各 10 個 token，每個 64 維
print("輸入", tuple(x.shape), "→ 輸出", tuple(block(x).shape))   # 形狀不變，可以一直疊下去
print("一個 Block 的引數：", sum(p.numel() for p in block.parameters()), "= 12d² + 13d =", 12 * 64**2 + 13 * 64)
```

輸入和輸出的形狀完全一樣，所以可以把任意多個 Block 疊在一起。

## 5.4 GPT-2 的 1.24 億個引數從哪裡來？

一個 Block 的引數（d 是模型維度）：

| 部分 | 引數 |
|------|------|
| 注意力：QKV 投影 `d × 3d` + 偏置 `3d` | 3d² + 3d |
| 注意力：輸出投影 `d × d` + 偏置 `d` | d² + d |
| MLP：`d × 4d` + `4d`，`4d × d` + `d` | 8d² + 5d |
| 兩個 LayerNorm（各有 γ、β） | 4d |
| **合計** | **12d² + 13d** |

再加上 token 嵌入（`V × d`）、位置嵌入（`上下文長度 × d`）和最後一個 LayerNorm（`2d`）；輸出層與 token 嵌入**共用**同一個矩陣（權重繫結），不額外算引數：

```python
def gpt2_params(n_layer, d, vocab=50257, ctx=1024):
    emb = vocab * d + ctx * d                       # token 嵌入 + 位置嵌入
    attn = (d * 3 * d + 3 * d) + (d * d + d)        # QKV 投影 + 輸出投影（含偏置）
    mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)     # d → 4d → d（含偏置）
    norms = 2 * (2 * d)                             # 兩個 LayerNorm，各有 γ 和 β
    per_layer = attn + mlp + norms
    final = 2 * d                                   # 最後一個 LayerNorm；輸出層與 token 嵌入共用
    return per_layer, emb + n_layer * per_layer + final

for name, n_layer, d in [("GPT-2 small", 12, 768), ("GPT-2 medium", 24, 1024),
                         ("GPT-2 large", 36, 1280), ("GPT-2 XL", 48, 1600)]:
    per_layer, total = gpt2_params(n_layer, d)
    print(f"{name:13s} {n_layer:2d} 層  d={d:4d}  每層 {per_layer / 1e6:5.1f}M  總計 {total / 1e6:7.1f}M")
```

算出來正好是 OpenAI 公佈的 124M、355M、774M、1.5B。可以看到：**引數主要來自 `12d²`**——維度翻倍，每層引數約變成 4 倍。

## 5.5 三種 Transformer

原始的 Transformer（2017）是為機器翻譯設計的，有**編碼器**和**解碼器**兩部分。後來它分化成三個家族：

| 家族 | 代表 | 注意力 | 訓練目標 | 擅長 |
|------|------|--------|----------|------|
| **僅編碼器** | BERT（2018） | **雙向**：每個 token 能看到整句話 | 完形填空：遮住一些詞讓模型猜 | 理解：分類、檢索、句向量 |
| **編碼器-解碼器** | 原始 Transformer、T5 | 編碼器雙向，解碼器因果，再透過「交叉注意力」讀取編碼器 | 把輸入序列變成輸出序列 | 翻譯、摘要 |
| **僅解碼器** | GPT 系列、Llama、Qwen、DeepSeek、Claude…… | **因果**：只能看前文 | 預測下一個 token | 生成；規模足夠大後幾乎什麼都能做 |

今天的大語言模型幾乎全部是**僅解碼器**結構：它最簡單、最容易擴充套件，而且「預測下一個 token」這個目標可以用網際網路上**任何文字**來訓練，不需要人工標註。

## 5.6 現代的改進：Llama 式 Transformer

GPT-2 之後，Transformer 的整體結構沒有大變，但很多細節都改進了。Llama（Meta，2023）的組合成為了事實標準，Qwen、Mistral、DeepSeek 等都大量採用：

| 部件 | GPT-2（2019） | Llama 式（2023 至今） | 為什麼 |
|------|---------------|------------------------|--------|
| 歸一化 | LayerNorm | **RMSNorm** | 更省計算，效果相當 |
| 位置資訊 | 可學習的位置嵌入（加在輸入上） | **RoPE**（在注意力裡旋轉 q、k） | 表示相對位置，更容易擴充套件上下文 |
| MLP | GELU，`d → 4d → d` | **SwiGLU** 門控 MLP | 同樣引數量下效果更好 |
| K、V 頭 | 和 query 頭一樣多 | **GQA**：幾個 query 頭共用一組 K、V | KV 快取小好幾倍，推理更快 |
| 偏置 | 線性層都有偏置 | 通常**不要偏置** | 幾乎沒影響，更簡單 |
| 輸出層 | 與嵌入共用 | 大模型通常**不共用** | 引數充足時效果略好 |

**SwiGLU** 是一種「門控」MLP：用一條支路算出「門」（經過 SiLU 啟用），去乘另一條支路的結果，再投影回 d 維。它有三個矩陣，所以中間維度通常取 `8d/3` 左右，讓引數量和普通 MLP 差不多：

```python requires=torch
import torch
from torch import nn
import torch.nn.functional as F

class SwiGLU(nn.Module):                        # Llama、Qwen、DeepSeek 等使用的 MLP
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))   # 「門」逐元素乘以「內容」

mlp = SwiGLU(d=64, hidden=172)                  # 172 ≈ 8 × 64 / 3
print(tuple(mlp(torch.randn(2, 10, 64)).shape), "引數：", sum(p.numel() for p in mlp.parameters()))
```

用同樣的方法可以算出 Llama 3 的引數量。Llama 3 8B：32 層，d = 4096，32 個 query 頭、8 個 KV 頭，SwiGLU 中間維度 14336，詞表 128256：

```python
def llama_params(n_layer, d, n_heads, n_kv_heads, ffn_hidden, vocab):
    head_dim = d // n_heads
    attn = d * d + 2 * d * (n_kv_heads * head_dim) + d * d      # W_Q、W_K、W_V（GQA 讓 K、V 變小）、W_O
    mlp = 3 * d * ffn_hidden                                    # SwiGLU 的三個矩陣
    norms = 2 * d                                               # 兩個 RMSNorm（只有 γ）
    per_layer = attn + mlp + norms
    return n_layer * per_layer + 2 * vocab * d + d              # 輸入嵌入、輸出層不共用；最後一個 RMSNorm

print(f"Llama 3 8B ： {llama_params(32, 4096, 32, 8, 14336, 128256) / 1e9:.2f}B")
print(f"Llama 3 70B： {llama_params(80, 8192, 64, 8, 28672, 128256) / 1e9:.2f}B")
```

結果與 Meta 公佈的 8.03B 和 70.6B 一致。可以看到，在 Llama 式模型裡，**MLP 佔了引數的大頭**（8B 模型每層約 1.76 億個 MLP 引數，注意力只有約 4200 萬）。

## 5.7 混合專家（MoE）：引數很多，每次只用一部分

模型越大越聰明，但每個 token 的計算量也越大。**混合專家**（Mixture of Experts，MoE）的思路是：把一個大 MLP 換成**很多個小 MLP（專家）**，再加一個**路由器**（router）。每個 token 只被送到得分最高的少數幾個專家那裡：

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

# 4 個「專家」（真實模型裡每個都是一個 MLP；這裡用簡單的函式代替）
experts = [lambda x: [2 * v for v in x],         # 專家 0
           lambda x: [v + 1 for v in x],         # 專家 1
           lambda x: [-v for v in x],            # 專家 2
           lambda x: [v * v for v in x]]         # 專家 3
router = [[0.5, -1.0], [1.0, 0.2], [-0.3, 0.8], [0.1, 0.1]]     # 路由器：每個專家一行權重

def moe(x, top_k=2):
    scores = [sum(w * v for w, v in zip(row, x)) for row in router]
    chosen = sorted(range(len(experts)), key=lambda i: -scores[i])[:top_k]   # 只選得分最高的 k 個
    gates = softmax([scores[i] for i in chosen])                            # 選中專家的權重
    outs = [experts[i](x) for i in chosen]                                  # 只有被選中的專家要計算
    y = [sum(g * o[j] for g, o in zip(gates, outs)) for j in range(len(x))]
    return chosen, [round(g, 3) for g in gates], [round(v, 3) for v in y]

for x in ([1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]):
    print("輸入", x, "→ 選中的專家、權重、輸出：", moe(x))
```

不同的輸入被路由到不同的專家。這樣，模型的**總引數**可以非常大（儲存更多知識），而每個 token **實際參與計算的引數**（啟用引數）只是一小部分：

| 模型 | 總引數 | 每個 token 啟用的引數 |
|------|--------|------------------------|
| Mixtral 8x7B | 約 470 億 | 約 130 億（8 個專家選 2 個） |
| DeepSeek-V3 / R1 | 6710 億 | 約 370 億 |
| Qwen3-235B-A22B | 2350 億 | 約 220 億 |

MoE 已經是最前沿的大模型的主流選擇。它的難點在於：要讓各個專家「分工均衡」（不能所有 token 都擠到同一個專家），而且全部專家的引數都要裝進視訊記憶體。

## 5.8 小結

- GPT 式 Transformer：token 嵌入（+ 位置）→ N 個 Block → 最後的歸一化 → 輸出層 → 每個位置下一個 token 的 logits。
- Block = `x + Attention(Norm(x))`，再 `x + MLP(Norm(x))`：注意力負責 token 之間**交流**，MLP 負責每個 token 各自**計算**；殘差流貫穿全部層，Pre-Norm 讓訓練穩定。
- 每層引數約 **12d²**；GPT-2 small = 12 層 × d=768 ≈ 1.24 億引數。
- 三個家族：僅編碼器（BERT，雙向，理解）、編碼器-解碼器（T5，翻譯）、**僅解碼器**（GPT、Llama、Qwen……，今天的 LLM）。
- 現代改進：RMSNorm、RoPE、SwiGLU、GQA、去掉偏置；**MoE** 用路由器讓每個 token 只啟用少數專家，總引數大而計算量小。

## 5.9 練習

<!-- exercise: gpt2-params -->
### 練習 1：數一數 GPT-2 式模型的引數

輸入一行四個整數 `n_layer d vocab ctx`（層數、維度、詞表大小、上下文長度），按 5.4 節的 GPT-2 結構（每層 `12d² + 13d`，token 嵌入 + 位置嵌入，最後一個 LayerNorm `2d`，輸出層與嵌入共用）輸出總引數量。

```text
樣例輸入：12 768 50257 1024
樣例輸出：124439808
```

<details>
<summary>提示</summary>

`vocab * d + ctx * d + n_layer * (12 * d * d + 13 * d) + 2 * d`
</details>

<!-- exercise: llama-layer -->
### 練習 2：Llama 式的一層

輸入一行四個整數 `d n_heads n_kv_heads ffn_hidden`，按 5.6 節的 Llama 結構（無偏置；`W_Q` 和 `W_O` 各 `d × d`；`W_K` 和 `W_V` 各 `d × (n_kv_heads × d / n_heads)`；SwiGLU 三個矩陣 `3 × d × ffn_hidden`；兩個 RMSNorm 共 `2d`）輸出**一層**的引數量。

```text
樣例輸入：4096 32 8 14336
樣例輸出：218112000
```

<details>
<summary>提示</summary>

先算 `head_dim = d // n_heads`，K、V 每個是 `d * n_kv_heads * head_dim`。
</details>

<!-- exercise: residual-role -->
### 練習 3：殘差連線的作用

在 Transformer 裡，每個子層寫成 `x = x + f(x)` 而不是 `x = f(x)`，主要好處是？

- 減少引數量
- 梯度可以沿著「加法」這條路直接傳到前面的層，很深的網路也能訓練；每層只需學習一個「修改量」
- 讓注意力看不到未來
- 讓輸出變成機率

<details>
<summary>答案與解析</summary>

`x + f(x)` 的導數是 `1 + f'(x)`，那個 1 讓梯度至少能原樣穿過每一層；所有層共同讀寫一條「殘差流」。
</details>

<!-- exercise: which-family -->
### 練習 4：哪種結構？

GPT、Llama、Qwen、DeepSeek 這些對話大模型用的是哪種 Transformer？

- 僅編碼器（像 BERT）
- 編碼器-解碼器（像 T5）
- 僅解碼器（因果注意力，預測下一個 token）
- 不是 Transformer

<details>
<summary>答案與解析</summary>

它們都是僅解碼器結構：因果注意力 + 預測下一個 token。這種結構最簡單、最容易擴大規模，也能用任何文字訓練。
</details>

<!-- exercise: moe-why -->
### 練習 5：為什麼用 MoE？

DeepSeek-V3 總共有 6710 億個引數，但每個 token 只啟用約 370 億個。MoE 的主要好處是？

- 模型檔案更小
- 總引數（知識容量）很大，但每個 token 的計算量只相當於一個小得多的模型
- 不需要訓練路由器
- 不需要 GPU

<details>
<summary>答案與解析</summary>

每個 token 只經過少數幾個專家，計算量由啟用引數決定，而知識可以儲存在全部專家裡。代價是所有專家的引數仍然要放進記憶體。
</details>

## 5.10 延伸閱讀

- Jay Alammar：[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- Anthropic：[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)（「殘差流」視角的來源）
- Sebastian Raschka：《從零構建大模型》（*Build a Large Language Model (From Scratch)*，有中文版）及其開原始碼 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- Meta：《The Llama 3 Herd of Models》（2024）；DeepSeek-AI：《DeepSeek-V3 Technical Report》（2024）
- 下一課：[親手訓練一個 GPT](../06-gpt-from-scratch/zh-CN.md)
