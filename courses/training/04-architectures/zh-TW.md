# 4. 現代架構：更快、更省、更長

> 🎯 **本課目標**：在第三部分第 5 課（Llama 式 Transformer、MoE 入門）的基礎上更進一步：理解 Pre-Norm、RMSNorm 和 QK-Norm 怎樣讓訓練更穩定；比較 MHA、MQA、GQA 和 DeepSeek 的 **MLA**，學會計算 **KV 快取**有多大；理解長上下文的技術（RoPE 插值、YaRN、滑動視窗、注意力匯聚點）；親手實現 **FlashAttention** 的核心——**線上 softmax**；理解 MoE 的**負載均衡**；瞭解多 token 預測、線性注意力和混合架構等新方向。
>
> ⏱ **預計用時**：90 分鐘　｜　📋 **前置知識**：第三部分第 4、5 課；本部分第 2、3 課

2017 年的 Transformer 和 2026 年最強的模型，整體結構幾乎一樣：嵌入 → 很多個「注意力 + MLP」Block → 輸出層。但幾乎每個零件都被改進過，目標都是三件事：**訓練更穩定**、**推理更快更省**、**能處理更長的上下文**。

## 4.1 歸一化：Pre-Norm、RMSNorm 和 QK-Norm

原始的 Transformer 把 LayerNorm 放在殘差相加**之後**（Post-Norm）。模型一深，訓練就容易發散。GPT-2 起改成了放在每個子層**之前**（Pre-Norm）：

```text
Post-Norm：x = Norm(x + Attention(x))        Pre-Norm：x = x + Attention(Norm(x))
```

Pre-Norm 讓殘差連線成為一條「乾淨的高速公路」，梯度可以暢通無阻地傳到底層，所以幾百層的模型也能穩定訓練。

**RMSNorm**（Llama 起成為標準）比 LayerNorm 更簡單：不減去平均值，只除以均方根，再乘以可學習的縮放 γ：

```python
import math

def layer_norm(x, eps=1e-6):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    return [(v - mean) / math.sqrt(var + eps) for v in x]

def rms_norm(x, eps=1e-6):
    rms = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v / rms for v in x]

x = [2.0, -1.0, 4.0, 3.0]
print("LayerNorm：", [round(v, 3) for v in layer_norm(x)])
print("RMSNorm：  ", [round(v, 3) for v in rms_norm(x)])
print("RMSNorm 後的均方根：", round(math.sqrt(sum(v * v for v in rms_norm(x)) / len(x)), 3))
```

少算一個平均值看似微不足道，但歸一化在每層都要做好幾次，省下的時間很可觀，效果卻幾乎一樣。

**QK-Norm**：訓練很大的模型時，注意力的 logits（q·k）有時會越來越大，softmax 變得極端，訓練就崩了。在算點積之前，先對 q 和 k 各做一次 RMSNorm，就能把它們控制住。Gemma 3、Qwen3、OLMo 2 等模型都用了 QK-Norm。

## 4.2 注意力的變體和 KV 快取

生成文字時，模型每生成一個 token，都要用到前面所有 token 的 K 和 V。為了不重複計算，它們會被儲存下來，叫做 **KV 快取**（第六部分會詳細講）。KV 快取的大小是：

```formula
KV 快取（每個 token）= 2 × 層數 × KV 頭數 × 每個頭的維度 × 每個數的位元組數
```

上下文很長、同時服務的使用者很多時，KV 快取往往比模型引數本身還佔視訊記憶體。於是出現了幾種減少 K、V 的方法：

- **MHA**（多頭注意力）：每個 query 頭都有自己的 K、V 頭。
- **MQA**（多查詢注意力，2019）：所有 query 頭**共用一組** K、V。快取最小，但效果有所下降。
- **GQA**（分組查詢注意力，2023）：query 頭分成幾組，每組共用一組 K、V。Llama 3 70B 有 64 個 query 頭、8 個 KV 頭。效果接近 MHA，快取小 8 倍——現在最常用。
- **MLA**（多頭潛在注意力，DeepSeek-V2，2024）：不直接快取 K、V，而是快取一個壓縮過的**低維「潛在」向量**（DeepSeek-V3 每層 512 維，加上 64 維給 RoPE 用），用的時候再還原成各個頭的 K、V。

```python
def kv_cache_gb(layers, values_per_layer, tokens, bytes_per_value=2):
    return layers * values_per_layer * tokens * bytes_per_value / 1e9

T = 128_000                                       # 一條 12.8 萬 token 的上下文，BF16
configs = [
    ("MHA, 64 KV heads", 80, 2 * 64 * 128),       # 80 層，K 和 V 各 64 個頭 × 128 維
    ("GQA, 8 KV heads", 80, 2 * 8 * 128),         # Llama 3 70B 的設定
    ("MQA, 1 KV head", 80, 2 * 1 * 128),
    ("MLA (DeepSeek-V3)", 61, 512 + 64),          # 61 層，每層快取 576 個數
]
for name, layers, per_layer in configs:
    kb = layers * per_layer * 2 / 1024
    print(f"{name:<18} 每個 token {kb:6.1f} KB，12.8 萬 token 共 {kv_cache_gb(layers, per_layer, T):6.1f} GB")
```

從 MHA 到 GQA，快取縮小了 8 倍；MLA 則在 6710 億引數的模型上把它壓到了每個 token 不到 70 KB。

## 4.3 長上下文

2023 年初，大多數模型只能處理 2K～4K 個 token；到 2025 年，128K 已是標配，不少模型支援 100 萬 token 以上。這需要幾方面的技術。

### RoPE 的外推問題和插值

第三部分第 4 課講過，RoPE 讓第 i 組維度按 `θᵢ = base^(−2i/d)` 的速度旋轉。轉得最慢的那一組，在訓練長度以內只轉過很小的角度。如果直接用到更長的上下文，這一組會轉到訓練時**從沒見過的角度**，模型就不知所措了：

```python
import math

d, base = 128, 10_000
theta_slow = base ** (-(d - 2) / d)          # 轉得最慢的一組
train_len, target_len = 8_192, 32_768

print(f"訓練長度 {train_len}：最慢的一組最多轉到 {train_len * theta_slow:.2f} 弧度")
print(f"直接用到 {target_len}：會轉到 {target_len * theta_slow:.2f} 弧度（訓練時沒見過）")

scale = train_len / target_len               # 位置插值：把位置「壓縮」回訓練範圍
print(f"位置插值後：最多 {target_len * scale * theta_slow:.2f} 弧度")

new_base = base * (target_len / train_len) ** (d / (d - 2))   # 「NTK」式：改大 base，讓慢的組轉得更慢
theta_new = new_base ** (-(d - 2) / d)
print(f"把 base 改成 {new_base:,.0f} 後：最多 {target_len * theta_new:.2f} 弧度")
```

- **位置插值**（Position Interpolation，2023）：把位置乘以一個縮放係數，讓角度回到訓練過的範圍，再用少量長文字微調。
- **NTK 縮放、YaRN**（2023）：改大 base 或者對不同頻率的組區別對待——轉得快的組（負責區分相鄰 token）保持不變，只調整轉得慢的組。現在的模型常用 YaRN 把上下文擴充套件 4～32 倍。
- 訓練上，通常先用較短的序列（比如 8K）預訓練，最後再專門用長文件訓練一個「長上下文階段」（Llama 3 就是這樣擴充套件到 128K 的）。

### 滑動視窗和注意力匯聚點

- **滑動視窗注意力**：每個 token 只看最近的 W 個 token（比如 4096 個），計算和快取都不再隨上下文無限增長。Mistral 7B 用過這種方法；Gemma 3 把「區域性」（滑動視窗）層和「全域性」層按 5:1 交替排列——大多數層只看附近，少數層看全域性，KV 快取因此小了很多。
- **注意力匯聚點**（attention sink，2023）：研究者發現，模型會把大量的注意力分給**最開頭的幾個 token**，哪怕它們沒什麼意義——這是 softmax 必須讓權重加起來等於 1 的副作用。如果滑動視窗把開頭的 token 丟掉了，模型就會崩潰；只要一直保留開頭幾個 token，就能穩定地處理很長的文字流。

能「放得下」長上下文，不等於能「用好」它。有研究發現，模型對放在長上下文**中間**的資訊利用得最差（「迷失在中間」，lost in the middle）。第 5 課會介紹怎樣評測長上下文能力。

## 4.4 FlashAttention：精確，但更省

注意力要算一個 n × n 的分數矩陣。n = 128K 時，這個矩陣有 160 億個元素。直接算的話，要把它寫進 GPU 的視訊記憶體（HBM）再讀出來——GPU 的計算其實很快，**真正慢的是搬運資料**。

**FlashAttention**（Tri Dao 等，2022）的思路是：把 Q、K、V 切成小塊，放進 GPU 晶片上又小又快的快取（SRAM）裡，**一塊一塊地**算，完全不把 n × n 的矩陣寫出來。難點在於 softmax 需要知道**所有**分數的最大值和總和。答案是**線上 softmax**：一邊讀入新的分數塊，一邊更新「目前的最大值」和「目前的總和」，舊的結果用一個修正係數調整：

```python
import math

scores = [2.0, -1.0, 0.5, 3.0, 1.0, -0.5, 2.5, 0.0]       # 一個 query 對 8 個 key 的分數
values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]  # 每個 key 對應的 value（這裡是一維的）

# 普通做法：先算完所有分數，再 softmax，再加權平均
m = max(scores)
w = [math.exp(s - m) for s in scores]
standard = sum(wi * v for wi, v in zip(w, values)) / sum(w)

# 線上做法：每次只看 3 個分數，從不同時拿著全部分數
run_max, run_sum, run_out = float("-inf"), 0.0, 0.0
for start in range(0, len(scores), 3):
    block_s, block_v = scores[start:start + 3], values[start:start + 3]
    new_max = max(run_max, max(block_s))
    fix = math.exp(run_max - new_max)                     # 最大值變了：舊的累計結果要按比例縮小
    run_sum = run_sum * fix + sum(math.exp(s - new_max) for s in block_s)
    run_out = run_out * fix + sum(math.exp(s - new_max) * v for s, v in zip(block_s, block_v))
    run_max = new_max
    print(f"讀完前 {start + len(block_s)} 個：當前最大值 {run_max}，當前結果 {run_out / run_sum:.4f}")

print(f"普通做法：{standard:.6f}，線上做法：{run_out / run_sum:.6f}")
```

兩種做法的結果**完全相同**——FlashAttention 不是近似，而是同一個計算的更聰明的順序。它讓注意力需要的額外視訊記憶體從 O(n²) 降到 O(n)，速度也快了好幾倍。FlashAttention-2、-3 針對新的 GPU 進一步最佳化，現在幾乎所有訓練和推理框架都在用它（PyTorch 的 `scaled_dot_product_attention` 會自動呼叫）。

## 4.5 MoE 的負載均衡

第三部分第 5 課介紹了 MoE：路由器為每個 token 選出得分最高的幾個專家。訓練 MoE 最大的麻煩是**負載不均衡**：一開始某幾個專家稍好一點，路由器就把更多 token 發給它們，它們越學越好，得到的 token 越來越多……最後大部分專家閒著，模型退化成了小模型，而且負責熱門專家的 GPU 忙不過來。

常見的解決辦法是加一個**輔助損失**（Switch Transformer，2021）：設 fᵢ 是被分給專家 i 的 token 比例，Pᵢ 是路由器給專家 i 的平均機率，E 是專家數：

```formula
輔助損失 = E · Σᵢ fᵢ · Pᵢ
```

完全均衡時它等於 1；越不均衡，它越大。把它加進總損失，模型就會被推向均衡：

```python
import math
import random

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

random.seed(0)
E, T = 4, 1000                                       # 4 個專家，1000 個 token
for name, bias in [("偏向專家 0", [2.0, 0, 0, 0]), ("均衡", [0, 0, 0, 0])]:
    counts, prob_sum = [0] * E, [0.0] * E
    for _ in range(T):
        p = softmax([random.gauss(0, 1) + b for b in bias])   # 路由器給這個 token 的機率
        counts[p.index(max(p))] += 1                          # top-1：分給機率最大的專家
        prob_sum = [s + pi for s, pi in zip(prob_sum, p)]
    f = [c / T for c in counts]
    P = [s / T for s in prob_sum]
    aux = E * sum(fi * pi for fi, pi in zip(f, P))
    print(f"{name}：各專家的 token 比例 {[round(v, 2) for v in f]}，輔助損失 {aux:.3f}")
```

DeepSeek 在此基礎上做了幾項改進：

- **細粒度專家 + 共享專家**（DeepSeekMoE，2024）：把專家切得更小、更多（DeepSeek-V3 每層 256 個路由專家，每個 token 選 8 個），再加一個所有 token 都會經過的**共享專家**，負責通用知識；
- **無輔助損失的均衡**（DeepSeek-V3）：輔助損失會干擾主要的訓練目標，於是改為給每個專家的路由分數加一個偏置：某個專家太忙就調低它的偏置，太閒就調高，偏置隻影響「選誰」，不影響輸出的加權。

## 4.6 更多新方向

- **多 token 預測**（MTP）：DeepSeek-V3 在訓練時，每個位置不僅預測下一個 token，還用額外的小模組預測再下一個 token。這給模型提供了更密集的訓練訊號，推理時這些模組還能用來加速（投機解碼，第六部分）。
- **線性注意力和狀態空間模型**：普通注意力的計算隨上下文長度平方增長。Mamba（2023）等**狀態空間模型**和各種**線性注意力**把它降到了線性，生成時只需要一個固定大小的「狀態」，不需要越來越大的 KV 快取。但它們在需要「精確回憶」上下文的任務上通常不如普通注意力。
- **混合架構**：於是很多新模型把兩者混合：大部分層用線性注意力或狀態空間層，每隔幾層放一層普通注意力。Jamba（2024）、MiniMax-01（2025）、Qwen3-Next（2025）等都採用了這樣的設計，在超長上下文上兼顧了速度和效果。

## 4.7 小結

- Pre-Norm 讓深層模型穩定；RMSNorm 更省計算；QK-Norm 防止注意力 logits 爆炸。
- KV 快取（每 token）= 2 × 層數 × KV 頭數 × 頭維度 × 位元組數；GQA 讓多個 query 頭共用 K、V，MLA 快取一個壓縮的潛在向量。
- 長上下文：位置插值、NTK、YaRN 調整 RoPE；先短後長地訓練；滑動視窗和區域性 / 全域性交替減少開銷；要保留注意力匯聚點。
- FlashAttention：分塊計算 + 線上 softmax，結果精確，視訊記憶體從 O(n²) 降到 O(n)，速度更快。
- MoE 負載均衡：輔助損失 E·Σfᵢ·Pᵢ，或 DeepSeek 的偏置調節；細粒度專家 + 共享專家。
- 新方向：多 token 預測、線性注意力和狀態空間模型、混合架構。

## 4.8 練習

<!-- exercise: rmsnorm -->
### 練習 1：RMSNorm

輸入一行數（一個向量），輸出它經過 RMSNorm（γ 全為 1，eps = 1e-6）之後的結果，每個數保留 4 位小數，用空格隔開。

```text
樣例輸入：2 -1 4 3
樣例輸出：0.7303 -0.3651 1.4606 1.0954
```

<details>
<summary>提示</summary>

`rms = math.sqrt(sum(v * v for v in x) / len(x) + 1e-6)`，然後每個數除以 `rms`。
</details>

<!-- exercise: kv-cache -->
### 練習 2：KV 快取有多大

輸入一行：層數、KV 頭數、每個頭的維度、上下文的 token 數、每個數的位元組數。輸出這條上下文的 KV 快取大小（GB，1 GB = 10⁹ 位元組），保留 2 位小數。

```text
樣例輸入：32 8 128 8192 2
樣例輸出：1.07
```

<details>
<summary>提示</summary>

`2 * 層數 * KV頭數 * 頭維度 * token數 * 位元組數 / 1e9`，前面的 2 是因為 K 和 V 各存一份。
</details>

<!-- exercise: online-softmax -->
### 練習 3：線上 softmax

第一行是一個 query 對各個 key 的注意力分數，第二行是對應的 value（每個 value 是一個數），第三行是塊的大小 B。按 4.4 節的方法，每次只處理 B 個分數，用線上 softmax 算出注意力的輸出（加權平均），保留 6 位小數。

```text
樣例輸入：
2 -1 0.5 3 1 -0.5 2.5 0
10 20 30 40 50 60 70 80
3
樣例輸出：44.332136
```

<details>
<summary>提示</summary>

維護 `run_max`、`run_sum`、`run_out` 三個量。每讀入一塊：先求新的最大值，再用 `math.exp(run_max - new_max)` 修正舊的 `run_sum` 和 `run_out`。注意一開始 `run_max` 是負無窮，`math.exp(float("-inf"))` 等於 0，正好可以用。
</details>

<!-- exercise: moe-balance -->
### 練習 4：MoE 的輔助損失

第一行是 token 數 T 和專家數 E，接下來 T 行，每行是路由器給這個 token 的 E 個機率。每個 token 分給機率最大的專家（top-1；機率相同時選編號小的）。按 4.5 節的公式輸出輔助損失 `E · Σ fᵢ · Pᵢ`，保留 4 位小數。

```text
樣例輸入：
4 2
0.9 0.1
0.8 0.2
0.6 0.4
0.3 0.7
樣例輸出：1.1500
```

<details>
<summary>提示</summary>

fᵢ = 分給專家 i 的 token 數 / T；Pᵢ = 所有 token 在第 i 個機率上的平均值。`p.index(max(p))` 就是機率最大（並且編號最小）的專家。
</details>

<!-- exercise: gqa-why -->
### 練習 5：為什麼用 GQA？

和標準的多頭注意力（MHA）相比，GQA 的主要好處是什麼？

- 讓每個 query 頭看到更多的 token
- 幾個 query 頭共用一組 K、V，KV 快取小了好幾倍，推理更快，效果幾乎不變
- 不再需要位置編碼
- 讓注意力的計算從 O(n²) 變成 O(n)

<details>
<summary>答案與解析</summary>

GQA 把 query 頭分組，每組共用一組 K、V，所以需要快取的 K、V 少了很多（Llama 3 70B 少了 8 倍）。它不改變注意力的 O(n²) 計算，也不影響位置編碼。
</details>

<!-- exercise: flash-exact -->
### 練習 6：FlashAttention 是近似嗎？

關於 FlashAttention，下面哪個說法是對的？

- 它只計算一部分注意力分數，所以結果是近似的
- 它算出的結果和普通注意力完全相同，只是透過分塊和線上 softmax 減少了視訊記憶體讀寫
- 它把 softmax 換成了另一個更快的函式
- 它只能在推理時使用，不能用於訓練

<details>
<summary>答案與解析</summary>

FlashAttention 是精確的：它改變的是計算順序，不再把 n × n 的分數矩陣寫進視訊記憶體，而是分塊計算並用線上 softmax 合併結果。訓練和推理都在用。
</details>

## 4.9 延伸閱讀

- Ainslie 等：《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》（2023）
- DeepSeek-AI：《DeepSeek-V2》（2024，MLA 和 DeepSeekMoE）、《DeepSeek-V3 Technical Report》（2024）
- Dao 等：《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》（2022）
- Peng 等：《YaRN: Efficient Context Window Extension of Large Language Models》（2023）
- Xiao 等：《Efficient Streaming Language Models with Attention Sinks》（2023）
- Gu、Dao：《Mamba: Linear-Time Sequence Modeling with Selective State Spaces》（2023）
- 下一課：[評測](../05-evaluation/zh-CN.md)
