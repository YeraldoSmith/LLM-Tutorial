# 2. 規模定律與算力

> 🎯 **本課目標**：會數一個 Transformer 有多少引數；會用 **C ≈ 6ND** 估算訓練要多少計算量、要多少張 GPU 訓練多少天；理解**規模定律**——損失隨引數量、資料量、計算量按冪律下降；理解 **Chinchilla** 的結論（每個引數約 20 個 token）以及為什麼今天的模型常常「過度訓練」；會估算訓練和推理需要多少視訊記憶體；瞭解「測試時計算」這條新的擴充套件路線。
>
> ⏱ **預計用時**：70 分鐘　｜　📋 **前置知識**：本部分第 1 課；第三部分第 5、6 課

訓練一個大模型要花幾百萬到上億美元。在花錢之前，研究者需要回答：模型做多大？用多少資料？要多少張 GPU、訓練多久？訓練完大概能有多好？這一課的工具讓你能用紙筆（和幾行 Python）回答這些問題。

## 2.1 數引數

第三部分第 5 課的 Transformer 裡，引數主要在兩個地方：注意力和前饋網路（MLP）。設模型維度為 d、層數為 L：

- 注意力：Q、K、V、輸出四個 d × d 的矩陣，共 **4d²**；
- MLP：d → 4d → d 兩個矩陣，共 **8d²**；
- 所以每層大約 **12d²**，再加上詞表大小 V × d 的嵌入矩陣。

現代的 Llama 式模型稍有不同：K、V 用**分組查詢注意力**（GQA，第 4 課）只佔較少的頭；MLP 用 SwiGLU，有三個矩陣；輸入嵌入和輸出層通常不共享。我們兩種都算一下：

```python
def gpt2_params(L, d, V, ctx):
    per_layer = 12 * d * d + 13 * d          # 12d² 的矩陣，加上偏置和 LayerNorm
    return L * per_layer + V * d + ctx * d + 2 * d

def llama_params(L, d, n_heads, n_kv, ffn, V):
    head_dim = d // n_heads
    attn = 2 * d * d + 2 * d * n_kv * head_dim    # Q 和輸出 d×d；K、V 只有 n_kv 個頭
    mlp = 3 * d * ffn                             # SwiGLU：門控、上投影、下投影
    per_layer = attn + mlp + 2 * d                # 加兩個 RMSNorm
    return L * per_layer + 2 * V * d + d          # 輸入嵌入 + 輸出層（不共享）+ 最後的 RMSNorm

print(f"GPT-2 small : {gpt2_params(12, 768, 50257, 1024) / 1e6:.0f}M")
print(f"GPT-2 XL    : {gpt2_params(48, 1600, 50257, 1024) / 1e9:.2f}B")
print(f"Llama 3 8B  : {llama_params(32, 4096, 32, 8, 14336, 128256) / 1e9:.2f}B")
print(f"Llama 3 70B : {llama_params(80, 8192, 64, 8, 28672, 128256) / 1e9:.2f}B")
```

算出的數字和官方公佈的引數量（1.24 億、15 億、80 億、706 億）完全吻合。可以看出：模型越大，引數越集中在每層的矩陣裡（和 d² 成正比），嵌入層所佔的比例越小。

## 2.2 計算量：C ≈ 6ND

訓練一個 N 個引數的模型、讀 D 個 token，總的計算量（浮點運算次數，FLOPs）大約是：

```formula
C ≈ 6 · N · D
```

為什麼是 6？每個引數在處理每個 token 時：

- **前向傳播**做一次乘法和一次加法 → 2 次運算；
- **反向傳播**要算對輸入的梯度和對引數的梯度，大約是前向的兩倍 → 4 次運算。

（注意力本身的計算和序列長度有關，在序列不太長時只佔一小部分，這裡忽略。）同理，**推理**時每生成一個 token 大約需要 **2N** 次運算。

有了這個公式，就能估算訓練時間：

```python
def train_days(N, D, gpus, peak_tflops, mfu):
    flops = 6 * N * D
    per_second = gpus * peak_tflops * 1e12 * mfu      # 整個叢集每秒實際完成的運算
    return flops / per_second / 86400

H100 = 989            # H100 的 BF16 稠密矩陣運算峰值，約 989 TFLOPS
print(f"80 億引數 × 15 萬億 token：{6 * 8e9 * 15e12:.1e} FLOPs")
for N, D, gpus in [(8e9, 15e12, 1024), (70e9, 15e12, 4096), (405e9, 15e12, 16384)]:
    days = train_days(N, D, gpus, H100, 0.40)
    print(f"{N / 1e9:>4.0f}B 引數、{D / 1e12:.0f}T token、{gpus:>5} 張 H100：約 {days:5.1f} 天")
```

- **MFU**（Model FLOPs Utilization，模型算力利用率）：GPU 的峰值算力只是理論值，實際訓練中要等資料、等通訊、做各種雜事，能用上 **30%～50%** 就算不錯了。
- Meta 訓練 Llama 3 405B 時用了 **1.6 萬張 H100**；DeepSeek 報告 DeepSeek-V3 的完整訓練用了約 **279 萬 H800 GPU 小時**。這些數字和我們的估算處在同一個量級。

## 2.3 規模定律

2020 年，OpenAI 的 Kaplan 等人發現了一個驚人的規律：在很大的範圍內（跨越好幾個數量級），語言模型的測試損失隨引數量 N、資料量 D、計算量 C 都按**冪律**平滑地下降。也就是說，在對數座標裡幾乎是一條直線——只要加大規模，效果就**可以預測地**變好。這就是**規模定律**（scaling laws），也是各家公司敢於投入巨資訓練更大模型的底氣。

2022 年，DeepMind 的 Hoffmann 等人（**Chinchilla** 論文）用 400 多個模型做實驗，把損失寫成引數量和資料量的函式：

```formula
L(N, D) = E + A / N^α + B / D^β
```

- E 是**不可消除的損失**：語言本身的隨機性，再大的模型也消除不了；
- A / N^α：模型**太小**帶來的損失；
- B / D^β：資料**太少**帶來的損失。

下面用 Besiroglu 等人（2024）重新擬合 Chinchilla 資料得到的係數，看看在**計算量固定**時，引數和資料該怎麼分配：

```python
E, A, B, alpha, beta = 1.82, 482.01, 2085.43, 0.3478, 0.3658

def loss(N, D):
    return E + A / N**alpha + B / D**beta

def best_split(C):                         # 計算量 C 固定：試遍各種 N，D 由 C = 6ND 決定
    candidates = [10 ** (k / 100) for k in range(700, 1300)]
    return min(candidates, key=lambda N: loss(N, C / (6 * N)))

print("計算量 C      最優引數 N   最優資料 D   D/N   預測損失")
for C in (1e21, 1e22, 1e23, 5.76e23, 1e24, 1e25):
    N = best_split(C)
    D = C / (6 * N)
    print(f"{C:9.1e}   {N / 1e9:8.1f}B   {D / 1e9:8.0f}B  {D / N:5.1f}   {loss(N, D):.3f}")

print()
print(f"Gopher     280B 引數 × 0.3T token：損失 {loss(280e9, 0.3e12):.3f}")
print(f"Chinchilla  70B 引數 × 1.4T token：損失 {loss(70e9, 1.4e12):.3f}")
```

結論：

- 計算量增加時，**引數量和資料量應該大致同比例增加**；最優點上，每個引數大約對應 **20 個 token**。
- 在此之前，大家主要在增加引數量。DeepMind 自己的 Gopher（2800 億引數，3000 億 token）其實**訓練得不夠**。用同樣的計算量（表裡 5.8e+23 那一行），訓練一個小 4 倍的 Chinchilla（700 億引數，1.4 萬億 token），效果反而更好。

## 2.4 為什麼今天的模型「過度訓練」

Chinchilla 的最優只考慮了**訓練**的花費。但模型訓練一次，卻要被使用千千萬萬次，**推理**的花費和引數量成正比。所以現在的常見做法是：訓練一個比「最優」更小的模型，餵給它遠多於 20 倍的資料——訓練時多花一些，換來以後每次推理都更便宜。

```python
E, A, B, alpha, beta = 1.82, 482.01, 2085.43, 0.3478, 0.3658

def loss(N, D):
    return E + A / N**alpha + B / D**beta

N = 8e9
print("80 億引數的模型，讀越來越多的資料：")
for D in (160e9, 1e12, 5e12, 15e12, 36e12):
    print(f"  {D / 1e12:5.2f}T token（每引數 {D / N:5.0f} 個）：預測損失 {loss(N, D):.3f}")
```

Llama 3 8B 讀了 15 萬億 token，是「最優」的近 100 倍，損失仍然在下降。（這裡的數字是用公式**外推**的，只能看趨勢：擬合公式時用的模型比這小得多。）

## 2.5 視訊記憶體：為什麼一張 GPU 訓練不了大模型

訓練時，每個引數要佔多少視訊記憶體？以常見的 **BF16 混合精度 + Adam 最佳化器**為例：

| 內容 | 每個引數的位元組數 |
|------|------------------|
| BF16 的引數（前向、反向用） | 2 |
| BF16 的梯度 | 2 |
| FP32 的「主引數」（更新時用，保證精度） | 4 |
| Adam 的一階動量 m（FP32） | 4 |
| Adam 的二階動量 v（FP32） | 4 |
| **合計** | **16** |

```python
import math

def train_memory_gb(N):
    return 16 * N / 1e9                      # 不含啟用值

def infer_memory_gb(N, bytes_per_param=2):
    return bytes_per_param * N / 1e9         # 只要存引數（不含 KV 快取，第六部分）

for N in (1e9, 8e9, 70e9, 405e9):
    need = train_memory_gb(N)
    print(f"{N / 1e9:>4.0f}B：訓練約 {need:>6.0f} GB（至少 {math.ceil(need / 80):>3} 張 80GB GPU），"
          f"BF16 推理約 {infer_memory_gb(N):>4.0f} GB")
```

這還沒算**啟用值**——前向傳播的中間結果，反向傳播時要用，和批大小、序列長度成正比，往往比引數還佔地方。所以 80 億引數的模型，一張 80GB 的 GPU 都放不下它的訓練狀態；700 億引數的模型至少要十幾張。下一課講怎樣把模型和資料分到成百上千張 GPU 上。

## 2.6 湧現、測試時計算和新的擴充套件方向

- **湧現能力**：有些能力（比如多位數算術）似乎在模型達到某個規模時「突然出現」。但 Schaeffer 等人（2023）指出，這很大程度上和評測指標有關：用「完全答對才算分」的指標，能力看起來是突變；換成連續的指標（比如每一位數字的正確率），提升其實是平滑的。
- **測試時計算**（test-time compute）：2024 年起，OpenAI o1、DeepSeek-R1 等**推理模型**展示了另一條擴充套件路線：不只加大訓練，還讓模型在回答前「多想一會兒」——生成長長的思考過程。思考得越多，難題的正確率越高，形成了新的規模定律（第五部分會詳細講）。
- **資料瓶頸**：高品質的人類文字是有限的，按現在的速度，幾年內可能被用完。這推動了合成資料、多模態資料和更高效訓練方法的研究。

## 2.7 小結

- 每層引數約 12d²（GPT 式）；Llama 式模型要考慮 GQA、SwiGLU 和不共享的嵌入層。
- 訓練計算量 C ≈ 6ND，推理每個 token ≈ 2N；訓練天數 = 6ND ÷（GPU 數 × 峰值算力 × MFU）。
- 規模定律：損失隨 N、D、C 按冪律平滑下降；L(N, D) = E + A/N^α + B/D^β。
- Chinchilla：計算量固定時，N 和 D 同比例增加，約 20 個 token / 引數；為了降低推理成本，今天的模型常常遠超這個比例。
- 訓練視訊記憶體約 16 位元組 / 引數（BF16 + Adam），另加啟用值；BF16 推理約 2 位元組 / 引數。
- 新方向：測試時計算、合成資料。

## 2.8 練習

<!-- exercise: param-count -->
### 練習 1：數引數

輸入層數 `L`、模型維度 `d` 和詞表大小 `V`。按「每層 12d²，加上嵌入層 V × d」估算引數量（忽略偏置、LayerNorm 和位置嵌入），以百萬為單位輸出，保留 1 位小數，後面加 `M`。

```text
樣例輸入：12 768 50257
樣例輸出：123.5M
```

<details>
<summary>提示</summary>

`params = 12 * L * d * d + V * d`，輸出 `f"{params / 1e6:.1f}M"`。
</details>

<!-- exercise: train-days -->
### 練習 2：要訓練多少天？

輸入一行：引數量 N（十億）、訓練 token 數 D（十億）、GPU 數量、每張 GPU 的峰值算力（TFLOPS）、MFU（0～1 之間的小數）。用 C = 6ND 估算訓練需要多少天，保留 1 位小數。

```text
樣例輸入：8 15000 1024 989 0.4
樣例輸出：20.6
```

<details>
<summary>提示</summary>

先把單位換算好：N 和 D 都乘以 `1e9`，算力乘以 `1e12`。天數 = `6 * N * D / (gpus * tflops * 1e12 * mfu) / 86400`。
</details>

<!-- exercise: chinchilla-opt -->
### 練習 3：Chinchilla 最優

輸入計算量 C（FLOPs，可能寫成 `1e23` 這樣的形式）。按「C = 6ND」和「D = 20N」兩條規則，求出最優的引數量 N 和資料量 D，以十億為單位輸出，各保留 1 位小數，用空格隔開。

```text
樣例輸入：1e23
樣例輸出：28.9 577.4
```

<details>
<summary>提示</summary>

把 D = 20N 代入 C = 6ND，得到 C = 120N²，所以 `N = (C / 120) ** 0.5`。`float("1e23")` 能直接讀入這種寫法。
</details>

<!-- exercise: adam-memory -->
### 練習 4：訓練要幾張 GPU？

輸入引數量 N（十億）和每張 GPU 的視訊記憶體（GB）。按每個引數 16 位元組（BF16 混合精度 + Adam，不算啟用值）計算訓練狀態需要多少 GB 視訊記憶體，以及至少需要幾張 GPU，用空格隔開輸出（視訊記憶體取整數）。

```text
樣例輸入：70 80
樣例輸出：1120 14
```

<details>
<summary>提示</summary>

`need = 16 * N`（N 以十億為單位時，結果正好是 GB）。GPU 數用 `math.ceil(need / mem)` 向上取整。
</details>

<!-- exercise: six-nd -->
### 練習 5：6ND 裡的 6

訓練計算量公式 C ≈ 6ND 中的「6」是怎麼來的？

- 6 是 Transformer 每層的矩陣個數
- 每個引數每處理一個 token：前向傳播約 2 次運算，反向傳播約 4 次運算
- 6 是經驗數字，沒有道理
- Adam 最佳化器每個引數要存 6 個數

<details>
<summary>答案與解析</summary>

前向傳播裡每個引數參與一次乘法和一次加法（2 次）；反向傳播既要算對輸入的梯度、又要算對引數的梯度，大約是前向的兩倍（4 次）。合計 6 次。推理只有前向，所以是每 token 約 2N。
</details>

<!-- exercise: chinchilla-lesson -->
### 練習 6：Chinchilla 的結論

在計算量固定的情況下，Chinchilla 論文得出了什麼結論？

- 應該把幾乎全部計算量用來增大模型
- 應該用儘可能小的模型、儘可能多的資料
- 引數量和訓練 token 數應該大致同比例增加，此前的大模型訓練資料太少了
- 資料量對損失沒有影響

<details>
<summary>答案與解析</summary>

Chinchilla 發現最優的做法是 N 和 D 同比例增加（約 20 個 token / 引數）。Gopher 等之前的模型引數很多、資料偏少；同樣的計算量，更小的模型配更多的資料效果更好。
</details>

## 2.9 延伸閱讀

- Kaplan 等：《Scaling Laws for Neural Language Models》（2020）
- Hoffmann 等：《Training Compute-Optimal Large Language Models》（2022，Chinchilla）
- Besiroglu 等：《Chinchilla Scaling: A replication attempt》（2024）
- Schaeffer 等：《Are Emergent Abilities of Large Language Models a Mirage?》（2023）
- 下一課：[分散式訓練與混合精度](../03-distributed/zh-CN.md)
