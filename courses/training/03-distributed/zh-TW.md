# 3. 分散式訓練與混合精度

> 🎯 **本課目標**：認識 FP32、FP16、BF16、FP8 等數字格式，理解**混合精度訓練**為什麼要保留 FP32 的主引數，FP16 為什麼需要損失縮放；理解**梯度累積**；理解**資料並行**和 **all-reduce**（親手模擬環形 all-reduce）；理解 **ZeRO / FSDP** 怎樣把最佳化器狀態、梯度和引數切分到多張 GPU 上；理解**張量並行**和**流水線並行**（以及流水線的「氣泡」）；瞭解啟用重計算和真實的大規模訓練是什麼樣子。
>
> ⏱ **預計用時**：90 分鐘　｜　📋 **前置知識**：本部分第 2 課；第二部分第 4、5 課

上一課算過：訓練一個 700 億引數的模型，光是引數和最佳化器狀態就要 1 TB 多的視訊記憶體，計算量要幾千張 GPU 算上一兩個月。這一課講兩件事：**怎樣讓每張 GPU 算得更快、更省視訊記憶體**（低精度），以及**怎樣讓成千上萬張 GPU 一起訓練一個模型**（並行）。

## 3.1 數字格式：位數越少，越快越省

計算機裡的浮點數由三部分組成：符號、**指數**（決定能表示多大、多小的數，即**範圍**）和**尾數**（決定有多少位有效數字，即**精度**）。

| 格式 | 總位數 | 指數位 | 尾數位 | 最大值 | 大約幾位有效數字 |
|------|--------|--------|--------|--------|------------------|
| FP32 | 32 | 8 | 23 | 3.4 × 10³⁸ | 7 |
| FP16 | 16 | 5 | 10 | 65504 | 3 |
| BF16 | 16 | 8 | 7 | 3.4 × 10³⁸ | 2～3 |
| FP8（E4M3） | 8 | 4 | 3 | 448 | 1 |

位數減半，視訊記憶體和資料傳輸減半，GPU 的矩陣運算速度通常翻倍（H100 的 BF16 約 989 TFLOPS，FP8 約 1979 TFLOPS）。代價是精度和範圍。用 Python 的 `struct` 模組模擬一下：

```python
import struct

def to_fp32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

def to_fp16(x):
    try:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    except OverflowError:                  # 超出 FP16 的範圍
        return float("inf")

def to_bf16(x):                            # BF16 就是 FP32 的高 16 位（四捨五入）
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]

print(f"{'x':>13} {'FP32':>14} {'FP16':>14} {'BF16':>14}")
for x in (1 / 3, 3.14159265, 1.001, 70000.0, 1e-8):
    print(f"{x:>13.6g} {to_fp32(x):>14.8g} {to_fp16(x):>14.8g} {to_bf16(x):>14.8g}")
```

- **FP16**：精度還行，但**範圍很小**——超過 65504 就變成無窮大，小於大約 6 × 10⁻⁸ 就變成 0。訓練時很小的梯度會直接「消失」。
- **BF16**（Google 為深度學習設計的格式）：和 FP32 一樣有 8 位指數，**範圍相同**，不會溢位；但只有 7 位尾數，精度很低——`1.001` 變成了 `1`。
- 現在訓練大模型的標準選擇是 **BF16**。

## 3.2 混合精度訓練

BF16 精度這麼低，為什麼還能訓練？關鍵是**混合精度**：

- **前向和反向傳播**中的矩陣乘法用 BF16 計算（佔絕大部分時間），速度快、省視訊記憶體；
- **引數的更新**用 FP32 的「主引數」完成。

為什麼更新一定要用 FP32？看一個例子：引數是 1.0，每一步加上一個很小的更新 0.0001：

```python
import struct

def to_bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]

w_bf16, w_fp32 = 1.0, 1.0
for step in range(1000):
    w_bf16 = to_bf16(w_bf16 + 0.0001)      # 每次加完都存回 BF16
    w_fp32 = w_fp32 + 0.0001
print("只用 BF16 存引數：", w_bf16)        # 一直是 1.0！
print("用 FP32 存主引數：", round(w_fp32, 4))
```

BF16 在 1.0 附近能表示的最小間隔是 2⁻⁷ ≈ 0.0078，`1.0 + 0.0001` 四捨五入後還是 1.0——更新**完全丟失了**。訓練後期的更新往往就是這麼小，所以引數必須用 FP32 儲存和更新（這正是上一課「每個引數 16 位元組」裡那 4 位元組主引數的來歷）。

**FP16 的損失縮放**：如果用 FP16，還有一個問題——很小的梯度會下溢成 0。解決辦法是先把損失乘以一個大數（比如 65536），梯度也跟著放大，算完再除回去：

```python
import struct

def to_fp16(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]

g = 1e-8                                   # 一個很小的梯度
print("直接存成 FP16：", to_fp16(g))
scale = 65536
print("先放大 65536 倍，存成 FP16，再縮小：", to_fp16(g * scale) / scale)
```

BF16 的範圍和 FP32 一樣，不需要損失縮放，這是它取代 FP16 的主要原因。

在 PyTorch 裡，混合精度只需要一個 `torch.autocast`：

```python requires=torch timeout=120
import torch

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(512, 64, device=device)
y = 2 * x[:, :1] + 0.5

for step in range(201):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):   # 這裡面的矩陣乘法用 BF16
        pred = model(x)
    loss = torch.nn.functional.mse_loss(pred.float(), y)
    opt.zero_grad()
    loss.backward()
    opt.step()                                                       # 引數和梯度仍是 FP32
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  輸出 {pred.dtype}  "
              f"引數 {model[0].weight.dtype}  梯度 {model[0].weight.grad.dtype}")
```

**FP8 訓練**：DeepSeek-V3（2024）把大部分矩陣乘法換成了 FP8，並用「細粒度縮放」（每一小塊資料配一個縮放因子）控制誤差，是第一批在超大規模上成功使用 FP8 訓練的模型之一。

## 3.3 梯度累積

批大小（batch size）越大，梯度越穩定。大模型訓練的全域性批大小通常是**幾百萬個 token**，一張 GPU 一次根本放不下。辦法之一是**梯度累積**：一次只算一小批（微批次），把梯度累加起來，攢夠了再更新一次引數：

```python
data = [(1.0, 2.1), (2.0, 3.9), (3.0, 6.2), (4.0, 7.8), (5.0, 10.1), (6.0, 12.2), (7.0, 13.8), (8.0, 16.1)]
w = 0.5

def grad(batch, w):                         # 均方誤差 mean((w·x - y)²) 對 w 的梯度
    return sum(2 * (w * x - y) * x for x, y in batch) / len(batch)

full = grad(data, w)                        # 一次算完整個批次

micro = [data[i:i + 2] for i in range(0, len(data), 2)]   # 4 個微批次，每個 2 條資料
acc = 0.0
for mb in micro:
    acc += grad(mb, w) / len(micro)         # 累加，併除以累積的步數
print(f"整批計算：{full:.4f}，梯度累積：{acc:.4f}")
```

兩者完全相等。梯度累積用**時間**換**視訊記憶體**：多算幾次前向和反向，才更新一次引數。

## 3.4 資料並行和 all-reduce

**資料並行**（Data Parallelism, DP）是最常用的並行方式：每張 GPU 上都放一份**完整的模型**，各自處理**不同的資料**，算出各自的梯度；然後所有 GPU 把梯度**求平均**，每張 GPU 都用這個平均梯度更新引數，所以模型始終保持一致。這和梯度累積在數學上是一回事——只不過微批次分給了不同的 GPU 同時計算。

「每張 GPU 都得到所有梯度之和」這個操作叫 **all-reduce**。最樸素的做法是全都發給一張 GPU 加起來再發回去，但那張 GPU 會成為瓶頸。實際中常用**環形 all-reduce**：GPU 排成一個環，每張只和下一張通訊。把資料切成 n 塊：

1. **reduce-scatter**：轉 n−1 圈，每一圈每張 GPU 把一塊資料傳給下一張，下一張把它加到自己的那塊上。結束時，每張 GPU 各擁有**一塊**完整的總和。
2. **all-gather**：再轉 n−1 圈，把這些完整的塊傳遍整個環。

```python
def ring_allreduce(data):
    n = len(data)                              # GPU 數
    c = len(data[0]) // n                      # 每塊的長度
    buf = [list(d) for d in data]
    sent = [0] * n                             # 每張 GPU 傳送了多少個數
    for step in range(n - 1):                  # ① reduce-scatter
        msgs = []
        for i in range(n):
            k = (i - step) % n                 # 這一步 GPU i 傳送第 k 塊
            msgs.append(((i + 1) % n, k, buf[i][k * c:(k + 1) * c]))
            sent[i] += c
        for dst, k, chunk in msgs:
            for j in range(c):
                buf[dst][k * c + j] += chunk[j]
    for step in range(n - 1):                  # ② all-gather
        msgs = []
        for i in range(n):
            k = (i + 1 - step) % n
            msgs.append(((i + 1) % n, k, buf[i][k * c:(k + 1) * c]))
            sent[i] += c
        for dst, k, chunk in msgs:
            buf[dst][k * c:(k + 1) * c] = chunk
    return buf, sent

grads = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400], [1000, 2000, 3000, 4000]]
result, sent = ring_allreduce(grads)
for i, r in enumerate(result):
    print(f"GPU {i}：{r}")
print("每張 GPU 傳送的資料量：", sent, "（梯度一共 4 個數）")
```

每張 GPU 只傳送了 `2(n−1)/n` 倍的梯度量——即使有 1000 張 GPU，每張傳送的資料也不到梯度總量的 2 倍，而且所有連線同時在工作。NVIDIA 的 **NCCL** 庫實現了這些「集合通訊」操作，PyTorch 的 `DistributedDataParallel`（DDP）就用它來同步梯度。

## 3.5 ZeRO 和 FSDP：把冗餘切掉

資料並行有一個浪費：每張 GPU 都存著**完整的**引數、梯度和最佳化器狀態（每引數 16 位元組）。微軟的 **ZeRO**（2019）想到：既然大家存的一樣，不如**切開分著存**，需要時再臨時交換：

| 階段 | 切分什麼 | 每張 GPU 的視訊記憶體（N 個引數、n 張 GPU） |
|------|----------|----------------------------------------|
| 普通 DP | 什麼都不切 | 16N |
| ZeRO-1 | 最佳化器狀態（12 位元組） | 4N + 12N / n |
| ZeRO-2 | 再加上梯度 | 2N + 14N / n |
| ZeRO-3 | 再加上引數本身 | 16N / n |

```python
def zero_gb(N, n, stage):
    per_param = {0: 16, 1: 4 + 12 / n, 2: 2 + 14 / n, 3: 16 / n}[stage]
    return N * per_param / 1e9

N, n = 70e9, 64
for stage in range(4):
    print(f"ZeRO-{stage}：每張 GPU {zero_gb(N, n, stage):7.1f} GB")
```

ZeRO-3 時，每張 GPU 只存 1/n 的引數；計算某一層之前，先把這一層的引數從其他 GPU 收集過來（all-gather），用完就扔掉。PyTorch 自帶的 **FSDP**（Fully Sharded Data Parallel）就是 ZeRO-3 的實現，DeepSpeed 庫也提供了完整的 ZeRO。

## 3.6 張量並行和流水線並行

當一層的矩陣太大、或者希望減少每張 GPU 上的計算量時，就要把**模型本身**拆開。

**張量並行**（Tensor Parallelism, TP，來自 NVIDIA 的 Megatron-LM）：把一個矩陣乘法拆到幾張 GPU 上。以 MLP 為例，第一個矩陣**按列**切開，第二個矩陣**按行**切開，每張 GPU 獨立算完自己的部分，最後只需要一次 all-reduce 把結果加起來：

```python
def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

def relu(M):
    return [[max(0, v) for v in row] for row in M]

x = [[1, 2], [3, -4]]                       # 2 個 token，每個 2 維
W = [[1, 0, 2, -1], [0, 1, 1, 3]]           # 2 → 4（MLP 第一層）
V = [[1, 2], [0, 1], [1, 0], [2, 1]]        # 4 → 2（MLP 第二層）
full = matmul(relu(matmul(x, W)), V)        # 一張 GPU 的完整計算

W0, W1 = [r[:2] for r in W], [r[2:] for r in W]   # GPU 0 拿前兩列，GPU 1 拿後兩列
V0, V1 = V[:2], V[2:]                              # 對應地，V 按行切開
part0 = matmul(relu(matmul(x, W0)), V0)            # 兩張 GPU 各算各的，中間不用通訊
part1 = matmul(relu(matmul(x, W1)), V1)
summed = [[a + b for a, b in zip(r0, r1)] for r0, r1 in zip(part0, part1)]   # 一次 all-reduce
print("完整計算：", full)
print("張量並行：", summed, "相同" if full == summed else "不同")
```

（ReLU 是逐個元素計算的，所以按列切開後各自做也完全正確。注意力也可以按「頭」切開。）張量並行每一層都要通訊，需要 GPU 之間非常快的連線，所以通常只在**同一臺機器**的 8 張 GPU 之間使用（它們用 NVLink 相連）。

**流水線並行**（Pipeline Parallelism, PP）：把模型**按層**切開，比如 80 層分給 8 張 GPU，每張 10 層，資料像流水線一樣依次經過。問題是：第 1 張 GPU 算完第一批資料，第 8 張 GPU 才剛開始；其他 GPU 在等待，這些空閒叫**氣泡**（bubble）。辦法是把一個批次切成很多個微批次，讓它們首尾相接地流過：

```python
def bubble(p, m):              # p 段流水線，m 個微批次
    return (p - 1) / (m + p - 1)

for m in (1, 4, 16, 64):
    print(f"8 段流水線、{m:>2} 個微批次：GPU 空閒 {bubble(8, m):.0%}")
```

## 3.7 啟用重計算，以及把它們組合起來

- **啟用重計算**（activation checkpointing）：前向傳播時不儲存所有中間結果，只儲存少數「檢查點」；反向傳播需要時再從檢查點**重新算一遍**。多花約 1/3 的計算，換來視訊記憶體的大幅節省。
- **組合使用**：真實的大規模訓練會同時使用好幾種並行。比如 Meta 訓練 Llama 3 405B 時，在 1.6 萬張 H100 上組合了張量並行（8 張，同一臺機器內）、流水線並行（16 段）、資料並行（FSDP），以及為長序列準備的**上下文並行**（把一條長序列切到多張 GPU 上）——被稱為「4D 並行」。混合專家模型（第 4 課）還有**專家並行**：不同的專家放在不同的 GPU 上。
- **故障**：幾萬張 GPU 一起工作好幾個月，硬體故障是家常便飯。Llama 3 的論文提到，在 54 天裡遇到了 400 多次意外中斷。所以訓練程式要經常儲存**檢查點**（checkpoint），出故障時能從最近的檢查點恢復。

## 3.8 小結

- FP32 / FP16 / BF16 / FP8：位數越少越快越省，代價是精度和範圍；BF16 和 FP32 範圍相同，是現在的標準選擇。
- 混合精度：矩陣乘法用 BF16，引數用 FP32 主副本更新（小的更新在 BF16 裡會丟失）；FP16 需要損失縮放。
- 梯度累積：多個微批次的梯度累加後再更新，用時間換視訊記憶體。
- 資料並行：每張 GPU 一份完整模型、不同資料，用 all-reduce 平均梯度；環形 all-reduce 每張 GPU 只傳送 2(n−1)/n 倍的資料。
- ZeRO / FSDP：把最佳化器狀態、梯度、引數切分到各 GPU，ZeRO-3 每張只存 1/n。
- 張量並行：一層的矩陣按行列切開，通訊多，用在機器內；流水線並行：按層切開，用微批次減少氣泡。
- 啟用重計算用計算換視訊記憶體；大規模訓練組合多種並行，並經常儲存檢查點。

## 3.9 練習

<!-- exercise: zero-memory -->
### 練習 1：ZeRO 的視訊記憶體

輸入引數量 N（十億）、GPU 數量 n 和 ZeRO 階段（0～3）。按 3.5 節的表格，輸出每張 GPU 需要的視訊記憶體（GB），保留 1 位小數（不算啟用值）。

```text
樣例輸入：70 64 3
樣例輸出：17.5
```

<details>
<summary>提示</summary>

每引數位元組數：階段 0 是 16，階段 1 是 `4 + 12 / n`，階段 2 是 `2 + 14 / n`，階段 3 是 `16 / n`。N 以十億為單位時，`N * 位元組數` 正好是 GB。
</details>

<!-- exercise: pipeline-bubble -->
### 練習 2：流水線的氣泡

輸入流水線的段數 `p` 和微批次數 `m`，輸出 GPU 空閒時間所佔的比例 `(p - 1) / (m + p - 1)`，保留 3 位小數。

```text
樣例輸入：8 16
樣例輸出：0.304
```

<details>
<summary>提示</summary>

直接套公式：`print(f"{(p - 1) / (m + p - 1):.3f}")`。
</details>

<!-- exercise: ring-traffic -->
### 練習 3：all-reduce 的通訊量

輸入梯度的大小 S（GB）和 GPU 數量 n，輸出用環形 all-reduce 時每張 GPU 需要傳送的資料量（GB），即 `2(n−1)/n × S`，保留 2 位小數。

```text
樣例輸入：16 8
樣例輸出：28.00
```

<details>
<summary>提示</summary>

`print(f"{2 * (n - 1) / n * S:.2f}")`。注意 n 很大時結果接近 `2S`。
</details>

<!-- exercise: global-batch -->
### 練習 4：全域性批大小

輸入序列長度、每張 GPU 的微批次大小（條）、梯度累積步數和資料並行的 GPU 數。每次引數更新一共處理多少個 token？再換算成百萬（M），保留 2 位小數。兩個數用空格隔開。

```text
樣例輸入：8192 2 4 128
樣例輸出：8388608 8.39
```

<details>
<summary>提示</summary>

token 數 = 序列長度 × 微批次大小 × 累積步數 × GPU 數。百萬：除以 `1e6`。
</details>

<!-- exercise: why-bf16 -->
### 練習 5：為什麼用 BF16？

訓練大模型時，BF16 取代 FP16 成為首選，主要原因是什麼？

- BF16 的精度比 FP16 高
- BF16 的指數位和 FP32 一樣多，範圍相同，不容易上溢或下溢，不需要損失縮放
- BF16 只佔 8 位
- BF16 不需要 FP32 的主引數

<details>
<summary>答案與解析</summary>

BF16 有 8 位指數，範圍和 FP32 相同；FP16 只有 5 位指數，最大 65504，很小的梯度又會下溢成 0。BF16 的精度其實比 FP16 **低**（7 位尾數對 10 位），所以引數更新仍要用 FP32 主引數。
</details>

<!-- exercise: zero3 -->
### 練習 6：ZeRO-3 切分什麼

ZeRO-3（或 FSDP）在資料並行的基礎上切分了哪些內容？

- 只切分資料
- 只切分最佳化器狀態
- 最佳化器狀態、梯度和引數都切分到各張 GPU 上
- 按層把模型切分到不同 GPU 上

<details>
<summary>答案與解析</summary>

ZeRO-1 切分最佳化器狀態，ZeRO-2 再加梯度，ZeRO-3 連引數也切分，每張 GPU 只存 1/n，用到時再臨時收集。按層切分是流水線並行。
</details>

## 3.10 延伸閱讀

- Micikevicius 等：《Mixed Precision Training》（2018）
- Rajbhandari 等：《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》（2019）
- Shoeybi 等：《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》（2019）
- Hugging Face：《The Ultra-Scale Playbook: Training LLMs on GPU Clusters》（2025，圖文並茂的並行訓練指南）
- 下一課：[現代架構](../04-architectures/zh-CN.md)
