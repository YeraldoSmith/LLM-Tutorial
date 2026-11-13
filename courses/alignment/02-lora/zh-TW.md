# 2. LoRA 與 QLoRA：只訓練一小部分引數

> 🎯 **本課目標**：明白全引數微調為什麼昂貴；理解**引數高效微調**（PEFT）的思路；掌握 **LoRA** 的原理——凍結原來的權重，只訓練兩個小矩陣 B、A，權重的變化是 `(α/r)·B·A`；理解為什麼低秩就夠用，為什麼 B 要初始化為 0，訓練完怎樣**合併**權重；親手用 PyTorch 實現一個 LoRA 層；理解 **QLoRA** 怎樣把基礎模型壓到 4 位，讓一張顯示卡就能微調大模型；會用 PEFT 庫，知道 LoRA 的常見設定和侷限。
>
> ⏱ **預計用時**：70 分鐘　｜　📋 **前置知識**：本部分第 1 課；第二部分第 5 課；第四部分第 2、3 課

上一課的 SFT 預設要更新模型的**全部**引數。第四部分第 2 課算過：用 Adam 訓練，每個引數要 16 位元組視訊記憶體——70 億引數的模型要 112 GB，遠超一張消費級顯示卡。而且，如果你要為 10 個不同的任務各微調一個版本，就要儲存 10 份完整的模型。

## 2.1 引數高效微調

**引數高效微調**（Parameter-Efficient Fine-Tuning，PEFT）的思路是：**凍結**原來的模型，只訓練少量新增的引數。這樣：

- 梯度和最佳化器狀態只需要為少量引數儲存，視訊記憶體大大減少；
- 每個任務只需儲存那一小部分引數（幾 MB 到幾百 MB），而不是整個模型；
- 原來的模型完好無損，可以在不同任務之間隨時切換。

各種 PEFT 方法中，最流行的是微軟 2021 年提出的 **LoRA**（Low-Rank Adaptation，低秩適配）。

## 2.2 LoRA 的原理

微調一個線性層，本質上是把它的權重 W 改成 W + ΔW。LoRA 的關鍵假設是：**ΔW 不需要是任意的矩陣，一個「低秩」矩陣就夠了**。於是把 ΔW 寫成兩個瘦長矩陣的乘積：

```formula
W' = W + (α / r) · B · A　　　（W：d_out × d_in，凍結；B：d_out × r；A：r × d_in；r 遠小於 d）
```

- **r** 叫**秩**（rank），通常取 8～64；
- **α** 是一個縮放係數（常取 r 的 1～2 倍），讓調整 r 時不必重新調學習率；
- 前向傳播時，`W'x = Wx + (α/r)·B(Ax)`：先把 x 壓到 r 維，再放大回去，計算量很小。

可訓練的引數從 d_out × d_in 個，變成了 r × (d_in + d_out) 個：

```python
def lora_params(d_in, d_out, r):
    return r * (d_in + d_out)

for d_in, d_out, r in [(4096, 4096, 8), (4096, 4096, 16), (4096, 14336, 16), (8192, 8192, 64)]:
    full = d_in * d_out
    lora = lora_params(d_in, d_out, r)
    print(f"{d_out:>5}×{d_in:<5} 的矩陣，r={r:>2}：完整 {full:>11,} 個引數，LoRA {lora:>9,} 個（{lora / full:.2%}）")
```

用純 Python 看看前向傳播：

```python
def matvec(M, x):
    return [sum(m * v for m, v in zip(row, x)) for row in M]

W = [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]      # 凍結的原權重，2×3
A = [[0.5, -0.5, 1.0]]                       # r = 1，A 是 1×3
B = [[2.0], [-1.0]]                          # B 是 2×1
alpha, r = 2, 1
x = [1.0, 2.0, 3.0]

base = matvec(W, x)
delta = [(alpha / r) * v for v in matvec(B, matvec(A, x))]
print("原模型輸出：", base)
print("LoRA 的修正：", delta)
print("微調後輸出：", [b + d for b, d in zip(base, delta)])

merged = [[W[i][j] + (alpha / r) * B[i][0] * A[0][j] for j in range(3)] for i in range(2)]
print("合併權重後直接算：", matvec(merged, x))
```

兩點要注意：

- **B 初始化為 0**（A 用小的隨機數）。這樣訓練剛開始時 B·A = 0，模型和原來**完全一樣**，微調從原模型平穩地出發。
- **訓練完可以合併**：把 `(α/r)·B·A` 直接加進 W，得到一個普通的模型，推理時沒有任何額外開銷。也可以不合並，在執行時根據請求切換不同的 LoRA。

## 2.3 為什麼低秩就夠了？

一個 4096 × 4096 的矩陣有 1600 多萬個數，r = 8 的 LoRA 只有 6.5 萬個可訓練引數，怎麼夠用？研究發現，微調時權重的變化 ΔW 往往集中在**少數幾個方向**上（它的「內在維度」很低）——畢竟微調只是在已有能力的基礎上做一些調整。

用奇異值分解（SVD）可以看出一個矩陣「有幾個主要方向」。下面構造一個「真正的變化只有 4 個方向、再加一點噪聲」的矩陣：

```python requires=numpy
import numpy as np

rng = np.random.default_rng(0)
d = 256
delta = rng.normal(size=(d, 4)) @ rng.normal(size=(4, d)) + 0.05 * rng.normal(size=(d, d))

U, S, Vt = np.linalg.svd(delta)
print("最大的 8 個奇異值：", np.round(S[:8], 1))
for r in (1, 2, 4, 8, 32):
    approx = (U[:, :r] * S[:r]) @ Vt[:r]                  # 只保留前 r 個方向
    err = np.linalg.norm(delta - approx) / np.linalg.norm(delta)
    print(f"r = {r:>2}：相對誤差 {err:.3f}")
```

前 4 個奇異值很大，之後驟然變小；r = 4 時誤差就只剩噪聲的水平，再加大 r 幾乎沒有幫助。LoRA 賭的就是：真實的微調也是這樣。

## 2.4 親手實現 LoRA

用 PyTorch 寫一個 LoRA 層：包住一個凍結的 `nn.Linear`，只訓練 A 和 B。我們故意讓目標權重和原權重之間只差一個**秩為 1** 的變化，看看 r = 4 的 LoRA 能不能學會：

```python requires=torch timeout=120
import torch
from torch import nn

torch.manual_seed(0)
d = 64

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                                   # 凍結原權重
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))      # B 從 0 開始
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

base = nn.Linear(d, d, bias=False)
u, v = torch.randn(d, 1), torch.randn(1, d)
target_W = base.weight.detach() + 0.5 * (u @ v) / d ** 0.5           # 目標 = 原權重 + 一個秩 1 的變化
x = torch.randn(2048, d)
y = x @ target_W.T

layer = LoRALinear(base, r=4, alpha=8)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f"可訓練引數 {trainable} / 總引數 {total}")

opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
for step in range(301):
    loss = nn.functional.mse_loss(layer(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"第 {step:3d} 步：損失 {loss.item():.6f}")

with torch.no_grad():
    merged = layer.base.weight + layer.scale * layer.B @ layer.A      # 合併成一個普通的權重
    print(f"合併後的權重與目標的最大誤差：{(merged - target_W).abs().max().item():.5f}")
```

只訓練了 512 個引數（原矩陣有 4096 個），損失就降到了幾乎為 0。

## 2.5 用 PEFT 庫給大模型加 LoRA

實際中用 Hugging Face 的 **PEFT** 庫，幾行程式碼就能給模型的指定層加上 LoRA（需要下載模型）：

```python norun
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
config = LoraConfig(
    r=16,                                    # 秩
    lora_alpha=32,                           # α
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # 注意力和 MLP 的線性層
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()           # 通常只有百分之一左右可訓練
```

常見的經驗設定：

- **加在哪些層**：最初的論文只加在注意力的 q、v 上；後來發現加在**所有線性層**（包括 MLP）上效果更好。
- **r 和 α**：r = 8～64，α = r 或 2r。任務越難、和原模型差別越大，r 越大。
- **學習率**：比全引數微調大一個數量級，常用 1e-4～2e-4。
- **得到的檔案很小**：一個 LoRA 介面卡通常只有幾十 MB，可以分享、組合，推理服務（如 vLLM）還能同時載入很多個 LoRA，按請求切換。

## 2.6 QLoRA：一張顯示卡微調大模型

LoRA 減少了梯度和最佳化器狀態，但凍結的基礎模型本身仍要放進視訊記憶體：BF16 下每個引數 2 位元組，700 億引數就是 140 GB。**QLoRA**（Dettmers 等，2023）進一步把凍結的基礎模型**量化成 4 位**：

- **NF4**（4 位 NormalFloat）：一種專門為「近似正態分佈的權重」設計的 4 位格式，比普通的 4 位整數更精確；
- **雙重量化**：連量化用的縮放係數也再量化一次，進一步省視訊記憶體；
- **分頁最佳化器**：視訊記憶體不夠時把最佳化器狀態暫時挪到記憶體裡，防止視訊記憶體峰值導致崩潰；
- LoRA 的 A、B 仍用 BF16 訓練，計算時把 4 位權重臨時還原成 BF16。

```python
def finetune_memory_gb(N, method, lora_fraction=0.01):
    if method == "全引數":
        return 16 * N                         # BF16 引數 + 梯度 + FP32 主引數和 Adam（第四部分第 2 課）
    base_bytes = {"LoRA": 2, "QLoRA": 0.5}[method]   # 凍結的基礎模型：BF16 或 4 位
    return base_bytes * N + 16 * N * lora_fraction   # 再加上 LoRA 引數自己的訓練狀態

for N in (7, 70):                              # 單位：十億引數，結果單位：GB（不含啟用值）
    row = [f"{m} {finetune_memory_gb(N, m):6.1f} GB" for m in ("全引數", "LoRA", "QLoRA")]
    print(f"{N:>2}B 模型：" + "，".join(row))
```

QLoRA 的論文在**一張 48 GB 的 GPU** 上微調了 650 億引數的模型，效果和 16 位的全引數微調相當。這讓個人和小團隊也能定製大模型。代價是速度：每次計算都要把 4 位權重還原，訓練比 LoRA 慢一些。

## 2.7 LoRA 的侷限和變體

- **學得少，也忘得少**：Biderman 等（2024）發現，在需要學習大量新知識的任務上（比如在大量程式碼或數學資料上繼續訓練），LoRA 明顯不如全引數微調；但 LoRA 也更少「忘記」原來的能力。所以 LoRA 很適合調整格式、風格和特定任務，不太適合大規模地注入新知識。
- **變體**：DoRA（2024）把權重分解成「大小」和「方向」分別調整；還有各種改進初始化和學習率的方法。原理都和 LoRA 一樣：凍結大的、訓練小的。

## 2.8 小結

- 全引數微調視訊記憶體大、每個任務都要存一份完整模型；PEFT 凍結原模型，只訓練少量新引數。
- LoRA：W' = W + (α/r)·B·A，可訓練引數從 d_out·d_in 降到 r·(d_in + d_out)；B 初始化為 0，訓練開始時模型不變。
- 微調的權重變化通常是低秩的，所以小的 r 就夠用；訓練完可以把 BA 合併進 W，推理沒有額外開銷。
- 常見設定：加在所有線性層，r = 8～64，α = r～2r，學習率 1e-4 左右；介面卡檔案很小，可以隨時切換。
- QLoRA：基礎模型量化為 4 位 NF4，加上雙重量化和分頁最佳化器，一張顯示卡就能微調幾百億引數的模型。
- LoRA 適合調整格式、風格和任務，大量注入新知識時不如全引數微調。

## 2.9 練習

<!-- exercise: lora-params -->
### 練習 1：LoRA 有多少引數

輸入一個線性層的 `d_in`、`d_out` 和 LoRA 的秩 `r`。輸出完整矩陣的引數量、LoRA 的可訓練引數量，以及後者佔前者的百分比（保留 2 位小數），用空格隔開。

```text
樣例輸入：4096 4096 8
樣例輸出：16777216 65536 0.39%
```

<details>
<summary>提示</summary>

完整：`d_in * d_out`；LoRA：`r * (d_in + d_out)`；百分比：`f"{lora / full:.2%}"`。
</details>

<!-- exercise: lora-forward -->
### 練習 2：LoRA 的前向傳播

第一行是 `d_out d_in r`；接下來 `d_out` 行是凍結的權重 W；再接下來 `r` 行是 A（每行 `d_in` 個數）；再接下來 `d_out` 行是 B（每行 `r` 個數）；然後一行是 α；最後一行是輸入向量 x。輸出 `W x + (α/r)·B(A x)`，每個數保留 3 位小數，用空格隔開。

```text
樣例輸入：
2 3 1
1 0 2
0 1 -1
0.5 -0.5 1
2
-1
2
1 2 3
樣例輸出：17.000 -6.000
```

<details>
<summary>提示</summary>

寫一個矩陣乘向量的函式 `matvec(M, x)`。先算 `Ax`（r 個數），再算 `B(Ax)`（d_out 個數），乘以 `alpha / r` 後加到 `Wx` 上。
</details>

<!-- exercise: model-lora -->
### 練習 3：整個模型的 LoRA 引數

一個模型有 `L` 層，模型維度為 `d`。假設每層有 `m` 個需要加 LoRA 的 `d × d` 線性層，秩為 `r`。輸出 LoRA 的總引數量，以百萬為單位，保留 2 位小數，後面加 `M`。

```text
樣例輸入：32 4096 4 16
樣例輸出：16.78M
```

<details>
<summary>提示</summary>

每個 d × d 的層有 `r * (d + d)` 個 LoRA 引數，一共有 `L * m` 個這樣的層。
</details>

<!-- exercise: finetune-memory -->
### 練習 4：三種微調要多少視訊記憶體

輸入模型的引數量 N（十億）。按 2.6 節的估算（全引數 16 位元組 / 引數；LoRA 的基礎模型 2 位元組 / 引數、QLoRA 的基礎模型 0.5 位元組 / 引數，兩者都再加上 1% 引數 × 16 位元組的 LoRA 訓練狀態），輸出三種方法需要的視訊記憶體（GB），各保留 1 位小數，用空格隔開。

```text
樣例輸入：7
樣例輸出：112.0 15.1 4.6
```

<details>
<summary>提示</summary>

LoRA：`2 * N + 16 * N * 0.01`；QLoRA：`0.5 * N + 16 * N * 0.01`。N 以十億為單位時結果就是 GB。
</details>

<!-- exercise: b-zero -->
### 練習 5：B 為什麼從 0 開始？

LoRA 中矩陣 B 初始化為 0、A 初始化為小的隨機數。這樣做的主要目的是什麼？

- 讓訓練速度加倍
- 讓訓練開始時 B·A = 0，模型的輸出和原模型完全一樣，從原模型平穩地出發
- 讓 LoRA 的引數量減半
- 因為 B 不需要訓練

<details>
<summary>答案與解析</summary>

B·A = 0 意味著一開始的權重就是 W 本身，微調不會一上來就破壞原模型。A 不能也是 0：那樣 B 和 A 的梯度都是 0，訓練無法開始。
</details>

<!-- exercise: qlora-what -->
### 練習 6：QLoRA 量化了什麼？

QLoRA 把什麼量化成了 4 位？

- LoRA 的 A、B 矩陣
- 凍結的基礎模型的權重；LoRA 的 A、B 仍用 16 位訓練
- 訓練資料
- 梯度

<details>
<summary>答案與解析</summary>

QLoRA 把佔視訊記憶體最多、又不需要更新的基礎模型壓成 4 位（NF4），可訓練的 LoRA 引數仍是 BF16，計算時臨時把 4 位權重還原。
</details>

## 2.10 延伸閱讀

- Hu 等：《LoRA: Low-Rank Adaptation of Large Language Models》（2021）
- Dettmers 等：《QLoRA: Efficient Finetuning of Quantized LLMs》（2023）
- Biderman 等：《LoRA Learns Less and Forgets Less》（2024）
- Liu 等：《DoRA: Weight-Decomposed Low-Rank Adaptation》（2024）
- Hugging Face [PEFT 文件](https://huggingface.co/docs/peft)
- 下一課：[獎勵模型與 RLHF](../03-rlhf/zh-CN.md)
