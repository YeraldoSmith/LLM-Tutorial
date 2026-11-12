# 6. 訓練技巧：讓網路學得更好

> 🎯 **本課目標**：學會識別**過擬合**與**欠擬合**；理解學習率、**預熱 + 餘弦衰減**排程；看懂 SGD、Momentum、Adam、**AdamW** 的區別；知道為什麼初始化、**LayerNorm/RMSNorm**、**殘差連線**能讓深層網路訓練起來；掌握權重衰減、Dropout、梯度裁剪和混合精度（bf16）；拿到一份訓練排錯清單。這些就是訓練大模型時每天都在用的「工程手藝」。
>
> ⏱ **預計用時**：90 分鐘　｜　📋 **前置知識**：本部分第 1～5 課

上一課的例子都很「乖」：網路小、資料乾淨，隨便設個學習率都能學會。真實的訓練往往會遇到：損失不降、損失變成 `nan`、訓練集上很好但新資料上很差、網路一深就學不動……本課介紹的每一個技巧都對應一個具體問題。本課大部分程式碼是純 Python，不需要安裝 PyTorch 也能執行。

## 6.1 過擬合與欠擬合

我們訓練模型，真正關心的是它在**沒見過的新資料**上表現如何，這叫**泛化**（generalization）。

- **欠擬合**（underfitting）：模型太簡單或訓練不夠，連訓練集都學不好。
- **過擬合**（overfitting）：模型把訓練集「背」了下來，連其中的噪聲和偶然也記住了，換成新資料就不行。

下面這個例子裡，真正的規律是「x > 5 就是類別 1」，但 20% 的標籤是錯的（現實資料總有噪聲）。分類方法是「看離 x 最近的 k 個訓練樣本，少數服從多數」：

```python
import random
random.seed(1)

def make(n):
    data = []
    for _ in range(n):
        x = random.uniform(0, 10)
        y = 1 if x > 5 else 0          # 真正的規律：x > 5 就是類別 1
        if random.random() < 0.2:      # 但有 20% 的標籤是錯的（噪聲）
            y = 1 - y
        data.append((x, y))
    return data

train, test = make(100), make(1000)

def predict(x, k):                     # 看離 x 最近的 k 個訓練樣本，少數服從多數
    near = sorted(train, key=lambda p: abs(p[0] - x))[:k]
    return 1 if sum(y for _, y in near) * 2 > k else 0

def accuracy(data, k):
    return sum(predict(x, k) == y for x, y in data) / len(data)

for k in (1, 5, 25):
    print(f"k={k:2d}  訓練準確率 {accuracy(train, k):.0%}   測試準確率 {accuracy(test, k):.0%}")
```

`k=1` 在訓練集上 100% 正確（每個點最近的就是它自己），但測試準確率最低——它把錯誤標籤也背下來了。**訓練集上的成績會騙人**，所以資料要分成三份：

| 資料集 | 用途 |
|--------|------|
| **訓練集**（train） | 用來計算梯度、更新引數 |
| **驗證集**（validation） | 訓練過程中檢查泛化效果，用來挑選超引數（學習率、模型大小……）和決定何時停止 |
| **測試集**（test） | 最後只用一次，報告最終成績 |

訓練時同時畫出兩條損失曲線，就能診斷問題：

```text
損失                                      損失
 │╲                                         │╲
 │ ╲___ 驗證                                │ ╲      ___── 驗證（開始上升 = 過擬合）
 │  ╲  ‾‾‾‾‾‾‾‾‾‾                           │  ╲___/
 │   ╲___ 訓練                              │   ╲___
 │       ‾‾‾‾‾‾‾‾‾  兩條都高 = 欠擬合        │       ‾‾‾‾‾‾─── 訓練（一直下降）
 └──────────────── 步數                     └──────────────── 步數
```

> [!NOTE]
> 大語言模型預訓練時資料極多（幾萬億 token），每條資料通常只看一到幾遍，所以**很少過擬合**，主要問題是欠擬合——這也是為什麼模型和資料越大越好（第四部分會講「縮放定律」）。但在**微調**時，資料少，過擬合非常常見。

## 6.2 學習率：最重要的超引數

用梯度下降最小化 `f(x) = x²`（最低點在 0），從 `x = 5` 出發，看看不同學習率會怎樣：

```python
def descend(lr, steps=10):
    x = 5.0                       # 最小化 f(x) = x²，最低點在 x = 0
    path = []
    for _ in range(steps):
        x -= lr * 2 * x           # f'(x) = 2x
        path.append(x)
    return path

for lr in (0.01, 0.1, 0.9, 1.1):
    xs = descend(lr)
    print(f"lr={lr:<5} 10 步後 x = {xs[-1]:10.4f}   前 5 步：", " ".join(f"{v:.1f}" for v in xs[:5]))
```

| 學習率 | 現象 |
|--------|------|
| 太小（0.01） | 穩定但非常慢，10 步才走了一點點 |
| 合適（0.1） | 穩步接近 0 |
| 偏大（0.9） | 在兩邊來回跳（震盪），但還在收斂 |
| 太大（1.1） | 越跳越遠，**發散**——真實訓練中表現為損失變成 `inf` / `nan` |

> [!TIP]
> 訓練出問題時，**第一個要懷疑的就是學習率**。常見的起點：Adam/AdamW 用 `1e-3`～`3e-4`；大模型預訓練用 `1e-4`～`6e-4`，模型越大學習率通常越小；微呼叫 `1e-5`～`2e-4`。

## 6.3 學習率排程：預熱 + 餘弦衰減

學習率不必一成不變。幾乎所有大模型訓練都用這樣的**排程**（schedule）：

1. **預熱**（warmup）：最開始的幾百到幾千步，學習率從接近 0 線性升到最大值。剛開始引數是隨機的，梯度很「亂」，一上來就邁大步容易把訓練搞崩。
2. **餘弦衰減**（cosine decay）：之後按餘弦曲線慢慢降到一個很小的值。訓練後期步子小一點，才能更精細地停在低谷裡。

```python
import math

def lr_at(step, max_lr=3e-4, warmup=100, total=1000, min_lr=3e-5):
    if step < warmup:                                   # 線性預熱
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / (total - warmup)       # 從 0 走到 1
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

for step in [0, 25, 50, 75] + list(range(100, 1001, 100)):
    lr = lr_at(step)
    print(f"step {step:4d}  lr {lr:.6f}  " + "█" * round(lr / 3e-4 * 40))
```

> [!NOTE]
> 一些新的訓練方案用 **WSD**（Warmup-Stable-Decay）排程：預熱後長時間保持最大學習率，最後一小段再快速衰減。好處是訓練可以隨時延長，不用事先定好總步數。

## 6.4 最佳化器：SGD、Momentum、Adam 與 AdamW

普通梯度下降（SGD）在「狹長山谷」形狀的損失上會很吃力：陡的方向來回震盪，平的方向又走得很慢。下面的損失函式 `x² + 100y²` 就是這樣一個山谷：

```python
import math

def loss_grad(x, y):                  # 狹長山谷：y 方向很陡，x 方向很平
    return x * x + 100 * y * y, (2 * x, 200 * y)

def run(update, steps=1000):
    p, state = [-4.0, 1.0], {}
    for t in range(1, steps + 1):
        loss, g = loss_grad(*p)
        if loss < 1e-4:
            return t
        update(p, g, state, t)
    return None

def sgd(p, g, s, t, lr=0.009):        # 學習率再大一點（0.01 以上）y 方向就會發散
    for i in range(2):
        p[i] -= lr * g[i]

def momentum(p, g, s, t, lr=0.005, beta=0.9):
    v = s.setdefault("v", [0.0, 0.0])
    for i in range(2):
        v[i] = beta * v[i] + g[i]                # 「速度」：過去梯度的累積，來回震盪的部分互相抵消
        p[i] -= lr * v[i]

def adam(p, g, s, t, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    m = s.setdefault("m", [0.0, 0.0])
    v = s.setdefault("v", [0.0, 0.0])
    for i in range(2):
        m[i] = b1 * m[i] + (1 - b1) * g[i]       # 一階矩：梯度的移動平均（方向）
        v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2  # 二階矩：梯度平方的移動平均（大小）
        m_hat = m[i] / (1 - b1 ** t)             # 偏差修正（開始時 m、v 偏向 0）
        v_hat = v[i] / (1 - b2 ** t)
        p[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)   # 每個引數有自己的步長

for name, fn in [("SGD", sgd), ("Momentum", momentum), ("Adam", adam)]:
    t = run(fn)
    print(f"{name:9s}", f"{t} 步到達最低點附近" if t else "1000 步內沒有到達")
```

| 最佳化器 | 思路 | 說明 |
|--------|------|------|
| **SGD** | 沿負梯度走一步 | 最簡單；對學習率很敏感 |
| **Momentum** | 加一個「慣性」：保留之前的速度 | 震盪相互抵消，平坦方向越走越快 |
| **Adam** | 動量 + 按每個引數梯度的大小**自動調整步長** | 梯度大的引數步子小，梯度小的步子大；對超引數不敏感 |
| **AdamW** | Adam + **解耦的權重衰減** | 大模型訓練的預設選擇；常用 `betas=(0.9, 0.95)`、`weight_decay=0.1` |

> [!NOTE]
> 最佳化器也在不斷進步。**Muon** 等新最佳化器已經在一些大模型（如 Kimi K2）的訓練中使用，號稱比 AdamW 更省算力。不過 AdamW 仍然是最常見、最穩妥的選擇。注意 Adam 要為每個引數額外儲存 `m` 和 `v` 兩個值，所以**最佳化器狀態要佔兩倍於引數的記憶體**——訓練比推理吃記憶體多得多的原因之一。

## 6.5 初始化：起點很重要

引數的初始值不能隨便設：

- **全部設成 0？** 不行。同一層的所有神經元會算出一樣的輸出、得到一樣的梯度，永遠一模一樣——相當於只有一個神經元。必須用**隨機數**打破對稱。
- **隨機數的大小也很關鍵**。一層層傳下去，太大的值會越乘越大（爆炸），太小的值會越乘越小（消失）。

```python
import math
import random
random.seed(0)

n, depth = 64, 10

def std(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / len(v))

def forward(scale):
    x = [random.gauss(0, 1) for _ in range(n)]
    stds = []
    for _ in range(depth):
        W = [[random.gauss(0, scale) for _ in range(n)] for _ in range(n)]
        x = [max(0.0, sum(w * a for w, a in zip(row, x))) for row in W]   # 線性層 + ReLU
        stds.append(std(x))
    return stds

print("初始化方式        第1層   第4層   第7層   第10層（每層輸出的標準差）")
for name, scale in [("std=1", 1.0), ("std=0.01", 0.01), ("Kaiming √(2/n)", math.sqrt(2 / n))]:
    s = forward(scale)
    print(f"{name:15s}", "  ".join(f"{v:7.2g}" for v in (s[0], s[3], s[6], s[9])))
```

**Kaiming（He）初始化**讓每層權重的標準差為 `√(2/n)`（`n` 是輸入維度），訊號一層層傳下去大小基本不變。PyTorch 的 `nn.Linear` 預設就用了類似的縮放，所以平時不用自己操心；GPT 類模型常用均值 0、標準差 0.02 的正態分佈初始化，並對殘差分支的輸出層再按層數縮小。

## 6.6 歸一化：LayerNorm 與 RMSNorm

即使初始化得好，訓練過程中各層數值的大小還是會漂移。**歸一化**層把每個向量重新拉回「標準大小」：

- **LayerNorm**：減去均值，再除以標準差 → 均值 0、方差 1。原始 Transformer、GPT-2 使用。
- **RMSNorm**：不減均值，只除以均方根 `√(mean(x²))`。計算更省，效果相當，**Llama、Qwen、DeepSeek 等現代大模型幾乎都用它**。

```python
import math

def layer_norm(x, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((a - mean) ** 2 for a in x) / len(x)
    return [(a - mean) / math.sqrt(var + eps) for a in x]

def rms_norm(x, eps=1e-5):
    rms = math.sqrt(sum(a * a for a in x) / len(x) + eps)
    return [a / rms for a in x]

x = [2.0, 4.0, 6.0, 8.0]
print("LayerNorm:", [round(v, 3) for v in layer_norm(x)])
print("RMSNorm:  ", [round(v, 3) for v in rms_norm(x)])
big = [v * 1000 for v in x]
print("放大 1000 倍後 RMSNorm：", [round(v, 3) for v in rms_norm(big)])   # 結果不變
```

實際的歸一化層後面還會乘一個**可學習**的縮放向量 `γ`（LayerNorm 還加一個偏移 `β`），讓網路自己決定需要多大的數值。`eps` 是一個很小的數，防止除以 0。

> [!NOTE]
> 你可能還聽說過 **BatchNorm**：它在一個 batch 的樣本之間做歸一化，在影象網路（CNN）裡常用。語言模型裡句子長短不一、推理時常常一次只處理一句，所以用的是對**每個向量自己**做歸一化的 LayerNorm / RMSNorm。

## 6.7 殘差連線：給梯度修一條高速公路

網路越深，反向傳播時梯度要連乘越多的區域性導數。如果每層導數都小於 1，乘幾十次就幾乎為 0——**梯度消失**，前面的層根本學不動。

**殘差連線**（residual connection，ResNet 2015 年提出）的辦法極其簡單：每層的輸出 = **輸入 + 這一層算出的變化**，即 `y = x + f(x)`。它的導數是 `1 + f'(x)`，那個「1」保證梯度至少能原樣穿過每一層。

```python
import math

def grad_through(depth, residual, w=0.5):
    x, grad = 1.0, 1.0
    for _ in range(depth):
        f = math.tanh(w * x)
        local = w * (1 - f * f)                   # f(x) = tanh(w·x) 的導數
        if residual:
            x, grad = x + f, grad * (1 + local)   # y = x + f(x)，導數 = 1 + f'(x)
        else:
            x, grad = f, grad * local
    return grad

for depth in (5, 20, 50):
    print(f"{depth:2d} 層  普通：{grad_through(depth, False):.2e}   殘差：{grad_through(depth, True):.2f}")
```

Transformer 的每一層都是兩個殘差塊，結構正是：

```formula
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

這種「先歸一化再計算，再加回去」的寫法叫 **Pre-Norm**，是現在大模型的標準做法，能穩定地訓練上百層的網路。第三部分會詳細講。

## 6.8 正則化：防止過擬合

| 方法 | 做法 | 直覺 |
|------|------|------|
| **權重衰減**（weight decay） | 每步把引數往 0 拉一點：`w -= lr * λ * w` | 不讓權重變得太大，模型更「平滑」 |
| **Dropout** | 訓練時隨機把一部分神經元輸出置 0 | 不能依賴任何單個神經元，迫使網路學到更穩健的特徵 |
| **早停**（early stopping） | 驗證損失不再下降就停止訓練，保留最好的那次 | 在開始「背題」之前停下 |
| **更多資料 / 資料增強** | 收集更多資料，或對已有資料做變換 | 最根本的辦法 |

Dropout 的實現很簡單。注意訓練時沒被置 0 的值要**除以 `1 − p`**，讓輸出的平均值保持不變；評估時關閉 Dropout（這就是 `model.eval()` 的作用之一）：

```python
import random
random.seed(0)

def dropout(x, p=0.5, training=True):
    if not training:
        return x
    return [0.0 if random.random() < p else a / (1 - p) for a in x]

x = [1.0] * 8
print("訓練時：", dropout(x))
print("評估時：", dropout(x, training=False))
avg = sum(sum(dropout(x)) for _ in range(10000)) / 10000 / len(x)
print("訓練時的平均值：", round(avg, 3))     # 約等於 1，和評估時一致
```

> [!NOTE]
> 大模型**預訓練**通常不用 Dropout（資料夠多，不怎麼過擬合），但會用 AdamW 的權重衰減。微調小資料集時，Dropout 和早停就很有用了。

## 6.9 梯度裁剪

偶爾遇到一批特別「怪」的資料，梯度會突然變得極大，一步就把引數推到很遠的地方，損失跳成 `nan`（叫「損失尖峰」）。**梯度裁剪**：如果所有梯度合在一起的長度（**全域性範數**）超過閾值，就按比例縮小，方向不變。

```python
import math

def clip_by_norm(grads, max_norm):
    norm = math.sqrt(sum(g * g for g in grads))      # 全域性範數：所有梯度合成一個向量的長度
    if norm <= max_norm:
        return grads
    return [g * max_norm / norm for g in grads]       # 方向不變，長度縮到 max_norm

print(clip_by_norm([0.3, -0.4], 1.0))     # 長度 0.5，不需要裁剪
print(clip_by_norm([30.0, -40.0], 1.0))   # 長度 50，縮到 1
```

大模型訓練幾乎都設 `max_norm=1.0`。PyTorch 裡一行：`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`，放在 `backward()` 之後、`step()` 之前。

## 6.10 混合精度：fp32、fp16、bf16

預設的 `float32`（fp32）每個數佔 4 位元組。用 16 位（2 位元組）的格式，記憶體減半，GPU 上的計算也快得多：

| 格式 | 位元組 | 範圍 | 精度 | 說明 |
|------|------|------|------|------|
| fp32 | 4 | 約 ±3.4×10³⁸ | 約 7 位有效數字 | 預設 |
| fp16 | 2 | 約 ±65504 | 約 3～4 位 | 範圍小，容易溢位，需要「損失縮放」 |
| **bf16** | 2 | 和 fp32 一樣 | 約 2～3 位 | 範圍大、不易溢位，**大模型訓練的主流** |
| fp8 / fp4 | 1 / 0.5 | 更小 | 更低 | 最新的 GPU 支援，DeepSeek-V3 等用 fp8 訓練 |

用 Python 自帶的 `struct` 模組就能親眼看到精度損失：

```python
import struct

def to_fp16(x):
    return struct.unpack("e", struct.pack("e", x))[0]       # "e" 表示 16 位半精度

def to_bf16(x):
    bits = struct.unpack("I", struct.pack("f", x))[0]       # 先拿到 32 位浮點數的二進位制
    return struct.unpack("f", struct.pack("I", bits & 0xFFFF0000))[0]   # bf16 = 只保留高 16 位

for x in (0.1, 3.14159, 1000.7):
    print(f"{x:<8}  fp16 → {to_fp16(x):<12.7g} bf16 → {to_bf16(x):.7g}")

try:
    to_fp16(70000.0)
except OverflowError:
    print("70000 超出了 fp16 的範圍（最大約 65504）")
print("bf16 能表示 70000 嗎？", to_bf16(70000.0))           # 能，只是不太精確
```

**混合精度訓練**：引數的「主副本」和最佳化器狀態保留 fp32，前向和反向的大部分計算用 bf16。在 PyTorch 裡（有 GPU 時）：

```python norun
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(xb), yb)
loss.backward()
```

> [!TIP]
> 估算記憶體的經驗法則：**推理**時 bf16 模型每個引數 2 位元組，70 億（7B）引數 ≈ 14 GB；**訓練**（AdamW + 混合精度）每個引數約 16 位元組，7B 模型 ≈ 112 GB，還不算中間啟用值。這就是為什麼微調大模型常用 LoRA 等省記憶體的方法（第五部分會講）。

## 6.11 放在一起：一個現代的訓練指令碼

下面用 PyTorch 把本課的技巧組合起來：殘差塊 + RMSNorm + GELU + Dropout + AdamW（權重衰減）+ 預熱餘弦排程 + 梯度裁剪。這幾乎就是一個迷你 Transformer 的訓練骨架（只差注意力層）：

```python requires=torch
import math
import torch
from torch import nn

torch.manual_seed(0)

class Block(nn.Module):                      # 現代的殘差塊：Norm → MLP → Dropout，再加回輸入
    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return x + self.drop(self.mlp(self.norm(x)))      # 殘差連線

model = nn.Sequential(nn.Linear(8, 32), *[Block(32) for _ in range(4)], nn.Linear(32, 1))
X = torch.randn(512, 8)
y = (X[:, :1] * X[:, 1:2]).sin()                         # 隨便造一個非線性的目標

total_steps, warmup = 300, 30
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
def schedule(step):                                       # 返回「學習率倍數」
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

for step in range(total_steps):
    idx = torch.randint(0, len(X), (64,))                 # 隨機取一個小批次
    loss = nn.functional.mse_loss(model(X[idx]), y[idx])
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # 梯度裁剪
    optimizer.step()
    scheduler.step()                                      # 更新學習率
    if step % 100 == 0 or step == total_steps - 1:
        print(f"step {step:3d}  loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.5f}  grad_norm {norm:.2f}")

model.eval()                                              # 評估模式：關閉 Dropout
with torch.no_grad():
    print("最終 MSE：", round(nn.functional.mse_loss(model(X), y).item(), 4))
```

## 6.12 訓練排錯清單

參考 Andrej Karpathy 的經典文章《A Recipe for Training Neural Networks》：

1. **先看資料**。列印幾條樣本和標籤，確認沒有讀錯、沒有錯位。
2. **從簡單開始**。先用最小的模型、最小的資料跑通整個流程。
3. **先在一個小批次上過擬合**。用同一個 batch 反覆訓練，損失應該能降到接近 0；降不下去說明程式碼有 bug。
4. **檢查初始損失**。比如 10 分類時，隨機猜的交叉熵應該約為 `ln 10 ≈ 2.30`；差太多說明初始化或損失函式有問題。
5. **損失變 `nan`**：降低學習率、加梯度裁剪、檢查資料裡有沒有 `inf` / `nan`、檢查有沒有 `log(0)`。
6. **損失完全不動**：檢查是否忘了 `zero_grad()` / `step()`、學習率是否太小、梯度是否被 `no_grad()` / `.detach()` 切斷。
7. **一次只改一個地方**，並把每次實驗的設定和結果記下來。

## 6.13 小結

- 關心的是**泛化**：用驗證集監控，訓練損失降、驗證損失升就是**過擬合**，兩者都高是**欠擬合**。
- **學習率**最重要：太大發散、太小太慢；大模型用**預熱 + 餘弦衰減**。
- **AdamW** = 動量 + 自適應步長 + 解耦權重衰減，是預設的最佳化器；它的狀態佔引數兩倍的記憶體。
- 好的**初始化**（隨機、尺度合適）、**歸一化**（RMSNorm/LayerNorm）和**殘差連線**（`x + f(x)`）讓深層網路能訓練起來。
- 正則化：權重衰減、Dropout、早停、更多資料；**梯度裁剪**（全域性範數 1.0）防止損失尖峰。
- **bf16** 混合精度：記憶體減半、速度更快，範圍和 fp32 一樣。

## 6.14 練習

<!-- exercise: lr-schedule -->
### 練習 1：預熱 + 餘弦排程

輸入一行三個整數 `step warmup total`，輸出這一步的學習率**倍數**（最大學習率記為 1，最小為 0），保留 4 位小數：

- `step < warmup` 時：`(step + 1) / warmup`
- 否則：`0.5 × (1 + cos(π × (step − warmup) / (total − warmup)))`

```text
樣例輸入：0 10 100
樣例輸出：0.1000
```

<details>
<summary>提示</summary>

用 `math.cos` 和 `math.pi`。注意 `step` 從 0 開始計數。
</details>

<!-- exercise: rmsnorm -->
### 練習 2：實現 RMSNorm

輸入一行若干個數，輸出 RMSNorm 的結果：每個數除以 `√(平方的平均值 + 1e-5)`，保留 3 位小數，空格隔開。

```text
樣例輸入：2 4 6 8
樣例輸出：0.365 0.730 1.095 1.461
```

<details>
<summary>提示</summary>

`rms = math.sqrt(sum(a * a for a in x) / len(x) + 1e-5)`，然後 `print(" ".join(f"{a / rms:.3f}" for a in x))`。
</details>

<!-- exercise: clip-norm -->
### 練習 3：梯度裁剪

第一行是若干個梯度值，第二行是 `max_norm`。如果梯度的全域性範數（所有值平方和的平方根）大於 `max_norm`，就把每個值乘以 `max_norm / 範數`；否則保持不變。輸出結果，保留 4 位小數，空格隔開。

```text
樣例輸入：
30 -40
1
樣例輸出：0.6000 -0.8000
```

<details>
<summary>提示</summary>

先算 `norm = math.sqrt(sum(g * g for g in grads))`，再判斷要不要縮放。
</details>

<!-- exercise: diagnose -->
### 練習 4：診斷訓練曲線

訓練損失一直在下降，但驗證損失從第 3 個 epoch 開始持續上升。最可能的情況是？

- 欠擬合，應該把模型變大
- 過擬合，可以早停、加正則化或增加資料
- 學習率太小
- 一切正常，繼續訓練就好

<details>
<summary>答案與解析</summary>

訓練集越來越好、驗證集越來越差，說明模型在「背」訓練資料——典型的過擬合。可以在驗證損失最低時停止（早停），或使用權重衰減、Dropout、更多資料。
</details>

<!-- exercise: zero-init -->
### 練習 5：全部初始化為 0

如果把一個 MLP 的所有權重都初始化為 0，會怎樣？

- 訓練會更快、更穩定
- 同一層的所有神經元始終完全相同，網路相當於每層只有一個神經元
- 沒有任何影響
- 程式會報錯

<details>
<summary>答案與解析</summary>

所有神經元的輸入、權重都一樣，算出的輸出和梯度也一樣，更新後依然一樣——這種「對稱性」永遠無法打破。所以必須用隨機數初始化。
</details>

<!-- exercise: bf16-memory -->
### 練習 6：估算記憶體

一個 70 億引數（7B）的模型，用 bf16 格式載入做推理，僅模型引數大約需要多少記憶體？

- 7 GB
- 14 GB
- 28 GB
- 56 GB

<details>
<summary>答案與解析</summary>

bf16 每個引數 2 位元組：7×10⁹ × 2 位元組 = 14×10⁹ 位元組 ≈ 14 GB。fp32 則需要 28 GB；訓練時加上梯度和 AdamW 狀態，需要的更多。
</details>

## 6.15 延伸閱讀

- Andrej Karpathy：[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)（經典的訓練經驗總結）
- 《動手學深度學習》：[最佳化演算法](https://zh.d2l.ai/chapter_optimization/index.html) 一章
- Google Research：[Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)（調參手冊）
- 下一部分：[第三部分 · 語言模型](../../lm/01-language-models/zh-CN.md)——終於要開始做語言模型了！
