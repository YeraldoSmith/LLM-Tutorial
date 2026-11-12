# 5. PyTorch 入門：張量、自動求導與訓練迴圈

> 🎯 **本課目標**：認識 PyTorch 的核心——**張量**；掌握張量的建立、運算、廣播和變形；看到 PyTorch 的自動求導和上一課的 micrograd 完全一致；用 `nn.Module` 搭模型；寫出標準的五步訓練迴圈；學會小批次訓練、使用 GPU、儲存模型，並認識最常見的報錯。
>
> ⏱ **預計用時**：90 分鐘　｜　📋 **前置知識**：本部分第 1～4 課；需要安裝 PyTorch（見第一部分第 2 課）

## 5.1 為什麼用 PyTorch？

上一課我們用純 Python 寫出了 micrograd：它原理完整，但一次只能處理**一個數字**，訓練一個 17 個引數的網路都要好幾秒。真正的大模型有幾十億個引數，需要：

1. 一次處理**整塊資料**（向量、矩陣乃至更高維的陣列）；
2. 在 **GPU** 上平行計算，速度快成千上萬倍；
3. 現成的網路層、損失函式、最佳化器。

**PyTorch** 就是這樣一個框架，由 Meta 開源，現在由 Linux 基金會下的 PyTorch 基金會維護。它是目前研究和工業界最主流的深度學習框架：Hugging Face Transformers、vLLM、Llama、DeepSeek、Qwen 等的訓練和推理程式碼，絕大多數都基於 PyTorch。（另一個常見的選擇是 Google 的 JAX。）

> [!TIP]
> **安裝**：在你的專案目錄執行 `uv add torch`（或 `pip install torch`），詳見第一部分第 2 課。中國大陸使用者可以加上清華映象：`pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple`。
>
> 如果學習程式提示找不到 `torch`，說明它用的 Python 不是你裝了 torch 的那個。開啟**設定**，把 **Python 路徑**改成專案虛擬環境裡的 Python：macOS/Linux 是 `專案目錄/.venv/bin/python`，Windows 是 `專案目錄\.venv\Scripts\python.exe`。

## 5.2 張量（Tensor）

**張量**就是「多維陣列」：0 維是一個數（標量），1 維是向量，2 維是矩陣，3 維以上統稱張量。它和 NumPy 的陣列幾乎一樣，但多了兩項本領：**能在 GPU 上執行**、**能自動求導**。

```python requires=torch
import torch

print(torch.__version__)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(x)
print("shape:", x.shape, " dtype:", x.dtype, " device:", x.device)
print(torch.zeros(2, 3))                 # 全 0
print(torch.ones(3))                     # 全 1
print(torch.arange(6).reshape(2, 3))     # 0..5，再排成 2 行 3 列
torch.manual_seed(0)                     # 固定隨機種子，讓結果可以復現
print(torch.randn(2, 2))                 # 標準正態分佈的隨機數
```

每個張量都有三個最重要的屬性，**出錯時先看它們**：

| 屬性 | 含義 | 例子 |
|------|------|------|
| `shape` | 形狀：每一維有多大 | `torch.Size([2, 3])`：2 行 3 列 |
| `dtype` | 資料型別 | `torch.float32`（預設小數）、`torch.int64`（預設整數）、`torch.bfloat16`（大模型常用） |
| `device` | 存放在哪裡 | `cpu`、`cuda`（NVIDIA 顯示卡）、`mps`（蘋果晶片） |

## 5.3 張量運算

```python requires=torch
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
print(a + b)          # 逐元素相加
print(a * b)          # 逐元素相乘（不是矩陣乘法！）
print(a @ b)          # 矩陣乘法
print(a.sum(), a.mean(), a.max())
print(a.sum(dim=0))   # 沿第 0 維求和（把各行加起來）→ 每列的和
print(a.sum(dim=1))   # 沿第 1 維求和 → 每行的和
print(a.T)            # 轉置
print(a[0], a[:, 1], a[1, 0].item())    # 取第 0 行、第 1 列、單個元素
```

> [!WARNING]
> `*` 是**逐元素**相乘，`@` 才是**矩陣乘法**。這是新手最常混淆的地方。另外 `.item()` 把只有一個元素的張量變成普通的 Python 數字，列印損失時經常用到。

`dim` 引數（維度）很重要：對形狀 `(2, 3)` 的張量，`sum(dim=0)` 會把第 0 維「壓扁」，結果形狀是 `(3,)`；`sum(dim=1)` 結果形狀是 `(2,)`。可以記成：**`dim` 指定哪一維消失。**

## 5.4 廣播與變形

**廣播**（broadcasting）：形狀不同的張量做逐元素運算時，PyTorch 會自動把較小的那個「複製擴充套件」成相同形狀。規則是**從最右邊一維開始對齊**，每一維要麼相等，要麼其中一個是 1（或不存在）。

```python requires=torch
import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])     # 形狀 (2, 3)
bias = torch.tensor([10.0, 20.0, 30.0])  # 形狀 (3,)
print(x + bias)                          # bias 被廣播到每一行
col = torch.tensor([[100.0], [200.0]])   # 形狀 (2, 1)
print(x + col)                           # col 被廣播到每一列
print(x.reshape(3, 2))                   # 改變形狀（元素總數不變）
print(x.reshape(-1))                     # -1 表示「自動推算」→ 展平成 6 個元素
print(x.unsqueeze(0).shape)              # 在最前面加一維 → (1, 2, 3)
```

```text
x     (2, 3)          x     (2, 3)
bias     (3,)  ✓      col   (2, 1)  ✓      (1 會被擴充套件成 3)
結果  (2, 3)          結果  (2, 3)

(2, 3) 和 (2,) ✗：最右邊一維 3 ≠ 2，而且都不是 1 → 報錯
```

神經網路裡到處都是廣播：線性層 `x @ W.T + b` 中，偏置 `b` 的形狀是 `(out,)`，會被廣播到 batch 裡的每個樣本上。

## 5.5 自動求導：就是 micrograd

把上一課的例子 `L = (a × b + c)²` 用 PyTorch 再算一遍：

```python requires=torch
import torch

a = torch.tensor(2.0, requires_grad=True)    # requires_grad=True：需要對它求梯度
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(10.0, requires_grad=True)
L = (a * b + c) ** 2
L.backward()
print(L.item(), a.grad, b.grad, c.grad)      # 16.0 -24 16 8，和手算、micrograd 都一樣
```

完全一樣！PyTorch 在每次運算時也在後臺記錄計算圖（每個結果張量都有一個 `grad_fn`），`backward()` 時按拓撲逆序把梯度傳回去——這正是你上一課親手實現的東西。

## 5.6 用 nn.Module 搭模型

`torch.nn` 裡有現成的網路層。最常用的是 **`nn.Linear(in_features, out_features)`**：它內部有權重 `weight`（形狀 `(out, in)`）和偏置 `bias`（形狀 `(out,)`），計算 `x @ weight.T + bias`。

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
layer = nn.Linear(3, 2)          # 3 維輸入 → 2 維輸出
print(layer.weight.shape, layer.bias.shape)
x = torch.randn(4, 3)            # 4 個樣本（一個 batch），每個 3 維
print(layer(x).shape)            # (4, 2)：每個樣本得到 2 個輸出

model = nn.Sequential(           # 按順序把層串起來
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
print(model)
print("引數量：", sum(p.numel() for p in model.parameters()))   # 3×16+16 + 16×2+2 = 98
```

> [!NOTE]
> 第一維幾乎總是 **batch**（一批樣本的數量）。模型一次處理一整批資料，GPU 才能發揮並行的優勢。`nn.Linear(3, 2)` 只關心**最後一維**是 3，前面可以有任意多維。

更靈活的寫法是**繼承 `nn.Module`**，自己寫 `forward`（前向計算）。所有大模型（包括 GPT、Llama）都是這樣寫的：

```python requires=torch
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                    # 必須先呼叫父類的初始化
        self.fc1 = nn.Linear(d_in, d_hidden)  # 作為屬性的層會被自動登記為引數
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):                     # 只需寫前向；反向由 autograd 自動完成
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

model = MLP(2, 8, 1)
print(model(torch.zeros(5, 2)).shape)         # 呼叫 model(x) 會自動執行 forward
for name, p in model.named_parameters():
    print(name, tuple(p.shape))
```

## 5.7 標準訓練迴圈

用 PyTorch 訓練 XOR。注意迴圈裡的五行，**幾乎所有 PyTorch 訓練程式碼都長這樣**：

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
loss_fn = nn.BCEWithLogitsLoss()                  # 二分類損失（內部自帶 sigmoid）
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for step in range(1, 501):
    logits = model(X)                             # ① 前向
    loss = loss_fn(logits, y)                     # ② 計算損失
    optimizer.zero_grad()                         # ③ 清零梯度
    loss.backward()                               # ④ 反向傳播
    optimizer.step()                              # ⑤ 更新引數
    if step % 100 == 0:
        print(f"第 {step} 步：損失 {loss.item():.4f}")

with torch.no_grad():
    print(torch.sigmoid(model(X)).round().squeeze().tolist())   # [0.0, 1.0, 1.0, 0.0]
```

對照上一課：

| micrograd | PyTorch |
|-----------|---------|
| 自己寫 `(p - y) ** 2` | `nn.MSELoss()`、`nn.BCEWithLogitsLoss()`、`nn.CrossEntropyLoss()` 等 |
| `p.grad = 0.0` | `optimizer.zero_grad()` |
| `p.data -= 0.1 * p.grad` | `optimizer.step()`（SGD）；更常用的是 `torch.optim.AdamW` |

> [!TIP]
> 分類任務的模型最後一層**不要**自己加 sigmoid/softmax：`BCEWithLogitsLoss` 和 `CrossEntropyLoss` 接收的是原始分數 **logits**，內部會用數值更穩定的方式處理。需要機率時（比如預測時）再手動 `torch.sigmoid` / `torch.softmax`。

## 5.8 完整例子：小批次、GPU 與評估

下面是一個更接近真實專案的例子：分類兩個同心圓環上的點。新概念：

- **Dataset / DataLoader**：把資料切成小批次（mini-batch），並在每輪打亂順序。用小批次的梯度下降叫**隨機梯度下降**（SGD），比每次用全部資料省記憶體，而且往往學得更好。
- **epoch**：把全部訓練資料過一遍叫一個 epoch。
- **device**：有 GPU 就用 GPU，模型和資料都要 `.to(device)`。
- **訓練集 / 測試集**：在模型**沒見過**的資料上評估，才能知道它是否真的學會了（而不是背下來了）。
- **`model.eval()` 與 `torch.no_grad()`**：評估時切換到評估模式，並且不記錄計算圖。

```python requires=torch
import math
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

# 1. 資料：兩個同心圓環。內圈是類別 0，外圈是類別 1
def make_rings(n):
    r = torch.cat([torch.rand(n) * 1.0, 1.5 + torch.rand(n) * 1.0])   # 半徑
    theta = torch.rand(2 * n) * 2 * math.pi                          # 角度
    X = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()            # 類別標籤必須是整數
    return X, y

X_train, y_train = make_rings(500)
X_test, y_test = make_rings(100)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# 2. 裝置：有 NVIDIA 顯示卡用 cuda，蘋果晶片用 mps，否則用 cpu
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print("裝置：", device)

# 3. 模型、損失、最佳化器
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()                     # 多分類交叉熵（內部自帶 softmax）
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

def accuracy(X, y):
    model.eval()                                    # 切換到評估模式
    with torch.no_grad():                           # 評估時不需要梯度，更快、更省記憶體
        pred = model(X.to(device)).argmax(dim=1)    # 分數最高的類別就是預測
    model.train()                                   # 切回訓練模式
    return (pred == y.to(device)).float().mean().item()

# 4. 訓練：每個 epoch 把全部資料按小批次過一遍
for epoch in range(1, 21):
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  損失 {total / len(X_train):.4f}  "
              f"測試準確率 {accuracy(X_test, y_test):.1%}")
```

線性模型永遠分不開兩個同心圓，而這個小小的 MLP 很快就在**測試集**上達到接近 100% 的準確率。

## 5.9 儲存與載入模型

```python requires=torch
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
torch.save(model.state_dict(), "rings.pt")        # 只儲存引數（推薦做法）

model2 = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
model2.load_state_dict(torch.load("rings.pt"))    # 先建好同樣結構的模型，再裝入引數
print(torch.equal(model[0].weight, model2[0].weight))   # True
```

`state_dict()` 是一個「引數名 → 張量」的字典。Hugging Face 上下載的模型檔案（如 `model.safetensors`）本質上也是這樣一個字典，只是換了一種更安全、更快的儲存格式。

## 5.10 最常見的報錯

| 報錯資訊（節選） | 原因 | 解決辦法 |
|------------------|------|----------|
| `mat1 and mat2 shapes cannot be multiplied (4x5 and 3x2)` | 輸入最後一維（5）≠ `nn.Linear` 的 `in_features`（3） | `print(x.shape)`，改模型或資料 |
| `The size of tensor a (3) must match the size of tensor b (2)` | 兩個張量無法廣播 | 檢查兩邊的 `shape` |
| `Expected all tensors to be on the same device` | 模型在 GPU、資料在 CPU（或反過來） | 兩者都 `.to(device)` |
| `expected target dtype to be Long` | `CrossEntropyLoss` 的標籤不是整數 | `y = y.long()` |
| `mat1 and mat2 must have the same dtype` | 一個 `float64`（常來自 NumPy）、一個 `float32` | `.float()` |
| 損失越來越大、變成 `nan` | 學習率太大，或忘了 `zero_grad()` | 減小 `lr`，檢查訓練迴圈 |

學習程式會識別這些報錯，並用你的語言解釋原因。

```python requires=torch
import torch
from torch import nn

layer = nn.Linear(3, 2)
x = torch.randn(4, 5)             # 故意給 5 維輸入
try:
    layer(x)
except RuntimeError as e:
    print("出錯了：", e)
```

## 5.11 小結

- **張量**是 PyTorch 的基本資料：多維陣列，能放在 GPU 上，能自動求導；出錯時先看 `shape`、`dtype`、`device`。
- `*` 是逐元素乘，`@` 是矩陣乘；`dim` 指定沿哪一維運算（那一維會消失）；**廣播**從右往左對齊，每維相等或為 1。
- `requires_grad=True` + `backward()` 就是 micrograd 的張量版。
- 用 `nn.Linear`、`nn.Sequential` 或繼承 `nn.Module` 搭模型；第一維通常是 batch。
- 訓練五步：**前向 → 損失 → `zero_grad()` → `backward()` → `step()`**；用 `DataLoader` 小批次訓練，在測試集上評估時用 `model.eval()` 和 `torch.no_grad()`。
- `torch.save(model.state_dict(), ...)` 儲存，`load_state_dict` 載入。

## 5.12 練習

<!-- exercise: linear-shape -->
### 練習 1：輸出形狀

`x` 的形狀是 `(32, 10)`，經過 `nn.Linear(10, 5)` 之後，輸出的形狀是？

- `(10, 5)`
- `(32, 5)`
- `(32, 10)`
- `(5, 32)`

<details>
<summary>答案與解析</summary>

`nn.Linear` 只改變**最後一維**：10 → 5；第一維 32 是 batch 大小，保持不變。
</details>

<!-- exercise: broadcast-shape -->
### 練習 2：廣播規則

不用 PyTorch，自己實現廣播規則。輸入兩行，每行是一個形狀（若干個用空格隔開的正整數）。如果兩個形狀可以廣播，輸出結果形狀（用空格隔開）；否則輸出 `error`。

```text
樣例輸入：
2 3
3
樣例輸出：2 3
```

<details>
<summary>提示</summary>

先在較短的形狀前面補 1，讓兩者一樣長；然後逐維比較：相等就取這個值，有一個是 1 就取另一個，否則報錯。

```python norun
a = list(map(int, input().split()))
b = list(map(int, input().split()))
n = max(len(a), len(b))
a = [1] * (n - len(a)) + a
b = [1] * (n - len(b)) + b
```
</details>

<!-- exercise: star-vs-at -->
### 練習 3：`*` 還是 `@`？

下面的程式會輸出什麼？

```python requires=torch
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 0], [0, 1]])
print((a * b).tolist())
print((a @ b).tolist())
```

<details>
<summary>答案與解析</summary>

`b` 是單位矩陣。逐元素乘 `a * b` 只保留對角線：`[[1, 0], [0, 4]]`；矩陣乘 `a @ b` 等於 `a` 本身：`[[1, 2], [3, 4]]`。
</details>

<!-- exercise: torch-grad -->
### 練習 4：用 PyTorch 求導

輸入一個數 `x`，用 PyTorch 的自動求導計算 `y = x³ + 2x` 及其導數 `dy/dx`，輸出 `y` 和 `dy/dx`（保留 2 位小數，空格隔開）。（需要安裝 `torch`）

```text
樣例輸入：2
樣例輸出：12.00 14.00
```

<details>
<summary>提示</summary>

`x = torch.tensor(float(input()), requires_grad=True)`，算出 `y` 後呼叫 `y.backward()`，導數就在 `x.grad` 裡。用 `.item()` 取出數字。
</details>

<!-- exercise: loop-order -->
### 練習 5：訓練迴圈的順序

下面哪一個是正確的訓練迴圈順序？

- `backward()` → `zero_grad()` → `step()` → 前向
- 前向 → 計算損失 → `zero_grad()` → `backward()` → `step()`
- 前向 → `step()` → `backward()` → `zero_grad()`
- `zero_grad()` → `step()` → 前向 → `backward()`

<details>
<summary>答案與解析</summary>

先前向算出損失，再清零舊梯度、反向傳播算出新梯度，最後用梯度更新引數。`zero_grad()` 放在 `backward()` 之前的任何位置都可以（有人習慣放在迴圈最開頭）。
</details>

<!-- exercise: no-grad -->
### 練習 6：為什麼評估時用 `torch.no_grad()`？

- 能讓模型預測得更準
- 不記錄計算圖，節省記憶體和時間
- 不加的話一定會報錯
- 會把所有梯度變成 0

<details>
<summary>答案與解析</summary>

評估時不需要反向傳播，所以沒必要記錄計算圖。`no_grad()` 不會改變預測結果，但能明顯節省視訊記憶體和時間——對大模型推理尤其重要。
</details>

## 5.13 延伸閱讀

- [PyTorch 官方教程](https://docs.pytorch.org/tutorials/)：從 “Learn the Basics” 開始
- 《動手學深度學習》（李沐等）：[zh.d2l.ai](https://zh.d2l.ai/)，有完整的 PyTorch 程式碼和中文講解影片
- Andrej Karpathy：[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列影片
- 下一課：[訓練技巧：讓網路學得更好](../06-training-tricks/zh-CN.md)
