# 4. 反向傳播與自動求導：親手寫一個 micrograd

> 🎯 **本課目標**：理解「計算圖」；用鏈式法則手算一個表示式的梯度；親手實現一個約 60 行的自動求導引擎（參考 Karpathy 的 micrograd）；用它從零訓練一個神經網路解決 XOR——這就是 PyTorch 的核心原理。
>
> ⏱ **預計用時**：90 分鐘　｜　📋 **前置知識**：第一部分第 5 課（鏈式法則），本部分第 3 課

上一課 XOR 網路的權重是手工設計的。真正的網路有成千上萬甚至上萬億個引數，必須**自動**算出每個引數的梯度。這件事叫**反向傳播**（backpropagation），實現它的程式叫**自動求導**（automatic differentiation，autograd）。這一課是整門課最重要的幾課之一——理解了它，你就理解了所有深度學習框架的心臟。

> [!NOTE]
> 本課的程式碼改編自 Andrej Karpathy 的開源專案 [micrograd](https://github.com/karpathy/micrograd)（MIT 許可證），強烈推薦配合他的影片 [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) 學習。

## 4.1 計算圖

任何計算都可以畫成一張**圖**：節點是數值，箭頭表示「由誰算出來」。例如 `L = (a × b + c)²`，令 `a = 2, b = −3, c = 10`：

```text
 a=2 ──┐
       ├─(×)── d=−6 ──┐
 b=−3 ─┘              ├─(+)── e=4 ──(²)── L=16
 c=10 ────────────────┘
```

**前向傳播**（forward）：從左到右算出每個節點的值。

**反向傳播**（backward）：從右往左，算出 L 對每個節點的導數（梯度）。

## 4.2 手算反向傳播

從最後開始，每一步只需要知道**區域性導數**，再用鏈式法則乘上「後面傳過來的梯度」：

```text
dL/dL = 1                                  （起點）
L = e²      → dL/de = 2e = 8
e = d + c   → dL/dd = dL/de × 1 = 8        （加法：梯度原樣分給兩個輸入）
              dL/dc = dL/de × 1 = 8
d = a × b   → dL/da = dL/dd × b = 8 × (−3) = −24    （乘法：梯度乘以「另一個」輸入）
              dL/db = dL/dd × a = 8 × 2 = 16
```

兩條非常好記的規律：

- **加法節點**把梯度**原樣**傳給每個輸入；
- **乘法節點**把梯度乘以**另一個輸入的值**再傳回去。

用數值導數驗證：

```python
def L(a, b, c):
    return (a * b + c) ** 2

a, b, c, h = 2.0, -3.0, 10.0, 1e-6
print("dL/da ≈", round((L(a + h, b, c) - L(a - h, b, c)) / (2 * h), 3))
print("dL/db ≈", round((L(a, b + h, c) - L(a, b - h, c)) / (2 * h), 3))
print("dL/dc ≈", round((L(a, b, c + h) - L(a, b, c - h)) / (2 * h), 3))
```

## 4.3 自動求導引擎：Value 類

現在讓電腦自動完成上面的過程。思路：

1. 每個數值包裝成一個 `Value` 物件，記住**它的值**、**它的梯度**、**它由哪些值算出來**，以及**怎樣把梯度傳回給它們**；
2. 前向計算時，每做一次運算就建立一個新的 `Value`，同時記下「反向時怎麼做」；
3. 呼叫 `backward()` 時，按**拓撲順序**（從輸出往輸入，保證每個節點在它所有「下游」之後處理）依次執行這些反向函式。

```python
import math

class Value:
    def __init__(self, data, children=(), op=""):
        self.data = data            # 數值
        self.grad = 0.0             # 損失對它的導數，開始為 0
        self._backward = lambda: None
        self._prev = set(children)  # 由哪些節點算出來
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad           # 加法：梯度原樣傳回
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad    # 乘法：乘以另一個輸入
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):           # k 是普通數字
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad   # tanh 的導數是 1 − tanh²
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other

    def backward(self):
        order, seen = [], set()
        def build(v):                       # 拓撲排序：先處理完所有輸入，再加入自己
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                order.append(v)
        build(self)
        self.grad = 1.0                     # dL/dL = 1
        for v in reversed(order):           # 從輸出往輸入
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) ** 2
L.backward()
print(L)
print(a.grad, b.grad, c.grad)     # -24.0 16.0 8.0：和手算完全一致
```

### 為什麼梯度要「累加」（`+=`）？

如果同一個值被用了多次，比如 `y = a + a`，那麼 `a` 透過兩條路徑影響 `y`，總梯度是兩條路徑之和（多元鏈式法則）。所以每個 `_backward` 都用 `+=` 而不是 `=`。這也是為什麼 PyTorch 每次反向傳播前要**把梯度清零**（`optimizer.zero_grad()`）——否則會把上一次的梯度也加進來。

```python norun
a = Value(3.0)
y = a + a
y.backward()
print(a.grad)     # 2.0
```

## 4.4 用 Value 搭神經網路

有了自動求導，搭網路就只需要「前向」怎麼算：

```python norun
import random

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, n_in, sizes):
        dims = [n_in] + sizes
        self.layers = [Layer(dims[i], dims[i + 1]) for i in range(len(sizes))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

## 4.5 從零訓練：學會 XOR

把 4.3 和 4.4 的程式碼放在一起，再加上訓練迴圈（本部分第 1 課的四步）——這次不需要手推任何梯度：

```python
import math
import random

class Value:
    def __init__(self, data, children=(), op=""):
        self.data, self.grad = data, 0.0
        self._backward, self._prev = (lambda: None), set(children)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad; other.grad += out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def backward(self):
        order, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                order.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()
    def parameters(self):
        return self.w + [self.b]

class MLP:
    def __init__(self, n_in, sizes):
        dims = [n_in] + sizes
        self.layers = [[Neuron(dims[i]) for _ in range(dims[i + 1])] for i in range(len(sizes))]
    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0]
    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]

random.seed(42)
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]                  # 用 tanh 輸出，所以標籤用 -1 / 1
model = MLP(2, [4, 1])
print("引數個數：", len(model.parameters()))

for step in range(1, 201):
    preds = [model(x) for x in xs]                              # ① 前向
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))         # ② 損失（平方誤差之和）
    for p in model.parameters():                                # ③ 清零梯度，再反向傳播
        p.grad = 0.0
    loss.backward()
    for p in model.parameters():                                # ④ 梯度下降
        p.data -= 0.1 * p.grad
    if step % 50 == 0:
        print(f"第 {step} 步：損失 {loss.data:.4f}")

for x, y in zip(xs, ys):
    print(x, "目標", y, "預測", round(model(x).data, 3))
```

網路**自己學會了** XOR。你剛剛從零實現了一個完整的深度學習框架的核心：**自動求導 + 神經網路 + 梯度下降**。

## 4.6 和 PyTorch 的對應關係

| 我們的 micrograd | PyTorch |
|------------------|---------|
| `Value`（一個數） | `torch.Tensor`（一個多維陣列，可以包含幾十億個數） |
| `v.grad` | `t.grad` |
| `loss.backward()` | `loss.backward()` |
| 手動把每個 `p.grad` 清零 | `optimizer.zero_grad()` |
| `p.data -= lr * p.grad` | `optimizer.step()` |
| 每個運算定義 `_backward` | PyTorch 為每個運算（矩陣乘法、softmax……）都寫好了反向函式，並在 GPU 上高效執行 |

區別只在於：PyTorch 操作的是**整個張量**而不是單個數字，所以快幾百萬倍；原理完全相同。下一課就來學 PyTorch。

## 4.7 小結

- 計算可以畫成**計算圖**；前向傳播算值，**反向傳播**從輸出往回算梯度。
- 每個節點只需知道**區域性導數**：加法把梯度原樣傳回，乘法把梯度乘以另一個輸入；再用鏈式法則連乘。
- 自動求導：每次運算記錄「怎麼反向」，`backward()` 按**拓撲順序的逆序**執行。
- 同一個值被多次使用時梯度要**累加**，所以每次反向傳播前要**清零**梯度。
- micrograd ≈ PyTorch 的核心思想；PyTorch 只是把它擴充套件到張量並放到 GPU 上。

## 4.8 練習

<!-- exercise: manual-backprop -->
### 練習 1：手算梯度

對 `f = (a × b + c)²`，輸入一行三個數 `a b c`，輸出 `df/da`、`df/db`、`df/dc`（保留 2 位小數，空格隔開）。請用 4.2 節的鏈式法則直接寫公式，不要用數值導數。

```text
樣例輸入：2 -3 10
樣例輸出：-24.00 16.00 8.00
```

<details>
<summary>提示</summary>

`e = a*b + c`，`df/de = 2e`；`df/da = 2e·b`，`df/db = 2e·a`，`df/dc = 2e`。
</details>

<!-- exercise: neuron-grad -->
### 練習 2：一個神經元的梯度

一個神經元 `y = tanh(w·x + b)`。輸入一行 `w x b`，輸出 `y`、`dy/dw`、`dy/db`（保留 4 位小數，空格隔開）。

```text
樣例輸入：1 1 0
樣例輸出：0.7616 0.4200 0.4200
```

<details>
<summary>提示</summary>

令 `z = w·x + b`。tanh 的區域性導數是 `1 − y²`；`z` 對 `w` 的導數是 `x`，對 `b` 的導數是 1。所以 `dy/dw = (1 − y²)·x`，`dy/db = 1 − y²`。
</details>

<!-- exercise: add-relu -->
### 練習 3：給 Value 加上 ReLU

起始程式碼裡有一個精簡的 `Value` 類（加法、乘法、反向傳播），讀入和輸出也已經寫好了。請在 `# TODO` 處新增 `relu()` 方法：前向是 `max(0, x)`；反向時，如果輸入大於 0，梯度原樣傳回，否則梯度為 0。程式會讀入 `a b c`，計算 `y = relu(a * b + c)`，輸出 `y` 的值以及 `a`、`b`、`c` 的梯度。

```text
樣例輸入：2 3 -1
樣例輸出：5.00 3.00 2.00 1.00
```

<details>
<summary>提示</summary>

```python norun
def relu(self):
    out = Value(max(0.0, self.data), (self,))
    def _backward():
        self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out
```
</details>

<!-- exercise: accumulate -->
### 練習 4：梯度累加

用本課的 `Value` 計算 `y = a * a + a`（`a = 3`），然後 `y.backward()`。`a.grad` 是多少？

- 3
- 6
- 7
- 10

<details>
<summary>答案與解析</summary>

`dy/da = 2a + 1 = 7`。`a` 在三處出現，三條路徑的梯度（3 + 3 + 1）被累加起來。
</details>

<!-- exercise: zero-grad -->
### 練習 5：為什麼要清零？

訓練迴圈中如果忘了在每次 `backward()` 之前把梯度清零，會發生什麼？

- 沒有影響
- 梯度會把之前所有步的梯度也累加進來，更新方向和大小都錯了
- 程式會報錯
- 訓練會變快

<details>
<summary>答案與解析</summary>

因為 `_backward` 用的是 `+=`，不清零就會越累越大。PyTorch 中對應的是忘記 `optimizer.zero_grad()`——一個非常常見的 bug。
</details>

## 4.9 延伸閱讀

- Andrej Karpathy：[micrograd](https://github.com/karpathy/micrograd) 與配套影片 [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)（B 站有中文字幕搬運）——本課的靈感來源。
- 3Blue1Brown：[反向傳播演算](https://www.bilibili.com/video/BV16x411V7Qg)（官方中文版）
- 下一課：[PyTorch 入門](../05-pytorch/zh-CN.md)
