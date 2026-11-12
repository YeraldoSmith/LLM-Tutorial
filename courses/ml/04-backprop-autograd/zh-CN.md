# 4. 反向传播与自动求导：亲手写一个 micrograd

> 🎯 **本课目标**：理解「计算图」；用链式法则手算一个表达式的梯度；亲手实现一个约 60 行的自动求导引擎（参考 Karpathy 的 micrograd）；用它从零训练一个神经网络解决 XOR——这就是 PyTorch 的核心原理。
>
> ⏱ **预计用时**：90 分钟　｜　📋 **前置知识**：第一部分第 5 课（链式法则），本部分第 3 课

上一课 XOR 网络的权重是手工设计的。真正的网络有成千上万甚至上万亿个参数，必须**自动**算出每个参数的梯度。这件事叫**反向传播**（backpropagation），实现它的程序叫**自动求导**（automatic differentiation，autograd）。这一课是整门课最重要的几课之一——理解了它，你就理解了所有深度学习框架的心脏。

> [!NOTE]
> 本课的代码改编自 Andrej Karpathy 的开源项目 [micrograd](https://github.com/karpathy/micrograd)（MIT 许可证），强烈推荐配合他的视频 [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) 学习。

## 4.1 计算图

任何计算都可以画成一张**图**：节点是数值，箭头表示「由谁算出来」。例如 `L = (a × b + c)²`，令 `a = 2, b = −3, c = 10`：

```text
 a=2 ──┐
       ├─(×)── d=−6 ──┐
 b=−3 ─┘              ├─(+)── e=4 ──(²)── L=16
 c=10 ────────────────┘
```

**前向传播**（forward）：从左到右算出每个节点的值。

**反向传播**（backward）：从右往左，算出 L 对每个节点的导数（梯度）。

## 4.2 手算反向传播

从最后开始，每一步只需要知道**局部导数**，再用链式法则乘上「后面传过来的梯度」：

```text
dL/dL = 1                                  （起点）
L = e²      → dL/de = 2e = 8
e = d + c   → dL/dd = dL/de × 1 = 8        （加法：梯度原样分给两个输入）
              dL/dc = dL/de × 1 = 8
d = a × b   → dL/da = dL/dd × b = 8 × (−3) = −24    （乘法：梯度乘以「另一个」输入）
              dL/db = dL/dd × a = 8 × 2 = 16
```

两条非常好记的规律：

- **加法节点**把梯度**原样**传给每个输入；
- **乘法节点**把梯度乘以**另一个输入的值**再传回去。

用数值导数验证：

```python
def L(a, b, c):
    return (a * b + c) ** 2

a, b, c, h = 2.0, -3.0, 10.0, 1e-6
print("dL/da ≈", round((L(a + h, b, c) - L(a - h, b, c)) / (2 * h), 3))
print("dL/db ≈", round((L(a, b + h, c) - L(a, b - h, c)) / (2 * h), 3))
print("dL/dc ≈", round((L(a, b, c + h) - L(a, b, c - h)) / (2 * h), 3))
```

## 4.3 自动求导引擎：Value 类

现在让电脑自动完成上面的过程。思路：

1. 每个数值包装成一个 `Value` 对象，记住**它的值**、**它的梯度**、**它由哪些值算出来**，以及**怎样把梯度传回给它们**；
2. 前向计算时，每做一次运算就创建一个新的 `Value`，同时记下「反向时怎么做」；
3. 调用 `backward()` 时，按**拓扑顺序**（从输出往输入，保证每个节点在它所有「下游」之后处理）依次执行这些反向函数。

```python
import math

class Value:
    def __init__(self, data, children=(), op=""):
        self.data = data            # 数值
        self.grad = 0.0             # 损失对它的导数，开始为 0
        self._backward = lambda: None
        self._prev = set(children)  # 由哪些节点算出来
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad           # 加法：梯度原样传回
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad    # 乘法：乘以另一个输入
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):           # k 是普通数字
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad   # tanh 的导数是 1 − tanh²
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
        def build(v):                       # 拓扑排序：先处理完所有输入，再加入自己
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                order.append(v)
        build(self)
        self.grad = 1.0                     # dL/dL = 1
        for v in reversed(order):           # 从输出往输入
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) ** 2
L.backward()
print(L)
print(a.grad, b.grad, c.grad)     # -24.0 16.0 8.0：和手算完全一致
```

### 为什么梯度要「累加」（`+=`）？

如果同一个值被用了多次，比如 `y = a + a`，那么 `a` 通过两条路径影响 `y`，总梯度是两条路径之和（多元链式法则）。所以每个 `_backward` 都用 `+=` 而不是 `=`。这也是为什么 PyTorch 每次反向传播前要**把梯度清零**（`optimizer.zero_grad()`）——否则会把上一次的梯度也加进来。

```python norun
a = Value(3.0)
y = a + a
y.backward()
print(a.grad)     # 2.0
```

## 4.4 用 Value 搭神经网络

有了自动求导，搭网络就只需要「前向」怎么算：

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

## 4.5 从零训练：学会 XOR

把 4.3 和 4.4 的代码放在一起，再加上训练循环（本部分第 1 课的四步）——这次不需要手推任何梯度：

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
ys = [-1, 1, 1, -1]                  # 用 tanh 输出，所以标签用 -1 / 1
model = MLP(2, [4, 1])
print("参数个数：", len(model.parameters()))

for step in range(1, 201):
    preds = [model(x) for x in xs]                              # ① 前向
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))         # ② 损失（平方误差之和）
    for p in model.parameters():                                # ③ 清零梯度，再反向传播
        p.grad = 0.0
    loss.backward()
    for p in model.parameters():                                # ④ 梯度下降
        p.data -= 0.1 * p.grad
    if step % 50 == 0:
        print(f"第 {step} 步：损失 {loss.data:.4f}")

for x, y in zip(xs, ys):
    print(x, "目标", y, "预测", round(model(x).data, 3))
```

网络**自己学会了** XOR。你刚刚从零实现了一个完整的深度学习框架的核心：**自动求导 + 神经网络 + 梯度下降**。

## 4.6 和 PyTorch 的对应关系

| 我们的 micrograd | PyTorch |
|------------------|---------|
| `Value`（一个数） | `torch.Tensor`（一个多维数组，可以包含几十亿个数） |
| `v.grad` | `t.grad` |
| `loss.backward()` | `loss.backward()` |
| 手动把每个 `p.grad` 清零 | `optimizer.zero_grad()` |
| `p.data -= lr * p.grad` | `optimizer.step()` |
| 每个运算定义 `_backward` | PyTorch 为每个运算（矩阵乘法、softmax……）都写好了反向函数，并在 GPU 上高效执行 |

区别只在于：PyTorch 操作的是**整个张量**而不是单个数字，所以快几百万倍；原理完全相同。下一课就来学 PyTorch。

## 4.7 小结

- 计算可以画成**计算图**；前向传播算值，**反向传播**从输出往回算梯度。
- 每个节点只需知道**局部导数**：加法把梯度原样传回，乘法把梯度乘以另一个输入；再用链式法则连乘。
- 自动求导：每次运算记录「怎么反向」，`backward()` 按**拓扑顺序的逆序**执行。
- 同一个值被多次使用时梯度要**累加**，所以每次反向传播前要**清零**梯度。
- micrograd ≈ PyTorch 的核心思想；PyTorch 只是把它扩展到张量并放到 GPU 上。

## 4.8 练习

<!-- exercise: manual-backprop -->
### 练习 1：手算梯度

对 `f = (a × b + c)²`，输入一行三个数 `a b c`，输出 `df/da`、`df/db`、`df/dc`（保留 2 位小数，空格隔开）。请用 4.2 节的链式法则直接写公式，不要用数值导数。

```text
样例输入：2 -3 10
样例输出：-24.00 16.00 8.00
```

<details>
<summary>提示</summary>

`e = a*b + c`，`df/de = 2e`；`df/da = 2e·b`，`df/db = 2e·a`，`df/dc = 2e`。
</details>

<!-- exercise: neuron-grad -->
### 练习 2：一个神经元的梯度

一个神经元 `y = tanh(w·x + b)`。输入一行 `w x b`，输出 `y`、`dy/dw`、`dy/db`（保留 4 位小数，空格隔开）。

```text
样例输入：1 1 0
样例输出：0.7616 0.4200 0.4200
```

<details>
<summary>提示</summary>

令 `z = w·x + b`。tanh 的局部导数是 `1 − y²`；`z` 对 `w` 的导数是 `x`，对 `b` 的导数是 1。所以 `dy/dw = (1 − y²)·x`，`dy/db = 1 − y²`。
</details>

<!-- exercise: add-relu -->
### 练习 3：给 Value 加上 ReLU

起始代码里有一个精简的 `Value` 类（加法、乘法、反向传播），读入和输出也已经写好了。请在 `# TODO` 处添加 `relu()` 方法：前向是 `max(0, x)`；反向时，如果输入大于 0，梯度原样传回，否则梯度为 0。程序会读入 `a b c`，计算 `y = relu(a * b + c)`，输出 `y` 的值以及 `a`、`b`、`c` 的梯度。

```text
样例输入：2 3 -1
样例输出：5.00 3.00 2.00 1.00
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
### 练习 4：梯度累加

用本课的 `Value` 计算 `y = a * a + a`（`a = 3`），然后 `y.backward()`。`a.grad` 是多少？

- 3
- 6
- 7
- 10

<details>
<summary>答案与解析</summary>

`dy/da = 2a + 1 = 7`。`a` 在三处出现，三条路径的梯度（3 + 3 + 1）被累加起来。
</details>

<!-- exercise: zero-grad -->
### 练习 5：为什么要清零？

训练循环中如果忘了在每次 `backward()` 之前把梯度清零，会发生什么？

- 没有影响
- 梯度会把之前所有步的梯度也累加进来，更新方向和大小都错了
- 程序会报错
- 训练会变快

<details>
<summary>答案与解析</summary>

因为 `_backward` 用的是 `+=`，不清零就会越累越大。PyTorch 中对应的是忘记 `optimizer.zero_grad()`——一个非常常见的 bug。
</details>

## 4.9 延伸阅读

- Andrej Karpathy：[micrograd](https://github.com/karpathy/micrograd) 与配套视频 [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)（B 站有中文字幕搬运）——本课的灵感来源。
- 3Blue1Brown：[反向传播演算](https://www.bilibili.com/video/BV16x411V7Qg)（官方中文版）
- 下一课：[PyTorch 入门](../05-pytorch/zh-CN.md)
