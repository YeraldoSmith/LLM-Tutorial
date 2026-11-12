# 4. Backpropagation and autograd: build your own micrograd

> 🎯 **Goals**: understand the "computational graph"; compute an expression's gradients by hand with the chain rule; implement a ~60-line automatic differentiation engine yourself (based on Karpathy's micrograd); use it to train a neural network on XOR from scratch. This is the core idea behind PyTorch.
>
> ⏱ **Time**: 90 minutes　|　📋 **You need**: part 1 lesson 5 (the chain rule) and lesson 3 of this part

The weights of last lesson's XOR network were designed by hand. Real networks have thousands to trillions of parameters, so the gradient of every parameter must be computed **automatically**. This is called **backpropagation**, and a program that does it is an **automatic differentiation** engine (autograd). This is one of the most important lessons in the whole course: once you understand it, you understand the heart of every deep learning framework.

> [!NOTE]
> The code in this lesson is adapted from Andrej Karpathy's open-source project [micrograd](https://github.com/karpathy/micrograd) (MIT licence). We strongly recommend his video [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) alongside this lesson.

## 4.1 The computational graph

Any calculation can be drawn as a **graph**: nodes are values and arrows show "computed from". For example `L = (a × b + c)²` with `a = 2, b = −3, c = 10`:

```text
 a=2 ──┐
       ├─(×)── d=−6 ──┐
 b=−3 ─┘              ├─(+)── e=4 ──(²)── L=16
 c=10 ────────────────┘
```

**Forward pass**: go left to right and compute every node's value.

**Backward pass**: go right to left and compute the derivative (gradient) of L with respect to every node.

## 4.2 Backpropagation by hand

Start from the end. Each step only needs the **local derivative**, multiplied (chain rule) by "the gradient flowing in from later":

```text
dL/dL = 1                                  (the starting point)
L = e²      → dL/de = 2e = 8
e = d + c   → dL/dd = dL/de × 1 = 8        (addition: the gradient is copied to both inputs)
              dL/dc = dL/de × 1 = 8
d = a × b   → dL/da = dL/dd × b = 8 × (−3) = −24    (multiplication: multiply by the OTHER input)
              dL/db = dL/dd × a = 8 × 2 = 16
```

Two rules that are easy to remember:

- an **addition node** passes the gradient **unchanged** to each input;
- a **multiplication node** multiplies the gradient by **the other input's value** before passing it back.

Check with numerical derivatives:

```python
def L(a, b, c):
    return (a * b + c) ** 2

a, b, c, h = 2.0, -3.0, 10.0, 1e-6
print("dL/da ≈", round((L(a + h, b, c) - L(a - h, b, c)) / (2 * h), 3))
print("dL/db ≈", round((L(a, b + h, c) - L(a, b - h, c)) / (2 * h), 3))
print("dL/dc ≈", round((L(a, b, c + h) - L(a, b, c - h)) / (2 * h), 3))
```

## 4.3 An autograd engine: the Value class

Now let the computer do this automatically. The idea:

1. wrap every number in a `Value` object that remembers **its value**, **its gradient**, **which values it was computed from**, and **how to pass gradients back to them**;
2. during the forward computation, every operation creates a new `Value` and records "what to do on the way back";
3. `backward()` runs these backward functions in **topological order** (from the output towards the inputs, making sure each node is handled after everything "downstream" of it).

```python
import math

class Value:
    def __init__(self, data, children=(), op=""):
        self.data = data            # the number
        self.grad = 0.0             # derivative of the loss w.r.t. this value, starts at 0
        self._backward = lambda: None
        self._prev = set(children)  # the nodes it was computed from
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad           # addition: pass the gradient through
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad    # multiplication: times the other input
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):           # k is a plain number
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad   # the derivative of tanh is 1 − tanh²
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
        def build(v):                       # topological sort: all inputs first, then the node itself
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                order.append(v)
        build(self)
        self.grad = 1.0                     # dL/dL = 1
        for v in reversed(order):           # from the output back to the inputs
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) ** 2
L.backward()
print(L)
print(a.grad, b.grad, c.grad)     # -24.0 16.0 8.0: exactly what we got by hand
```

### Why do gradients "accumulate" (`+=`)?

If one value is used several times, e.g. `y = a + a`, then `a` affects `y` along two paths, and the total gradient is the sum of both paths (the multivariable chain rule). That's why every `_backward` uses `+=` instead of `=`. It's also why PyTorch needs you to **reset the gradients to zero** before every backward pass (`optimizer.zero_grad()`) — otherwise the previous step's gradients get added in too.

```python norun
a = Value(3.0)
y = a + a
y.backward()
print(a.grad)     # 2.0
```

## 4.4 Building a neural network out of Values

With autograd, building a network only requires describing the forward computation:

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

## 4.5 Training from scratch: learning XOR

Put the code from 4.3 and 4.4 together and add a training loop (the four steps from lesson 1 of this part). This time we derive no gradients by hand:

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
ys = [-1, 1, 1, -1]                  # the output is tanh, so the labels are -1 / 1
model = MLP(2, [4, 1])
print("parameters:", len(model.parameters()))

for step in range(1, 201):
    preds = [model(x) for x in xs]                              # ① forward
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))         # ② loss (sum of squared errors)
    for p in model.parameters():                                # ③ zero the gradients, then backprop
        p.grad = 0.0
    loss.backward()
    for p in model.parameters():                                # ④ gradient descent
        p.data -= 0.1 * p.grad
    if step % 50 == 0:
        print(f"step {step}: loss {loss.data:.4f}")

for x, y in zip(xs, ys):
    print(x, "target", y, "prediction", round(model(x).data, 3))
```

The network **learned XOR by itself**. You have just implemented, from scratch, the core of a complete deep learning framework: **autograd + neural networks + gradient descent**.

## 4.6 How this maps to PyTorch

| Our micrograd | PyTorch |
|---------------|---------|
| `Value` (one number) | `torch.Tensor` (a multi-dimensional array that can hold billions of numbers) |
| `v.grad` | `t.grad` |
| `loss.backward()` | `loss.backward()` |
| setting each `p.grad` to zero by hand | `optimizer.zero_grad()` |
| `p.data -= lr * p.grad` | `optimizer.step()` |
| defining `_backward` for each operation | PyTorch ships a backward function for every operation (matrix multiply, softmax…) and runs them efficiently on the GPU |

The only difference is that PyTorch works on **whole tensors** instead of single numbers, which makes it millions of times faster; the principle is exactly the same. The next lesson is about PyTorch.

## 4.7 Summary

- A computation can be drawn as a **computational graph**; the forward pass computes values, and **backpropagation** computes gradients from the output backwards.
- Each node only needs its **local derivative**: addition passes the gradient through, multiplication multiplies it by the other input; the chain rule multiplies these together.
- Autograd: every operation records "how to go backwards", and `backward()` runs them in **reverse topological order**.
- When a value is used several times its gradient **accumulates**, so gradients must be **zeroed** before every backward pass.
- micrograd ≈ the core idea of PyTorch; PyTorch extends it to tensors and runs it on GPUs.

## 4.8 Exercises

<!-- exercise: manual-backprop -->
### Exercise 1: Gradients by hand

For `f = (a × b + c)²`, read a line with three numbers `a b c` and print `df/da`, `df/db` and `df/dc` (2 decimal places, separated by spaces). Write the formulas directly using the chain rule from section 4.2 — don't use numerical derivatives.

```text
Sample input: 2 -3 10
Sample output: -24.00 16.00 8.00
```

<details>
<summary>Hint</summary>

`e = a*b + c`, `df/de = 2e`; `df/da = 2e·b`, `df/db = 2e·a`, `df/dc = 2e`.
</details>

<!-- exercise: neuron-grad -->
### Exercise 2: The gradient of one neuron

A neuron computes `y = tanh(w·x + b)`. Read a line `w x b` and print `y`, `dy/dw` and `dy/db` (4 decimal places, separated by spaces).

```text
Sample input: 1 1 0
Sample output: 0.7616 0.4200 0.4200
```

<details>
<summary>Hint</summary>

Let `z = w·x + b`. The local derivative of tanh is `1 − y²`; the derivative of `z` w.r.t. `w` is `x`, and w.r.t. `b` it is 1. So `dy/dw = (1 − y²)·x` and `dy/db = 1 − y²`.
</details>

<!-- exercise: add-relu -->
### Exercise 3: Add ReLU to Value

The starter code contains a minimal `Value` class (addition, multiplication, backpropagation), and the input and output code is already written. Add a `relu()` method at the `# TODO`: the forward pass is `max(0, x)`; on the way back, the gradient passes through unchanged if the input was greater than 0, and is 0 otherwise. The program reads `a b c`, computes `y = relu(a * b + c)` and prints the value of `y` and the gradients of `a`, `b` and `c`.

```text
Sample input: 2 3 -1
Sample output: 5.00 3.00 2.00 1.00
```

<details>
<summary>Hint</summary>

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
### Exercise 4: Accumulating gradients

Using this lesson's `Value`, compute `y = a * a + a` with `a = 3`, then call `y.backward()`. What is `a.grad`?

- 3
- 6
- 7
- 10

<details>
<summary>Answer and explanation</summary>

`dy/da = 2a + 1 = 7`. `a` appears in three places, and the gradients along the three paths (3 + 3 + 1) are added together.
</details>

<!-- exercise: zero-grad -->
### Exercise 5: Why zero the gradients?

What happens if a training loop forgets to reset the gradients to zero before each `backward()`?

- nothing
- the gradients of all previous steps are added in, so both the direction and the size of the updates are wrong
- the program crashes
- training gets faster

<details>
<summary>Answer and explanation</summary>

Because `_backward` uses `+=`, the gradients keep growing if you don't reset them. In PyTorch this is forgetting `optimizer.zero_grad()` — a very common bug.
</details>

## 4.9 Further reading

- Andrej Karpathy: [micrograd](https://github.com/karpathy/micrograd) and the companion video [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) — the inspiration for this lesson.
- 3Blue1Brown: [Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus)
- Next lesson: [Getting started with PyTorch](../05-pytorch/en.md)
