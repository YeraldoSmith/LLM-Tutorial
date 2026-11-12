# 5. Getting started with PyTorch: tensors, autograd and the training loop

> 🎯 **Goals**: meet the core of PyTorch, the **tensor**; create, compute with, broadcast and reshape tensors; see that PyTorch's autograd works exactly like last lesson's micrograd; build models with `nn.Module`; write the standard five-step training loop; train in mini-batches, use a GPU, save models, and recognise the most common errors.
>
> ⏱ **Time**: 90 minutes　|　📋 **You need**: lessons 1–4 of this part; PyTorch installed (see part 1 lesson 2)

## 5.1 Why PyTorch?

Last lesson we wrote micrograd in pure Python. It is complete in principle, but it handles **one number at a time** — training a 17-parameter network already takes seconds. Real large models have billions of parameters and need:

1. operations on **whole blocks of data** at once (vectors, matrices and higher-dimensional arrays);
2. parallel computation on a **GPU**, thousands of times faster;
3. ready-made layers, loss functions and optimizers.

**PyTorch** is such a framework. It was open-sourced by Meta and is now maintained by the PyTorch Foundation under the Linux Foundation. It is the dominant deep learning framework in research and industry today: the training and inference code of Hugging Face Transformers, vLLM, Llama, DeepSeek, Qwen and most others is built on PyTorch. (Another common choice is Google's JAX.)

> [!TIP]
> **Install**: run `uv add torch` (or `pip install torch`) in your project folder — see part 1 lesson 2 for details.
>
> If the learning app says it can't find `torch`, it is using a different Python from the one where you installed torch. Open **Settings** and set the **Python path** to the Python inside your project's virtual environment: `project-folder/.venv/bin/python` on macOS/Linux, `project-folder\.venv\Scripts\python.exe` on Windows.

## 5.2 Tensors

A **tensor** is a "multi-dimensional array": 0 dimensions is a single number (a scalar), 1 dimension is a vector, 2 dimensions is a matrix, and anything with more dimensions is just called a tensor. Tensors are almost the same as NumPy arrays, with two extra powers: they **can run on a GPU** and they **support automatic differentiation**.

```python requires=torch
import torch

print(torch.__version__)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(x)
print("shape:", x.shape, " dtype:", x.dtype, " device:", x.device)
print(torch.zeros(2, 3))                 # all zeros
print(torch.ones(3))                     # all ones
print(torch.arange(6).reshape(2, 3))     # 0..5 arranged as 2 rows × 3 columns
torch.manual_seed(0)                     # fix the random seed so results are reproducible
print(torch.randn(2, 2))                 # random numbers from a standard normal distribution
```

Every tensor has three key attributes — **check them first when something goes wrong**:

| Attribute | Meaning | Example |
|-----------|---------|---------|
| `shape` | the size of each dimension | `torch.Size([2, 3])`: 2 rows, 3 columns |
| `dtype` | data type | `torch.float32` (default for decimals), `torch.int64` (default for integers), `torch.bfloat16` (common in LLMs) |
| `device` | where it lives | `cpu`, `cuda` (NVIDIA GPU), `mps` (Apple silicon) |

## 5.3 Tensor operations

```python requires=torch
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
print(a + b)          # element-wise addition
print(a * b)          # element-wise multiplication (NOT matrix multiplication!)
print(a @ b)          # matrix multiplication
print(a.sum(), a.mean(), a.max())
print(a.sum(dim=0))   # sum along dimension 0 (add the rows together) → column sums
print(a.sum(dim=1))   # sum along dimension 1 → row sums
print(a.T)            # transpose
print(a[0], a[:, 1], a[1, 0].item())    # row 0, column 1, a single element
```

> [!WARNING]
> `*` multiplies **element-wise**; `@` is **matrix multiplication**. Beginners mix these up more than anything else. Also, `.item()` turns a one-element tensor into a plain Python number — you'll use it all the time when printing losses.

The `dim` argument (dimension) matters: for a tensor of shape `(2, 3)`, `sum(dim=0)` "squashes" dimension 0 and the result has shape `(3,)`; `sum(dim=1)` gives shape `(2,)`. A handy way to remember it: **`dim` names the dimension that disappears.**

## 5.4 Broadcasting and reshaping

**Broadcasting**: when you combine tensors of different shapes element-wise, PyTorch automatically "stretches" the smaller one to the same shape. The rule: **align shapes from the rightmost dimension**; each dimension must be equal, or one of them must be 1 (or missing).

```python requires=torch
import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])     # shape (2, 3)
bias = torch.tensor([10.0, 20.0, 30.0])  # shape (3,)
print(x + bias)                          # bias is broadcast to every row
col = torch.tensor([[100.0], [200.0]])   # shape (2, 1)
print(x + col)                           # col is broadcast to every column
print(x.reshape(3, 2))                   # change the shape (same number of elements)
print(x.reshape(-1))                     # -1 means "work it out" → flatten to 6 elements
print(x.unsqueeze(0).shape)              # add a dimension at the front → (1, 2, 3)
```

```text
x       (2, 3)        x       (2, 3)
bias       (3,)  ✓    col     (2, 1)  ✓    (the 1 is stretched to 3)
result  (2, 3)        result  (2, 3)

(2, 3) with (2,) ✗: the rightmost dimensions 3 ≠ 2 and neither is 1 → error
```

Broadcasting is everywhere in neural networks: in a linear layer `x @ W.T + b`, the bias `b` has shape `(out,)` and is broadcast to every sample in the batch.

## 5.5 Autograd: it's micrograd

Let's redo last lesson's example `L = (a × b + c)²` in PyTorch:

```python requires=torch
import torch

a = torch.tensor(2.0, requires_grad=True)    # requires_grad=True: we want its gradient
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(10.0, requires_grad=True)
L = (a * b + c) ** 2
L.backward()
print(L.item(), a.grad, b.grad, c.grad)      # 16.0 -24 16 8, same as by hand and micrograd
```

Identical! PyTorch also records a computational graph behind the scenes for every operation (each result tensor has a `grad_fn`), and `backward()` passes gradients back in reverse topological order — exactly what you implemented yourself last lesson.

## 5.6 Building models with nn.Module

`torch.nn` contains ready-made layers. The most common is **`nn.Linear(in_features, out_features)`**: it holds a `weight` (shape `(out, in)`) and a `bias` (shape `(out,)`) and computes `x @ weight.T + bias`.

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
layer = nn.Linear(3, 2)          # 3-dimensional input → 2-dimensional output
print(layer.weight.shape, layer.bias.shape)
x = torch.randn(4, 3)            # 4 samples (one batch), 3 dimensions each
print(layer(x).shape)            # (4, 2): 2 outputs per sample

model = nn.Sequential(           # chain layers in order
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))   # 3×16+16 + 16×2+2 = 98
```

> [!NOTE]
> The first dimension is almost always the **batch** (how many samples are processed together). Models process a whole batch at once so the GPU can work in parallel. `nn.Linear(3, 2)` only cares that the **last** dimension is 3; there can be any number of dimensions before it.

A more flexible style is to **subclass `nn.Module`** and write `forward` (the forward computation) yourself. Every large model, GPT and Llama included, is written this way:

```python requires=torch
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                    # always call the parent's __init__ first
        self.fc1 = nn.Linear(d_in, d_hidden)  # layers stored as attributes are registered as parameters
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):                     # only the forward pass; autograd handles backward
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

model = MLP(2, 8, 1)
print(model(torch.zeros(5, 2)).shape)         # calling model(x) runs forward
for name, p in model.named_parameters():
    print(name, tuple(p.shape))
```

## 5.7 The standard training loop

Let's train XOR with PyTorch. Look at the five lines inside the loop — **almost all PyTorch training code looks like this**:

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
loss_fn = nn.BCEWithLogitsLoss()                  # binary classification loss (sigmoid built in)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for step in range(1, 501):
    logits = model(X)                             # ① forward
    loss = loss_fn(logits, y)                     # ② compute the loss
    optimizer.zero_grad()                         # ③ reset the gradients
    loss.backward()                               # ④ backpropagate
    optimizer.step()                              # ⑤ update the parameters
    if step % 100 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

with torch.no_grad():
    print(torch.sigmoid(model(X)).round().squeeze().tolist())   # [0.0, 1.0, 1.0, 0.0]
```

Compared with last lesson:

| micrograd | PyTorch |
|-----------|---------|
| writing `(p - y) ** 2` yourself | `nn.MSELoss()`, `nn.BCEWithLogitsLoss()`, `nn.CrossEntropyLoss()` and more |
| `p.grad = 0.0` | `optimizer.zero_grad()` |
| `p.data -= 0.1 * p.grad` | `optimizer.step()` (SGD); `torch.optim.AdamW` is used more often |

> [!TIP]
> For classification, do **not** put a sigmoid/softmax at the end of the model yourself: `BCEWithLogitsLoss` and `CrossEntropyLoss` take raw scores (**logits**) and handle them in a numerically stable way internally. Apply `torch.sigmoid` / `torch.softmax` manually only when you need probabilities (for example when predicting).

## 5.8 A complete example: mini-batches, GPUs and evaluation

Here is an example closer to a real project: classifying points on two concentric rings. New ideas:

- **Dataset / DataLoader**: split the data into mini-batches and shuffle it every round. Gradient descent on mini-batches is called **stochastic gradient descent** (SGD); it uses less memory than the full dataset and often learns better.
- **epoch**: one pass over the entire training set.
- **device**: use the GPU if there is one; both the model and the data need `.to(device)`.
- **training set / test set**: only evaluation on data the model has **never seen** tells you whether it really learned (rather than memorised).
- **`model.eval()` and `torch.no_grad()`**: switch to evaluation mode and stop recording the graph while evaluating.

```python requires=torch
import math
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

# 1. Data: two concentric rings. The inner ring is class 0, the outer ring is class 1
def make_rings(n):
    r = torch.cat([torch.rand(n) * 1.0, 1.5 + torch.rand(n) * 1.0])   # radius
    theta = torch.rand(2 * n) * 2 * math.pi                          # angle
    X = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()            # class labels must be integers
    return X, y

X_train, y_train = make_rings(500)
X_test, y_test = make_rings(100)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# 2. Device: cuda for NVIDIA GPUs, mps for Apple silicon, otherwise cpu
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

# 3. Model, loss and optimizer
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()                     # multi-class cross-entropy (softmax built in)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

def accuracy(X, y):
    model.eval()                                    # switch to evaluation mode
    with torch.no_grad():                           # no gradients needed: faster, less memory
        pred = model(X.to(device)).argmax(dim=1)    # the class with the highest score is the prediction
    model.train()                                   # back to training mode
    return (pred == y.to(device)).float().mean().item()

# 4. Training: each epoch goes over all the data in mini-batches
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
        print(f"epoch {epoch:2d}  loss {total / len(X_train):.4f}  "
              f"test accuracy {accuracy(X_test, y_test):.1%}")
```

A linear model can never separate two concentric rings, but this tiny MLP quickly reaches close to 100% accuracy on the **test set**.

## 5.9 Saving and loading models

```python requires=torch
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
torch.save(model.state_dict(), "rings.pt")        # save only the parameters (recommended)

model2 = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
model2.load_state_dict(torch.load("rings.pt"))    # build the same architecture, then load the parameters
print(torch.equal(model[0].weight, model2[0].weight))   # True
```

`state_dict()` is a dictionary from parameter names to tensors. The model files you download from Hugging Face (such as `model.safetensors`) are essentially the same kind of dictionary, stored in a safer and faster format.

## 5.10 The most common errors

| Error message (excerpt) | Cause | Fix |
|-------------------------|-------|-----|
| `mat1 and mat2 shapes cannot be multiplied (4x5 and 3x2)` | the input's last dimension (5) ≠ `in_features` of `nn.Linear` (3) | `print(x.shape)`, then fix the model or the data |
| `The size of tensor a (3) must match the size of tensor b (2)` | the two tensors can't be broadcast | check both `shape`s |
| `Expected all tensors to be on the same device` | model on GPU, data on CPU (or vice versa) | `.to(device)` both |
| `expected target dtype to be Long` | `CrossEntropyLoss` labels aren't integers | `y = y.long()` |
| `mat1 and mat2 must have the same dtype` | one is `float64` (often from NumPy), the other `float32` | `.float()` |
| loss keeps growing or becomes `nan` | learning rate too high, or `zero_grad()` forgotten | lower `lr`, check the training loop |

The learning app recognises these errors and explains the cause in your language.

```python requires=torch
import torch
from torch import nn

layer = nn.Linear(3, 2)
x = torch.randn(4, 5)             # deliberately 5-dimensional input
try:
    layer(x)
except RuntimeError as e:
    print("error:", e)
```

## 5.11 Summary

- The **tensor** is PyTorch's basic data type: a multi-dimensional array that can live on a GPU and supports autograd. When something breaks, check `shape`, `dtype` and `device` first.
- `*` is element-wise, `@` is matrix multiplication; `dim` names the dimension to operate along (it disappears); **broadcasting** aligns shapes from the right, and each dimension must be equal or 1.
- `requires_grad=True` + `backward()` is micrograd for tensors.
- Build models with `nn.Linear`, `nn.Sequential` or a subclass of `nn.Module`; the first dimension is usually the batch.
- The five training steps: **forward → loss → `zero_grad()` → `backward()` → `step()`**; train in mini-batches with `DataLoader` and evaluate on a test set with `model.eval()` and `torch.no_grad()`.
- Save with `torch.save(model.state_dict(), ...)`, load with `load_state_dict`.

## 5.12 Exercises

<!-- exercise: linear-shape -->
### Exercise 1: Output shape

`x` has shape `(32, 10)`. What is the shape of the output after `nn.Linear(10, 5)`?

- `(10, 5)`
- `(32, 5)`
- `(32, 10)`
- `(5, 32)`

<details>
<summary>Answer and explanation</summary>

`nn.Linear` changes only the **last** dimension: 10 → 5. The first dimension, 32, is the batch size and stays the same.
</details>

<!-- exercise: broadcast-shape -->
### Exercise 2: The broadcasting rule

Implement the broadcasting rule yourself, without PyTorch. Read two lines, each a shape (positive integers separated by spaces). If the shapes can be broadcast, print the resulting shape (separated by spaces); otherwise print `error`.

```text
Sample input:
2 3
3
Sample output: 2 3
```

<details>
<summary>Hint</summary>

First pad the shorter shape with 1s at the front so both have the same length; then compare dimension by dimension: if equal, take that value; if one is 1, take the other; otherwise it's an error.

```python norun
a = list(map(int, input().split()))
b = list(map(int, input().split()))
n = max(len(a), len(b))
a = [1] * (n - len(a)) + a
b = [1] * (n - len(b)) + b
```
</details>

<!-- exercise: star-vs-at -->
### Exercise 3: `*` or `@`?

What does this program print?

```python requires=torch
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 0], [0, 1]])
print((a * b).tolist())
print((a @ b).tolist())
```

<details>
<summary>Answer and explanation</summary>

`b` is the identity matrix. Element-wise `a * b` keeps only the diagonal: `[[1, 0], [0, 4]]`; the matrix product `a @ b` equals `a` itself: `[[1, 2], [3, 4]]`.
</details>

<!-- exercise: torch-grad -->
### Exercise 4: Derivatives with PyTorch

Read a number `x`. Use PyTorch's autograd to compute `y = x³ + 2x` and its derivative `dy/dx`, and print `y` and `dy/dx` (2 decimal places, separated by a space). (Requires `torch`.)

```text
Sample input: 2
Sample output: 12.00 14.00
```

<details>
<summary>Hint</summary>

`x = torch.tensor(float(input()), requires_grad=True)`; compute `y`, call `y.backward()`, and the derivative is in `x.grad`. Use `.item()` to get plain numbers.
</details>

<!-- exercise: loop-order -->
### Exercise 5: The order of the training loop

Which of these is the correct order for a training loop?

- `backward()` → `zero_grad()` → `step()` → forward
- forward → compute the loss → `zero_grad()` → `backward()` → `step()`
- forward → `step()` → `backward()` → `zero_grad()`
- `zero_grad()` → `step()` → forward → `backward()`

<details>
<summary>Answer and explanation</summary>

First compute the loss with a forward pass, then clear the old gradients and backpropagate to get new ones, and finally update the parameters with them. `zero_grad()` can go anywhere before `backward()` (some people put it at the very start of the loop).
</details>

<!-- exercise: no-grad -->
### Exercise 6: Why use `torch.no_grad()` during evaluation?

- it makes the model's predictions more accurate
- it doesn't record the computational graph, saving memory and time
- without it you always get an error
- it sets all gradients to 0

<details>
<summary>Answer and explanation</summary>

Evaluation needs no backpropagation, so there is no reason to record the graph. `no_grad()` doesn't change the predictions, but it saves a lot of memory and time — especially important for large-model inference.
</details>

## 5.13 Further reading

- [Official PyTorch tutorials](https://docs.pytorch.org/tutorials/): start with "Learn the Basics"
- *Dive into Deep Learning*: [d2l.ai](https://d2l.ai/), a free book with complete PyTorch code
- Andrej Karpathy: the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) video series
- Next lesson: [Training tricks: helping networks learn better](../06-training-tricks/en.md)
