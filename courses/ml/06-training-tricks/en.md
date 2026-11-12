# 6. Training tricks: helping networks learn better

> 🎯 **Goals**: recognise **overfitting** and **underfitting**; understand the learning rate and the **warmup + cosine decay** schedule; see how SGD, Momentum, Adam and **AdamW** differ; know why initialization, **LayerNorm/RMSNorm** and **residual connections** make deep networks trainable; master weight decay, dropout, gradient clipping and mixed precision (bf16); get a checklist for debugging training. This is the everyday craft of training large models.
>
> ⏱ **Time**: 90 minutes　|　📋 **You need**: lessons 1–5 of this part

The examples so far were well-behaved: small networks, clean data, almost any learning rate worked. Real training runs into problems: the loss doesn't go down, the loss becomes `nan`, the model is great on training data but poor on new data, a deeper network refuses to learn… Each trick in this lesson answers one specific problem. Most of the code here is pure Python and runs without PyTorch.

## 6.1 Overfitting and underfitting

What we really care about is how a model performs on **new data it hasn't seen**. This is called **generalization**.

- **Underfitting**: the model is too simple or not trained enough; it can't even learn the training set.
- **Overfitting**: the model has "memorised" the training set, including its noise and accidents, and fails on new data.

In the example below the true rule is "x > 5 means class 1", but 20% of the labels are wrong (real data always has noise). The classifier "looks at the k training samples closest to x and takes a majority vote":

```python
import random
random.seed(1)

def make(n):
    data = []
    for _ in range(n):
        x = random.uniform(0, 10)
        y = 1 if x > 5 else 0          # the true rule: x > 5 means class 1
        if random.random() < 0.2:      # but 20% of the labels are wrong (noise)
            y = 1 - y
        data.append((x, y))
    return data

train, test = make(100), make(1000)

def predict(x, k):                     # look at the k closest training samples, majority vote
    near = sorted(train, key=lambda p: abs(p[0] - x))[:k]
    return 1 if sum(y for _, y in near) * 2 > k else 0

def accuracy(data, k):
    return sum(predict(x, k) == y for x, y in data) / len(data)

for k in (1, 5, 25):
    print(f"k={k:2d}  train accuracy {accuracy(train, k):.0%}   test accuracy {accuracy(test, k):.0%}")
```

`k=1` is 100% correct on the training set (each point's nearest neighbour is itself), yet it has the lowest test accuracy — it memorised the wrong labels too. **Training-set scores can lie**, so data is split three ways:

| Split | Purpose |
|-------|---------|
| **training set** | used to compute gradients and update parameters |
| **validation set** | checks generalization during training; used to choose hyperparameters (learning rate, model size…) and when to stop |
| **test set** | used once at the very end to report the final result |

Plotting both loss curves during training diagnoses the problem:

```text
loss                                      loss
 │╲                                         │╲
 │ ╲___ validation                          │ ╲      ___── validation (rising = overfitting)
 │  ╲  ‾‾‾‾‾‾‾‾‾‾                           │  ╲___/
 │   ╲___ training                          │   ╲___
 │       ‾‾‾‾‾‾‾‾‾  both high = underfitting │       ‾‾‾‾‾‾─── training (keeps falling)
 └──────────────── steps                    └──────────────── steps
```

> [!NOTE]
> LLM pre-training uses enormous amounts of data (trillions of tokens), and each example is usually seen only once or a few times, so it **rarely overfits**; the main problem is underfitting — which is why bigger models and more data keep helping (part 4 covers "scaling laws"). During **fine-tuning**, however, data is scarce and overfitting is very common.

## 6.2 The learning rate: the most important hyperparameter

Minimise `f(x) = x²` (minimum at 0) with gradient descent, starting from `x = 5`, and see what different learning rates do:

```python
def descend(lr, steps=10):
    x = 5.0                       # minimise f(x) = x², minimum at x = 0
    path = []
    for _ in range(steps):
        x -= lr * 2 * x           # f'(x) = 2x
        path.append(x)
    return path

for lr in (0.01, 0.1, 0.9, 1.1):
    xs = descend(lr)
    print(f"lr={lr:<5} after 10 steps x = {xs[-1]:10.4f}   first 5 steps:", " ".join(f"{v:.1f}" for v in xs[:5]))
```

| Learning rate | What happens |
|---------------|--------------|
| too small (0.01) | stable but very slow; barely moved after 10 steps |
| good (0.1) | steadily approaches 0 |
| large (0.9) | jumps back and forth (oscillates) but still converges |
| too large (1.1) | jumps further each time and **diverges** — in real training the loss becomes `inf` / `nan` |

> [!TIP]
> When training goes wrong, **suspect the learning rate first**. Common starting points: `1e-3`–`3e-4` for Adam/AdamW; `1e-4`–`6e-4` for LLM pre-training (bigger models usually need smaller rates); `1e-5`–`2e-4` for fine-tuning.

## 6.3 Learning-rate schedules: warmup + cosine decay

The learning rate doesn't have to stay fixed. Almost every large model is trained with this **schedule**:

1. **Warmup**: for the first few hundred or thousand steps, the learning rate rises linearly from almost 0 to its maximum. At the start the parameters are random and the gradients are "messy"; taking big steps immediately can wreck training.
2. **Cosine decay**: afterwards it slowly falls along a cosine curve to a small value. Smaller steps late in training let the model settle precisely into the valley.

```python
import math

def lr_at(step, max_lr=3e-4, warmup=100, total=1000, min_lr=3e-5):
    if step < warmup:                                   # linear warmup
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / (total - warmup)       # goes from 0 to 1
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

for step in [0, 25, 50, 75] + list(range(100, 1001, 100)):
    lr = lr_at(step)
    print(f"step {step:4d}  lr {lr:.6f}  " + "█" * round(lr / 3e-4 * 40))
```

> [!NOTE]
> Some newer recipes use a **WSD** (Warmup-Stable-Decay) schedule: after warmup the learning rate stays at its maximum for a long time and decays quickly only at the very end. The advantage is that training can be extended at any time without fixing the total number of steps in advance.

## 6.4 Optimizers: SGD, Momentum, Adam and AdamW

Plain gradient descent (SGD) struggles on losses shaped like a "long narrow valley": it oscillates along the steep direction and crawls along the flat one. The loss `x² + 100y²` below is such a valley:

```python
import math

def loss_grad(x, y):                  # a narrow valley: steep in y, flat in x
    return x * x + 100 * y * y, (2 * x, 200 * y)

def run(update, steps=1000):
    p, state = [-4.0, 1.0], {}
    for t in range(1, steps + 1):
        loss, g = loss_grad(*p)
        if loss < 1e-4:
            return t
        update(p, g, state, t)
    return None

def sgd(p, g, s, t, lr=0.009):        # any larger (0.01 and up) and the y direction diverges
    for i in range(2):
        p[i] -= lr * g[i]

def momentum(p, g, s, t, lr=0.005, beta=0.9):
    v = s.setdefault("v", [0.0, 0.0])
    for i in range(2):
        v[i] = beta * v[i] + g[i]                # "velocity": accumulated past gradients; oscillations cancel out
        p[i] -= lr * v[i]

def adam(p, g, s, t, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    m = s.setdefault("m", [0.0, 0.0])
    v = s.setdefault("v", [0.0, 0.0])
    for i in range(2):
        m[i] = b1 * m[i] + (1 - b1) * g[i]       # first moment: moving average of the gradient (direction)
        v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2  # second moment: moving average of the squared gradient (size)
        m_hat = m[i] / (1 - b1 ** t)             # bias correction (m and v start biased towards 0)
        v_hat = v[i] / (1 - b2 ** t)
        p[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)   # every parameter gets its own step size

for name, fn in [("SGD", sgd), ("Momentum", momentum), ("Adam", adam)]:
    t = run(fn)
    print(f"{name:9s}", f"reached the bottom in {t} steps" if t else "didn't reach it in 1000 steps")
```

| Optimizer | Idea | Notes |
|-----------|------|-------|
| **SGD** | step along the negative gradient | simplest; very sensitive to the learning rate |
| **Momentum** | add "inertia": keep the previous velocity | oscillations cancel out; flat directions speed up |
| **Adam** | momentum + **step size adapted to each parameter's gradient size** | parameters with large gradients take small steps and vice versa; robust to hyperparameters |
| **AdamW** | Adam + **decoupled weight decay** | the default for training large models; typically `betas=(0.9, 0.95)`, `weight_decay=0.1` |

> [!NOTE]
> Optimizers keep improving. Newer ones such as **Muon** have been used to train some large models (for example Kimi K2) and are claimed to need less compute than AdamW. Still, AdamW remains the most common and safest choice. Note that Adam stores two extra values, `m` and `v`, for every parameter, so **the optimizer state takes twice as much memory as the parameters** — one reason training needs far more memory than inference.

## 6.5 Initialization: the starting point matters

Initial parameter values can't be arbitrary:

- **All zeros?** No. Every neuron in a layer would compute the same output and get the same gradient, and they'd stay identical forever — as if there were only one neuron. **Random** values are needed to break the symmetry.
- **The size of the random values matters too.** Passing through layer after layer, values that are too large keep growing (exploding) and values that are too small keep shrinking (vanishing).

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
        x = [max(0.0, sum(w * a for w, a in zip(row, x))) for row in W]   # linear layer + ReLU
        stds.append(std(x))
    return stds

print("initialization    layer 1  layer 4  layer 7  layer 10 (std of each layer's output)")
for name, scale in [("std=1", 1.0), ("std=0.01", 0.01), ("Kaiming √(2/n)", math.sqrt(2 / n))]:
    s = forward(scale)
    print(f"{name:15s}", "  ".join(f"{v:7.2g}" for v in (s[0], s[3], s[6], s[9])))
```

**Kaiming (He) initialization** gives each layer's weights a standard deviation of `√(2/n)` (`n` is the input dimension), so the signal keeps roughly the same size from layer to layer. PyTorch's `nn.Linear` uses a similar scaling by default, so you rarely need to worry about it; GPT-style models usually initialise from a normal distribution with mean 0 and standard deviation 0.02, further scaling down the output layers of residual branches by depth.

## 6.6 Normalization: LayerNorm and RMSNorm

Even with good initialization, the size of values drifts during training. **Normalization** layers pull each vector back to a "standard size":

- **LayerNorm**: subtract the mean, divide by the standard deviation → mean 0, variance 1. Used by the original Transformer and GPT-2.
- **RMSNorm**: no mean subtraction; just divide by the root mean square `√(mean(x²))`. Cheaper, works as well, and **almost every modern LLM (Llama, Qwen, DeepSeek…) uses it**.

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
print("RMSNorm after scaling by 1000:", [round(v, 3) for v in rms_norm(big)])   # unchanged
```

Real normalization layers then multiply by a **learnable** scale vector `γ` (LayerNorm also adds a shift `β`), so the network can decide what size it needs. `eps` is a tiny number that prevents division by zero.

> [!NOTE]
> You may also hear of **BatchNorm**: it normalizes across the samples in a batch and is common in image networks (CNNs). In language models, sentences have different lengths and inference often processes a single sequence, so LayerNorm / RMSNorm, which normalize **each vector on its own**, are used instead.

## 6.7 Residual connections: a highway for gradients

The deeper the network, the more local derivatives backpropagation multiplies together. If each is below 1, a few dozen multiplications give almost 0 — **vanishing gradients** — and the early layers can't learn at all.

**Residual connections** (introduced by ResNet in 2015) fix this with a very simple idea: each layer's output = **its input + the change this layer computes**, i.e. `y = x + f(x)`. Its derivative is `1 + f'(x)`, and that "1" guarantees the gradient can pass through every layer at least unchanged.

```python
import math

def grad_through(depth, residual, w=0.5):
    x, grad = 1.0, 1.0
    for _ in range(depth):
        f = math.tanh(w * x)
        local = w * (1 - f * f)                   # the derivative of f(x) = tanh(w·x)
        if residual:
            x, grad = x + f, grad * (1 + local)   # y = x + f(x), derivative = 1 + f'(x)
        else:
            x, grad = f, grad * local
    return grad

for depth in (5, 20, 50):
    print(f"{depth:2d} layers  plain: {grad_through(depth, False):.2e}   residual: {grad_through(depth, True):.2f}")
```

Every Transformer layer consists of exactly two residual blocks:

```formula
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

This "normalize, compute, add back" style is called **Pre-Norm**. It is the standard for today's large models and makes networks with a hundred-plus layers train stably. Part 3 covers it in detail.

## 6.8 Regularization: fighting overfitting

| Method | How | Intuition |
|--------|-----|-----------|
| **weight decay** | pull parameters slightly towards 0 each step: `w -= lr * λ * w` | keeps weights from growing too large; a "smoother" model |
| **dropout** | randomly set some neuron outputs to 0 during training | the network can't rely on any single neuron and learns more robust features |
| **early stopping** | stop when the validation loss stops improving and keep the best checkpoint | stop before it starts memorising |
| **more data / data augmentation** | collect more data or transform existing data | the most fundamental fix |

Dropout is simple to implement. Values that are kept during training are **divided by `1 − p`** so that the average output stays the same; during evaluation dropout is switched off (one of the things `model.eval()` does):

```python
import random
random.seed(0)

def dropout(x, p=0.5, training=True):
    if not training:
        return x
    return [0.0 if random.random() < p else a / (1 - p) for a in x]

x = [1.0] * 8
print("training:  ", dropout(x))
print("evaluation:", dropout(x, training=False))
avg = sum(sum(dropout(x)) for _ in range(10000)) / 10000 / len(x)
print("average during training:", round(avg, 3))     # about 1, same as during evaluation
```

> [!NOTE]
> LLM **pre-training** usually doesn't use dropout (there's so much data that overfitting is rare), but it does use AdamW weight decay. When fine-tuning on small datasets, dropout and early stopping become very useful.

## 6.9 Gradient clipping

Occasionally a particularly "weird" batch produces a huge gradient that throws the parameters far away in a single step, and the loss jumps to `nan` (a "loss spike"). **Gradient clipping**: if the combined length of all gradients (the **global norm**) exceeds a threshold, scale them down proportionally, keeping the direction.

```python
import math

def clip_by_norm(grads, max_norm):
    norm = math.sqrt(sum(g * g for g in grads))      # global norm: the length of all gradients as one vector
    if norm <= max_norm:
        return grads
    return [g * max_norm / norm for g in grads]       # same direction, length shrunk to max_norm

print(clip_by_norm([0.3, -0.4], 1.0))     # length 0.5: no clipping needed
print(clip_by_norm([30.0, -40.0], 1.0))   # length 50: shrunk to 1
```

Almost every large model is trained with `max_norm=1.0`. In PyTorch it's one line — `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — placed after `backward()` and before `step()`.

## 6.10 Mixed precision: fp32, fp16, bf16

The default `float32` (fp32) uses 4 bytes per number. 16-bit (2-byte) formats halve the memory and are much faster on GPUs:

| Format | Bytes | Range | Precision | Notes |
|--------|-------|-------|-----------|-------|
| fp32 | 4 | about ±3.4×10³⁸ | about 7 significant digits | the default |
| fp16 | 2 | about ±65504 | about 3–4 digits | small range, overflows easily, needs "loss scaling" |
| **bf16** | 2 | same as fp32 | about 2–3 digits | wide range, hard to overflow; **the mainstream for LLM training** |
| fp8 / fp4 | 1 / 0.5 | smaller | lower | supported by the newest GPUs; DeepSeek-V3 and others train in fp8 |

Python's built-in `struct` module lets you see the precision loss yourself:

```python
import struct

def to_fp16(x):
    return struct.unpack("e", struct.pack("e", x))[0]       # "e" means 16-bit half precision

def to_bf16(x):
    bits = struct.unpack("I", struct.pack("f", x))[0]       # get the bits of the 32-bit float
    return struct.unpack("f", struct.pack("I", bits & 0xFFFF0000))[0]   # bf16 = keep only the top 16 bits

for x in (0.1, 3.14159, 1000.7):
    print(f"{x:<8}  fp16 → {to_fp16(x):<12.7g} bf16 → {to_bf16(x):.7g}")

try:
    to_fp16(70000.0)
except OverflowError:
    print("70000 is outside fp16's range (max about 65504)")
print("Can bf16 represent 70000?", to_bf16(70000.0))           # yes, just not precisely
```

**Mixed-precision training** keeps a "master copy" of the parameters and the optimizer state in fp32 and does most of the forward and backward computation in bf16. In PyTorch (with a GPU):

```python norun
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(xb), yb)
loss.backward()
```

> [!TIP]
> A rule of thumb for memory: for **inference**, a bf16 model needs 2 bytes per parameter, so 7 billion (7B) parameters ≈ 14 GB; for **training** (AdamW + mixed precision) it's about 16 bytes per parameter, so a 7B model ≈ 112 GB, not counting intermediate activations. That's why memory-saving methods such as LoRA are popular for fine-tuning (part 5).

## 6.11 Putting it together: a modern training script

Here is everything from this lesson combined in PyTorch: residual blocks + RMSNorm + GELU + dropout + AdamW (weight decay) + warmup-cosine schedule + gradient clipping. It is almost the training skeleton of a mini Transformer (only attention is missing):

```python requires=torch
import math
import torch
from torch import nn

torch.manual_seed(0)

class Block(nn.Module):                      # a modern residual block: Norm → MLP → Dropout, added back to the input
    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return x + self.drop(self.mlp(self.norm(x)))      # residual connection

model = nn.Sequential(nn.Linear(8, 32), *[Block(32) for _ in range(4)], nn.Linear(32, 1))
X = torch.randn(512, 8)
y = (X[:, :1] * X[:, 1:2]).sin()                         # an arbitrary non-linear target

total_steps, warmup = 300, 30
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
def schedule(step):                                       # returns a learning-rate multiplier
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

for step in range(total_steps):
    idx = torch.randint(0, len(X), (64,))                 # pick a random mini-batch
    loss = nn.functional.mse_loss(model(X[idx]), y[idx])
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # gradient clipping
    optimizer.step()
    scheduler.step()                                      # update the learning rate
    if step % 100 == 0 or step == total_steps - 1:
        print(f"step {step:3d}  loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.5f}  grad_norm {norm:.2f}")

model.eval()                                              # evaluation mode: dropout off
with torch.no_grad():
    print("final MSE:", round(nn.functional.mse_loss(model(X), y).item(), 4))
```

## 6.12 A checklist for debugging training

Based on Andrej Karpathy's classic post "A Recipe for Training Neural Networks":

1. **Look at the data first.** Print a few samples and labels; make sure nothing is misread or misaligned.
2. **Start simple.** Get the whole pipeline working with the smallest model and the smallest data.
3. **Overfit a single batch first.** Train on the same batch repeatedly; the loss should drop close to 0. If it doesn't, there's a bug.
4. **Check the initial loss.** For 10 classes, random guessing gives a cross-entropy of about `ln 10 ≈ 2.30`; if it's far off, the initialization or the loss function is wrong.
5. **Loss becomes `nan`**: lower the learning rate, add gradient clipping, check the data for `inf` / `nan`, look for `log(0)`.
6. **Loss doesn't move at all**: check for a forgotten `zero_grad()` / `step()`, a learning rate that's too small, or gradients cut off by `no_grad()` / `.detach()`.
7. **Change one thing at a time**, and write down the settings and results of every experiment.

## 6.13 Summary

- What matters is **generalization**: monitor a validation set; training loss falling while validation loss rises means **overfitting**, both high means **underfitting**.
- The **learning rate** matters most: too high diverges, too low is slow; large models use **warmup + cosine decay**.
- **AdamW** = momentum + adaptive step sizes + decoupled weight decay, the default optimizer; its state takes twice the memory of the parameters.
- Good **initialization** (random, well-scaled), **normalization** (RMSNorm/LayerNorm) and **residual connections** (`x + f(x)`) make deep networks trainable.
- Regularization: weight decay, dropout, early stopping, more data; **gradient clipping** (global norm 1.0) prevents loss spikes.
- **bf16** mixed precision halves memory and speeds things up, with the same range as fp32.

## 6.14 Exercises

<!-- exercise: lr-schedule -->
### Exercise 1: Warmup + cosine schedule

Read a line with three integers `step warmup total` and print the learning-rate **multiplier** for that step (the maximum learning rate is 1, the minimum 0), with 4 decimal places:

- when `step < warmup`: `(step + 1) / warmup`
- otherwise: `0.5 × (1 + cos(π × (step − warmup) / (total − warmup)))`

```text
Sample input: 0 10 100
Sample output: 0.1000
```

<details>
<summary>Hint</summary>

Use `math.cos` and `math.pi`. Note that `step` counts from 0.
</details>

<!-- exercise: rmsnorm -->
### Exercise 2: Implement RMSNorm

Read a line of numbers and print their RMSNorm: each number divided by `√(mean of the squares + 1e-5)`, with 3 decimal places, separated by spaces.

```text
Sample input: 2 4 6 8
Sample output: 0.365 0.730 1.095 1.461
```

<details>
<summary>Hint</summary>

`rms = math.sqrt(sum(a * a for a in x) / len(x) + 1e-5)`, then `print(" ".join(f"{a / rms:.3f}" for a in x))`.
</details>

<!-- exercise: clip-norm -->
### Exercise 3: Gradient clipping

The first line contains some gradient values; the second line is `max_norm`. If the global norm of the gradients (the square root of the sum of squares) is larger than `max_norm`, multiply every value by `max_norm / norm`; otherwise leave them unchanged. Print the result with 4 decimal places, separated by spaces.

```text
Sample input:
30 -40
1
Sample output: 0.6000 -0.8000
```

<details>
<summary>Hint</summary>

Compute `norm = math.sqrt(sum(g * g for g in grads))` first, then decide whether to scale.
</details>

<!-- exercise: diagnose -->
### Exercise 4: Diagnose the training curves

The training loss keeps falling, but the validation loss has been rising since epoch 3. What is most likely going on?

- underfitting — make the model bigger
- overfitting — use early stopping, regularization or more data
- the learning rate is too small
- everything is fine, keep training

<details>
<summary>Answer and explanation</summary>

Getting better on the training set while getting worse on the validation set means the model is memorising the training data — classic overfitting. Stop where the validation loss is lowest (early stopping), or use weight decay, dropout or more data.
</details>

<!-- exercise: zero-init -->
### Exercise 5: Initialising everything to 0

What happens if every weight of an MLP is initialised to 0?

- training becomes faster and more stable
- all neurons in a layer stay exactly identical, as if each layer had a single neuron
- nothing changes
- the program crashes

<details>
<summary>Answer and explanation</summary>

All neurons have the same inputs and weights, so they compute the same outputs and gradients and remain the same after every update — this "symmetry" can never be broken. That's why initialization must be random.
</details>

<!-- exercise: bf16-memory -->
### Exercise 6: Estimate the memory

A model with 7 billion parameters (7B) is loaded in bf16 for inference. Roughly how much memory do the parameters alone need?

- 7 GB
- 14 GB
- 28 GB
- 56 GB

<details>
<summary>Answer and explanation</summary>

bf16 uses 2 bytes per parameter: 7×10⁹ × 2 bytes = 14×10⁹ bytes ≈ 14 GB. fp32 would need 28 GB; training needs much more once gradients and AdamW state are added.
</details>

## 6.15 Further reading

- Andrej Karpathy: [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) (a classic summary of hard-won training experience)
- *Dive into Deep Learning*: the [Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html) chapter
- Google Research: [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- Next part: [Part 3 · Language models](../../lm/01-language-models/en.md) — time to build language models at last!
