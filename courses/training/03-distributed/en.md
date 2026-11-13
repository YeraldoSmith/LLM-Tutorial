# 3. Distributed and mixed-precision training

> 🎯 **Goals**: know the number formats FP32, FP16, BF16 and FP8, and understand why **mixed-precision training** keeps FP32 master weights and why FP16 needs loss scaling; understand **gradient accumulation**; understand **data parallelism** and **all-reduce** (simulating ring all-reduce yourself); understand how **ZeRO / FSDP** shard optimiser states, gradients and parameters across GPUs; understand **tensor parallelism** and **pipeline parallelism** (and the pipeline "bubble"); learn about activation recomputation and what large-scale training really looks like.
>
> ⏱ **Time**: 90 minutes　|　📋 **You need**: lesson 2 of this part; Part 2, lessons 4 and 5

Last lesson's numbers: training a 70B model needs over 1 TB of memory just for parameters and optimiser state, and enough compute to keep thousands of GPUs busy for a month or two. This lesson covers two things: **making each GPU faster and leaner** (lower precision), and **getting thousands of GPUs to train one model together** (parallelism).

## 3.1 Number formats: fewer bits, faster and smaller

A floating-point number has three parts: a sign, an **exponent** (how big or small a number can be — the **range**) and a **mantissa** (how many significant digits — the **precision**).

| Format | Total bits | Exponent bits | Mantissa bits | Largest value | Roughly how many significant digits |
|--------|------------|---------------|---------------|---------------|-------------------------------------|
| FP32 | 32 | 8 | 23 | 3.4 × 10³⁸ | 7 |
| FP16 | 16 | 5 | 10 | 65504 | 3 |
| BF16 | 16 | 8 | 7 | 3.4 × 10³⁸ | 2–3 |
| FP8 (E4M3) | 8 | 4 | 3 | 448 | 1 |

Halving the bits halves memory and data transfer, and GPU matrix maths typically runs twice as fast (an H100 does about 989 TFLOPS in BF16 and about 1979 in FP8). The price is precision and range. Let's simulate with Python's `struct` module:

```python
import struct

def to_fp32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

def to_fp16(x):
    try:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    except OverflowError:                  # outside FP16's range
        return float("inf")

def to_bf16(x):                            # BF16 is the top 16 bits of FP32 (rounded)
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]

print(f"{'x':>13} {'FP32':>14} {'FP16':>14} {'BF16':>14}")
for x in (1 / 3, 3.14159265, 1.001, 70000.0, 1e-8):
    print(f"{x:>13.6g} {to_fp32(x):>14.8g} {to_fp16(x):>14.8g} {to_bf16(x):>14.8g}")
```

- **FP16**: decent precision but a **tiny range** — above 65504 becomes infinity, and below about 6 × 10⁻⁸ becomes 0. Very small gradients simply "vanish" during training.
- **BF16** (a format Google designed for deep learning): 8 exponent bits like FP32, so **the same range** and no overflow; but only 7 mantissa bits, so very low precision — `1.001` becomes `1`.
- **BF16** is now the standard choice for training large models.

## 3.2 Mixed-precision training

With such low precision, how can BF16 train anything? The key is **mixed precision**:

- the matrix multiplications of the **forward and backward passes** (nearly all the time) run in BF16 — fast and lean;
- the **parameter update** is done on FP32 "master" weights.

Why must the update be FP32? An example: a parameter of 1.0 gets a tiny update of 0.0001 at every step:

```python
import struct

def to_bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]

w_bf16, w_fp32 = 1.0, 1.0
for step in range(1000):
    w_bf16 = to_bf16(w_bf16 + 0.0001)      # stored back in BF16 after every step
    w_fp32 = w_fp32 + 0.0001
print("parameters kept only in BF16:", w_bf16)      # stuck at 1.0!
print("FP32 master weights:         ", round(w_fp32, 4))
```

Near 1.0, the smallest step BF16 can represent is 2⁻⁷ ≈ 0.0078, so `1.0 + 0.0001` rounds back to 1.0 — the update is **lost completely**. Late in training, updates are often this small, so parameters must be stored and updated in FP32 (that's where the 4 bytes of master weights in last lesson's "16 bytes per parameter" come from).

**Loss scaling for FP16**: FP16 has an extra problem — very small gradients underflow to 0. The fix is to multiply the loss by a big number (say 65536), which scales the gradients up too, and divide back afterwards:

```python
import struct

def to_fp16(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]

g = 1e-8                                   # a very small gradient
print("stored directly in FP16:", to_fp16(g))
scale = 65536
print("scaled up 65536×, stored in FP16, scaled back:", to_fp16(g * scale) / scale)
```

BF16 has FP32's range and needs no loss scaling — the main reason it replaced FP16.

In PyTorch, mixed precision takes just one `torch.autocast`:

```python requires=torch timeout=120
import torch

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(512, 64, device=device)
y = 2 * x[:, :1] + 0.5

for step in range(201):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):   # matrix maths in here runs in BF16
        pred = model(x)
    loss = torch.nn.functional.mse_loss(pred.float(), y)
    opt.zero_grad()
    loss.backward()
    opt.step()                                                       # parameters and gradients stay FP32
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  output {pred.dtype}  "
              f"weights {model[0].weight.dtype}  grads {model[0].weight.grad.dtype}")
```

**FP8 training**: DeepSeek-V3 (2024) moved most of its matrix multiplications to FP8, controlling the error with "fine-grained scaling" (one scale factor for each small block of data), making it one of the first models trained successfully in FP8 at very large scale.

## 3.3 Gradient accumulation

Bigger batches give steadier gradients. The global batch for large-model training is usually **millions of tokens**, far more than one GPU can hold. One answer is **gradient accumulation**: compute a small batch (a micro-batch) at a time, add up the gradients, and update the parameters once enough has accumulated:

```python
data = [(1.0, 2.1), (2.0, 3.9), (3.0, 6.2), (4.0, 7.8), (5.0, 10.1), (6.0, 12.2), (7.0, 13.8), (8.0, 16.1)]
w = 0.5

def grad(batch, w):                         # gradient of the mean squared error mean((w·x - y)²) with respect to w
    return sum(2 * (w * x - y) * x for x, y in batch) / len(batch)

full = grad(data, w)                        # the whole batch at once

micro = [data[i:i + 2] for i in range(0, len(data), 2)]   # 4 micro-batches of 2 examples
acc = 0.0
for mb in micro:
    acc += grad(mb, w) / len(micro)         # accumulate, dividing by the number of steps
print(f"whole batch: {full:.4f}, accumulated: {acc:.4f}")
```

They're exactly equal. Gradient accumulation trades **time** for **memory**: several forward and backward passes for one parameter update.

## 3.4 Data parallelism and all-reduce

**Data parallelism** (DP) is the most common form of parallelism: every GPU holds a **complete copy of the model**, processes **different data**, and computes its own gradients; then all GPUs **average** their gradients and each updates its parameters with that average, so the copies stay identical. Mathematically it's the same as gradient accumulation — the micro-batches just run on different GPUs at the same time.

The operation "every GPU ends up with the sum of all gradients" is called **all-reduce**. The naive way sends everything to one GPU to add up and send back, but that GPU becomes a bottleneck. In practice **ring all-reduce** is common: the GPUs form a ring and each talks only to the next. Split the data into n chunks:

1. **reduce-scatter**: n−1 rounds; in each, every GPU passes one chunk to the next, which adds it to its own chunk. At the end, each GPU holds **one chunk** of the complete sum.
2. **all-gather**: n−1 more rounds pass these complete chunks all the way round the ring.

```python
def ring_allreduce(data):
    n = len(data)                              # number of GPUs
    c = len(data[0]) // n                      # chunk length
    buf = [list(d) for d in data]
    sent = [0] * n                             # how many numbers each GPU has sent
    for step in range(n - 1):                  # ① reduce-scatter
        msgs = []
        for i in range(n):
            k = (i - step) % n                 # GPU i sends chunk k at this step
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
    print(f"GPU {i}: {r}")
print("numbers sent by each GPU:", sent, "(the gradient has 4 numbers)")
```

Each GPU sends only `2(n−1)/n` times the gradient size — even with 1000 GPUs, less than twice the gradient — and every link is busy at once. NVIDIA's **NCCL** library implements these "collective" operations, and PyTorch's `DistributedDataParallel` (DDP) uses it to synchronise gradients.

## 3.5 ZeRO and FSDP: cutting out the redundancy

Data parallelism has a waste: every GPU stores a **full** copy of the parameters, gradients and optimiser state (16 bytes per parameter). Microsoft's **ZeRO** (2019) observed that since everyone stores the same thing, it can be **split up and shared**, swapping pieces in only when needed:

| Stage | What's sharded | Memory per GPU (N parameters, n GPUs) |
|-------|----------------|---------------------------------------|
| plain DP | nothing | 16N |
| ZeRO-1 | optimiser state (12 bytes) | 4N + 12N / n |
| ZeRO-2 | plus gradients | 2N + 14N / n |
| ZeRO-3 | plus the parameters themselves | 16N / n |

```python
def zero_gb(N, n, stage):
    per_param = {0: 16, 1: 4 + 12 / n, 2: 2 + 14 / n, 3: 16 / n}[stage]
    return N * per_param / 1e9

N, n = 70e9, 64
for stage in range(4):
    print(f"ZeRO-{stage}: {zero_gb(N, n, stage):7.1f} GB per GPU")
```

With ZeRO-3 each GPU stores only 1/n of the parameters; before computing a layer, it gathers that layer's parameters from the other GPUs (all-gather) and discards them afterwards. PyTorch's built-in **FSDP** (Fully Sharded Data Parallel) implements ZeRO-3, and the DeepSpeed library provides all of ZeRO.

## 3.6 Tensor parallelism and pipeline parallelism

When a layer's matrices are too big, or you want less computation on each GPU, the **model itself** has to be split.

**Tensor parallelism** (TP, from NVIDIA's Megatron-LM) splits one matrix multiplication across several GPUs. In an MLP, the first matrix is split **by columns** and the second **by rows**; each GPU finishes its own part independently, and a single all-reduce adds up the results at the end:

```python
def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

def relu(M):
    return [[max(0, v) for v in row] for row in M]

x = [[1, 2], [3, -4]]                       # 2 tokens of 2 dimensions
W = [[1, 0, 2, -1], [0, 1, 1, 3]]           # 2 → 4 (first MLP layer)
V = [[1, 2], [0, 1], [1, 0], [2, 1]]        # 4 → 2 (second MLP layer)
full = matmul(relu(matmul(x, W)), V)        # the whole computation on one GPU

W0, W1 = [r[:2] for r in W], [r[2:] for r in W]   # GPU 0 takes the first two columns, GPU 1 the last two
V0, V1 = V[:2], V[2:]                              # and V is split by rows to match
part0 = matmul(relu(matmul(x, W0)), V0)            # each GPU computes its part with no communication in between
part1 = matmul(relu(matmul(x, W1)), V1)
summed = [[a + b for a, b in zip(r0, r1)] for r0, r1 in zip(part0, part1)]   # one all-reduce
print("whole:          ", full)
print("tensor parallel:", summed, "same" if full == summed else "different")
```

(ReLU works element by element, so applying it to each column shard separately is exactly right. Attention can be split by heads.) Tensor parallelism communicates at every layer and needs very fast links between GPUs, so it's normally used only among the 8 GPUs **inside one machine** (connected by NVLink).

**Pipeline parallelism** (PP) splits the model **by layers** — say 80 layers over 8 GPUs, 10 each — and the data flows through them like a production line. The problem: when GPU 1 finishes the first batch, GPU 8 is only just starting; the other GPUs sit waiting, and this idle time is the **bubble**. The fix is to split each batch into many micro-batches that flow through back to back:

```python
def bubble(p, m):              # p pipeline stages, m micro-batches
    return (p - 1) / (m + p - 1)

for m in (1, 4, 16, 64):
    print(f"8-stage pipeline, {m:>2} micro-batches: GPUs idle {bubble(8, m):.0%} of the time")
```

## 3.7 Activation recomputation, and putting it all together

- **Activation recomputation** (activation checkpointing): instead of keeping every intermediate result of the forward pass, keep only a few "checkpoints" and **recompute** the rest from them when the backward pass needs it. About a third more computation buys a big memory saving.
- **Combining them**: real large-scale training uses several kinds of parallelism at once. When Meta trained Llama 3 405B on 16,000 H100s, it combined tensor parallelism (8 GPUs inside a machine), pipeline parallelism (16 stages), data parallelism (FSDP) and, for long sequences, **context parallelism** (splitting one long sequence across GPUs) — "4D parallelism". Mixture-of-experts models (lesson 4) add **expert parallelism**, with different experts on different GPUs.
- **Failures**: with tens of thousands of GPUs working for months, hardware failures are routine. The Llama 3 paper reports over 400 unexpected interruptions in 54 days. So training saves frequent **checkpoints** and resumes from the latest one after a failure.

## 3.8 Summary

- FP32 / FP16 / BF16 / FP8: fewer bits are faster and leaner, at the cost of precision and range; BF16 has FP32's range and is today's standard.
- Mixed precision: matrix maths in BF16, updates on FP32 master weights (small updates are lost in BF16); FP16 needs loss scaling.
- Gradient accumulation: add up the gradients of several micro-batches before updating — time for memory.
- Data parallelism: a full model on every GPU, different data, gradients averaged with all-reduce; in a ring all-reduce each GPU sends only 2(n−1)/n times the data.
- ZeRO / FSDP: shard optimiser state, gradients and parameters across GPUs; with ZeRO-3 each holds 1/n.
- Tensor parallelism splits a layer's matrices by rows and columns, with heavy communication, inside a machine; pipeline parallelism splits by layers, with micro-batches to shrink the bubble.
- Activation recomputation trades computation for memory; large-scale training combines several kinds of parallelism and checkpoints often.

## 3.9 Exercises

<!-- exercise: zero-memory -->
### Exercise 1: Memory with ZeRO

Read the parameter count N (billions), the number of GPUs n and the ZeRO stage (0–3). Using the table in section 3.5, print the memory each GPU needs (GB) to 1 decimal place (activations not included).

```text
Sample input: 70 64 3
Sample output: 17.5
```

<details>
<summary>Hint</summary>

Bytes per parameter: 16 for stage 0, `4 + 12 / n` for stage 1, `2 + 14 / n` for stage 2 and `16 / n` for stage 3. With N in billions, `N * bytes` is in GB.
</details>

<!-- exercise: pipeline-bubble -->
### Exercise 2: The pipeline bubble

Read the number of pipeline stages `p` and of micro-batches `m`, and print the fraction of time GPUs sit idle, `(p - 1) / (m + p - 1)`, to 3 decimal places.

```text
Sample input: 8 16
Sample output: 0.304
```

<details>
<summary>Hint</summary>

Just apply the formula: `print(f"{(p - 1) / (m + p - 1):.3f}")`.
</details>

<!-- exercise: ring-traffic -->
### Exercise 3: All-reduce traffic

Read the gradient size S (GB) and the number of GPUs n, and print how much each GPU sends in a ring all-reduce (GB), that is `2(n−1)/n × S`, to 2 decimal places.

```text
Sample input: 16 8
Sample output: 28.00
```

<details>
<summary>Hint</summary>

`print(f"{2 * (n - 1) / n * S:.2f}")`. For large n, the result approaches `2S`.
</details>

<!-- exercise: global-batch -->
### Exercise 4: The global batch size

Read the sequence length, the micro-batch size per GPU (sequences), the number of gradient accumulation steps and the number of data-parallel GPUs. How many tokens does each parameter update process in total? Also give it in millions (M) to 2 decimal places, separated by a space.

```text
Sample input: 8192 2 4 128
Sample output: 8388608 8.39
```

<details>
<summary>Hint</summary>

Tokens = sequence length × micro-batch size × accumulation steps × GPUs. Millions: divide by `1e6`.
</details>

<!-- exercise: why-bf16 -->
### Exercise 5: Why BF16?

What is the main reason BF16 replaced FP16 as the first choice for training large models?

- BF16 is more precise than FP16
- BF16 has as many exponent bits as FP32, so the same range: it rarely overflows or underflows and needs no loss scaling
- BF16 takes only 8 bits
- BF16 doesn't need FP32 master weights

<details>
<summary>Answer and explanation</summary>

BF16 has 8 exponent bits and FP32's range; FP16 has only 5, maxes out at 65504, and lets very small gradients underflow to 0. BF16 is actually **less** precise than FP16 (7 mantissa bits against 10), which is why updates still use FP32 master weights.
</details>

<!-- exercise: zero3 -->
### Exercise 6: What ZeRO-3 shards

On top of data parallelism, what does ZeRO-3 (or FSDP) shard?

- only the data
- only the optimiser state
- the optimiser state, the gradients and the parameters, across all GPUs
- the model's layers, across different GPUs

<details>
<summary>Answer and explanation</summary>

ZeRO-1 shards the optimiser state, ZeRO-2 adds the gradients, and ZeRO-3 shards the parameters too, so each GPU stores 1/n and gathers what it needs on the fly. Splitting by layers is pipeline parallelism.
</details>

## 3.10 Further reading

- Micikevicius et al., *Mixed Precision Training* (2018)
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2019)
- Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2019)
- Hugging Face, *The Ultra-Scale Playbook: Training LLMs on GPU Clusters* (2025) — an illustrated guide to parallel training
- Next lesson: [Modern architectures](../04-architectures/en.md)
