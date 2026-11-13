# 2. LoRA and QLoRA: training only a small part

> 🎯 **Goals**: see why full fine-tuning is expensive; understand the idea of **parameter-efficient fine-tuning** (PEFT); master how **LoRA** works — freeze the original weights and train only two small matrices B and A, so the weight change is `(α/r)·B·A`; understand why low rank is enough, why B starts at 0, and how weights are **merged** after training; implement a LoRA layer yourself in PyTorch; understand how **QLoRA** squeezes the base model into 4 bits so that one GPU can fine-tune a large model; use the PEFT library, and know LoRA's usual settings and limits.
>
> ⏱ **Time**: 70 minutes　|　📋 **You need**: lesson 1 of this part; Part 2, lesson 5; Part 4, lessons 2 and 3

Last lesson's SFT updates **all** of a model's parameters. As Part 4, lesson 2 worked out, training with Adam takes 16 bytes of memory per parameter — 112 GB for a 7B model, far beyond a consumer GPU. And fine-tuning a version for each of 10 different tasks means storing 10 complete models.

## 2.1 Parameter-efficient fine-tuning

The idea of **parameter-efficient fine-tuning** (PEFT): **freeze** the original model and train only a small number of new parameters. Then:

- gradients and optimiser state are only kept for those few parameters, which saves a great deal of memory;
- each task needs only its small set of parameters stored (a few MB to a few hundred MB), not a whole model;
- the original model stays intact, and you can switch between tasks at will.

Of the many PEFT methods, the most popular is **LoRA** (Low-Rank Adaptation), proposed by Microsoft in 2021.

## 2.2 How LoRA works

Fine-tuning a linear layer amounts to changing its weight W into W + ΔW. LoRA's key assumption: **ΔW needn't be an arbitrary matrix — a "low-rank" one is enough**. So ΔW is written as the product of two thin matrices:

```formula
W' = W + (α / r) · B · A　　　(W: d_out × d_in, frozen; B: d_out × r; A: r × d_in; r much smaller than d)
```

- **r** is the **rank**, usually 8–64;
- **α** is a scale factor (often 1–2 times r) so that changing r doesn't require re-tuning the learning rate;
- in the forward pass, `W'x = Wx + (α/r)·B(Ax)`: x is squeezed down to r dimensions and expanded back — very little computation.

The trainable parameters go from d_out × d_in to r × (d_in + d_out):

```python
def lora_params(d_in, d_out, r):
    return r * (d_in + d_out)

for d_in, d_out, r in [(4096, 4096, 8), (4096, 4096, 16), (4096, 14336, 16), (8192, 8192, 64)]:
    full = d_in * d_out
    lora = lora_params(d_in, d_out, r)
    print(f"{d_out:>5}×{d_in:<5} matrix, r={r:>2}: full {full:>11,} parameters, LoRA {lora:>9,} ({lora / full:.2%})")
```

Here's the forward pass in plain Python:

```python
def matvec(M, x):
    return [sum(m * v for m, v in zip(row, x)) for row in M]

W = [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]      # the frozen original weight, 2×3
A = [[0.5, -0.5, 1.0]]                       # r = 1, so A is 1×3
B = [[2.0], [-1.0]]                          # and B is 2×1
alpha, r = 2, 1
x = [1.0, 2.0, 3.0]

base = matvec(W, x)
delta = [(alpha / r) * v for v in matvec(B, matvec(A, x))]
print("original output:", base)
print("LoRA correction:", delta)
print("fine-tuned output:", [b + d for b, d in zip(base, delta)])

merged = [[W[i][j] + (alpha / r) * B[i][0] * A[0][j] for j in range(3)] for i in range(2)]
print("with merged weights:", matvec(merged, x))
```

Two things to note:

- **B starts at 0** (A gets small random numbers). So at the start of training B·A = 0 and the model is **exactly** the original: fine-tuning sets off smoothly from it.
- **The weights can be merged after training**: add `(α/r)·B·A` straight into W to get an ordinary model with no extra inference cost. Or keep them separate and switch between different LoRAs per request at run time.

## 2.3 Why is low rank enough?

A 4096 × 4096 matrix holds over 16 million numbers, and an r = 8 LoRA has only 65 thousand trainable parameters — how can that be enough? Research found that the weight change ΔW during fine-tuning tends to concentrate in **a few directions** (its "intrinsic dimension" is low) — after all, fine-tuning only adjusts abilities the model already has.

The singular value decomposition (SVD) shows how many "main directions" a matrix has. Below we build a matrix whose "real change has only 4 directions, plus a little noise":

```python requires=numpy
import numpy as np

rng = np.random.default_rng(0)
d = 256
delta = rng.normal(size=(d, 4)) @ rng.normal(size=(4, d)) + 0.05 * rng.normal(size=(d, d))

U, S, Vt = np.linalg.svd(delta)
print("the 8 largest singular values:", np.round(S[:8], 1))
for r in (1, 2, 4, 8, 32):
    approx = (U[:, :r] * S[:r]) @ Vt[:r]                  # keep only the first r directions
    err = np.linalg.norm(delta - approx) / np.linalg.norm(delta)
    print(f"r = {r:>2}: relative error {err:.3f}")
```

The first 4 singular values are large, then they drop sharply; at r = 4 the error is already down to the noise, and a larger r barely helps. LoRA is betting that real fine-tuning looks like this.

## 2.4 Implementing LoRA yourself

Let's write a LoRA layer in PyTorch: it wraps a frozen `nn.Linear` and trains only A and B. We deliberately make the target weight differ from the original by a change of **rank 1**, and see whether an r = 4 LoRA can learn it:

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
            p.requires_grad = False                                   # freeze the original weight
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))      # B starts at 0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

base = nn.Linear(d, d, bias=False)
u, v = torch.randn(d, 1), torch.randn(1, d)
target_W = base.weight.detach() + 0.5 * (u @ v) / d ** 0.5           # target = original + a rank-1 change
x = torch.randn(2048, d)
y = x @ target_W.T

layer = LoRALinear(base, r=4, alpha=8)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f"trainable parameters {trainable} / total {total}")

opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
for step in range(301):
    loss = nn.functional.mse_loss(layer(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}: loss {loss.item():.6f}")

with torch.no_grad():
    merged = layer.base.weight + layer.scale * layer.B @ layer.A      # merge into an ordinary weight
    print(f"largest error of the merged weight against the target: {(merged - target_W).abs().max().item():.5f}")
```

Training just 512 parameters (the matrix has 4096) brings the loss practically to 0.

## 2.5 Adding LoRA to a large model with PEFT

In practice, Hugging Face's **PEFT** library adds LoRA to chosen layers of a model in a few lines (it downloads the model):

```python norun
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
config = LoraConfig(
    r=16,                                    # rank
    lora_alpha=32,                           # α
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # the linear layers of attention and the MLP
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()           # usually only about one percent is trainable
```

Common settings from experience:

- **Which layers**: the original paper only adapted attention's q and v; later work found that **all linear layers** (the MLP included) work better.
- **r and α**: r = 8–64, α = r or 2r. The harder the task and the further from the original model, the larger r.
- **Learning rate**: an order of magnitude higher than full fine-tuning, typically 1e-4 to 2e-4.
- **The result is tiny**: a LoRA adapter is usually a few tens of MB, easy to share and combine, and serving systems (such as vLLM) can load many LoRAs at once and switch per request.

## 2.6 QLoRA: fine-tuning a large model on one GPU

LoRA cuts the gradients and optimiser state, but the frozen base model still has to fit in memory: at 2 bytes per parameter in BF16, 70 billion parameters take 140 GB. **QLoRA** (Dettmers et al., 2023) goes further and **quantises the frozen base model to 4 bits**:

- **NF4** (4-bit NormalFloat): a 4-bit format designed for "roughly normally distributed weights", more accurate than plain 4-bit integers;
- **double quantisation**: even the scale factors used for quantisation are quantised again, saving more memory;
- **paged optimisers**: when memory runs short, optimiser state is moved temporarily to main memory, preventing crashes at memory peaks;
- LoRA's A and B are still trained in BF16, with the 4-bit weights temporarily restored to BF16 for each computation.

```python
def finetune_memory_gb(N, method, lora_fraction=0.01):
    if method == "full":
        return 16 * N                         # BF16 params + grads + FP32 master and Adam (Part 4, lesson 2)
    base_bytes = {"LoRA": 2, "QLoRA": 0.5}[method]   # the frozen base model: BF16 or 4-bit
    return base_bytes * N + 16 * N * lora_fraction   # plus the LoRA parameters' own training state

for N in (7, 70):                              # in billions of parameters; results in GB (activations not included)
    row = [f"{m} {finetune_memory_gb(N, m):6.1f} GB" for m in ("full", "LoRA", "QLoRA")]
    print(f"{N:>2}B model: " + ", ".join(row))
```

The QLoRA paper fine-tuned a 65-billion-parameter model on **a single 48 GB GPU**, matching 16-bit full fine-tuning. That let individuals and small teams customise large models. The cost is speed: restoring the 4-bit weights for every computation makes training somewhat slower than LoRA.

## 2.7 LoRA's limits and variants

- **Learns less, forgets less**: Biderman et al. (2024) found that on tasks needing lots of new knowledge (such as continued training on large amounts of code or maths), LoRA clearly falls behind full fine-tuning; but LoRA also "forgets" fewer of the original abilities. So LoRA is well suited to adjusting format, style and specific tasks, and less suited to injecting large amounts of new knowledge.
- **Variants**: DoRA (2024) decomposes weights into "magnitude" and "direction" and adjusts them separately; other methods improve the initialisation and learning rates. The principle is always LoRA's: freeze the big, train the small.

## 2.8 Summary

- Full fine-tuning needs lots of memory and a full model per task; PEFT freezes the original model and trains a few new parameters.
- LoRA: W' = W + (α/r)·B·A, cutting trainable parameters from d_out·d_in to r·(d_in + d_out); B starts at 0, so the model is unchanged at the start.
- Fine-tuning's weight changes are usually low-rank, so a small r is enough; after training, BA can be merged into W with no inference cost.
- Typical settings: all linear layers, r = 8–64, α = r–2r, learning rate around 1e-4; adapters are small and easy to switch.
- QLoRA: the base model is quantised to 4-bit NF4, with double quantisation and paged optimisers, so one GPU can fine-tune models with tens of billions of parameters.
- LoRA suits format, style and task adjustments, but falls behind full fine-tuning when injecting lots of new knowledge.

## 2.9 Exercises

<!-- exercise: lora-params -->
### Exercise 1: How many LoRA parameters?

Read a linear layer's `d_in` and `d_out` and the LoRA rank `r`. Print the full matrix's parameter count, LoRA's trainable parameter count, and the latter as a percentage of the former (2 decimal places), separated by spaces.

```text
Sample input: 4096 4096 8
Sample output: 16777216 65536 0.39%
```

<details>
<summary>Hint</summary>

Full: `d_in * d_out`; LoRA: `r * (d_in + d_out)`; percentage: `f"{lora / full:.2%}"`.
</details>

<!-- exercise: lora-forward -->
### Exercise 2: LoRA's forward pass

The first line is `d_out d_in r`; the next `d_out` lines are the frozen weight W; the next `r` lines are A (`d_in` numbers each); the next `d_out` lines are B (`r` numbers each); then a line with α; and finally a line with the input vector x. Print `W x + (α/r)·B(A x)`, each number to 3 decimal places, separated by spaces.

```text
Sample input:
2 3 1
1 0 2
0 1 -1
0.5 -0.5 1
2
-1
2
1 2 3
Sample output: 17.000 -6.000
```

<details>
<summary>Hint</summary>

Write a matrix-times-vector function `matvec(M, x)`. Compute `Ax` (r numbers), then `B(Ax)` (d_out numbers), multiply by `alpha / r` and add it to `Wx`.
</details>

<!-- exercise: model-lora -->
### Exercise 3: LoRA parameters for a whole model

A model has `L` layers and model dimension `d`. Suppose each layer has `m` linear layers of size `d × d` getting LoRA with rank `r`. Print LoRA's total parameter count in millions, to 2 decimal places, followed by `M`.

```text
Sample input: 32 4096 4 16
Sample output: 16.78M
```

<details>
<summary>Hint</summary>

Each d × d layer has `r * (d + d)` LoRA parameters, and there are `L * m` such layers.
</details>

<!-- exercise: finetune-memory -->
### Exercise 4: Memory for three kinds of fine-tuning

Read the parameter count N (billions). Using section 2.6's estimate (full: 16 bytes per parameter; LoRA's base model 2 bytes per parameter and QLoRA's 0.5, each plus a LoRA training state of 1% of the parameters × 16 bytes), print the memory each method needs (GB), each to 1 decimal place, separated by spaces.

```text
Sample input: 7
Sample output: 112.0 15.1 4.6
```

<details>
<summary>Hint</summary>

LoRA: `2 * N + 16 * N * 0.01`; QLoRA: `0.5 * N + 16 * N * 0.01`. With N in billions, the results are in GB.
</details>

<!-- exercise: b-zero -->
### Exercise 5: Why does B start at 0?

In LoRA, B is initialised to 0 and A to small random numbers. What is the main purpose?

- to double training speed
- so that B·A = 0 at the start of training and the model's output is exactly the original's, setting off smoothly from it
- to halve LoRA's parameter count
- because B doesn't need training

<details>
<summary>Answer and explanation</summary>

B·A = 0 means the initial weight is W itself, so fine-tuning doesn't wreck the original model right away. A can't also be 0: then the gradients of both B and A would be 0 and training couldn't start.
</details>

<!-- exercise: qlora-what -->
### Exercise 6: What does QLoRA quantise?

What does QLoRA quantise to 4 bits?

- LoRA's A and B matrices
- the frozen base model's weights; LoRA's A and B are still trained in 16 bits
- the training data
- the gradients

<details>
<summary>Answer and explanation</summary>

QLoRA compresses the base model — which takes the most memory and never needs updating — into 4 bits (NF4); the trainable LoRA parameters stay in BF16, with the 4-bit weights temporarily restored for computation.
</details>

## 2.10 Further reading

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- Biderman et al., *LoRA Learns Less and Forgets Less* (2024)
- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024)
- Hugging Face [PEFT documentation](https://huggingface.co/docs/peft)
- Next lesson: [Reward models and RLHF](../03-rlhf/en.md)
