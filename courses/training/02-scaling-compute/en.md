# 2. Scaling laws and compute

> 🎯 **Goals**: count a Transformer's parameters; use **C ≈ 6ND** to estimate how much computation training takes, and how many GPUs for how many days; understand **scaling laws** — loss falls as a power law in parameters, data and compute; understand **Chinchilla**'s conclusion (about 20 tokens per parameter) and why today's models are often "over-trained"; estimate the memory needed for training and inference; meet "test-time compute", a new direction for scaling.
>
> ⏱ **Time**: 70 minutes　|　📋 **You need**: lesson 1 of this part; Part 3, lessons 5 and 6

Training a large model costs millions to hundreds of millions of dollars. Before spending that, researchers need answers: how big should the model be? How much data? How many GPUs, for how long? And roughly how good will it be? This lesson's tools let you answer with pencil and paper (and a few lines of Python).

## 2.1 Counting parameters

In the Transformer of Part 3, lesson 5, the parameters live mainly in two places: attention and the feed-forward network (MLP). With model dimension d and L layers:

- attention: four d × d matrices for Q, K, V and the output, **4d²** in total;
- MLP: two matrices, d → 4d → d, **8d²** in total;
- so each layer has about **12d²**, plus a V × d embedding matrix for a vocabulary of size V.

Modern Llama-style models differ a little: K and V use **grouped-query attention** (GQA, lesson 4) with fewer heads; the MLP is a SwiGLU with three matrices; and the input embedding and output layer are usually not shared. Let's count both:

```python
def gpt2_params(L, d, V, ctx):
    per_layer = 12 * d * d + 13 * d          # the 12d² matrices, plus biases and LayerNorms
    return L * per_layer + V * d + ctx * d + 2 * d

def llama_params(L, d, n_heads, n_kv, ffn, V):
    head_dim = d // n_heads
    attn = 2 * d * d + 2 * d * n_kv * head_dim    # Q and output are d×d; K and V have only n_kv heads
    mlp = 3 * d * ffn                             # SwiGLU: gate, up and down projections
    per_layer = attn + mlp + 2 * d                # plus two RMSNorms
    return L * per_layer + 2 * V * d + d          # input embedding + output layer (not shared) + final RMSNorm

print(f"GPT-2 small : {gpt2_params(12, 768, 50257, 1024) / 1e6:.0f}M")
print(f"GPT-2 XL    : {gpt2_params(48, 1600, 50257, 1024) / 1e9:.2f}B")
print(f"Llama 3 8B  : {llama_params(32, 4096, 32, 8, 14336, 128256) / 1e9:.2f}B")
print(f"Llama 3 70B : {llama_params(80, 8192, 64, 8, 28672, 128256) / 1e9:.2f}B")
```

The results match the official parameter counts exactly (124M, 1.5B, 8.0B, 70.6B). Notice that the bigger the model, the more its parameters concentrate in the per-layer matrices (which grow as d²) and the smaller the embedding's share.

## 2.2 Compute: C ≈ 6ND

Training a model with N parameters on D tokens takes about this much computation (floating-point operations, FLOPs):

```formula
C ≈ 6 · N · D
```

Why 6? For each parameter and each token:

- the **forward pass** does one multiplication and one addition → 2 operations;
- the **backward pass** computes gradients with respect to both the inputs and the parameters, about twice the forward work → 4 operations.

(Attention's own computation depends on sequence length and is a small share unless sequences are very long; it's ignored here.) Likewise, **inference** costs about **2N** operations per generated token.

With this formula you can estimate training time:

```python
def train_days(N, D, gpus, peak_tflops, mfu):
    flops = 6 * N * D
    per_second = gpus * peak_tflops * 1e12 * mfu      # operations the cluster really completes per second
    return flops / per_second / 86400

H100 = 989            # an H100's peak dense BF16 matrix throughput, about 989 TFLOPS
print(f"8B parameters × 15T tokens: {6 * 8e9 * 15e12:.1e} FLOPs")
for N, D, gpus in [(8e9, 15e12, 1024), (70e9, 15e12, 4096), (405e9, 15e12, 16384)]:
    days = train_days(N, D, gpus, H100, 0.40)
    print(f"{N / 1e9:>4.0f}B params, {D / 1e12:.0f}T tokens, {gpus:>5} H100s: about {days:5.1f} days")
```

- **MFU** (Model FLOPs Utilization): a GPU's peak is only a theoretical figure. Real training waits for data and communication and does all sorts of housekeeping; **30%–50%** is considered good.
- Meta trained Llama 3 405B on **16,000 H100s**; DeepSeek reports that DeepSeek-V3's full training took about **2.79 million H800 GPU hours**. These are the same order of magnitude as our estimates.

## 2.3 Scaling laws

In 2020, Kaplan et al. at OpenAI found a striking regularity: across a huge range (several orders of magnitude), a language model's test loss falls smoothly as a **power law** in the parameter count N, the amount of data D and the compute C. On log-log axes it's almost a straight line — make things bigger and results improve **predictably**. These are the **scaling laws**, and they're why companies dare to invest so much in ever larger models.

In 2022, Hoffmann et al. at DeepMind (the **Chinchilla** paper) trained over 400 models and wrote the loss as a function of parameters and data:

```formula
L(N, D) = E + A / N^α + B / D^β
```

- E is the **irreducible loss**: the inherent randomness of language, which no model can remove;
- A / N^α: the loss from a model that's **too small**;
- B / D^β: the loss from **too little data**.

Using the coefficients from Besiroglu et al.'s 2024 refit of the Chinchilla data, let's see how to split a **fixed amount of compute** between parameters and data:

```python
E, A, B, alpha, beta = 1.82, 482.01, 2085.43, 0.3478, 0.3658

def loss(N, D):
    return E + A / N**alpha + B / D**beta

def best_split(C):                         # fixed compute C: try many N, with D set by C = 6ND
    candidates = [10 ** (k / 100) for k in range(700, 1300)]
    return min(candidates, key=lambda N: loss(N, C / (6 * N)))

print("compute C     best params N   best data D   D/N   predicted loss")
for C in (1e21, 1e22, 1e23, 5.76e23, 1e24, 1e25):
    N = best_split(C)
    D = C / (6 * N)
    print(f"{C:9.1e}   {N / 1e9:10.1f}B   {D / 1e9:10.0f}B  {D / N:5.1f}   {loss(N, D):.3f}")

print()
print(f"Gopher     280B params × 0.3T tokens: loss {loss(280e9, 0.3e12):.3f}")
print(f"Chinchilla  70B params × 1.4T tokens: loss {loss(70e9, 1.4e12):.3f}")
```

The conclusions:

- As compute grows, **parameters and data should grow roughly in proportion**; at the optimum there are about **20 tokens per parameter**.
- Before this, people mostly grew the parameter count. DeepMind's own Gopher (280B parameters, 300B tokens) was actually **under-trained**. With the same compute (the 5.8e+23 row of the table), a model four times smaller — Chinchilla, 70B parameters on 1.4T tokens — did better.

## 2.4 Why today's models are "over-trained"

Chinchilla's optimum only counts the cost of **training**. But a model trained once is used millions of times, and the cost of **inference** is proportional to its parameter count. So the common practice now is to train a model smaller than "optimal" on far more than 20 times as many tokens — spending a bit more on training to make every future inference cheaper.

```python
E, A, B, alpha, beta = 1.82, 482.01, 2085.43, 0.3478, 0.3658

def loss(N, D):
    return E + A / N**alpha + B / D**beta

N = 8e9
print("An 8B model reading more and more data:")
for D in (160e9, 1e12, 5e12, 15e12, 36e12):
    print(f"  {D / 1e12:5.2f}T tokens ({D / N:5.0f} per parameter): predicted loss {loss(N, D):.3f}")
```

Llama 3 8B read 15 trillion tokens, nearly 100 times the "optimum", and its loss was still going down. (These numbers are **extrapolated** from the formula and only show the trend: the models used to fit it were much smaller.)

## 2.5 Memory: why one GPU can't train a large model

How much GPU memory does each parameter need during training? With the common **BF16 mixed precision + Adam optimiser**:

| Item | Bytes per parameter |
|------|---------------------|
| BF16 parameters (for the forward and backward passes) | 2 |
| BF16 gradients | 2 |
| FP32 "master" parameters (for the update, to keep precision) | 4 |
| Adam's first moment m (FP32) | 4 |
| Adam's second moment v (FP32) | 4 |
| **Total** | **16** |

```python
import math

def train_memory_gb(N):
    return 16 * N / 1e9                      # activations not included

def infer_memory_gb(N, bytes_per_param=2):
    return bytes_per_param * N / 1e9         # parameters only (no KV cache — see Part 6)

for N in (1e9, 8e9, 70e9, 405e9):
    need = train_memory_gb(N)
    print(f"{N / 1e9:>4.0f}B: training about {need:>6.0f} GB (at least {math.ceil(need / 80):>3} 80GB GPUs), "
          f"BF16 inference about {infer_memory_gb(N):>4.0f} GB")
```

That doesn't even count **activations** — the intermediate results of the forward pass, which the backward pass needs. They grow with batch size and sequence length and often take more room than the parameters. So a single 80GB GPU can't hold the training state of an 8B model, and a 70B model needs at least a dozen or so. The next lesson shows how models and data are spread across hundreds or thousands of GPUs.

## 2.6 Emergence, test-time compute and new directions

- **Emergent abilities**: some abilities (such as multi-digit arithmetic) seem to "appear suddenly" once a model reaches a certain size. But Schaeffer et al. (2023) pointed out that this depends largely on the metric: with "all or nothing" scoring the ability looks like a jump; with a continuous metric (say, the accuracy of each digit) the improvement is actually smooth.
- **Test-time compute**: from 2024, **reasoning models** such as OpenAI o1 and DeepSeek-R1 showed another way to scale: not only bigger training, but letting the model "think longer" before answering by generating a long chain of reasoning. The more it thinks, the more hard problems it gets right — a new kind of scaling law (Part 5 covers this in detail).
- **The data wall**: high-quality human text is finite and, at the current pace, may run out within a few years. This drives research into synthetic data, multimodal data and more efficient training.

## 2.7 Summary

- About 12d² parameters per layer (GPT style); Llama-style models also need GQA, SwiGLU and unshared embeddings taken into account.
- Training compute C ≈ 6ND; inference ≈ 2N per token; training days = 6ND ÷ (GPUs × peak throughput × MFU).
- Scaling laws: loss falls smoothly as a power law in N, D and C; L(N, D) = E + A/N^α + B/D^β.
- Chinchilla: with fixed compute, grow N and D in proportion, about 20 tokens per parameter; to cut inference costs, today's models often go far beyond that ratio.
- Training memory is about 16 bytes per parameter (BF16 + Adam), plus activations; BF16 inference about 2 bytes per parameter.
- New directions: test-time compute and synthetic data.

## 2.8 Exercises

<!-- exercise: param-count -->
### Exercise 1: Counting parameters

Read the number of layers `L`, the model dimension `d` and the vocabulary size `V`. Estimate the parameter count as "12d² per layer plus a V × d embedding" (ignoring biases, LayerNorms and position embeddings), and print it in millions with 1 decimal place followed by `M`.

```text
Sample input: 12 768 50257
Sample output: 123.5M
```

<details>
<summary>Hint</summary>

`params = 12 * L * d * d + V * d`, printed as `f"{params / 1e6:.1f}M"`.
</details>

<!-- exercise: train-days -->
### Exercise 2: How many days of training?

Read one line: the parameter count N (billions), the number of training tokens D (billions), the number of GPUs, each GPU's peak throughput (TFLOPS) and the MFU (a decimal between 0 and 1). Estimate the training time in days with C = 6ND, to 1 decimal place.

```text
Sample input: 8 15000 1024 989 0.4
Sample output: 20.6
```

<details>
<summary>Hint</summary>

Convert the units first: multiply N and D by `1e9` and the throughput by `1e12`. Days = `6 * N * D / (gpus * tflops * 1e12 * mfu) / 86400`.
</details>

<!-- exercise: chinchilla-opt -->
### Exercise 3: The Chinchilla optimum

Read a compute budget C (in FLOPs, possibly written like `1e23`). Using the two rules "C = 6ND" and "D = 20N", find the optimal parameter count N and amount of data D, and print both in billions with 1 decimal place, separated by a space.

```text
Sample input: 1e23
Sample output: 28.9 577.4
```

<details>
<summary>Hint</summary>

Substituting D = 20N into C = 6ND gives C = 120N², so `N = (C / 120) ** 0.5`. `float("1e23")` reads that notation directly.
</details>

<!-- exercise: adam-memory -->
### Exercise 4: How many GPUs for training?

Read the parameter count N (billions) and each GPU's memory (GB). At 16 bytes per parameter (BF16 mixed precision + Adam, activations not counted), print how many GB the training state needs (as an integer) and the minimum number of GPUs, separated by a space.

```text
Sample input: 70 80
Sample output: 1120 14
```

<details>
<summary>Hint</summary>

`need = 16 * N` (with N in billions, the result is in GB). Round the number of GPUs up with `math.ceil(need / mem)`.
</details>

<!-- exercise: six-nd -->
### Exercise 5: The 6 in 6ND

Where does the "6" in the training compute formula C ≈ 6ND come from?

- it's the number of matrices in each Transformer layer
- per parameter and per token: about 2 operations in the forward pass and about 4 in the backward pass
- it's an empirical number with no reason behind it
- Adam stores 6 numbers per parameter

<details>
<summary>Answer and explanation</summary>

In the forward pass each parameter takes part in one multiplication and one addition (2 operations); the backward pass computes gradients for both the inputs and the parameters, about twice as much (4 operations). That's 6 in total. Inference is only a forward pass, hence about 2N per token.
</details>

<!-- exercise: chinchilla-lesson -->
### Exercise 6: Chinchilla's conclusion

What did the Chinchilla paper conclude about a fixed compute budget?

- nearly all of it should go into making the model bigger
- the model should be as small as possible, with as much data as possible
- parameters and training tokens should grow roughly in proportion; earlier large models had been trained on too little data
- the amount of data has no effect on the loss

<details>
<summary>Answer and explanation</summary>

Chinchilla found that the optimum grows N and D in proportion (about 20 tokens per parameter). Earlier models such as Gopher had many parameters and too little data; for the same compute, a smaller model with more data did better.
</details>

## 2.9 Further reading

- Kaplan et al., *Scaling Laws for Neural Language Models* (2020)
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022, Chinchilla)
- Besiroglu et al., *Chinchilla Scaling: A replication attempt* (2024)
- Schaeffer et al., *Are Emergent Abilities of Large Language Models a Mirage?* (2023)
- Next lesson: [Distributed and mixed-precision training](../03-distributed/en.md)
