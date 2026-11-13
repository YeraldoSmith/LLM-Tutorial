# 4. Modern architectures: faster, leaner, longer

> 🎯 **Goals**: build on Part 3, lesson 5 (Llama-style Transformers, an introduction to MoE): understand how Pre-Norm, RMSNorm and QK-Norm make training more stable; compare MHA, MQA, GQA and DeepSeek's **MLA**, and calculate the size of the **KV cache**; understand long-context techniques (RoPE interpolation, YaRN, sliding windows, attention sinks); implement the heart of **FlashAttention** — the **online softmax** — yourself; understand **load balancing** in MoE; meet new directions such as multi-token prediction, linear attention and hybrid architectures.
>
> ⏱ **Time**: 90 minutes　|　📋 **You need**: Part 3, lessons 4 and 5; lessons 2 and 3 of this part

The Transformer of 2017 and the strongest models of 2026 have almost the same overall shape: embeddings → many "attention + MLP" blocks → an output layer. But nearly every part has been improved, always towards three goals: **more stable training**, **faster and cheaper inference**, and **longer context**.

## 4.1 Normalisation: Pre-Norm, RMSNorm and QK-Norm

The original Transformer put LayerNorm **after** the residual addition (Post-Norm). Deep models trained that way easily diverged. From GPT-2 on, it moved to **before** each sub-layer (Pre-Norm):

```text
Post-Norm: x = Norm(x + Attention(x))        Pre-Norm: x = x + Attention(Norm(x))
```

Pre-Norm turns the residual connections into a "clean highway" along which gradients flow freely to the bottom layers, so even models hundreds of layers deep train stably.

**RMSNorm** (standard since Llama) is simpler than LayerNorm: it doesn't subtract the mean, it just divides by the root mean square and multiplies by a learnable scale γ:

```python
import math

def layer_norm(x, eps=1e-6):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    return [(v - mean) / math.sqrt(var + eps) for v in x]

def rms_norm(x, eps=1e-6):
    rms = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v / rms for v in x]

x = [2.0, -1.0, 4.0, 3.0]
print("LayerNorm:", [round(v, 3) for v in layer_norm(x)])
print("RMSNorm:  ", [round(v, 3) for v in rms_norm(x)])
print("RMS after RMSNorm:", round(math.sqrt(sum(v * v for v in rms_norm(x)) / len(x)), 3))
```

Skipping one mean looks trivial, but normalisation happens several times in every layer, so the time saved adds up — with practically the same results.

**QK-Norm**: when training very large models, the attention logits (q·k) sometimes grow and grow, softmax becomes extreme, and training collapses. Applying RMSNorm to q and k before the dot product keeps them in check. Gemma 3, Qwen3, OLMo 2 and others use QK-Norm.

## 4.2 Attention variants and the KV cache

When generating text, every new token needs the K and V of all earlier tokens. To avoid recomputing them they're stored — the **KV cache** (covered in detail in Part 6). Its size is:

```formula
KV cache (per token) = 2 × layers × KV heads × dimension per head × bytes per number
```

With long contexts and many users served at once, the KV cache often takes more memory than the model's parameters. Hence several ways to shrink K and V:

- **MHA** (multi-head attention): every query head has its own K and V heads.
- **MQA** (multi-query attention, 2019): all query heads **share one** set of K and V. The smallest cache, but quality drops somewhat.
- **GQA** (grouped-query attention, 2023): query heads are split into groups, each sharing one set of K and V. Llama 3 70B has 64 query heads and 8 KV heads. Quality close to MHA with an 8× smaller cache — today's most common choice.
- **MLA** (multi-head latent attention, DeepSeek-V2, 2024): instead of caching K and V directly, it caches a compressed **low-dimensional "latent" vector** (512 dimensions per layer in DeepSeek-V3, plus 64 for RoPE) and reconstructs each head's K and V from it when needed.

```python
def kv_cache_gb(layers, values_per_layer, tokens, bytes_per_value=2):
    return layers * values_per_layer * tokens * bytes_per_value / 1e9

T = 128_000                                       # one context of 128K tokens, in BF16
configs = [
    ("MHA, 64 KV heads", 80, 2 * 64 * 128),       # 80 layers, 64 heads of 128 dims each for K and V
    ("GQA, 8 KV heads", 80, 2 * 8 * 128),         # Llama 3 70B's setting
    ("MQA, 1 KV head", 80, 2 * 1 * 128),
    ("MLA (DeepSeek-V3)", 61, 512 + 64),          # 61 layers, 576 numbers cached per layer
]
for name, layers, per_layer in configs:
    kb = layers * per_layer * 2 / 1024
    print(f"{name:<18} {kb:6.1f} KB per token, {kv_cache_gb(layers, per_layer, T):6.1f} GB for 128K tokens")
```

GQA shrinks the cache 8× compared with MHA; MLA gets a 671-billion-parameter model down to under 70 KB per token.

## 4.3 Long context

In early 2023 most models handled 2K–4K tokens; by 2025, 128K was standard and many models supported a million tokens or more. That takes several techniques.

### RoPE extrapolation, and interpolation

As Part 3, lesson 4 explained, RoPE rotates dimension pair i at speed `θᵢ = base^(−2i/d)`. The slowest pair only turns through a small angle within the training length. Used directly on a longer context, it turns to **angles never seen in training**, and the model is lost:

```python
import math

d, base = 128, 10_000
theta_slow = base ** (-(d - 2) / d)          # the slowest-turning pair
train_len, target_len = 8_192, 32_768

print(f"training length {train_len}: the slowest pair turns at most {train_len * theta_slow:.2f} radians")
print(f"used directly at {target_len}: it turns to {target_len * theta_slow:.2f} radians (never seen in training)")

scale = train_len / target_len               # position interpolation: "squeeze" positions back into the trained range
print(f"after position interpolation: at most {target_len * scale * theta_slow:.2f} radians")

new_base = base * (target_len / train_len) ** (d / (d - 2))   # "NTK" style: a bigger base slows the slow pairs down
theta_new = new_base ** (-(d - 2) / d)
print(f"with base changed to {new_base:,.0f}: at most {target_len * theta_new:.2f} radians")
```

- **Position interpolation** (2023): multiply positions by a scale factor so the angles return to the trained range, then fine-tune briefly on long texts.
- **NTK scaling, YaRN** (2023): enlarge the base, or treat frequencies differently — keep the fast pairs (which tell neighbouring tokens apart) as they are and adjust only the slow ones. Models today often use YaRN to extend their context 4–32×.
- In training, models are usually pre-trained on shorter sequences (say 8K) and finish with a dedicated "long-context phase" on long documents (that's how Llama 3 reached 128K).

### Sliding windows and attention sinks

- **Sliding-window attention**: each token only looks at the most recent W tokens (say 4096), so computation and cache stop growing with the context. Mistral 7B used this; Gemma 3 alternates "local" (sliding-window) and "global" layers 5:1 — most layers look only nearby, a few look at everything — which shrinks the KV cache a lot.
- **Attention sinks** (2023): researchers found that models give a lot of attention to **the first few tokens**, even when they mean little — a side effect of softmax having to make the weights add up to 1. If a sliding window drops those first tokens the model breaks down; keeping a few of them permanently lets it handle very long streams stably.

Being able to "fit" a long context isn't the same as using it well. Research found models make the least use of information placed in the **middle** of a long context ("lost in the middle"). Lesson 5 shows how long-context ability is evaluated.

## 4.4 FlashAttention: exact, but leaner

Attention computes an n × n matrix of scores. At n = 128K that's 16 billion entries. Computed directly, it has to be written to the GPU's memory (HBM) and read back — and a GPU computes very fast; **what's slow is moving data**.

**FlashAttention** (Tri Dao et al., 2022) cuts Q, K and V into small blocks that fit in the GPU chip's small, fast on-chip memory (SRAM) and computes **block by block**, never writing out the n × n matrix. The difficulty is that softmax needs the maximum and the sum of **all** the scores. The answer is the **online softmax**: as each new block of scores arrives, update "the maximum so far" and "the sum so far", adjusting the earlier results with a correction factor:

```python
import math

scores = [2.0, -1.0, 0.5, 3.0, 1.0, -0.5, 2.5, 0.0]       # one query's scores against 8 keys
values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]  # each key's value (one-dimensional here)

# the usual way: compute all scores, softmax, then a weighted average
m = max(scores)
w = [math.exp(s - m) for s in scores]
standard = sum(wi * v for wi, v in zip(w, values)) / sum(w)

# the online way: 3 scores at a time, never holding all of them at once
run_max, run_sum, run_out = float("-inf"), 0.0, 0.0
for start in range(0, len(scores), 3):
    block_s, block_v = scores[start:start + 3], values[start:start + 3]
    new_max = max(run_max, max(block_s))
    fix = math.exp(run_max - new_max)                     # the maximum changed: scale the old totals down
    run_sum = run_sum * fix + sum(math.exp(s - new_max) for s in block_s)
    run_out = run_out * fix + sum(math.exp(s - new_max) * v for s, v in zip(block_s, block_v))
    run_max = new_max
    print(f"after {start + len(block_s)} scores: max so far {run_max}, result so far {run_out / run_sum:.4f}")

print(f"usual way: {standard:.6f}, online way: {run_out / run_sum:.6f}")
```

The results are **identical** — FlashAttention isn't an approximation but a smarter order for the same computation. It cuts the extra memory attention needs from O(n²) to O(n) and runs several times faster. FlashAttention-2 and -3 optimise further for newer GPUs, and nearly every training and inference framework uses it (PyTorch's `scaled_dot_product_attention` calls it automatically).

## 4.5 Load balancing in MoE

Part 3, lesson 5 introduced MoE: a router picks the top-scoring experts for each token. The biggest headache in training MoE is **load imbalance**: at first a few experts are slightly better, so the router sends them more tokens, so they learn faster and get still more tokens… until most experts sit idle, the model degenerates into a small one, and the GPUs holding the popular experts are overwhelmed.

A common fix is an **auxiliary loss** (Switch Transformer, 2021). Let fᵢ be the fraction of tokens sent to expert i, Pᵢ the router's average probability for expert i, and E the number of experts:

```formula
auxiliary loss = E · Σᵢ fᵢ · Pᵢ
```

It equals 1 when perfectly balanced and grows with imbalance. Added to the total loss, it pushes the model towards balance:

```python
import math
import random

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

random.seed(0)
E, T = 4, 1000                                       # 4 experts, 1000 tokens
for name, bias in [("biased to expert 0", [2.0, 0, 0, 0]), ("balanced", [0, 0, 0, 0])]:
    counts, prob_sum = [0] * E, [0.0] * E
    for _ in range(T):
        p = softmax([random.gauss(0, 1) + b for b in bias])   # the router's probabilities for this token
        counts[p.index(max(p))] += 1                          # top-1: send it to the most probable expert
        prob_sum = [s + pi for s, pi in zip(prob_sum, p)]
    f = [c / T for c in counts]
    P = [s / T for s in prob_sum]
    aux = E * sum(fi * pi for fi, pi in zip(f, P))
    print(f"{name}: share of tokens per expert {[round(v, 2) for v in f]}, auxiliary loss {aux:.3f}")
```

DeepSeek improved on this in several ways:

- **Fine-grained experts + shared experts** (DeepSeekMoE, 2024): cut the experts smaller and more numerous (DeepSeek-V3 has 256 routed experts per layer, with 8 chosen per token), plus a **shared expert** every token passes through, for general knowledge;
- **Auxiliary-loss-free balancing** (DeepSeek-V3): an auxiliary loss interferes with the main training objective, so instead each expert's routing score gets a bias: lowered when the expert is too busy, raised when it's too idle. The bias only affects *which* experts are chosen, not how their outputs are weighted.

## 4.6 More new directions

- **Multi-token prediction** (MTP): during training, DeepSeek-V3 not only predicts the next token at each position but uses extra small modules to predict the token after that too. This gives a denser training signal, and at inference time those modules can speed generation up (speculative decoding, Part 6).
- **Linear attention and state-space models**: ordinary attention's cost grows with the square of the context length. **State-space models** such as Mamba (2023) and various **linear attention** schemes bring it down to linear, needing only a fixed-size "state" during generation instead of an ever-growing KV cache. But they usually fall short of ordinary attention on tasks that need "exact recall" of the context.
- **Hybrid architectures**: so many new models mix the two: most layers use linear attention or state-space layers, with an ordinary attention layer every few layers. Jamba (2024), MiniMax-01 (2025), Qwen3-Next (2025) and others take this approach, getting both speed and quality on very long contexts.

## 4.7 Summary

- Pre-Norm keeps deep models stable; RMSNorm is cheaper; QK-Norm stops attention logits blowing up.
- KV cache per token = 2 × layers × KV heads × head dimension × bytes; GQA lets several query heads share K and V, and MLA caches one compressed latent vector.
- Long context: position interpolation, NTK and YaRN adjust RoPE; train short first, then long; sliding windows and local/global interleaving cut costs; keep the attention sinks.
- FlashAttention: block-wise computation + online softmax; exact results, memory down from O(n²) to O(n), and faster.
- MoE load balancing: an auxiliary loss E·Σfᵢ·Pᵢ, or DeepSeek's bias adjustment; fine-grained plus shared experts.
- New directions: multi-token prediction, linear attention and state-space models, hybrid architectures.

## 4.8 Exercises

<!-- exercise: rmsnorm -->
### Exercise 1: RMSNorm

Read one line of numbers (a vector) and print it after RMSNorm (γ all 1, eps = 1e-6), each number to 4 decimal places, separated by spaces.

```text
Sample input: 2 -1 4 3
Sample output: 0.7303 -0.3651 1.4606 1.0954
```

<details>
<summary>Hint</summary>

`rms = math.sqrt(sum(v * v for v in x) / len(x) + 1e-6)`, then divide each number by `rms`.
</details>

<!-- exercise: kv-cache -->
### Exercise 2: How big is the KV cache?

Read one line: layers, KV heads, dimension per head, number of tokens in the context, bytes per number. Print the size of the KV cache for this context in GB (1 GB = 10⁹ bytes) to 2 decimal places.

```text
Sample input: 32 8 128 8192 2
Sample output: 1.07
```

<details>
<summary>Hint</summary>

`2 * layers * KV heads * head dim * tokens * bytes / 1e9`; the 2 is there because K and V are both stored.
</details>

<!-- exercise: online-softmax -->
### Exercise 3: Online softmax

The first line holds one query's attention scores against the keys, the second the corresponding values (each value is one number), and the third the block size B. As in section 4.4, process B scores at a time and use the online softmax to compute the attention output (the weighted average), to 6 decimal places.

```text
Sample input:
2 -1 0.5 3 1 -0.5 2.5 0
10 20 30 40 50 60 70 80
3
Sample output: 44.332136
```

<details>
<summary>Hint</summary>

Keep three quantities: `run_max`, `run_sum` and `run_out`. For each block, find the new maximum, then correct the old `run_sum` and `run_out` with `math.exp(run_max - new_max)`. At the start `run_max` is minus infinity, and `math.exp(float("-inf"))` is 0, which works out exactly right.
</details>

<!-- exercise: moe-balance -->
### Exercise 4: The MoE auxiliary loss

The first line holds the number of tokens T and of experts E; each of the next T lines holds the router's E probabilities for one token. Each token goes to its most probable expert (top-1; on a tie, the lower number). Print the auxiliary loss `E · Σ fᵢ · Pᵢ` from section 4.5 to 4 decimal places.

```text
Sample input:
4 2
0.9 0.1
0.8 0.2
0.6 0.4
0.3 0.7
Sample output: 1.1500
```

<details>
<summary>Hint</summary>

fᵢ = tokens sent to expert i / T; Pᵢ = the average of every token's i-th probability. `p.index(max(p))` gives the most probable expert (and the lowest-numbered on a tie).
</details>

<!-- exercise: gqa-why -->
### Exercise 5: Why GQA?

Compared with standard multi-head attention (MHA), what is GQA's main benefit?

- each query head can see more tokens
- several query heads share one set of K and V, so the KV cache is several times smaller and inference faster, with almost no loss of quality
- position encodings are no longer needed
- attention's cost drops from O(n²) to O(n)

<details>
<summary>Answer and explanation</summary>

GQA groups the query heads, with each group sharing one set of K and V, so far fewer K and V need caching (8× fewer in Llama 3 70B). It doesn't change attention's O(n²) computation or affect position encoding.
</details>

<!-- exercise: flash-exact -->
### Exercise 6: Is FlashAttention an approximation?

Which statement about FlashAttention is true?

- it computes only some of the attention scores, so its results are approximate
- its results are identical to ordinary attention; it just reduces memory traffic with blocks and an online softmax
- it replaces softmax with a faster function
- it only works for inference, not training

<details>
<summary>Answer and explanation</summary>

FlashAttention is exact: it changes the order of computation, never writing the n × n score matrix to memory but computing in blocks and combining results with an online softmax. It's used for both training and inference.
</details>

## 4.9 Further reading

- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)
- DeepSeek-AI, *DeepSeek-V2* (2024, MLA and DeepSeekMoE) and *DeepSeek-V3 Technical Report* (2024)
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023)
- Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (2023)
- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023)
- Next lesson: [Evaluation](../05-evaluation/en.md)
