# 5. The Transformer: putting the blocks together

> 🎯 **Goals**: read the complete diagram of a GPT-style Transformer; understand the two jobs inside a **Block** — attention (tokens communicate) and the MLP (each token computes on its own) — plus the **residual stream** and **Pre-Norm**; write a Block in PyTorch; work out where GPT-2's 124 million parameters come from; tell encoders (BERT), decoders (GPT) and encoder-decoders (T5) apart; meet the modern Llama-style improvements (RMSNorm, SwiGLU, RoPE, GQA) and **mixture of experts** (MoE).
>
> ⏱ **Time**: 90 minutes　|　📋 **You need**: lessons 3–4 of this part; part 2 lessons 3 and 6

We now have every part: embeddings (lesson 3), attention (lesson 4), MLPs (part 2 lesson 3), normalization and residual connections (part 2 lesson 6). This lesson assembles them into a **Transformer** — more precisely, the **decoder-only** Transformer used by the GPT family. Next lesson we'll use it to train a real tiny GPT.

## 5.1 The big picture

```text
input text  "The weather today"
   │  tokenizer
   ▼
token numbers  [464, 6193, 1909]                      (T tokens)
   │  token embedding + position information
   ▼
vectors  (T × d)  ──────────────────────┐
   │                                     │  residual stream
   │   ┌──────────── Block 1 ──────────┐ │  each Block reads it
   │   │ x = x + Attention(Norm(x))    │ │  and "adds" its result back
   │   │ x = x + MLP(Norm(x))          │ │
   │   └───────────────────────────────┘ │
   │          … repeated N times …        │
   │   ┌──────────── Block N ──────────┐ │
   │   └───────────────────────────────┘ │
   ▼                                     │
final Norm  ◄────────────────────────────┘
   │  output layer (d → vocabulary size)
   ▼
logits  (T × V)  →  softmax  →  probability of the "next token" at every position
```

Key points:

- Throughout, every token corresponds to a d-dimensional vector; the shape stays `T × d` until it becomes `T × V` logits at the very end.
- The output at position t predicts token **t+1**. Generation only uses the last position's prediction; **during training every position makes a prediction**, so one forward pass yields T training signals (more next lesson).
- A model's "depth" is the number of Blocks N: GPT-2 small has 12 layers, Llama 3 70B has 80.

## 5.2 The Block: communicate, then think

Each Block has two sublayers, each of the form "**normalize → compute → add back to the residual**":

```formula
x = x + Attention(Norm(x))     ← tokens communicate: gather information from the context
x = x + MLP(Norm(x))           ← each token thinks on its own: process what it gathered
```

1. **Attention** is the only place where tokens at different positions **exchange information** (last lesson).
2. The **MLP** (also called the FFN, feed-forward network) applies the same computation to **each token independently**: `d → 4d → d` with an activation such as GELU in between. Research suggests much of a model's "knowledge" (like "Paris is the capital of France") is stored in the MLP weights.
3. **Residual connections**: each sublayer only computes a "change" that is added to x rather than replacing it. Think of x as a **residual stream** running through all the layers, which every layer reads from and writes into. Gradients can flow straight back to the first layers (part 2 lesson 6), so even networks with a hundred-plus layers train.
4. **Pre-Norm**: normalization is applied to the **input** of each sublayer (not after the addition, as in the original Transformer). It trains more stably and has been standard since GPT-2.

## 5.3 A Block in PyTorch

```python requires=torch
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):           # last lesson's multi-head causal self-attention (PyTorch's fast kernel)
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

class MLP(nn.Module):                           # d → 4d → d
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 4 * d)
        self.proj = nn.Linear(4 * d, d)
    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # ① tokens exchange information
        x = x + self.mlp(self.ln2(x))           # ② each token "thinks" on its own
        return x

torch.manual_seed(0)
block = Block(d=64, n_heads=4)
x = torch.randn(2, 10, 64)                      # 2 sequences of 10 tokens, 64 dimensions each
print("input", tuple(x.shape), "→ output", tuple(block(x).shape))   # same shape, so blocks can be stacked
print("parameters in one Block:", sum(p.numel() for p in block.parameters()), "= 12d² + 13d =", 12 * 64**2 + 13 * 64)
```

The output has exactly the same shape as the input, so any number of Blocks can be stacked.

## 5.4 Where do GPT-2's 124 million parameters come from?

The parameters of one Block (d is the model dimension):

| Part | Parameters |
|------|------------|
| attention: QKV projection `d × 3d` + bias `3d` | 3d² + 3d |
| attention: output projection `d × d` + bias `d` | d² + d |
| MLP: `d × 4d` + `4d`, `4d × d` + `d` | 8d² + 5d |
| two LayerNorms (each with γ and β) | 4d |
| **total** | **12d² + 13d** |

Add the token embedding (`V × d`), the position embedding (`context length × d`) and the final LayerNorm (`2d`); the output layer **shares** the token embedding matrix (weight tying) and adds no parameters:

```python
def gpt2_params(n_layer, d, vocab=50257, ctx=1024):
    emb = vocab * d + ctx * d                       # token embedding + position embedding
    attn = (d * 3 * d + 3 * d) + (d * d + d)        # QKV projection + output projection (with biases)
    mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)     # d → 4d → d (with biases)
    norms = 2 * (2 * d)                             # two LayerNorms, each with γ and β
    per_layer = attn + mlp + norms
    final = 2 * d                                   # final LayerNorm; the output layer shares the token embedding
    return per_layer, emb + n_layer * per_layer + final

for name, n_layer, d in [("GPT-2 small", 12, 768), ("GPT-2 medium", 24, 1024),
                         ("GPT-2 large", 36, 1280), ("GPT-2 XL", 48, 1600)]:
    per_layer, total = gpt2_params(n_layer, d)
    print(f"{name:13s} {n_layer:2d} layers  d={d:4d}  per layer {per_layer / 1e6:5.1f}M  total {total / 1e6:7.1f}M")
```

The results are exactly the 124M, 355M, 774M and 1.5B that OpenAI published. Notice that **the parameters come mainly from `12d²`** — double the dimension and each layer gets about 4 times as many parameters.

## 5.5 Three kinds of Transformer

The original Transformer (2017) was designed for machine translation and had an **encoder** and a **decoder**. It later split into three families:

| Family | Examples | Attention | Training objective | Good at |
|--------|----------|-----------|--------------------|---------|
| **encoder-only** | BERT (2018) | **bidirectional**: every token sees the whole sentence | fill in the blanks: hide some words and have the model guess | understanding: classification, retrieval, sentence embeddings |
| **encoder-decoder** | the original Transformer, T5 | bidirectional encoder, causal decoder that reads the encoder through "cross-attention" | turn an input sequence into an output sequence | translation, summarization |
| **decoder-only** | GPT, Llama, Qwen, DeepSeek, Claude… | **causal**: sees only earlier text | predict the next token | generation; at large scale, almost anything |

Almost all of today's LLMs are **decoder-only**: the design is the simplest and the easiest to scale, and "predict the next token" can be trained on **any text** on the internet, with no human labelling.

## 5.6 Modern improvements: the Llama-style Transformer

Since GPT-2 the overall structure hasn't changed much, but many details have improved. The combination used by Llama (Meta, 2023) became the de facto standard and is widely shared by Qwen, Mistral, DeepSeek and others:

| Component | GPT-2 (2019) | Llama style (2023–today) | Why |
|-----------|--------------|--------------------------|-----|
| normalization | LayerNorm | **RMSNorm** | cheaper, works as well |
| position information | learned position embeddings added to the input | **RoPE** (rotate q and k inside attention) | relative positions; easier to extend the context |
| MLP | GELU, `d → 4d → d` | **SwiGLU** gated MLP | better results for the same parameter count |
| K, V heads | as many as query heads | **GQA**: several query heads share one set of K, V | KV cache several times smaller, faster inference |
| biases | every linear layer has them | usually **no biases** | hardly matters, simpler |
| output layer | shared with the embedding | usually **not shared** in big models | slightly better when parameters are plentiful |

**SwiGLU** is a "gated" MLP: one branch computes a "gate" (passed through SiLU) that multiplies the result of another branch, which is then projected back to d dimensions. With three matrices, the hidden size is usually around `8d/3` so the parameter count matches an ordinary MLP:

```python requires=torch
import torch
from torch import nn
import torch.nn.functional as F

class SwiGLU(nn.Module):                        # the MLP used by Llama, Qwen, DeepSeek and others
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))   # the "gate" times the "content", element-wise

mlp = SwiGLU(d=64, hidden=172)                  # 172 ≈ 8 × 64 / 3
print(tuple(mlp(torch.randn(2, 10, 64)).shape), "parameters:", sum(p.numel() for p in mlp.parameters()))
```

The same method gives Llama 3's parameter count. Llama 3 8B: 32 layers, d = 4096, 32 query heads, 8 KV heads, SwiGLU hidden size 14336, vocabulary 128256:

```python
def llama_params(n_layer, d, n_heads, n_kv_heads, ffn_hidden, vocab):
    head_dim = d // n_heads
    attn = d * d + 2 * d * (n_kv_heads * head_dim) + d * d      # W_Q, W_K and W_V (smaller thanks to GQA), W_O
    mlp = 3 * d * ffn_hidden                                    # SwiGLU's three matrices
    norms = 2 * d                                               # two RMSNorms (γ only)
    per_layer = attn + mlp + norms
    return n_layer * per_layer + 2 * vocab * d + d              # separate input embedding and output layer; final RMSNorm

print(f"Llama 3 8B:  {llama_params(32, 4096, 32, 8, 14336, 128256) / 1e9:.2f}B")
print(f"Llama 3 70B: {llama_params(80, 8192, 64, 8, 28672, 128256) / 1e9:.2f}B")
```

These match the 8.03B and 70.6B Meta published. In Llama-style models **the MLP holds most of the parameters** (about 176 million MLP parameters per layer in the 8B model, against about 42 million for attention).

## 5.7 Mixture of experts (MoE): many parameters, only some used each time

Bigger models are smarter, but every token costs more computation. **Mixture of experts** (MoE) replaces one big MLP with **many small MLPs (experts)** plus a **router**. Each token is sent only to the few experts with the highest scores:

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

# 4 "experts" (in a real model each is an MLP; simple functions stand in here)
experts = [lambda x: [2 * v for v in x],         # expert 0
           lambda x: [v + 1 for v in x],         # expert 1
           lambda x: [-v for v in x],            # expert 2
           lambda x: [v * v for v in x]]         # expert 3
router = [[0.5, -1.0], [1.0, 0.2], [-0.3, 0.8], [0.1, 0.1]]     # the router: one row of weights per expert

def moe(x, top_k=2):
    scores = [sum(w * v for w, v in zip(row, x)) for row in router]
    chosen = sorted(range(len(experts)), key=lambda i: -scores[i])[:top_k]   # pick only the k best
    gates = softmax([scores[i] for i in chosen])                            # weights of the chosen experts
    outs = [experts[i](x) for i in chosen]                                  # only chosen experts compute
    y = [sum(g * o[j] for g, o in zip(gates, outs)) for j in range(len(x))]
    return chosen, [round(g, 3) for g in gates], [round(v, 3) for v in y]

for x in ([1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]):
    print("input", x, "→ chosen experts, weights, output:", moe(x))
```

Different inputs are routed to different experts. The model's **total parameters** can then be huge (storing more knowledge), while the parameters **actually used for each token** (the active parameters) are only a small fraction:

| Model | Total parameters | Active per token |
|-------|------------------|------------------|
| Mixtral 8x7B | about 47 billion | about 13 billion (2 of 8 experts) |
| DeepSeek-V3 / R1 | 671 billion | about 37 billion |
| Qwen3-235B-A22B | 235 billion | about 22 billion |

MoE is now the mainstream choice for frontier models. The difficulties: the experts must share the work evenly (not every token piling onto one expert), and all the experts' parameters still have to fit in memory.

## 5.8 Summary

- A GPT-style Transformer: token embeddings (+ positions) → N Blocks → final normalization → output layer → logits for the next token at every position.
- Block = `x + Attention(Norm(x))`, then `x + MLP(Norm(x))`: attention lets tokens **communicate**, the MLP lets each token **compute** on its own; the residual stream runs through every layer and Pre-Norm keeps training stable.
- About **12d²** parameters per layer; GPT-2 small = 12 layers × d=768 ≈ 124 million parameters.
- Three families: encoder-only (BERT, bidirectional, understanding), encoder-decoder (T5, translation), **decoder-only** (GPT, Llama, Qwen…, today's LLMs).
- Modern improvements: RMSNorm, RoPE, SwiGLU, GQA, no biases; **MoE** routes each token to a few experts, so total parameters are large but computation stays small.

## 5.9 Exercises

<!-- exercise: gpt2-params -->
### Exercise 1: Count a GPT-2-style model's parameters

Read a line with four integers `n_layer d vocab ctx` (layers, dimension, vocabulary size, context length) and print the total parameter count for the GPT-2 structure of section 5.4 (`12d² + 13d` per layer, token + position embeddings, a final LayerNorm of `2d`, output layer shared with the embedding).

```text
Sample input: 12 768 50257 1024
Sample output: 124439808
```

<details>
<summary>Hint</summary>

`vocab * d + ctx * d + n_layer * (12 * d * d + 13 * d) + 2 * d`
</details>

<!-- exercise: llama-layer -->
### Exercise 2: One Llama-style layer

Read a line with four integers `d n_heads n_kv_heads ffn_hidden` and print the parameter count of **one layer** of the Llama structure from section 5.6 (no biases; `W_Q` and `W_O` are `d × d` each; `W_K` and `W_V` are `d × (n_kv_heads × d / n_heads)` each; SwiGLU's three matrices are `3 × d × ffn_hidden`; two RMSNorms total `2d`).

```text
Sample input: 4096 32 8 14336
Sample output: 218112000
```

<details>
<summary>Hint</summary>

First compute `head_dim = d // n_heads`; K and V are `d * n_kv_heads * head_dim` each.
</details>

<!-- exercise: residual-role -->
### Exercise 3: What residual connections do

In a Transformer each sublayer is written `x = x + f(x)` rather than `x = f(x)`. What is the main benefit?

- fewer parameters
- gradients can flow straight back to earlier layers along the "addition" path, so very deep networks can train; each layer only has to learn a "change"
- attention can't see the future
- the output becomes a probability

<details>
<summary>Answer and explanation</summary>

The derivative of `x + f(x)` is `1 + f'(x)`, and that 1 lets the gradient pass through each layer at least unchanged; all the layers read from and write to one shared "residual stream".
</details>

<!-- exercise: which-family -->
### Exercise 4: Which architecture?

Which kind of Transformer do chat models such as GPT, Llama, Qwen and DeepSeek use?

- encoder-only (like BERT)
- encoder-decoder (like T5)
- decoder-only (causal attention, predicting the next token)
- not a Transformer at all

<details>
<summary>Answer and explanation</summary>

They are all decoder-only: causal attention + next-token prediction. This design is the simplest, scales most easily and can be trained on any text.
</details>

<!-- exercise: moe-why -->
### Exercise 5: Why MoE?

DeepSeek-V3 has 671 billion parameters in total, but each token activates only about 37 billion. What is the main benefit of MoE?

- a smaller model file
- the total parameters (knowledge capacity) are huge, but each token costs only as much computation as a much smaller model
- the router needs no training
- no GPU is needed

<details>
<summary>Answer and explanation</summary>

Each token passes through only a few experts, so the computation is set by the active parameters, while knowledge can be stored across all the experts. The cost: every expert's parameters must still be held in memory.
</details>

## 5.10 Further reading

- Jay Alammar: [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- Anthropic: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (where the "residual stream" view comes from)
- Sebastian Raschka: *Build a Large Language Model (From Scratch)* and its open-source code [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- Meta: "The Llama 3 Herd of Models" (2024); DeepSeek-AI: "DeepSeek-V3 Technical Report" (2024)
- Next lesson: [Train a GPT yourself](../06-gpt-from-scratch/en.md)
