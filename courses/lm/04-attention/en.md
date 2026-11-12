# 4. Attention: letting every token "see" its context

> 🎯 **Goals**: understand why attention is needed; derive **self-attention** step by step starting from "take the average"; build intuition for **Query, Key and Value** (a "fuzzy dictionary lookup"); master scaled dot-product attention `softmax(QKᵀ/√d)·V`, the **causal mask** and **multi-head attention**; implement it once in pure Python and once in PyTorch; understand attention's O(n²) cost and why **RoPE** rotary position embeddings represent relative positions.
>
> ⏱ **Time**: 100 minutes　|　📋 **You need**: lessons 1–3 of this part; part 1 lesson 4 (matrix multiplication)

This is the most central lesson of the whole course. The **Transformer**, introduced in the 2017 paper "Attention Is All You Need", is built almost entirely from attention, and every large model today — GPT, Claude, Llama, Qwen, DeepSeek — stands on it. Read slowly and run every piece of code.

> [!NOTE]
> The order of this lesson's derivation follows Andrej Karpathy's video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), which we highly recommend watching alongside.

## 4.1 Why do we need attention?

Last lesson's neural language model looked at **exactly the previous 3** characters and concatenated them. That has two problems:

1. **A fixed window**: to look at the previous 1000 tokens, the input would be 1000 times longer and the parameters would explode;
2. **Everything treated alike**: each position is processed in a fixed way, but the useful information could be anywhere.

Consider: "Alice returned the book to Mary because **she** has an exam tomorrow." To understand who "she" is, the model must **look back** for the relevant word (Mary), which may be far away. What we want: **every token should be able to "pick out" relevant information from all previous tokens, as needed, and gather it into itself.** That is **attention**.

## 4.2 Step one: average the previous information

The simplest way to "gather the previous text": the output at position t = the **average** of the vectors at positions up to t (including itself).

```python
# 4 tokens, each a 2-dimensional vector (pretend they're embeddings)
X = [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [1.0, 1.0]]

# output at position t = average of the first t+1 vectors (including itself)
out = []
for t in range(len(X)):
    seen = X[:t + 1]
    out.append([sum(v[i] for v in seen) / len(seen) for i in range(2)])
for t, v in enumerate(out):
    print(f"position {t}: averaged {t + 1} tokens →", [round(a, 3) for a in v])
```

Note that each position can **only see itself and earlier positions**, never later ones — a language model predicts the next token, and if it could "peek" at the answer it would learn nothing. This is the **causal** constraint.

Averaging has an obvious flaw: all previous tokens get **the same weight**. We want relevant tokens to get large weights and irrelevant ones small weights, with the weights **depending on the content**.

## 4.3 Step two: weight by relevance — Query, Key, Value

Think of attention as a **fuzzy dictionary lookup** (or a search engine):

- **Query**: "what am I looking for?" — the question the current token asks;
- **Key**: "what do I have?" — the label each token puts up;
- **Value**: "what I give you if you pick me" — the content each token provides.

Take the **dot product** of the query with every key to get match scores, turn the scores into weights that add up to 1 with **softmax**, then take the **weighted average** of the values:

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# a "mini search engine": each record has a key (label) and a value (content)
keys   = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
values = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
query  = [1.0, 0.2]                          # what I'm looking for

scores = [dot(query, k) for k in keys]       # how well it matches each key
weights = softmax(scores)                    # turn into weights that add up to 1
output = [sum(w * v[i] for w, v in zip(weights, values)) for i in range(2)]
print("match scores:     ", [round(s, 3) for s in scores])
print("attention weights:", [round(w, 3) for w in weights])
print("output (weighted average of the values):", [round(o, 3) for o in output])
```

An ordinary dictionary needs an "exact match" to return one entry; attention returns "**a mixture of all entries, weighted by how well they match**" — and the whole process is differentiable, so it can be trained with gradient descent.

In **self-attention**, the queries, keys and values **all come from the same sequence of tokens**: each token's embedding `x` is multiplied by three learnable matrices to get its q, k and v:

```formula
q = x·W_Q　　k = x·W_K　　v = x·W_V
```

Through training, the model learns by itself what "questions" to ask, what "labels" to put up and what "content" to provide when picked.

## 4.4 Scaled dot-product attention

Stack the q, k, v of all tokens into matrices Q, K, V (one row per token), and the attention for all positions can be computed at once with matrices:

```formula
Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```

- `Q·Kᵀ` is an n×n matrix: row i, column j = the dot product of token i's query with token j's key;
- dividing by `√d` (d is the dimension of q and k) is the **scaling**, explained below;
- softmax is applied to **each row**, so every token's weights add up to 1;
- multiplying by V: each token's output = a weighted average of all the values.

### Why divide by √d?

The dot product of two random d-dimensional vectors grows roughly like `√d`. In high dimensions the scores differ a lot and softmax becomes very "peaked": almost all the weight lands on one token, the other tokens get gradients close to 0, and learning is hard. Dividing by `√d` brings the scores back into a reasonable range:

```python
import math
import random

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

random.seed(0)
for d in (4, 64, 1024):
    q = [random.gauss(0, 1) for _ in range(d)]
    keys = [[random.gauss(0, 1) for _ in range(d)] for _ in range(5)]
    raw = [sum(a * b for a, b in zip(q, k)) for k in keys]
    scaled = [s / math.sqrt(d) for s in raw]
    print(f"d={d:4d}  largest weight unscaled {max(softmax(raw)):.3f}   divided by √d {max(softmax(scaled)):.3f}")
```

### The causal mask

In a language model, token i may only look at tokens at positions ≤ i. The trick: **before** softmax, set the scores of "future" positions to **−∞**. Since e^(−∞) = 0, those positions get weight 0 after softmax. The weight matrix is therefore **lower-triangular**.

## 4.5 Complete causal self-attention in pure Python

```python
import math
import random

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def matmul(A, B):                       # (n×k) @ (k×m)
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

random.seed(0)
tokens = ["I", "love", "big", "models"]
n, d = len(tokens), 4
X = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]          # embeddings of 4 tokens (n×d)
Wq = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]       # three learnable matrices (d×d)
Wk = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]
Wv = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]

Q, K, V = matmul(X, Wq), matmul(X, Wk), matmul(X, Wv)                   # every token's query, key and value
scores = matmul(Q, [list(c) for c in zip(*K)])                          # Q @ Kᵀ: n×n match scores
scale = 1 / math.sqrt(d)
weights = []
for i in range(n):
    row = [scores[i][j] * scale if j <= i else float("-inf") for j in range(n)]   # causal mask: no peeking ahead
    weights.append(softmax(row))
out = matmul(weights, V)                                                # weighted average of the values

print("attention weights (row i = whom token i is looking at):")
print("        " + "".join(f"{t:>8}" for t in tokens))
for t, row in zip(tokens, weights):
    print(f"{t:>6}  " + "".join(f"{w:8.3f}" for w in row))
print("output shape:", len(out), "×", len(out[0]))
```

The weight matrix is lower-triangular and every row adds up to 1. These weights come from random matrices and mean nothing yet; **after training**, the model learns useful patterns, such as pronouns "looking at" the nouns they refer to and verbs "looking at" their subjects.

> [!TIP]
> Attention's output has the same shape as its input (n tokens of d dimensions each), so layers can be stacked one after another. After a layer of attention, each token's vector has **absorbed information from its context**: the same word "apple" gets different vectors in "I ate an apple" and "Apple released a new phone".

## 4.6 Multi-head attention

A token may need several kinds of information at once: who its subject is, what the previous word was, what the sentence is about… One set of Q, K, V can only compute **one** weight distribution. **Multi-head attention** runs attention h times in parallel:

1. split the d dimensions into h parts; each "head" does attention independently with its own Q, K, V (d/h dimensions each);
2. **concatenate** the h heads' outputs back to d dimensions;
3. multiply by an output matrix `W_O` to mix the heads' information.

The computation is about the same as a single head, but the model can attend to different relationships simultaneously. Research has found genuinely specialised heads in trained models: some only look at the previous token, others "copy" patterns seen earlier in the text (**induction heads**)…

Here is multi-head causal self-attention in PyTorch, structured almost exactly like GPT-2's code:

```python requires=torch
import math
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads          # number of heads, dimension per head
        self.qkv = nn.Linear(d_model, 3 * d_model)             # compute Q, K and V in one go
        self.proj = nn.Linear(d_model, d_model)                # output projection W_O

    def forward(self, x):
        B, T, C = x.shape                                      # batch, sequence length, dimension
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, h, T, dh): cut C into h heads, each doing attention on its own
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.dh)    # (B, h, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # lower triangle: only self and earlier
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                            # (B, h, T, dh)
        y = y.transpose(1, 2).reshape(B, T, C)                 # put the heads back together
        return self.proj(y), att

torch.manual_seed(0)
attn = CausalSelfAttention(d_model=32, n_heads=4)
x = torch.randn(2, 5, 32)                                      # 2 sequences of 5 tokens, 32 dimensions each
y, att = attn(x)
print("output:", tuple(y.shape), " attention weights:", tuple(att.shape))
print("attention weights of sequence 1, head 1:")
print(att[0, 0].detach().round(decimals=2))
print("parameters:", sum(p.numel() for p in attn.parameters()))   # 4 × 32 × 32 + 4 × 32

# PyTorch's built-in efficient implementation (picks FlashAttention etc. automatically) gives the same result
q, k, v = attn.qkv(x).split(32, dim=2)
q, k, v = (t.view(2, 5, 4, 8).transpose(1, 2) for t in (q, k, v))
fast = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 5, 32)
print("matches F.scaled_dot_product_attention:", torch.allclose(attn.proj(fast), y, atol=1e-6))
```

Multi-head attention's parameters are the four d×d matrices `W_Q`, `W_K`, `W_V`, `W_O`: **4d²** in total (plus 4d biases). Each attention layer of GPT-2 small (d = 768) has about 2.36 million parameters.

## 4.7 The cost of attention: O(n²)

The attention score matrix is **n×n**: double the sequence length and computation and memory grow **four**-fold. A context of a million tokens has 10¹² scores — which is why "long context" is expensive. Researchers have found many ways around this:

| Technique | Idea | Used by |
|-----------|------|---------|
| **FlashAttention** | never store the full n×n matrix; compute it block by block in fast GPU memory — identical results, much faster, less memory | almost every modern training and inference framework |
| **KV cache** | during generation, store the K and V already computed, so each new token only needs one new row | all LLM inference (details in part 6) |
| **GQA / MQA** | several query heads **share** one set of K, V heads, making the KV cache several times smaller | Llama 3, Qwen and others |
| **MLA** (multi-head latent attention) | compress K and V into a low-dimensional "latent vector" before caching | DeepSeek-V2/V3 |
| **sliding-window / sparse attention** | each token only looks at a nearby window, or at a selected subset of tokens | Mistral, Gemma, DeepSeek's sparse attention and others |
| **linear attention / state-space models** | replace softmax attention with structures whose cost grows linearly with length | Mamba and some hybrid architectures |

## 4.8 Position information: RoPE rotary position embeddings

As we said last lesson, attention by itself doesn't know order — shuffle the tokens and each token's weight distribution stays the same. GPT-2 **adds** position embeddings to the input. Almost all modern models (Llama, Qwen, DeepSeek, Mistral…) use **RoPE** (Rotary Position Embedding, proposed by Su Jianlin et al. in 2021) instead:

- group the dimensions of the q and k vectors in pairs, treating each pair as a point in a plane;
- for a token at position m, **rotate** every pair by an angle `m × θ` (different pairs rotate at different speeds θ);
- after rotating both vectors, their dot product depends only on the **difference of angles** — that is, on the **relative position m − n**.

```python
import math

def rotate(v, angle):                      # rotate a 2-D vector anticlockwise by angle radians
    x, y = v
    return [x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)]

theta = 0.5                                 # each later position rotates another 0.5 radians
q, k = [1.0, 0.5], [0.3, 1.0]
for m, n in [(5, 3), (10, 8), (102, 100), (3, 5), (7, 7)]:
    qm, kn = rotate(q, m * theta), rotate(k, n * theta)   # q at position m, k at position n
    score = qm[0] * kn[0] + qm[1] * kn[1]
    print(f"q at position {m:3d}, k at position {n:3d}, distance {m - n:2d}: q·k = {score:.4f}")
```

As long as the distance is the same (2 here), the attention score is the same whether the tokens sit at position 5 or 102. That's exactly what we want: **relative relationships** like "the previous word" or "three words back" are what matter in language. In real RoPE, pair i rotates at speed `θᵢ = base^(−2i/d)` (base is usually 10000; Llama 3 uses 500000), so some pairs rotate fast (to tell nearby positions apart) and some very slowly (to tell distant positions apart). Adjusting the base and similar parameters (e.g. the YaRN method) also lets models handle contexts longer than those they were trained on.

## 4.9 Summary

- Attention lets every token gather a **weighted summary** of (earlier) tokens based on content; the weights come from how well queries match keys.
- Self-attention: `q = xW_Q, k = xW_K, v = xW_V`; `Attention = softmax(QKᵀ/√d)·V`; dividing by `√d` keeps softmax from becoming too peaked.
- **Causal mask**: set future positions to −∞ before softmax; the weight matrix is lower-triangular.
- **Multi-head attention**: h heads attend to different relationships in parallel, then are concatenated and multiplied by `W_O`; about 4d² parameters.
- The cost grows as **O(n²)** in the sequence length; FlashAttention, the KV cache, GQA/MLA and sparse attention all aim to reduce it.
- **RoPE** rotates q and k by position so attention scores depend only on relative position — the standard in modern LLMs.

## 4.10 Exercises

<!-- exercise: attn-weights -->
### Exercise 1: Attention weights

The first line is the query vector, the second line an integer `n`, and the next `n` lines are `n` key vectors (as long as the query). Print the scaled dot-product attention weights `softmax(q·kⱼ / √d)` (d is the vector length), 4 decimal places, separated by spaces.

```text
Sample input:
1 0
2
1 0
0 1
Sample output: 0.6698 0.3302
```

<details>
<summary>Hint</summary>

Scores `s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q)) for k in keys]`, then softmax (remember to subtract the maximum first to avoid overflow).
</details>

<!-- exercise: causal-mask -->
### Exercise 2: The causal mask

Read an integer `n` and print the `n×n` causal mask: row i, column j is `1` if token i can see token j (j ≤ i) and `0` otherwise. Separate the numbers in each row with spaces.

```text
Sample input: 3
Sample output:
1 0 0
1 1 0
1 1 1
```

<details>
<summary>Hint</summary>

Two nested loops with `1 if j <= i else 0`, or `" ".join("1" if j <= i else "0" for j in range(n))`.
</details>

<!-- exercise: attn-output -->
### Exercise 3: The attention output

The first line is the query vector, the second an integer `n`, then `n` lines of key vectors and then `n` lines of value vectors. Compute the scaled dot-product attention output (the weighted average of the values), 4 decimal places, separated by spaces.

```text
Sample input:
1 0
2
1 0
0 1
10 0
0 10
Sample output: 6.6976 3.3024
```

<details>
<summary>Hint</summary>

First compute the weights `w` as in exercise 1; component i of the output is `sum(wj * v[i] for wj, v in zip(w, values))`.
</details>

<!-- exercise: mha-params -->
### Exercise 4: Parameters of multi-head attention

Read a line with two integers `d bias`: the model dimension and whether there are biases (1 = yes, 0 = no). Print the total number of parameters in one multi-head attention layer (four d×d matrices `W_Q`, `W_K`, `W_V`, `W_O`, plus d biases per matrix if biases are used). Note: the count doesn't depend on the number of heads.

```text
Sample input: 768 1
Sample output: 2362368
```

<details>
<summary>Hint</summary>

`4 * d * d + (4 * d if bias else 0)`
</details>

<!-- exercise: why-scale -->
### Exercise 5: Why divide by √d?

- to make the computation faster
- the larger the dimension, the larger the dot products; softmax becomes very "peaked" and gradients go to 0, so dividing by √d brings the scores back into a good range
- so the weights add up to 1
- to implement the causal mask

<details>
<summary>Answer and explanation</summary>

The dot product of random d-dimensional vectors has a standard deviation of about √d. Without scaling, softmax becomes nearly one-hot in high dimensions and most positions get no gradient. "Adding up to 1" is guaranteed by softmax itself.
</details>

<!-- exercise: quadratic -->
### Exercise 6: Doubling the context

All else equal, if the sequence length grows from 4,000 to 8,000 tokens, how many times more elements does the attention score matrix have?

- 1× (unchanged)
- 2×
- 4×
- 8×

<details>
<summary>Answer and explanation</summary>

The score matrix is n×n: 8000² / 4000² = 4. That's what "attention cost grows quadratically with length" means, and why long contexts are expensive.
</details>

## 4.11 Further reading

- Vaswani et al.: "Attention Is All You Need" (2017) — the original Transformer paper
- Jay Alammar: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (the classic illustrated explanation)
- 3Blue1Brown: [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention) (episode 6 of the deep learning series)
- Su et al.: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021, the RoPE paper)
- Next lesson: [The Transformer: putting the blocks together](../05-transformer/en.md)
