# 7. Decoding: how the model picks the next word

> 🎯 **Goals**: understand that a model only provides "the probability of the next token" — **how to choose** is a separate step (decoding); see why greedy decoding is repetitive and dull; master **temperature**, **top-k**, **top-p** and **min-p** sampling and **repetition penalties**; learn about beam search, stopping conditions and **constrained decoding** (structured output); know how to set these parameters when calling an LLM API.
>
> ⏱ **Time**: 60 minutes　|　📋 **You need**: lessons 1 and 6 of this part

A trained language model outputs only a **probability distribution** at each step: 63% that the next token is "nice", 14% that it is "hot"… To actually generate text, something has to decide **which one to pick**. This step is called **decoding** or **sampling**. The same model can give answers of very different style, quality and variety depending on how it is decoded. Parameters like `temperature` and `top_p` in LLM APIs control exactly this step.

## 7.1 Greedy decoding: always pick the most likely

The simplest method: at every step pick the token with the **highest probability** (last lesson's addition GPT did this). It's deterministic and reproducible, good for tasks with "one right answer" (arithmetic, classification, information extraction). For writing, though, the problems are obvious:

- **dull**: it always picks the most common phrasing, with no variety;
- **repetitive**: once it falls into a loop, it can't get out.

Look at a small hand-made "word-level" model. After `cat` its most likely next word is `the`, and after `the` it is `cat`…

```python
# a hand-made "word-level" model: the logits of each word that can follow each word
LM = {
    "<s>": {"the": 2.0, "a": 1.0},
    "the": {"cat": 2.0, "dog": 1.5, "mat": 0.5},
    "a":   {"cat": 1.0, "dog": 1.0},
    "cat": {"the": 1.6, "sat": 1.4, ".": 0.5},
    "dog": {"the": 1.6, "ran": 1.2, ".": 0.5},
    "sat": {"on": 2.0, ".": 0.5},
    "ran": {"away": 2.0, ".": 0.5},
    "on":  {"the": 2.0, "a": 1.0},
    "away": {".": 2.0},
    "mat": {".": 2.0, "the": 0.5},
}

def greedy(penalty=1.0, max_len=12):
    out, cur = [], "<s>"
    while len(out) < max_len:
        scores = dict(LM[cur])
        for w in set(out):                        # repetition penalty: discount words that already appeared
            if w in scores:
                scores[w] = scores[w] / penalty if scores[w] > 0 else scores[w] * penalty
        cur = max(scores, key=scores.get)
        out.append(cur)
        if cur == ".":
            break
    return " ".join(out)

print("greedy:                ", greedy())
print("repetition penalty 1.5:", greedy(penalty=1.5))
```

Greedy decoding gets stuck in `the cat the cat …` until the length limit. A **repetition penalty** discounts the scores of words that have already appeared, breaking the loop. (Too strong a penalty is bad too: the model is forced to avoid words that should repeat, such as "the".)

## 7.2 Temperature: how "bold" to be

For variety, we **draw at random** according to the probabilities (sampling). **Temperature** (T) divides the logits by T before the softmax:

```formula
P(token i) = softmax(zᵢ / T)
```

```python
import math

def softmax(z, T=1.0):
    z = [v / T for v in z]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

words = ["nice", "hot", "cold", "lovely", "awful", "purple"]
logits = [3.0, 1.5, 1.2, 1.0, 0.2, -2.0]          # the model's scores for "The weather today is really ___"

for T in (0.2, 0.7, 1.0, 1.5):
    probs = softmax(logits, T)
    print(f"T={T:<4}", "  ".join(f"{w}:{p:5.1%}" for w, p in zip(words, probs)))
```

- **T < 1**: the distribution gets "sharper" and likely tokens get likelier — more conservative and stable. As T → 0 it becomes greedy.
- **T = 1**: the model's original distribution.
- **T > 1**: the distribution gets "flatter" and unlikely options (even "purple") get more chance — more creative, and more prone to nonsense.

Get a feel for it with the bigram name model from lesson 1:

```python
import math
import random

NAMES = """emma olivia ava isabella sophia mia charlotte amelia harper evelyn abigail emily
elizabeth sofia avery ella madison scarlett victoria aria grace chloe camila penelope riley
layla lillian nora zoey mila aubrey hannah lily addison eleanor natalie luna savannah leah
stella hazel ellie audrey violet claire bella aurora lucy anna samantha caroline sarah maya
liam noah william james oliver benjamin elijah lucas mason logan alexander ethan jacob
michael daniel henry jackson sebastian aiden matthew samuel david joseph carter owen wyatt
john jack luke dylan levi isaac gabriel julian mateo anthony lincoln joshua andrew caleb
ryan nathan thomas leo isaiah charles hudson hunter connor eli ezra aaron adrian nolan""".split()

chars = list(".abcdefghijklmnopqrstuvwxyz")
counts = {a: {b: 0.1 for b in chars} for a in chars}      # lesson 1's bigram model (smoothed)
for name in NAMES:
    seq = ["."] + list(name) + ["."]
    for a, b in zip(seq, seq[1:]):
        counts[a][b] += 1
logits = {a: [math.log(counts[a][b]) for b in chars] for a in chars}   # turn probabilities back into "logits"

def generate(T=1.0, top_k=None, greedy=False, max_len=15):
    out, cur = "", "."
    while len(out) < max_len:
        z = logits[cur]
        if greedy:
            cur = chars[max(range(len(z)), key=lambda i: z[i])]
        else:
            z = [v / T for v in z]                         # temperature
            if top_k:                                      # top-k: draw only from the k most likely
                kth = sorted(z, reverse=True)[top_k - 1]
                z = [v if v >= kth else float("-inf") for v in z]
            m = max(z)
            w = [math.exp(v - m) for v in z]
            cur = random.choices(chars, weights=w)[0]
        if cur == ".":
            break
        out += cur
    return out

random.seed(0)
print("greedy           ", [generate(greedy=True) for _ in range(5)])
for T in (0.5, 1.0, 1.5):
    print(f"temperature {T:<4} ", [generate(T=T) for _ in range(5)])
print("temp 1 + top-5   ", [generate(T=1.0, top_k=5) for _ in range(5)])
```

Greedy produces the same name, `a`, every time (the most common first letter is `a`, and the most common thing after `a` is the end); at low temperature names are short and safe, at high temperature they turn into long gibberish; adding top-k clearly reduces the gibberish.

## 7.3 Cutting off the "long tail": top-k, top-p, min-p

Even a good model gives tens of thousands of tokens small but non-zero probabilities. When sampling, the odd words in this "long tail" add up to a real chance of being drawn — and once one is drawn, everything after it may go wrong. So we usually **keep only some of the candidates** before sampling:

- **top-k**: keep only the k most likely tokens;
- **top-p** (nucleus sampling): add up probabilities starting from the most likely and keep the smallest set whose **total reaches p**. When the model is confident only one or two remain; when it is unsure many do — more flexible than a fixed k;
- **min-p** (a newer method): keep only tokens whose probability is **at least "the top probability × min_p"**.

The surviving candidates are renormalized (to add up to 1) and then sampled:

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

def top_k(probs, k):
    keep = sorted(range(len(probs)), key=lambda i: -probs[i])[:k]
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]

def top_p(probs, p):
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    keep, cum = [], 0.0
    for i in order:                      # add up from the most likely until the total reaches p
        keep.append(i)
        cum += probs[i]
        if cum >= p:
            break
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]

def min_p(probs, ratio):
    limit = ratio * max(probs)           # keep only tokens with probability ≥ top probability × ratio
    total = sum(q for q in probs if q >= limit)
    return [q / total if q >= limit else 0.0 for q in probs]

words = ["nice", "hot", "cold", "lovely", "awful", "purple"]
probs = softmax([3.0, 1.5, 1.2, 1.0, 0.2, -2.0])
show = lambda ps: "  ".join(f"{w}:{q:5.1%}" for w, q in zip(words, ps) if q > 0)
print("original ", show(probs))
print("top-k=3  ", show(top_k(probs, 3)))
print("top-p=0.9", show(top_p(probs, 0.9)))
print("min-p=0.1", show(min_p(probs, 0.1)))
```

In practice **temperature is usually combined with top-p (or top-k, or min-p)**: adjust the distribution with temperature, cut off the long tail, then sample.

## 7.4 Penalizing repetition: repetition, frequency and presence penalties

Section 7.1 showed a **repetition penalty**. APIs such as OpenAI's offer two similar parameters:

| Parameter | How | Effect |
|-----------|-----|--------|
| `frequency_penalty` | subtract "number of occurrences × penalty" from a token's logit | the more often a token appears, the less it's wanted; less long-winded repetition |
| `presence_penalty` | subtract a fixed amount once a token has appeared at all | encourages new topics |
| `repetition_penalty` (Hugging Face, vLLM etc.) | for tokens that appeared, divide positive logits by the penalty and multiply negative ones | breaks loops |

Leave these at their defaults (0 or 1.0) and raise them a little only if you see obvious repetition.

## 7.5 Beam search

Greedy decoding only looks at the best choice right now, but "best at every step" isn't the same as "best sentence overall". **Beam search** keeps the **b** candidate sequences with the highest total probability at every step (b is the beam width) and finally picks the most probable one overall:

```text
beam width b = 2
step 1: the (0.7)         a (0.3)
step 2: the cat (0.42)    the dog (0.21)     ← of all extensions, keep only the best 2
step 3: the cat sat (…)   the dog ran (…)
…
```

Beam search is widely used for tasks with fairly definite answers, such as **machine translation and speech recognition**. In open-ended writing and dialogue it tends to produce bland, repetitive text, so chatbots almost always use **sampling**.

## 7.6 When to stop?

Generation continues until a **stopping condition** is met:

- an **end token** is generated (such as `<|endoftext|>` or `<|im_end|>`, the special tokens from lesson 2) — the model itself "decides" it's done;
- the **maximum length** is reached (`max_tokens` / `max_output_tokens` in APIs);
- a **stop sequence** you specified appears (the `stop` parameter), e.g. stop at `"\n\n"`.

> [!WARNING]
> If `max_tokens` is too small, the answer is cut off mid-sentence. Reasoning models (part 5) generate long "thinking" before answering, and that counts towards the token budget too, so give them enough room.

## 7.7 Constrained decoding: guaranteeing the format

Many applications need output in a **strict format**: valid JSON, only "yes / no", a number. **Constrained decoding** sets the **logits of tokens that would break the format to −∞** before each sampling step, so the model simply cannot produce invalid output:

```python
import math

vocab = ["1", "2", "3", "seven", "abc", ".", "<end>"]
logits = [1.0, 0.5, 0.2, 2.5, 1.8, 0.3, 0.1]       # the model most wants "seven", but we need digits

allowed = {"1", "2", "3", "<end>"}                   # format rule: digits and the end token only
masked = [z if tok in allowed else float("-inf") for tok, z in zip(vocab, logits)]
m = max(masked)
e = [math.exp(z - m) for z in masked]
probs = [v / sum(e) for v in e]
for tok, p in zip(vocab, probs):
    print(f"{tok:6s}{p:6.1%}")
```

Real systems work out from a JSON Schema or grammar which tokens are allowed at each step. The "structured output" features of APIs from OpenAI, Anthropic and others, and the grammar constraints in inference engines like vLLM and llama.cpp, all work on this principle. We'll use it again for tool calling in part 7.

## 7.8 Practical advice

| Use case | Suggestion |
|----------|------------|
| code, maths, extraction, classification | low temperature (0–0.3) or plain greedy |
| general Q&A and chat | temperature around 0.7, top-p 0.9–0.95 |
| stories and brainstorming | temperature 0.8–1.2 |
| reasoning models | **follow the official recommendations**: DeepSeek-R1, for example, recommends a temperature of 0.5–0.7 (0.6 preferred); some reasoning-model APIs don't even let you change the temperature |

> [!NOTE]
> **Temperature 0 isn't always perfectly reproducible**: the order of floating-point operations on GPUs and servers batching several requests together can cause tiny differences, and once one token differs everything after it may change. When you need reproducibility, fix the random seed (a `seed` parameter, if the API offers one) and record the model version.

## 7.9 Summary

- A model only gives a **probability distribution** over the next token; **decoding** decides which one to pick, so the same model's answers can vary greatly in quality.
- **Greedy**: deterministic, good for tasks with one answer, but dull and repetitive. **Sampling**: draws at random by probability; varied, but may talk nonsense.
- **Temperature** T divides the logits; T < 1 is more conservative (T → 0 is greedy), T > 1 is bolder.
- **top-k / top-p / min-p** sample only among likely candidates, cutting off the "long tail"; usually combined with temperature.
- **Repetition, frequency and presence penalties** reduce repetition; **beam search** suits translation and similar tasks; **stopping conditions** are end tokens, maximum length and stop sequences.
- **Constrained decoding** masks tokens that would break the format, guaranteeing valid output (structured output, JSON).

## 7.10 Exercises

<!-- exercise: temperature -->
### Exercise 1: Softmax with temperature

The first line contains some logits; the second line is the temperature `T`. Print `softmax(z / T)` with 4 decimal places, separated by spaces.

```text
Sample input:
2 1 0
0.5
Sample output: 0.8668 0.1173 0.0159
```

<details>
<summary>Hint</summary>

First `z = [v / T for v in logits]`, then softmax (subtract the maximum first).
</details>

<!-- exercise: top-k -->
### Exercise 2: top-k

The first line contains some logits; the second line is an integer `k`. Apply softmax (temperature 1), keep only the `k` most likely tokens and renormalize them, setting the others' probabilities to 0. Print the probabilities of all tokens in their original order, 4 decimal places, separated by spaces.

```text
Sample input:
2 1 0 -1
2
Sample output: 0.7311 0.2689 0.0000 0.0000
```

<details>
<summary>Hint</summary>

`keep = sorted(range(n), key=lambda i: -p[i])[:k]`, then divide by the sum of the kept probabilities.
</details>

<!-- exercise: top-p -->
### Exercise 3: What does top-p keep?

The first line contains probabilities that already add up to 1; the second line is `p`. Add candidates in order of decreasing probability (on a tie, the smaller index first) until the running total is **greater than or equal to** `p`. Print the indices (from 0) of the kept tokens in the order they were added, separated by spaces.

```text
Sample input:
0.5 0.3 0.15 0.05
0.75
Sample output: 0 1
```

<details>
<summary>Hint</summary>

`sorted(range(n), key=lambda i: -probs[i])` is a stable sort, so on a tie the indices stay in increasing order.
</details>

<!-- exercise: greedy-repeat -->
### Exercise 4: The trouble with greedy decoding

What is the most common problem when writing long texts with greedy decoding?

- it's too slow
- the text is bland and easily falls into repetitive loops
- every run gives a different result
- generation never stops

<details>
<summary>Answer and explanation</summary>

Greedy always picks the most common phrasing, and once it enters a loop like "after A the likeliest is B, after B the likeliest is A" it can't escape. Its result is deterministic (the same every time), which is actually an advantage for tasks with one right answer.
</details>

<!-- exercise: low-temp -->
### Exercise 5: Temperature close to 0

If the temperature is set very close to 0, what does sampling become similar to?

- choosing a token uniformly at random
- greedy decoding (always taking the most likely token)
- always taking the least likely token
- beam search

<details>
<summary>Answer and explanation</summary>

Dividing the logits by a tiny T hugely magnifies the gap between the maximum and the rest, so softmax gives almost all the probability to the largest — that's greedy. Only a very large temperature approaches a uniform distribution.
</details>

## 7.11 Further reading

- Holtzman et al.: "The Curious Case of Neural Text Degeneration" (2019, introduced top-p nucleus sampling)
- Hugging Face blog: [How to generate text: using different decoding methods](https://huggingface.co/blog/how-to-generate)
- Nguyen et al.: "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs" (2024)
- That's the end of part 3! Next part: [Part 4 · Training at scale](../../training/01-data/en.md) — data, compute, distributed training and evaluation.
