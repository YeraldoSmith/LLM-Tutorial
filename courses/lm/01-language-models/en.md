# 1. Language models: predicting the next character

> 🎯 **Goals**: understand what a "language model" actually does — **predict the probability of the next token**; score a sentence with the chain rule of probability; build a **character-level bigram model** yourself: count, sample new names and evaluate with **perplexity**; then have the simplest possible neural network learn the same thing, paving the way to GPT.
>
> ⏱ **Time**: 75 minutes　|　📋 **You need**: part 1 lesson 6 (probability and information); part 2 lesson 2 (softmax and cross-entropy)

In part 1 we said that ChatGPT, DeepSeek, Qwen and the rest all do one thing over and over: **predict the next token from what came before**. Starting with this lesson, we'll build such a model ourselves, step by step — beginning with the simplest and oldest method.

> [!NOTE]
> This part follows the approach of Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) and [nanoGPT](https://github.com/karpathy/nanoGPT) projects (MIT licence) and their companion videos, which we highly recommend.

## 1.1 What is a language model?

A **language model** is a model that assigns probabilities to text. Its most common form: given what came before, output **the probability of every possible next token**.

```text
input:  "The weather today is really"
output: nice 41%   hot 17%   cold 9%   bad 6%   … (the probabilities of all possible tokens add up to 1)
```

Your phone keyboard's suggestions and a search box's autocomplete are small language models. Large language models (LLMs) are just **much bigger** (billions to trillions of parameters), **see much further back** (hundreds of thousands of tokens of context), and are trained on enormous amounts of text.

A **token** is the basic unit of text a model works with: a character, a word or part of a word. The next lesson covers how text is split. To keep this lesson simple, **each single character is a token**.

## 1.2 Scoring a sentence: the chain rule of probability

The probability of a sentence can be split into the probabilities of "predicting the next token" at each step, multiplied together:

```formula
P(w₁ w₂ … wₙ) = P(w₁) × P(w₂ | w₁) × P(w₃ | w₁ w₂) × … × P(wₙ | w₁ … wₙ₋₁)
```

`P(w₂ | w₁)` reads "the probability of w₂ given w₁" (a conditional probability). So **if you can predict the next token, you can score any text, and you can also generate text one token at a time** — that is all GPT does.

```python
import math

# suppose these are the model's probabilities at each step of "I love you"
steps = [("I", 0.05), ("love", 0.10), ("you", 0.60)]
p = 1.0
for token, prob in steps:
    p *= prob
    print(f"P({token} | context) = {prob}")
print("sentence probability:", round(p, 6))
print("negative log-likelihood (NLL):", round(-math.log(p), 3))   # lower probability → higher NLL
```

The probability of a long sentence becomes almost 0 (many numbers below 1 multiplied together), so in practice we always use **logarithms**: multiplying probabilities becomes adding logs, and the **negative log-likelihood** (NLL) serves as the loss — exactly the cross-entropy from part 2.

## 1.3 The simplest language model: look only at the previous character

The full conditional probability `P(wₙ | w₁ … wₙ₋₁)` is too complex. The simplest approximation: **look only at the previous token**.

```formula
P(wₙ | w₁ … wₙ₋₁) ≈ P(wₙ | wₙ₋₁)
```

This is a **bigram** model. Looking at the previous 2 tokens gives a trigram model, and in general an **n-gram model**. N-grams are the classic statistical language model; Claude Shannon used them in 1948 to generate text that "looked like English".

Let's train a **character-level bigram model** that learns what English first names look like and then generates new ones. The data is about a hundred common names. Each name gets a special character `.` at the start and the end, so `emma` becomes `.emma.`, which contains 5 pairs of neighbouring characters: `.e`, `em`, `mm`, `ma`, `a.`.

"Training" is just **counting**: count how often each character is followed by each other character, then divide by the totals to get probabilities.

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

chars = list(".abcdefghijklmnopqrstuvwxyz")    # "." marks the start and end of a name
print(len(NAMES), "names,", len(chars), "characters:", "".join(chars))

# 1. Count how often each character is followed by each character
counts = {a: {b: 0 for b in chars} for a in chars}
for name in NAMES:
    seq = ["."] + list(name) + ["."]
    for a, b in zip(seq, seq[1:]):
        counts[a][b] += 1

# 2. Turn counts into probabilities: P(b | a) = count(a, b) / count(a, anything); add 0.1 to every count (smoothing)
probs = {}
for a in chars:
    total = sum(counts[a][b] + 0.1 for b in chars)
    probs[a] = {b: (counts[a][b] + 0.1) / total for b in chars}

top = sorted(probs["."].items(), key=lambda kv: -kv[1])[:5]
print("most common first letters:", [(b, round(p, 3)) for b, p in top])

# 3. Sampling: start from ".", draw characters one by one according to the probabilities until "." comes up again
random.seed(7)
def sample():
    out, cur = "", "."
    while True:
        cur = random.choices(chars, weights=[probs[cur][b] for b in chars])[0]
        if cur == ".":
            return out
        out += cur
print("generated names:", [sample() for _ in range(10)])

# 4. Evaluation: average negative log-likelihood (NLL) and perplexity
def avg_nll(words):
    nll, n = 0.0, 0
    for w in words:
        seq = ["."] + list(w) + ["."]
        for a, b in zip(seq, seq[1:]):
            nll += -math.log(probs[a][b])
            n += 1
    return nll / n

loss = avg_nll(NAMES)
print(f"training set: average NLL = {loss:.3f}, perplexity = {math.exp(loss):.2f}")
print(f"random guessing (uniform): average NLL = {math.log(len(chars)):.3f}, perplexity = {len(chars)}")
print(f"new name 'sophie': {avg_nll(['sophie']):.3f}   gibberish 'zqxj': {avg_nll(['zqxj']):.3f}")
```

Let's go through what this program does.

### Sampling (generation)

This way of generating text is called **autoregressive**: start from the start symbol, **randomly draw** a character according to the model's probabilities, append it, draw the next one conditioned on it… until the end symbol comes up. `random.choices(candidates, weights=probabilities)` is exactly "draw by probability". GPT generates its answers the same way, one token after another.

Some generated names look plausible, others are odd — because the model **only remembers the previous character**; it doesn't know how long the name already is or which syllables it used. Doing better requires a longer context.

### Evaluation: perplexity

How do you measure whether a language model is good? Check how much probability it gives to **real text**: use the average NLL (lower is better), or its exponential — **perplexity**:

```formula
perplexity = e^(average NLL)
```

Intuitively, perplexity is **how many options the model is "torn between" at each step, on average**. Pure guessing (all 27 characters equally likely) has a perplexity of exactly 27; our bigram model gets about 9, so it has clearly learned something. The gibberish `zqxj` gets a much higher NLL than the name-like `sophie`.

> [!NOTE]
> Perplexity is the classic evaluation metric for language models and is still used to compare pre-trained models (it depends on the tokenization, so only compare models that use the same one).

### Smoothing: why add 0.1?

If a pair of characters never appears in the training data (say `qx`), its count is 0, its probability is 0, and `−log(0)` is **infinite** — a single occurrence ruins the whole evaluation. Adding a small number to every count (**smoothing**) gives "unseen" combinations a little probability.

## 1.4 Doing the same with a neural network

Counting is simple, but it doesn't scale: to look at the previous 10 characters there would be 27¹⁰ ≈ 2×10¹⁴ possible combinations, almost none of them ever seen — impossible to count. The **neural network** approach is not to count but to **learn a function** that takes the context and outputs the probabilities of the next character.

First, let's redo the bigram model with a neural network and see whether it learns the same thing as counting:

- input: the current character (number 0–26) as a **one-hot vector** (1 at position i, 0 elsewhere);
- model: a 27×27 weight matrix `W`; multiplying the one-hot vector by `W` gives exactly row i of `W`, used as the **logits**;
- output: softmax of the logits = the probabilities of the next character;
- loss: cross-entropy (the average NLL above), trained with gradient descent.

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
idx = {c: i for i, c in enumerate(chars)}          # character → number
V = len(chars)

pairs = []                                          # training data: (current character, next character)
for name in NAMES:
    seq = ["."] + list(name) + ["."]
    pairs += [(idx[a], idx[b]) for a, b in zip(seq, seq[1:])]
print("training examples:", len(pairs))

# The model: a V×V weight matrix. One-hot(a) times W picks out row a — that row is the logits
random.seed(0)
W = [[random.gauss(0, 0.1) for _ in range(V)] for _ in range(V)]

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

lr = 20.0
for epoch in range(1, 201):
    grad = [[0.0] * V for _ in range(V)]
    loss = 0.0
    for a, b in pairs:
        p = softmax(W[a])                          # forward: distribution over the next character
        loss += -math.log(p[b])                    # cross-entropy
        for j in range(V):                         # backward: the gradient of softmax + cross-entropy is p − one_hot(b)
            grad[a][j] += p[j] - (1.0 if j == b else 0.0)
    for i in range(V):                             # gradient descent (on the averaged gradient)
        for j in range(V):
            W[i][j] -= lr * grad[i][j] / len(pairs)
    if epoch in (1, 10, 50, 100, 200):
        print(f"epoch {epoch:3d}  average NLL {loss / len(pairs):.3f}")

p = softmax(W[idx["a"]])
best = sorted(range(V), key=lambda j: -p[j])[:3]
print("learned P(next | 'a'):", [(chars[j], round(p[j], 3)) for j in best])
```

The loss falls from about 3.3 (guessing) to about 2.14, practically the same as counting, and the learned probability table matches the counts. **Both methods learn the same thing.** The differences:

| | Counting (n-gram) | Neural network |
|---|---|---|
| "training" | count occurrences | gradient descent |
| longer context | combinations explode exponentially; never enough data | parameter count stays manageable; can take arbitrarily long context |
| generalization | unseen combinations rely on smoothing | similar contexts get similar predictions |

> [!TIP]
> A neat correspondence: adding L2 regularization (weight decay) to the neural version has the same effect as smoothing in the counting version — both keep the probability distributions from becoming too "extreme".

## 1.5 From n-grams to GPT: a short history

| Year | Milestone | Key idea |
|------|-----------|----------|
| 1948 | Shannon's n-gram experiments | predict the next letter / word from statistical frequencies |
| 1980s–2000s | statistical n-gram models | the core of speech recognition and machine translation |
| 2003 | Bengio's neural probabilistic language model | **word vectors** + a neural network instead of counting (word vectors come in lesson 3) |
| 2010s | RNNs / LSTMs | recurrent structures that "remember" long contexts |
| 2017 | **The Transformer** ("Attention Is All You Need") | **attention** sees the whole context at once and trains in parallel |
| 2018– | the GPT and BERT families | pre-train huge Transformers on enormous amounts of text |
| 2022– | the LLM era after ChatGPT | instruction tuning, alignment, reasoning models… |

The next lessons follow this road step by step: **tokenization** (lesson 2) → **embeddings** (lesson 3) → **attention** (lesson 4) → **the Transformer** (lesson 5) → training a **tiny GPT** yourself (lesson 6) → **decoding strategies** (lesson 7).

## 1.6 Summary

- A language model predicts **a probability distribution over the next token**; the chain rule scores whole texts, and **autoregressive sampling** generates text.
- In practice we use the **negative log-likelihood** (NLL, i.e. cross-entropy); **perplexity** = e^(average NLL), roughly "how many options the model is torn between at each step" — lower is better.
- **N-gram** models only look at the previous n−1 tokens and are trained by counting; unseen combinations need **smoothing**; longer contexts can't be counted.
- A neural network with a one-hot input, a linear layer and softmax learns exactly the same bigram model with gradient descent — and neural networks scale to much longer contexts.

## 1.7 Exercises

<!-- exercise: perplexity -->
### Exercise 1: Compute the perplexity

Read a line of probabilities: the probability the model gave to the correct token at each step of a sentence. Print three numbers separated by spaces: the sentence probability (6 decimal places), the average NLL (4 decimal places) and the perplexity (4 decimal places).

```text
Sample input: 0.5 0.25 0.5
Sample output: 0.062500 0.9242 2.5198
```

<details>
<summary>Hint</summary>

Average NLL = `-sum(math.log(p) for p in ps) / len(ps)`, perplexity = `math.exp(average NLL)`.
</details>

<!-- exercise: bigram-next -->
### Exercise 2: The most likely next character

The first line contains some words separated by spaces; the second line is a character `c`. As in this lesson, add `.` before and after each word and count pairs of neighbouring characters. Print the character that **follows `c` most often**; if there is a tie, print the smallest one in character order (`.` comes before all letters).

```text
Sample input:
emma ella eva
e
Sample output: l
```

<details>
<summary>Hint</summary>

`e` is followed by `m`, `l` and `v` once each; in a tie, take the smallest, `l`. You can use `sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]`.
</details>

<!-- exercise: why-weird -->
### Exercise 3: Why are the generated names odd?

Names generated by the bigram model are often very long or hard to read. What is the main reason?

- the training data contains typos
- the model only looks at the previous character and knows nothing earlier (such as how long the name already is)
- the random seed was badly chosen
- the character set lacks capital letters

<details>
<summary>Answer and explanation</summary>

A bigram model predicts the next character from the previous one alone and completely "forgets" everything earlier, so it can't make the whole thing look like a name. Seeing a longer context is the key improvement — exactly what neural networks and Transformers will do.
</details>

<!-- exercise: zero-prob -->
### Exercise 4: What happens without smoothing?

Without smoothing, if the test text contains a pair of characters that never appeared in the training set, the average NLL on the test set is:

- 0
- infinite
- unaffected
- negative

<details>
<summary>Answer and explanation</summary>

An unseen combination has probability 0, `−log(0) = ∞`, so the average is infinite too (and so is the perplexity). That's why smoothing is needed.
</details>

<!-- exercise: uniform-ppl -->
### Exercise 5: The perplexity of guessing

A model that "learned nothing" gives the same probability to all 50,000 tokens in its vocabulary. What is its perplexity?

- 1
- about 10.8 (that is, ln 50000)
- 50000
- 25000

<details>
<summary>Answer and explanation</summary>

Every token has probability 1/50000, so the NLL is ln 50000 and the perplexity is e^(ln 50000) = 50000. A uniform distribution's perplexity equals the number of options — which is where "how many options it is torn between" comes from.
</details>

## 1.8 Further reading

- Andrej Karpathy: [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) (the inspiration for this lesson)
- Claude Shannon: "A Mathematical Theory of Communication" (1948) — section 3 generates English with n-grams
- *Dive into Deep Learning*: [Language Models](https://d2l.ai/chapter_recurrent-neural-networks/language-model.html)
- Next lesson: [Tokenization: how text becomes tokens](../02-tokenization/en.md)
