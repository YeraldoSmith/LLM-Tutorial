# 6. Probability and information: how models are scored

> 🎯 **Goals**: understand probability distributions; turn any scores into probabilities with softmax; see why `log` is everywhere; learn cross-entropy loss — the number almost every language model minimises during training; use perplexity as an intuitive measure of how good a model is.
>
> ⏱ **Time**: 60 minutes　|　📋 **You need**: lessons 3–5

Lesson 1 said a language model outputs **a probability distribution over the next token**. This lesson makes every word of that sentence clear and shows how to score a model's predictions.

## 6.1 Probability distributions

A **probability** is a number between 0 and 1 saying how likely something is. List the probabilities of all possible outcomes and you have a **probability distribution**; they **add up to 1**.

```text
a die:     1      2      3      4      5      6
p:        1/6    1/6    1/6    1/6    1/6    1/6     (uniform distribution)

the model's next word for "The weather today is very ___":
         nice   hot    cold   bad    odd
p:       0.55   0.25   0.12   0.05   0.03            (sums to 1)
```

### Sampling by probability

Given a distribution you can **draw one at random** — likelier outcomes come up more often. This is how a language model picks words when generating text:

```python
import random
random.seed(0)

words = ["nice", "hot", "cold", "bad", "odd"]
probs = [0.55, 0.25, 0.12, 0.05, 0.03]
samples = random.choices(words, weights=probs, k=1000)
for w in words:
    print(w, samples.count(w) / 1000)       # sample frequencies approach the probabilities
```

Always taking the most probable word ("nice") is **greedy decoding**; sampling by probability makes the text more varied. Part 3, lesson 7 covers sampling strategies in depth.

## 6.2 Softmax: turning scores into probabilities

The last layer of a neural network outputs arbitrary real numbers called **logits** (they can be negative and don't sum to 1). To turn them into probabilities, use **softmax**:

```formula
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

Two steps: ① exponentiate each number, `e^z` (makes them positive and stretches differences); ② divide by the total (so they sum to 1).

```python
import math

def softmax(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

logits = [2.0, 1.0, 0.1]
probs = softmax(logits)
print([round(p, 4) for p in probs])     # [0.659, 0.2424, 0.0986]
print(sum(probs))                        # 1.0 (up to tiny float error)
```

The highest score gets the highest probability, but the others get some too — "softer" than picking the maximum, hence *soft*max.

### The numerically stable version

`math.exp(1000)` overflows (too big to store). Real logits can be large, so in practice you **subtract the maximum first** — the result is identical (numerator and denominator are both multiplied by `e^(−max)`) but nothing overflows:

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

print([round(p, 4) for p in softmax([1000.0, 999.0, 998.1])])   # no error
```

### Temperature: how sure to be

Divide the logits by a **temperature** `T` before softmax:

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]

logits = [2.0, 1.0, 0.1]
for T in [0.5, 1.0, 2.0]:
    print(f"T={T}:", [round(p, 3) for p in softmax([z / T for z in logits])])
```

- `T < 1`: a "sharper" distribution — almost always the top choice; output is conservative and stable;
- `T > 1`: a "flatter" distribution — more random, more creative, and more likely to ramble.

The `temperature` parameter of chatbot APIs is exactly this.

## 6.3 Why is log everywhere?

The **logarithm** is the inverse of the exponential: `e^x = y` ⇔ `x = ln y` (in this course `log` means the natural log). It's everywhere in probability for two reasons:

**① It turns products into sums.** The probability of a sentence is the **product** of its words' probabilities. With 100 words at probability 0.1 each, that's 10⁻¹⁰⁰ — too small to represent precisely (underflow). Logs turn it into a **sum**: `log(a × b) = log a + log b`, which is numerically far more stable.

```python
import math
probs = [0.1] * 100
product = 1.0
for p in probs:
    product *= p
print(product)                              # 1e-100; a little smaller and it becomes 0.0
print(sum(math.log(p) for p in probs))      # -230.26: perfectly safe with logs
```

**② It measures surprise.** `−log p` is the **information content** or **surprisal** of an event:

| probability p | −log p | intuition |
|---------------|--------|-----------|
| 1.0 | 0 | totally expected, no surprise |
| 0.5 | 0.69 | a coin flip |
| 0.1 | 2.30 | somewhat surprising |
| 0.01 | 4.61 | very surprising |
| 0.0001 | 9.21 | astonishing |

## 6.4 Cross-entropy: scoring a prediction

During training we know the **correct next token**. The higher the probability the model gave it, the better the prediction. The **cross-entropy loss** is simply **the surprisal of the correct answer**:

```formula
loss = −log p(correct token)
```

```python
import math

def cross_entropy(probs, target):
    return -math.log(probs[target])

probs = [0.7, 0.2, 0.1]          # the model's prediction for three candidates
print(round(cross_entropy(probs, 0), 4))    # correct answer is #0: 0.3567 (a good prediction)
print(round(cross_entropy(probs, 2), 4))    # correct answer is #2: 2.3026 (a bad prediction)
```

- probability 1 for the correct answer → loss 0 (perfect);
- probability near 0 → loss heads to infinity (very wrong).

To train a language model, **average** the cross-entropy over **every position** in a piece of text: that's the loss for the batch. Gradient descent (lesson 5) keeps adjusting the parameters to shrink this average — **that is, to give the correct next word a higher probability**.

> [!IMPORTANT]
> That is the complete objective of LLM pre-training: **minimise the average cross-entropy of the next token**. Billions of parameters, thousands of GPUs, months of training — all optimising this one number.

A useful baseline: with a vocabulary of `V` tokens, a model that has learned nothing and treats every token equally gives each probability `1/V`, so its loss is `log V`. GPT-2's vocabulary has about 50,000 tokens and `log 50257 ≈ 10.8` — roughly the loss at the very start of training.

## 6.5 Perplexity: how many options the model hesitates between

Cross-entropy isn't very intuitive. Exponentiate the average cross-entropy to get **perplexity**:

```formula
perplexity = e^(average cross-entropy)
```

Intuition: **the number of options the model is, on average, torn between**. Perplexity 1 means always certain (and right); 50 means it's as if it guessed among 50 words.

```python
import math

def perplexity(target_probs):
    """target_probs: the probability the model gave the correct token at each position"""
    avg_ce = sum(-math.log(p) for p in target_probs) / len(target_probs)
    return math.exp(avg_ce)

print(round(perplexity([0.25, 0.25, 0.25, 0.25]), 2))    # 4.0: like guessing among 4
print(round(perplexity([0.9, 0.8, 0.95, 0.7]), 2))       # about 1.2: very confident
```

## 6.6 Entropy (good to know)

The uncertainty of a distribution itself is its **entropy**: `H = −Σ pᵢ log pᵢ`, the "average surprise". A uniform distribution has the most entropy (most uncertain); a distribution with one certain outcome has zero. *Cross*-entropy is called that because it measures the average surprise of **real data** using **the model's distribution**.

## 6.7 Summary

- A **probability distribution** lists all outcomes' probabilities, summing to 1; `random.choices` samples from it.
- **Softmax** `e^z / Σ e^z` turns any logits into probabilities; subtract the max first to avoid overflow.
- **Temperature**: divide logits by T; small T is surer, large T more random.
- **log** turns products into sums and avoids underflow; `−log p` is surprisal.
- **Cross-entropy loss** = `−log p(correct)`; training an LM minimises the average over all positions.
- **Perplexity** = `e^(average cross-entropy)`, the number of options the model hesitates between; guessing among V gives V.

## 6.8 Exercises

<!-- exercise: softmax -->
### Exercise 1: Implement softmax

Read a line of numbers (logits) and print the softmax probabilities, each with 4 decimal places, separated by spaces. Use the stable "subtract the max" version.

```text
Sample input: 2 1 0.1
Sample output: 0.6590 0.2424 0.0986
```

<details>
<summary>Hint</summary>

`m = max(z)`; `exps = [math.exp(v - m) for v in z]`; divide by `sum(exps)`. Format with `f"{p:.4f}"`.
</details>

<!-- exercise: cross-entropy -->
### Exercise 2: Cross-entropy

Line 1 is the model's probability distribution (numbers separated by spaces); line 2 is the index of the correct answer (from 0). Print the cross-entropy loss with 4 decimal places.

```text
Sample input:
0.7 0.2 0.1
2
Sample output:
2.3026
```

<details>
<summary>Hint</summary>

`-math.log(probs[target])`
</details>

<!-- exercise: perplexity -->
### Exercise 3: Perplexity

Read a line of numbers: the probability the model gave the correct token at each position. Print the perplexity with 2 decimal places.

```text
Sample input: 0.5 0.25 0.125
Sample output: 4.00
```

<details>
<summary>Hint</summary>

Average cross-entropy first, `sum(-math.log(p) for p in ps) / len(ps)`, then `math.exp(...)`.
</details>

<!-- exercise: temperature -->
### Exercise 4: Temperature

To make a chatbot's answers **more stable and less random** (say, for code or factual questions), how should you set the temperature?

- higher, e.g. 1.5
- lower, e.g. 0.2
- temperature has no effect
- negative

<details>
<summary>Answer and explanation</summary>

A lower temperature makes the softmax distribution sharper, favouring the most likely token, so the output is steadier. For poems or brainstorming you can raise it.
</details>

<!-- exercise: uniform-loss -->
### Exercise 5: The loss of guessing

A model always gives the 4 tokens of its vocabulary equal probability (0.25 each). What does this program print (2 decimal places)?

```python
import math
print(f"{-math.log(0.25):.2f}")
print(f"{math.exp(-math.log(0.25)):.2f}")
```

<details>
<summary>Answer and explanation</summary>

Cross-entropy `−log 0.25 = log 4 ≈ 1.39`; perplexity `e^1.39 = 4.00` — exactly "guessing among 4".
</details>

## 6.9 Further reading

- [Dive into Deep Learning · 4.1 Softmax regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html)
- Chris Olah: [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) — a beautiful picture-based explanation of entropy and cross-entropy.
- Part 1 complete! Next: [Part 2 · Fitting a line with gradient descent](../../ml/01-linear-regression/en.md)
