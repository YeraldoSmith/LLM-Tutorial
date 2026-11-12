# 2. Classification and softmax regression

> 🎯 **Goals**: move from predicting numbers to predicting categories; understand the sigmoid and logistic regression; extend binary classification to many classes (softmax regression); remember a crucial fact — the gradient of softmax + cross-entropy is "predicted probabilities − correct answer"; train a classifier in plain Python and measure its accuracy.
>
> ⏱ **Time**: 60 minutes　|　📋 **You need**: part 1 lesson 6 (softmax, cross-entropy) and lesson 1 of this part

Last lesson's model output a **number** (a price). Many problems need a **category** instead: is this e-mail spam? Is this picture a cat, a dog or a bird? And most importantly — **which token in the vocabulary comes next?** A language model is, at heart, a classifier with tens of thousands of classes.

## 2.1 Binary classification: output a probability

To decide whether an e-mail is spam we'd like the model to output the **probability** that it is — a number between 0 and 1. But a linear model `z = w·x + b` can output any real number. We need a function that squashes it into 0–1: the **sigmoid**.

```formula
σ(z) = 1 / (1 + e^(−z))
```

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

for z in [-6, -2, 0, 2, 6]:
    print(f"σ({z:>2}) = {sigmoid(z):.4f}")
```

```text
       1 ┤                 ●●●●●
         │              ●●
     0.5 ┤ - - - - - - ●- - - - -     large z → close to 1
         │          ●●                 z = 0   → 0.5
       0 ┤ ●●●●●●●●                    small z → close to 0
         └────────────┬───────────
                      0
```

"Linear model + sigmoid" is **logistic regression** — called *regression*, but used for classification.

### Binary cross-entropy

With binary labels `y` (0 or 1) and the model's `p = σ(z)`, the loss is the cross-entropy of part 1 lesson 6:

```formula
L = −[ y·log p + (1 − y)·log(1 − p) ]
```

For `y = 1` this is `−log p`; for `y = 0` it's `−log(1 − p)` — still "the surprisal of the correct answer".

## 2.2 Many classes: softmax regression

With `K` classes (10 for handwritten digits, or 50,000 tokens for a language model), the model outputs `K` scores (logits) and **softmax** turns them into a probability distribution:

```text
input x ──► linear layer: K scores z = W·x + b ──► softmax ──► K probabilities ──► cross-entropy loss
            W has shape K × d                                   (sum to 1)          −log p[correct class]
```

This is **softmax regression** (multinomial logistic regression). The **last layer** of a language model has exactly this structure: the Transformer's output vector is multiplied by a `V × d` matrix to give `V` logits, followed by softmax.

## 2.3 The most important gradient: p − y

For softmax + cross-entropy, the gradient with respect to the logits is **astonishingly simple**:

```formula
∂L/∂zₖ = pₖ − yₖ
```

where `p` are the softmax probabilities and `y` is the **one-hot vector** of the correct answer (1 at the correct class, 0 elsewhere).

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

z = [2.0, 1.0, 0.1]          # scores for 3 classes
target = 0                    # the correct answer is class 0
p = softmax(z)
y = [1 if k == target else 0 for k in range(len(z))]     # one-hot: [1, 0, 0]
grad = [pk - yk for pk, yk in zip(p, y)]
print("probabilities:", [round(v, 4) for v in p])
print("gradient:     ", [round(v, 4) for v in grad])
```

Intuition:

- for the **correct class** the gradient `p − 1` is negative → gradient descent **raises** its score;
- for **wrong classes** the gradient `p` is positive → gradient descent **lowers** their scores;
- the better the prediction (the closer the correct p is to 1), the smaller the gradient and the smaller the adjustment.

> [!IMPORTANT]
> Remember **p − y**. When training a language model, the output-layer gradient at every position is computed exactly like this: the model's probability distribution minus the vector with a 1 at the correct token.

Let's verify the formula numerically (the method from part 1 lesson 5):

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def loss(z, target):
    return -math.log(softmax(z)[target])

z, target, h = [2.0, 1.0, 0.1], 0, 1e-5
numeric = []
for k in range(len(z)):
    zp = z[:]; zp[k] += h
    zm = z[:]; zm[k] -= h
    numeric.append((loss(zp, target) - loss(zm, target)) / (2 * h))
print("numerical gradient:", [round(v, 4) for v in numeric])
```

Both methods agree.

## 2.4 Hands on: train a 2-D classifier

We generate two clouds of points — class 0 bottom-left, class 1 top-right — and train softmax regression to tell them apart.

```python
import math
import random
random.seed(0)

# data: each sample is (x1, x2) with label 0 or 1
data = [((random.gauss(-1, 0.7), random.gauss(-1, 0.7)), 0) for _ in range(50)]
data += [((random.gauss(1, 0.7), random.gauss(1, 0.7)), 1) for _ in range(50)]

K, D = 2, 2
W = [[0.0] * D for _ in range(K)]      # K × D
b = [0.0] * K
lr = 0.5

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]

def forward(x):
    z = [sum(W[k][d] * x[d] for d in range(D)) + b[k] for k in range(K)]
    return softmax(z)

for epoch in range(1, 31):
    gW = [[0.0] * D for _ in range(K)]
    gb = [0.0] * K
    total_loss = 0.0
    for x, label in data:
        p = forward(x)
        total_loss += -math.log(p[label])
        for k in range(K):
            g = p[k] - (1 if k == label else 0)        # p − y
            gb[k] += g
            for d in range(D):
                gW[k][d] += g * x[d]                   # chain rule: ∂z/∂W = x
    n = len(data)
    for k in range(K):
        b[k] -= lr * gb[k] / n
        for d in range(D):
            W[k][d] -= lr * gW[k][d] / n
    if epoch % 10 == 0:
        acc = sum(max(range(K), key=lambda k: forward(x)[k]) == label for x, label in data) / n
        print(f"epoch {epoch}: average loss {total_loss / n:.4f}, accuracy {acc:.0%}")
```

After a few dozen epochs the accuracy is close to 100%. An **epoch** means one complete pass over the training data.

### Accuracy

Besides the loss, classification is often judged by **accuracy**: the fraction of samples whose predicted class (the most probable one, the argmax) equals the true class. The loss is what training differentiates; accuracy is easier for people to interpret.

## 2.5 From classification to language models

Replace `K` with the vocabulary size (say 50,000) and the input `x` with "a vector representing the preceding text", and you have the last step of a language model:

| | 2-D point classifier | language model |
|---|---|---|
| number of classes K | 2 | vocabulary size (tens to hundreds of thousands) |
| input x | coordinates (x1, x2) | the Transformer's understanding of the context (a vector of thousands of dimensions) |
| output | 2 probabilities | the probability that each token comes next |
| loss | cross-entropy | cross-entropy (averaged over positions) |
| gradient | p − y | p − y |

## 2.6 Summary

- The **sigmoid** `1/(1+e^(−z))` squashes any number into 0–1; linear model + sigmoid = **logistic regression** (binary), trained with binary cross-entropy.
- Many classes: a linear layer gives K logits → **softmax** → cross-entropy: **softmax regression**, which is a language model's output layer.
- The gradient of softmax + cross-entropy with respect to the logits is **p − y** (y one-hot).
- Training sums the per-sample gradients, averages, then updates; an **epoch** is one full pass over the data.
- Classification is often evaluated by **accuracy**: the fraction where the argmax matches the label.

## 2.7 Exercises

<!-- exercise: sigmoid -->
### Exercise 1: The sigmoid

Read a line of numbers z and print the sigmoid of each (4 decimal places, separated by spaces).

```text
Sample input: 0 2 -2
Sample output: 0.5000 0.8808 0.1192
```

<details>
<summary>Hint</summary>

`1 / (1 + math.exp(-z))`
</details>

<!-- exercise: softmax-grad -->
### Exercise 2: The gradient of softmax cross-entropy

Line 1 holds logits, line 2 the index of the correct class. Print the gradient of the cross-entropy loss with respect to each logit (`p − y`), 4 decimal places, separated by spaces.

```text
Sample input:
2 1 0.1
0
Sample output:
-0.3410 0.2424 0.0986
```

<details>
<summary>Hint</summary>

Compute p with a stable softmax, then subtract 1 at the correct class.
</details>

<!-- exercise: accuracy -->
### Exercise 3: Accuracy

Line 1 holds integers `n` and `K` (samples and classes). Each of the next `n` lines holds `K` predicted probabilities followed by the true class index. Print the accuracy (the predicted class is the most probable one; on a tie, the smaller index) as a percentage with 1 decimal place, e.g. `75.0%`.

```text
Sample input:
4 3
0.7 0.2 0.1 0
0.1 0.3 0.6 2
0.5 0.4 0.1 1
0.2 0.5 0.3 1
Sample output:
75.0%
```

<details>
<summary>Hint</summary>

For each line `vals = input().split()`: the first `K` are probabilities, the last is the label. Prediction: `max(range(K), key=lambda k: probs[k])` — on a tie `max` returns the first, i.e. the smaller index.
</details>

<!-- exercise: grad-sign -->
### Exercise 4: Which way does the gradient point?

When training with softmax + cross-entropy, the gradient `p − 1` of the **correct class's** logit is always:

- positive, so its score is lowered
- negative (or 0), so gradient descent raises its score
- always 0
- dependent on the learning rate

<details>
<summary>Answer and explanation</summary>

p is at most 1, so `p − 1 ≤ 0`. Gradient descent subtracts "learning rate × a negative number", raising that class's score — the model leans more and more towards the right answer.
</details>

<!-- exercise: lm-classes -->
### Exercise 5: A language model's classes

Viewing a language model as a classifier, what are its "classes"?

- 0 and 1
- all tokens in the vocabulary
- the sentiment of a sentence (positive / negative)
- the topic of the text

<details>
<summary>Answer and explanation</summary>

At every step it picks the next token among all tokens of the vocabulary; the number of classes equals the vocabulary size.
</details>

## 2.8 Further reading

- [Dive into Deep Learning · 4.1 Softmax regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html)
- CS231n: [Linear classification notes](https://cs231n.github.io/linear-classify/)
- Next lesson: [Neural networks](../03-neural-networks/en.md)
