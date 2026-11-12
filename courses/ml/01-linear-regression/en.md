# 1. Your first learning model: fitting a line with gradient descent

> 🎯 **Goals**: train your first model yourself; understand the five ideas that run through the whole course — model, parameters, loss function, gradient, training loop; derive and implement the gradients of linear regression; see how data scaling and the learning rate affect training; know the difference between training and test sets.
>
> ⏱ **Time**: 60 minutes　|　📋 **You need**: part 1 (especially lesson 5, "Derivatives and gradients")

From this lesson on we really *train* models. Today's model is very simple — a straight line — but the method is **exactly** how GPT is trained.

## 1.1 The problem: estimate house prices from size

Suppose we have data on 5 houses:

| size x (tens of m²) | price y (hundred thousands) |
|---------------------|-----------------------------|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |

We want a rule that estimates the price of a new house from its size. Looking closely, price ≈ 2 × size + 1. But we want the **computer to discover** this from the data.

## 1.2 Five core ideas

**① Model**: a function with parameters that takes x and outputs a prediction ŷ ("y-hat"). Here, a straight line:

```formula
ŷ = w · x + b
```

**② Parameters**: the adjustable numbers in the model — here **w** (the weight, i.e. slope) and **b** (the bias, i.e. intercept). GPT has billions of parameters; same idea.

**③ Loss function**: a single number measuring **how wrong** the predictions are. For regression the usual choice is **mean squared error** (MSE):

```formula
L = (1/N) · Σ (ŷᵢ − yᵢ)²
```

Each sample's error is squared (so positive and negative errors both count, and big errors are punished heavily), then averaged.

**④ Gradient**: the derivative of the loss with respect to each parameter, telling us which way to adjust it.

**⑤ Training loop**: repeat "predict → compute the loss → compute the gradients → update the parameters".

```text
         ┌────────────────────────────────────────────────┐
         ▼                                                │
  predict ŷ with the current parameters ──► loss L ──► gradients ──► params -= lr × gradient
```

## 1.3 Try by hand first

```python
xs = [1, 2, 3, 4, 5]
ys = [3, 5, 7, 9, 11]

def predict(w, b, x):
    return w * x + b

def mse(w, b):
    return sum((predict(w, b, x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

for w, b in [(0, 0), (1, 1), (2, 0), (2, 1)]:
    print(f"w={w}, b={b}  loss={mse(w, b):.2f}")
```

With `w=2, b=1` the loss is 0 — perfect. But in real problems we can't guess like this, and there are more than two parameters. Hence gradient descent.

## 1.4 Deriving the gradients

The loss is `L = (1/N) Σ (w·xᵢ + b − yᵢ)²`. Differentiating with respect to w and b (with the chain rule from part 1 lesson 5: the derivative of `(something)²` is `2 × something × the derivative of something`):

```formula
∂L/∂w = (2/N) · Σ (ŷᵢ − yᵢ) · xᵢ
∂L/∂b = (2/N) · Σ (ŷᵢ − yᵢ)
```

Intuition: `ŷᵢ − yᵢ` is each sample's **error**. If predictions are generally too high (positive errors), both gradients are positive and gradient descent lowers w and b — exactly right.

## 1.5 Train!

```python
xs = [1, 2, 3, 4, 5]
ys = [3, 5, 7, 9, 11]
N = len(xs)

w, b = 0.0, 0.0          # parameters start at 0
lr = 0.05                # learning rate

for step in range(1, 501):
    preds = [w * x + b for x in xs]                       # ① predict
    errors = [p - y for p, y in zip(preds, ys)]
    loss = sum(e * e for e in errors) / N                 # ② loss
    grad_w = 2 / N * sum(e * x for e, x in zip(errors, xs))   # ③ gradients
    grad_b = 2 / N * sum(errors)
    w -= lr * grad_w                                      # ④ update
    b -= lr * grad_b
    if step in (1, 10, 50, 100, 500):
        print(f"step {step:3d}: loss = {loss:.6f}, w = {w:.4f}, b = {b:.4f}")

print(f"predicted price for size 6: {w * 6 + b:.2f}")
```

The loss keeps falling, w approaches 2 and b approaches 1. **The model "learned" the rule from data** — nobody told it the answer.

> [!IMPORTANT]
> These 15 lines are the skeleton of all deep-learning training. When training GPT, the model changes from "a line" to "a Transformer", the loss from MSE to "cross-entropy of the next token", and backpropagation computes the gradients automatically — but the loop has **exactly the same shape**.

## 1.6 What about noisy data?

Real data never lies perfectly on a line. Let's generate data **with random noise**:

```python
import random
random.seed(1)

xs = [i / 10 for i in range(50)]                              # 0.0 to 4.9
ys = [3 * x - 2 + random.gauss(0, 0.5) for x in xs]          # true rule y = 3x - 2, plus noise

w, b, lr = 0.0, 0.0, 0.05
N = len(xs)
for step in range(2000):
    errors = [(w * x + b) - y for x, y in zip(xs, ys)]
    w -= lr * 2 / N * sum(e * x for e, x in zip(errors, xs))
    b -= lr * 2 / N * sum(errors)

loss = sum(((w * x + b) - y) ** 2 for x, y in zip(xs, ys)) / N
print(f"learned w = {w:.3f}, b = {b:.3f} (true values 3 and -2), loss = {loss:.3f}")
```

The learned parameters are close to the true values but not identical — noise makes the data itself uncertain. The loss doesn't reach 0 either; it settles around the noise level.

## 1.7 Learning rate and data scaling

Change `lr` in the example above to `0.5` and run it — the loss becomes `inf` or `nan` ("not a number"): **divergence**. As in part 1 lesson 5, the learning rate is too big and each step overshoots.

Another often overlooked cause is the **scale of the features**. If size were in square metres (tens to hundreds) instead of tens of m², `grad_w` would be multiplied by a large `x`, the gradient would be huge, and the same learning rate would diverge. That's why inputs are usually **standardised** (subtract the mean, divide by the standard deviation) so all features are near 0 and on similar scales.

## 1.8 Several features: the vector form

Price depends on more than size — floor, age… With `d` features the model becomes:

```formula
ŷ = w₁x₁ + w₂x₂ + … + w_d·x_d + b = w · x + b
```

That's the **dot product** from part 1 lesson 4! Stack several samples into a matrix and prediction becomes **matrix times vector** — exactly the computation of a "linear layer" (`Linear` / `Dense`) in a neural network.

```python
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# each sample: [size, floor, age]
X = [[8, 3, 10], [12, 5, 2], [6, 1, 30]]
w = [2.0, 0.5, -0.1]
b = 1.0
print([round(dot(w, x) + b, 2) for x in X])
```

## 1.9 Training and test sets

Doing well on the training data **doesn't mean doing well on new data**. A student who memorises the answers to the homework fails when the exam has different questions — this is **overfitting**.

So data is always split in two:

- the **training set**: used to adjust the parameters;
- the **test set** (or **validation set**): **never seen** during training, used only to evaluate.

A model is judged by its **loss on the test set**. Part 2 lesson 6 shows how to fight overfitting.

## 1.10 Summary

- A **model** is a function with **parameters** (adjustable numbers); the **loss function** measures how wrong it is (MSE for regression).
- Gradients of linear regression `ŷ = w·x + b`: `∂L/∂w = (2/N)Σ(ŷ−y)x`, `∂L/∂b = (2/N)Σ(ŷ−y)`.
- The **training loop** — predict → loss → gradients → update, repeated — is the skeleton of all deep-learning training.
- Too big a learning rate or badly scaled features make training diverge; inputs are usually standardised.
- With several features the prediction is a dot product `w·x + b`: a neural network's linear layer.
- Evaluate on a **test set** and watch out for **overfitting**.

## 1.11 Exercises

<!-- exercise: mse -->
### Exercise 1: Mean squared error

Line 1 holds predictions, line 2 the true values (separated by spaces). Print the mean squared error with 4 decimal places.

```text
Sample input:
2.5 0.0 2.1
3.0 -0.5 2.0
Sample output:
0.1700
```

<details>
<summary>Hint</summary>

`sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(ys)`
</details>

<!-- exercise: one-step -->
### Exercise 2: One step of gradient descent

The starter code contains data `xs` and `ys`. Read three numbers on one line: the current `w`, `b` and the learning rate `lr`. For the model `ŷ = w·x + b` with MSE loss, compute the gradients once, update the parameters, and print the new `w` and `b` (4 decimal places each, separated by a space).

```text
Sample input: 0 0 0.05
Sample output: 2.5000 0.7000
```

<details>
<summary>Hint</summary>

Compute `errors = [(w * x + b) - y ...]`, then `grad_w` and `grad_b` with the formulas of 1.4, then `w -= lr * grad_w` and `b -= lr * grad_b`.
</details>

<!-- exercise: train -->
### Exercise 3: Full training

Input: line 1 is an integer `n` (number of samples), line 2 holds `n` x values, line 3 `n` y values, line 4 the learning rate and the number of steps. Starting from `w = 0, b = 0`, train linear regression with gradient descent and print the final `w` and `b` (3 decimal places, separated by a space).

```text
Sample input:
5
1 2 3 4 5
3 5 7 9 11
0.05 1000
Sample output:
2.000 1.000
```

<details>
<summary>Hint</summary>

Turn the loop from 1.5 into a version that reads the data. Both `w` and `b` must be updated using errors from **the same** pass.
</details>

<!-- exercise: overfit -->
### Exercise 4: Overfitting

A model's training loss is almost 0 but its test loss is large. What does this tell you?

- The model is perfect; ship it
- The model overfits: it memorised the training data instead of learning a general rule
- The test set is wrong and should be deleted
- The learning rate is too small

<details>
<summary>Answer and explanation</summary>

Classic overfitting. Remedies include more data, a simpler model, regularisation and early stopping (lesson 6).
</details>

<!-- exercise: loss-meaning -->
### Exercise 5: What the loss is for

What does the loss function do during training?

- It decides how many parameters the model has
- It measures how wrong the current predictions are as one number, whose gradient guides the parameter updates
- It generates training data
- It speeds up the GPU

<details>
<summary>Answer and explanation</summary>

The loss turns "how good" into a differentiable number; gradient descent keeps making that number smaller.
</details>

## 1.12 Further reading

- Andrew Ng: [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction), week 1 — linear regression and gradient descent in great detail.
- [Dive into Deep Learning · 3.1 Linear regression](https://d2l.ai/chapter_linear-regression/linear-regression.html)
- Next lesson: [Classification and softmax regression](../02-classification/en.md)
