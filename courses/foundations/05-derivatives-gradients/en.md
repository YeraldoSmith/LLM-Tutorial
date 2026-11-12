# 5. Derivatives and gradients: how models "learn"

> 🎯 **Goals**: understand a derivative as a "rate of change"; approximate derivatives with code; understand gradient descent — how every neural network learns; see the effect of the learning rate; meet partial derivatives, the gradient vector and the chain rule (the basis of backpropagation).
>
> ⏱ **Time**: 60 minutes　|　📋 **You need**: lessons 3 and 4; knowing what x² means

Training an LLM ultimately means: **there's a number that measures "how wrong" the model is (the loss), and we nudge billions of parameters again and again to make that number smaller.** How do we know whether to nudge each parameter up or down? **Derivatives.**

## 5.1 Derivatives: how fast things change

Imagine standing on a hillside where your height depends on your position. The **derivative** answers: **if I take a tiny step forward, how much does my height change?**

- positive derivative: stepping right goes up (uphill);
- negative derivative: stepping right goes down (downhill);
- zero: flat ground (maybe a peak or a valley floor);
- the bigger its size, the steeper the slope.

The derivative of `f` at `x` is written `f'(x)`: change `x` by a tiny amount `h`, see how much `f` changes, and divide by `h`.

```formula
f'(x) ≈ ( f(x + h) − f(x − h) ) / (2h)        (h tiny, e.g. 0.00001)
```

No need to derive formulas — **code can compute it**:

```python
def f(x):
    return x ** 2

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

for x in [-2, 0, 1, 3]:
    print(f"x = {x:>2}  f(x) = {f(x):>2}  slope ≈ {derivative(f, x):.4f}")
```

Output:

```text
x = -2  f(x) =  4  slope ≈ -4.0000
x =  0  f(x) =  0  slope ≈ 0.0000
x =  1  f(x) =  1  slope ≈ 2.0000
x =  3  f(x) =  9  slope ≈ 6.0000
```

The derivative of `x²` is exactly `2x`. Here are some common derivatives — **know the results, skip the derivations**:

| Function | Derivative |
|----------|------------|
| `c` (constant) | `0` |
| `x` | `1` |
| `x²` | `2x` |
| `xⁿ` | `n·xⁿ⁻¹` |
| `eˣ` | `eˣ` (itself!) |
| `ln x` | `1/x` |
| `a·f(x)` | `a·f'(x)` |
| `f(x) + g(x)` | `f'(x) + g'(x)` |

## 5.2 Gradient descent: walk downhill

Say we want the minimum of `f(x) = (x − 3)²`. (Obviously `x = 3`, but pretend we don't know — just as we don't know the best values of billions of parameters.)

**Gradient descent** is beautifully simple: **walk down a hill blindfolded** — feel which way the ground slopes, take a small step downhill, repeat.

```formula
x_new = x_old − learning_rate × f'(x_old)
```

- positive derivative (uphill to the right) → `x` decreases, step left;
- negative derivative (uphill to the left) → `x` increases, step right;
- the **learning rate** (often `lr`) controls how big each step is.

```python
def f(x):
    return (x - 3) ** 2

def df(x):
    return 2 * (x - 3)          # the derivative of (x - 3)²

x = 0.0                          # start anywhere
lr = 0.1
for step in range(1, 21):
    x = x - lr * df(x)
    if step % 4 == 0:
        print(f"step {step:2d}: x = {x:.4f}, f(x) = {f(x):.6f}")
```

`x` creeps towards 3 and `f(x)` towards 0. **That is the whole secret of how neural networks learn** — except there are billions of parameters instead of one, and the function isn't `(x−3)²` but "how wrong the model's predictions are on the training data".

### Learning rate: neither too small nor too big

```python
def df(x):
    return 2 * (x - 3)

for lr in [0.01, 0.1, 0.9, 1.1]:
    x = 0.0
    for _ in range(20):
        x = x - lr * df(x)
    print(f"learning rate {lr:<4}: after 20 steps x = {x:.4f}")
```

- `0.01`: too small — slow, not there after 20 steps;
- `0.1`: about right;
- `0.9`: each step jumps over the minimum and oscillates, but still converges;
- `1.1`: too big — every jump goes further out: it **diverges**!

> [!TIP]
> When training an LLM, the learning rate is one of the most important "hyperparameters" (settings chosen by people). Training usually starts by slowly raising it from a tiny value (warmup) and later lowers it gradually (decay) — part 2 covers this.

## 5.3 Several variables: partial derivatives and the gradient

Real models have many parameters. For `f(x, y) = x² + 3y²` we look separately at how `f` changes when we move only `x` and only `y` — the **partial derivatives**:

```text
∂f/∂x = 2x          (treat y as a constant)
∂f/∂y = 6y          (treat x as a constant)
```

Put all partial derivatives into a vector and you have the **gradient**: `∇f = [∂f/∂x, ∂f/∂y]`. The gradient points in the direction of **steepest ascent**, so moving along the **negative gradient** descends fastest.

```python
def f(x, y):
    return x ** 2 + 3 * y ** 2

def grad(x, y):
    return 2 * x, 6 * y

x, y = 4.0, -2.0
lr = 0.1
for step in range(1, 31):
    gx, gy = grad(x, y)
    x, y = x - lr * gx, y - lr * gy
    if step % 10 == 0:
        print(f"step {step}: x = {x:.4f}, y = {y:.4f}, f = {f(x, y):.6f}")
```

For a model with 7 billion parameters the gradient is a vector of 7 billion numbers, each telling its parameter "which way and how far to move".

## 5.4 The chain rule: one link after another

A neural network is functions nested inside functions. With `y = g(x)` and `z = f(y)`, what is the derivative of `z` with respect to `x`?

The **chain rule**: **rates of change multiply**.

```formula
dz/dx = dz/dy × dy/dx
```

Intuition: gears. If `y` turns 3 times for each turn of `x`, and `z` turns 2 times for each turn of `y`, then `z` turns 3 × 2 = 6 times for each turn of `x`.

Example: `z = (3x + 1)²`. Let `y = 3x + 1`, so `z = y²`:

```text
dy/dx = 3
dz/dy = 2y
dz/dx = 2y × 3 = 6(3x + 1)
```

Check with code:

```python
def z_of_x(x):
    return (3 * x + 1) ** 2

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

x = 2.0
print("numerical: ", round(derivative(z_of_x, x), 4))
print("chain rule:", 6 * (3 * x + 1))
```

> [!IMPORTANT]
> The chain rule is the mathematical basis of **backpropagation**: starting from the final loss and multiplying backwards layer by layer gives the gradient of every parameter efficiently. You'll implement it yourself in part 2, lesson 4.

## 5.5 Why not train with numerical derivatives?

`(f(x+h) − f(x−h)) / 2h` runs the whole model twice for *each* parameter's derivative. With 7 billion parameters that's 14 billion runs — far too slow. Backpropagation uses the chain rule to get **all gradients for roughly the cost of two runs**. Numerical derivatives remain useful for debugging, though: they check that backpropagation is correct (a *gradient check*).

## 5.6 Summary

- A **derivative** is a rate of change: positive uphill, negative downhill, zero flat; approximate it with `(f(x+h) − f(x−h)) / 2h`.
- **Gradient descent**: `parameter ← parameter − learning rate × derivative`, repeated, makes the function smaller and smaller — this is how models learn.
- Too small a **learning rate** converges slowly; too big oscillates or diverges.
- With several variables, the **partial derivatives** form the **gradient** vector; the negative gradient descends fastest.
- The **chain rule**: the derivative of a composition is the product of the links' derivatives — the basis of backpropagation.

## 5.7 Exercises

<!-- exercise: numeric-derivative -->
### Exercise 1: Numerical derivative

Read a number `x`. Using `(f(x+h) − f(x−h)) / (2h)` with `h = 1e-5`, compute the derivative of `f(x) = x³ − 2x` at `x` and print it with 4 decimal places.

```text
Sample input: 2
Sample output: 10.0000
```

(The derivative of `x³ − 2x` is `3x² − 2`, which is 10 at 2.)

<details>
<summary>Hint</summary>

Define `def f(x): return x ** 3 - 2 * x` and apply the formula. Read with `float(input())`.
</details>

<!-- exercise: gradient-descent -->
### Exercise 2: Gradient descent

Read three numbers on one line: start `x0`, learning rate `lr`, number of steps `n`. Run `n` steps of gradient descent on `f(x) = (x − 5)²` (derivative `2(x − 5)`) and print the final `x` with 4 decimal places.

```text
Sample input: 0 0.1 10
Sample output: 4.4631
```

<details>
<summary>Hint</summary>

```python
x0, lr, n = input().split()
x, lr, n = float(x0), float(lr), int(n)
for _ in range(n):
    x = x - lr * 2 * (x - 5)
```
</details>

<!-- exercise: lr-too-big -->
### Exercise 3: Too big a learning rate

During gradient-descent training the loss grows instead of shrinking and finally becomes `inf`. The most likely cause?

- the learning rate is too small
- the learning rate is too big
- there's too much data
- the computer is too slow

<details>
<summary>Answer and explanation</summary>

With too big a learning rate each step overshoots the minimum and lands further away, so training diverges. Lowering the learning rate usually fixes it.
</details>

<!-- exercise: chain-rule -->
### Exercise 4: The chain rule

`y = 4x` and `z = y²`. What is `dz/dx` at `x = 1`?

- 4
- 8
- 16
- 32

<details>
<summary>Answer and explanation</summary>

`dz/dy = 2y = 8` (since `y = 4`), `dy/dx = 4`; the product is 32. Or expand: `z = 16x²`, derivative `32x`, which is 32 at 1.
</details>

## 5.8 Further reading

- 3Blue1Brown: [Essence of calculus](https://www.3blue1brown.com/topics/calculus) — episodes 1–4 are enough.
- [Dive into Deep Learning · 2.4 Calculus](https://d2l.ai/chapter_preliminaries/calculus.html), [2.5 Automatic differentiation](https://d2l.ai/chapter_preliminaries/autograd.html)
- Next lesson: [Probability and information](../06-probability-information/en.md)
