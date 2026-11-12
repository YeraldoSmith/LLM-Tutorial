# 3. Neural networks: stacking simple things

> 🎯 **Goals**: see the limit of linear models (the XOR problem); understand what a "neuron" computes; meet activation functions such as ReLU, GELU and SiLU and why they are essential; build a multi-layer perceptron (MLP) and compute a forward pass; learn to count parameters.
>
> ⏱ **Time**: 60 minutes　|　📋 **You need**: lessons 1–2 of this part; part 1 lesson 4 (matrix multiplication)

## 3.1 One line isn't enough: the XOR problem

A linear model (regression or softmax classification) can only separate classes with a **straight line** (a plane in higher dimensions). Consider a classic puzzle — **XOR**: output 0 when both inputs are equal, 1 when they differ.

```text
 x2
 1 ┤  ●(0,1)→1        ○(1,1)→0
   │
 0 ┤  ○(0,0)→0        ●(1,0)→1
   └──┴─────────────────┴── x1
      0                 1
```

No straight line can separate the two ● from the two ○. In 1969 this problem made many people lose faith in neural networks. The fix: **add a "hidden layer" with a non-linear function in the middle.**

## 3.2 The neuron: weighted sum + activation

A **neuron** does two things:

```formula
output = activation( w₁x₁ + w₂x₂ + … + b ) = f(w · x + b)
```

1. a **weighted sum**, just like a linear model (dot product plus bias);
2. an **activation function**: a **non-linear** function `f` deciding how strongly the neuron "fires".

Many neurons side by side form a **layer**; feeding one layer's outputs into the next gives a **neural network**.

## 3.3 Activation functions

```python
import math

def relu(x):                 # negatives become 0, positives unchanged
    return max(0.0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):                 # output between -1 and 1
    return math.tanh(x)

def gelu(x):                 # used by the GPT family; a "smooth ReLU"
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

def silu(x):                 # also called Swish; used by Llama, Qwen and others (part of SwiGLU)
    return x * sigmoid(x)

print(" x     ReLU    GELU    SiLU    tanh")
for x in [-3, -1, -0.5, 0, 0.5, 1, 3]:
    print(f"{x:>4}  {relu(x):6.3f}  {gelu(x):6.3f}  {silu(x):6.3f}  {tanh(x):6.3f}")
```

| Activation | Properties | Typical use |
|------------|-----------|-------------|
| **ReLU** `max(0, x)` | simple, fast, less prone to vanishing gradients | early deep networks, CNNs |
| **sigmoid** | output 0–1; gradients near 0 at both ends (vanishing gradients) | binary output layers, gates |
| **tanh** | output -1–1 | RNNs/LSTMs |
| **GELU** | a smooth ReLU with a little output for negatives | BERT, GPT-2/3 |
| **SiLU / Swish** | `x · σ(x)`, smooth | Llama, Qwen, DeepSeek and others (SwiGLU) |

### Why non-linearity is essential

Without activations, two linear layers in a row give `W₂(W₁x) = (W₂W₁)x` — **still one linear map**! A hundred such layers are no better than one and still can't draw curves. Activations break this, letting networks represent bent, complex boundaries.

```python
def matvec(W, x):
    return [sum(w * v for w, v in zip(row, x)) for row in W]

W1 = [[1, 2], [3, 4]]
W2 = [[1, -1], [0, 2]]
x = [1, 1]
two_layers = matvec(W2, matvec(W1, x))
W21 = [[sum(W2[i][k] * W1[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
print(two_layers, matvec(W21, x))       # identical: two linear layers = one linear layer
```

## 3.4 Solving XOR with two layers

This network has 2 hidden neurons with ReLU. The weights are designed by hand (next lesson the network learns them itself):

```text
            h1 = ReLU(x1 + x2)
x1, x2 ──►                         ──► y = h1 − 2·h2
            h2 = ReLU(x1 + x2 − 1)
```

```python
def relu(v):
    return max(0, v)

def xor_net(x1, x2):
    h1 = relu(1 * x1 + 1 * x2 + 0)
    h2 = relu(1 * x1 + 1 * x2 - 1)
    return 1 * h1 - 2 * h2

for x1 in (0, 1):
    for x2 in (0, 1):
        print(x1, x2, "→", xor_net(x1, x2))
```

The hidden layer *transforms* the inputs into a new space where the problem becomes linearly separable. That is what hidden layers do: **learn better representations**.

## 3.5 The multi-layer perceptron (MLP)

Stack several fully connected layers (every input connected to every neuron) with activations between them and you have a **multi-layer perceptron** (MLP), also called a feed-forward network:

```text
input x (d_in)
   │  linear: h = W₁x + b₁        W₁ has shape d_hidden × d_in
   ▼  activate: h = ReLU(h)
hidden h (d_hidden)
   │  linear: z = W₂h + b₂        W₂ has shape d_out × d_hidden
   ▼
output z (d_out) ── followed by softmax for classification
```

```python
import math
import random
random.seed(0)

def linear(W, b, x):
    return [sum(w * v for w, v in zip(row, x)) + bi for row, bi in zip(W, b)]

def relu_vec(v):
    return [max(0.0, a) for a in v]

def softmax(z):
    m = max(z)
    e = [math.exp(a - m) for a in z]
    s = sum(e)
    return [a / s for a in e]

d_in, d_hidden, d_out = 3, 4, 2
W1 = [[random.gauss(0, 0.5) for _ in range(d_in)] for _ in range(d_hidden)]
b1 = [0.0] * d_hidden
W2 = [[random.gauss(0, 0.5) for _ in range(d_hidden)] for _ in range(d_out)]
b2 = [0.0] * d_out

x = [1.0, 0.5, -1.0]
h = relu_vec(linear(W1, b1, x))      # 4 hidden values
z = linear(W2, b2, h)                # 2 logits
print("hidden:       ", [round(v, 3) for v in h])
print("probabilities:", [round(v, 3) for v in softmax(z)])
```

The parameters are random, so the probabilities mean nothing yet — **training** (next lesson: backpropagation) adjusts them into useful values.

### Universal approximation

In theory, a two-layer MLP with a wide enough hidden layer can approximate almost any continuous function to any precision (the **universal approximation theorem**). In practice, **deeper** networks (more layers) are usually more efficient than merely wider ones — which is where the "deep" in "deep learning" comes from.

## 3.6 Counting parameters

A linear layer from `d_in` to `d_out` dimensions has:

```formula
parameters = d_in × d_out (weights) + d_out (biases)
```

The MLP above: `3×4 + 4 = 16` plus `4×2 + 2 = 10`, 26 parameters in total.

Every Transformer layer contains such an MLP (the FFN, feed-forward network), usually `d → 4d → d`. For GPT-2 small, `d = 768`:

```python
def linear_params(d_in, d_out, bias=True):
    return d_in * d_out + (d_out if bias else 0)

d = 768
ffn = linear_params(d, 4 * d) + linear_params(4 * d, d)
print(f"GPT-2 small FFN parameters per layer: {ffn:,}")     # about 4.7 million
print(f"over 12 layers: {12 * ffn:,}")                       # about 57 million, nearly half of all 124M
```

A large share of an LLM's parameters live in these MLPs, and research suggests the MLP layers do much of the work of "storing knowledge".

## 3.7 Summary

- Linear models only draw straight lines and can't solve problems like **XOR**.
- A **neuron** = weighted sum + **activation function**; neurons form **layers**, stacked layers form a **neural network**.
- Common activations: ReLU, GELU (GPT), SiLU/Swish (Llama and others); without them, many linear layers equal one.
- An **MLP** is linear → activation → linear…; hidden layers learn better **representations**; deeper networks are often more efficient.
- Linear-layer parameters = `d_in × d_out + d_out`; Transformer FFNs (`d → 4d → d`) hold many of a model's parameters.

## 3.8 Exercises

<!-- exercise: activations -->
### Exercise 1: Activation functions

Read a number `x` and print its ReLU, sigmoid and GELU (use `0.5·x·(1 + erf(x/√2))`), one per line, 4 decimal places.

```text
Sample input: 1
Sample output:
1.0000
0.7311
0.8413
```

<details>
<summary>Hint</summary>

`math.erf` is the error function and `math.sqrt(2)` is √2.
</details>

<!-- exercise: mlp-params -->
### Exercise 2: Count the parameters

Read a line of integers: the width of each layer of an MLP (first the input dimension, last the output dimension). Between each pair of neighbouring layers there is a linear layer with bias. Print the total number of parameters.

```text
Sample input: 3 4 2
Sample output: 26
```

<details>
<summary>Hint</summary>

`for a, b in zip(sizes, sizes[1:]): total += a * b + b`
</details>

<!-- exercise: forward -->
### Exercise 3: A forward pass

The starter code gives the weights of a 2 → 2 → 1 network (ReLU in the hidden layer, no activation at the output). Read two numbers `x1 x2` and print the network's output with 2 decimal places.

```text
Sample input: 1 2
Sample output: 1.50
```

<details>
<summary>Hint</summary>

Hidden layer first: `h = [relu(W1[i][0]*x1 + W1[i][1]*x2 + b1[i]) for i in range(2)]`, then the output: `W2[0]*h[0] + W2[1]*h[1] + b2`.
</details>

<!-- exercise: no-activation -->
### Exercise 4: Remove the activations

If you remove every activation function from a 10-layer network (leaving only linear layers), it becomes as expressive as:

- a network 10 times more powerful
- a single linear model
- a 5-layer network
- nothing changes

<details>
<summary>Answer and explanation</summary>

A composition of linear maps is linear: `W₁₀⋯W₂W₁x = (W₁₀⋯W₁)x`. Without non-linearity, any number of layers equals one.
</details>

<!-- exercise: xor-predict -->
### Exercise 5: Predict the output

```python
def relu(v):
    return max(0, v)

def net(x1, x2):
    h1 = relu(x1 + x2)
    h2 = relu(x1 + x2 - 1)
    return h1 - 2 * h2

print(net(1, 1), net(0, 1), net(2, 0))
```

<details>
<summary>Answer and explanation</summary>

`net(1,1)`: h1 = 2, h2 = 1, output 0; `net(0,1)`: h1 = 1, h2 = 0, output 1; `net(2,0)`: h1 = 2, h2 = 1, output 0.
</details>

## 3.9 Further reading

- 3Blue1Brown: [But what is a neural network?](https://www.3blue1brown.com/lessons/neural-networks)
- [TensorFlow Playground](https://playground.tensorflow.org/): change layers and neurons in your browser and watch the network carve up the data — very intuitive.
- Next lesson: [Backpropagation and autograd](../04-backprop-autograd/en.md)
