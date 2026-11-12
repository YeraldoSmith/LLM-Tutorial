# 4. Vectors and matrices

> 🎯 **Goals**: see a vector as "a list of numbers" and as "an arrow in space"; compute addition, scaling, dot products and lengths; measure how similar two vectors are with cosine similarity; understand what matrix multiplication does and implement it in plain Python; learn to check shapes.
>
> ⏱ **Time**: 60 minutes　|　📋 **You need**: lesson 3; school maths (arithmetic, squares, square roots)

Almost every computation in a neural network is **vector and matrix arithmetic**. In an LLM each token is represented by a vector of thousands of numbers, and every layer is essentially a matrix multiplication. No proofs here — just **intuition + code**.

## 4.1 Vectors: a list of meaningful numbers

A **vector** is simply **an ordered list of numbers**. For example, describing a person:

```text
alex = [height 175, weight 68, age 20]
```

Together these three numbers form a 3-dimensional vector — a Python list does the job:

```python
alex = [175, 68, 20]
print(len(alex))     # the dimension: 3
```

A vector is also **an arrow in space**: the 2-D vector `[3, 4]` is the arrow from the origin going 3 right and 4 up.

```text
 y
 4 ┤        ↗ (3, 4)
 3 ┤      ╱
 2 ┤    ╱
 1 ┤  ╱
 0 ┼──┬──┬──┬── x
   0  1  2  3
```

In a language model each token is represented by a vector (an **embedding**, part 3 lesson 3). The magic: words with similar meanings get vectors pointing in similar directions.

## 4.2 Basic vector operations

### Addition: add matching positions

```python
a = [1, 2, 3]
b = [10, 20, 30]
c = [x + y for x, y in zip(a, b)]
print(c)        # [11, 22, 33]
```

Geometrically: put the arrows head to tail.

### Scaling: multiply every number by the same number

```python
a = [1, 2, 3]
print([2 * x for x in a])     # [2, 4, 6]: twice as long, same direction
```

### Length (norm): Pythagoras

The length of `[3, 4]` is √(3² + 4²) = 5. In any dimension: **square each number, add them up, take the square root**.

```python
import math

def norm(v):
    return math.sqrt(sum(x * x for x in v))

print(norm([3, 4]))        # 5.0
print(norm([1, 2, 2]))     # 3.0
```

## 4.3 The dot product: the most important operation in neural networks

The **dot product**: multiply matching positions, then add everything up — giving **one number**.

```formula
a · b = a₁b₁ + a₂b₂ + … + aₙbₙ
```

```python
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

print(dot([1, 2, 3], [4, 5, 6]))    # 1×4 + 2×5 + 3×6 = 32
```

Why does it matter? Two ways to see it:

**① A weighted sum.** A "neuron" in a neural network computes a dot product of its inputs `x` with its weights `w`; inputs with bigger weights matter more.

```python
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# will you like a film? [action, romance, comedy] content × how much you like each
movie = [0.9, 0.1, 0.4]
my_taste = [2.0, -1.0, 1.0]
print(dot(movie, my_taste))     # 1.8 - 0.1 + 0.4 = 2.1: a high score, you'll probably like it
```

**② Do the directions agree?** The more two vectors point the same way, the bigger the dot product; perpendicular vectors give 0; opposite directions give a negative number. This is the heart of **attention** in part 3: a dot product decides how related two words are.

## 4.4 Cosine similarity: how alike are two vectors?

The dot product also depends on the vectors' lengths. To compare only **directions**, divide by both lengths to get the **cosine similarity**, which ranges from -1 to 1:

```formula
cos(a, b) = (a · b) / (|a| × |b|)
```

| Value | Meaning |
|-------|---------|
| 1 | same direction (very similar) |
| 0 | perpendicular (unrelated) |
| -1 | opposite directions |

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def cosine(a, b):
    return dot(a, b) / (math.sqrt(dot(a, a)) * math.sqrt(dot(b, b)))

# imaginary 3-D "word vectors": [animal-ness, royalty, gender (male+ female-)]
king  = [0.1, 0.9, 0.8]
queen = [0.1, 0.9, -0.8]
cat   = [0.9, 0.0, 0.0]
print(f"king·queen: {cosine(king, queen):.2f}")
print(f"king·cat:   {cosine(king, cat):.2f}")
```

In part 7's **RAG** (retrieval-augmented generation) we'll use cosine similarity to find the documents most relevant to a question among thousands.

## 4.5 Matrices: a table of numbers

A **matrix** is a table of numbers in rows and columns. A `2 × 3` matrix has 2 rows and 3 columns:

```python
M = [
    [1, 2, 3],
    [4, 5, 6],
]
rows, cols = len(M), len(M[0])
print(rows, cols)     # 2 3: the matrix's "shape"
print(M[1][2])        # 6: row 2, column 3 (indexes start at 0)
```

A very useful view: **a matrix is a stack of vectors**, one per row. Five words with 4-dimensional vectors each form a `5 × 4` matrix.

### Transpose

Turn rows into columns: `m × n` becomes `n × m`.

```python
M = [[1, 2, 3],
     [4, 5, 6]]
T = [list(col) for col in zip(*M)]
print(T)      # [[1, 4], [2, 5], [3, 6]]
```

`zip(*M)` passes each row as a separate argument to `zip`, which then yields columns — the classic Python transpose trick.

## 4.6 Matrix times vector: many dot products at once

A matrix `W` (`m × n`) times a vector `x` (length `n`) gives a vector of length `m`: **each row of W is dotted with x**.

```text
        x = [1, 1, 2]
W = [[1, 0, 2],    →  1×1 + 0×1 + 2×2 = 5
     [0, 3, 1]]    →  0×1 + 3×1 + 1×2 = 5
W·x = [5, 5]
```

```python
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def matvec(W, x):
    return [dot(row, x) for row in W]

W = [[1, 0, 2],
     [0, 3, 1]]
print(matvec(W, [1, 1, 2]))    # [5, 5]
```

That's exactly what **one layer of a neural network** does: input vector `x`, times weight matrix `W`, gives the output vector. The shape of `W` sets the dimensions: 3 in, 2 out.

## 4.7 Matrix times matrix

Multiplying `A` (`m × n`) by `B` (`n × p`) gives `C` (`m × p`): **C at row i, column j = row i of A · column j of B**.

```text
A (2×3)        B (3×2)         C = A·B (2×2)
[1 2 3]        [7  8]          [1·7+2·9+3·11   1·8+2·10+3·12]   [ 58  64]
[4 5 6]   ×    [9 10]     =    [4·7+5·9+6·11   4·8+5·10+6·12] = [139 154]
               [11 12]
```

```python
def matmul(A, B):
    n = len(B)
    assert all(len(row) == n for row in A), "A's column count must equal B's row count"
    cols = list(zip(*B))                 # the columns of B
    return [[sum(a * b for a, b in zip(row, col)) for col in cols] for row in A]

A = [[1, 2, 3],
     [4, 5, 6]]
B = [[7, 8],
     [9, 10],
     [11, 12]]
print(matmul(A, B))     # [[58, 64], [139, 154]]
```

> [!IMPORTANT]
> **The shape rule**: `(m × n) · (n × p) = (m × p)`. The inner `n`s must match; the result takes the outer `m` and `p`. The most common bug in AI code is a shape mismatch. Get into the habit of **writing the shape in a comment at every step**.

Also, matrix multiplication is **not commutative**: in general `A·B ≠ B·A` (the shapes may not even fit).

## 4.8 Why are GPUs good at this?

The thousands of dot products in a matrix multiplication are **independent**, so they can all be computed at the same time — the GPU's speciality (lesson 2). Generating one token with an LLM takes billions to hundreds of billions of multiply-adds. Our hand-written version is slow, but the idea is identical: in NumPy and PyTorch, `A @ B` does exactly what `matmul` does.

```python requires=numpy
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
print(A @ B)
print(A.shape, B.shape, (A @ B).shape)
```

## 4.9 Summary

- A **vector** is a list of numbers and an arrow in space; operations: addition, scaling, length (root of the sum of squares).
- The **dot product** multiplies matching entries and sums them; it's a weighted sum and a measure of whether directions agree.
- **Cosine similarity** = dot product ÷ both lengths, from -1 to 1, compares how alike two vectors are.
- A **matrix** is a table (a stack of vectors); transpose with `zip(*M)`.
- Matrix × vector = dot each row with the vector = one neural-network layer.
- Matrix multiplication: `(m×n)·(n×p) = (m×p)`, C[i][j] = row i of A · column j of B; not commutative; always check shapes.

## 4.10 Exercises

<!-- exercise: dot -->
### Exercise 1: Dot product

Read two lines, each with the same number of integers separated by spaces, and print their dot product.

```text
Sample input:
1 2 3
4 5 6
Sample output:
32
```

<details>
<summary>Hint</summary>

`sum(x * y for x, y in zip(a, b))`
</details>

<!-- exercise: cosine -->
### Exercise 2: Cosine similarity

Read two lines, each a vector (numbers separated by spaces), and print their cosine similarity with 4 decimal places. Neither vector is all zeros.

```text
Sample input:
1 0
1 1
Sample output:
0.7071
```

<details>
<summary>Hint</summary>

Write `dot` first; the length is `math.sqrt(dot(v, v))`; then divide. Numbers may be decimals, so read them with `float`.
</details>

<!-- exercise: matmul -->
### Exercise 3: Matrix multiplication

Input: the first line holds `m n p`; then `m` lines of `n` integers (matrix A); then `n` lines of `p` integers (matrix B). Print `A·B`: `m` lines of `p` integers separated by spaces.

```text
Sample input:
2 3 2
1 2 3
4 5 6
7 8
9 10
11 12
Sample output:
58 64
139 154
```

<details>
<summary>Hint 1</summary>

Reading: `A = [list(map(int, input().split())) for _ in range(m)]`, and the same for B.
</details>

<details>
<summary>Hint 2</summary>

`C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))` — three nested loops, or follow 4.7.
</details>

<!-- exercise: shape -->
### Exercise 4: Shapes

A `4 × 3` matrix times a `3 × 5` matrix gives what shape?

- `4 × 5`
- `3 × 3`
- `5 × 4`
- they can't be multiplied

<details>
<summary>Answer and explanation</summary>

`(4×3)·(3×5) = (4×5)`. The inner 3s match, so they can be multiplied; the result takes the outer sizes.
</details>

<!-- exercise: dot-sign -->
### Exercise 5: The sign of a dot product

What is the dot product of two **perpendicular** vectors?

- 1
- 0
- -1
- infinity

<details>
<summary>Answer and explanation</summary>

0, e.g. `[1, 0]·[0, 1] = 0`. A positive dot product means roughly the same direction, a negative one roughly opposite.
</details>

## 4.11 Further reading

- 3Blue1Brown: [Essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra) — animations that build geometric intuition for vectors and matrices. Highly recommended.
- [Dive into Deep Learning · 2.3 Linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
- Next lesson: [Derivatives and gradients](../05-derivatives-gradients/en.md)
