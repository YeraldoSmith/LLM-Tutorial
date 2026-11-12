# 3. The Python you need for AI

> 🎯 **Goals**: review the Python idioms that appear most in AI code (comprehensions, counting with dictionaries, sorting, functions, classes); experiment with `random` and `math`; read and write text files; get a first look at NumPy and why "vectorisation" is fast.
>
> ⏱ **Time**: 45 minutes　|　📋 **You need**: Python basics (variables, conditions, loops, lists, dictionaries, functions)

This lesson doesn't teach Python from zero (that's [Our-tutorial's Python course](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/python)); it gathers **the idioms you need most to read and write AI code**. If something is unclear, revisit the matching Python lesson.

## 3.1 List comprehensions: process a batch in one line

AI code constantly does "the same thing to a group of numbers":

```python
scores = [2.0, -1.0, 0.5, 3.0]
doubled = [x * 2 for x in scores]
positive = [x for x in scores if x > 0]
relu = [max(0.0, x) for x in scores]        # ReLU: negatives become 0 (common in neural networks)
print(doubled)
print(positive)
print(relu)
```

Nested comprehensions build 2-D lists (matrices):

```python
rows, cols = 2, 3
zeros = [[0.0 for _ in range(cols)] for _ in range(rows)]
table = [[r * cols + c for c in range(cols)] for r in range(rows)]
print(zeros)
print(table)
```

> [!WARNING]
> Don't write `[[0.0] * cols] * rows`! The outer `* rows` copies a reference to **the same** inner list, so changing one row changes them all (Python lesson 8).

## 3.2 Dictionaries: counting and lookup

Tokenizing, counting words and mapping "word → number" all rely on dictionaries.

```python
text = "to be or not to be"
counts = {}
for w in text.split():
    counts[w] = counts.get(w, 0) + 1
print(counts)

from collections import Counter
print(Counter(text.split()).most_common(2))   # the standard library's counter is handier
```

A **vocabulary** is the foundation of a language model: give each distinct word (or character) a number, turning text into numbers:

```python
text = "hello world hello"
vocab = sorted(set(text.split()))                  # unique, sorted
stoi = {w: i for i, w in enumerate(vocab)}        # string to index
itos = {i: w for w, i in stoi.items()}            # index to string
ids = [stoi[w] for w in text.split()]
print(stoi)
print(ids)
print(" ".join(itos[i] for i in ids))             # back to text
```

You'll see the names `stoi` / `itos` in many open-source projects, such as Karpathy's nanoGPT.

## 3.3 Sorting with `key`

```python
words = {"the": 7, "cat": 3, "sat": 3, "on": 5}
by_count = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
print(by_count)
top2 = [w for w, c in by_count[:2]]
print(top2)
```

`key=lambda kv: kv[1]` means "sort by the second element of each item (the count)". A `lambda` is a tiny one-line function: `lambda arguments: result`.

To break ties alphabetically, return a tuple from `key`: `key=lambda kv: (-kv[1], kv[0])` — first by count, largest first (the minus sign), then by word.

## 3.4 Functions and classes: packaging a model

Each neural-network layer is usually a **class**: it stores its parameters when created and computes the output when called.

```python
class Linear:
    """The simplest possible "layer": y = w * x + b"""
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, x):          # lets the object be called like a function
        return self.w * x + self.b

layer = Linear(w=2.0, b=1.0)
print(layer(3.0))                   # 7.0
print([layer(x) for x in [0, 1, 2]])
```

`__call__` makes `layer(3.0)` possible — exactly how PyTorch models are used: `output = model(input)`.

## 3.5 random and math: tools for experiments

```python
import math
import random

random.seed(42)                         # fix the seed: same results every run, easy to reproduce
print(random.random())                  # a random float in [0, 1)
print(random.randint(1, 6))             # roll a die
print(random.choice(["rock", "paper", "scissors"]))
print(random.choices(["a", "b"], weights=[0.9, 0.1], k=5))   # sample by probability (how an LM picks words)
print(random.gauss(0, 1))               # normally distributed (used to initialise network weights)

print(math.exp(1), math.log(math.e))    # exp and natural log: you'll use them daily for probabilities
print(math.sqrt(2), math.pi)
```

> [!TIP]
> A **seed** makes "random" repeatable: the same seed gives the same sequence. Always set one in AI experiments; otherwise you can't tell whether a change in results came from your code or from luck.

## 3.6 Reading and writing text files

The first step in training a language model is reading text:

```python
from pathlib import Path

Path("tiny.txt").write_text("line one\nline two\nline three\n", encoding="utf-8")

text = Path("tiny.txt").read_text(encoding="utf-8")
print(len(text), "characters")
for i, line in enumerate(text.splitlines(), 1):
    print(i, line)
```

Always pass `encoding="utf-8"`, or text may get garbled on Windows ([Start here, lesson 6](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/06-numbers-and-text)).

## 3.7 A first look at NumPy: why "vectorise"?

Heavy number-crunching with Python lists is slow because the interpreter handles every number individually. **NumPy** stores data in contiguous memory and uses compiled C code to **process whole arrays at once** — often tens to hundreds of times faster. Practically every AI library (PyTorch included) thinks in arrays like this.

```python requires=numpy
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])
print(a + b)            # element-wise: [11. 22. 33.]
print(a * 2)            # every element times 2
print(a @ b)            # dot product (next lesson's star): 140.0
m = np.zeros((2, 3))    # a 2×3 matrix of zeros
print(m.shape)
print(np.exp(a))        # e to the power of each element
```

This needs NumPy installed (`pip install numpy` or `uv add numpy`). The exercises in parts 1–3 use **plain Python** so you can see and change every step; once you understand the ideas you'll find NumPy/PyTorch simply do the same things faster.

### Compare the speed

```python requires=numpy
import time
import numpy as np

n = 1_000_000
xs = list(range(n))
t = time.perf_counter()
total = sum(x * x for x in xs)
t_list = time.perf_counter() - t

arr = np.arange(n, dtype=np.float64)
t = time.perf_counter()
total_np = float((arr * arr).sum())
t_np = time.perf_counter() - t
print(f"Python list: {t_list * 1000:.1f} ms, NumPy: {t_np * 1000:.1f} ms")
```

## 3.8 Summary

- Comprehensions `[f(x) for x in xs if condition]` process batches; build 2-D lists with nested comprehensions, never `[[0]*n]*m`.
- Dictionaries count and build vocabularies (`stoi`, `itos`); `Counter.most_common` is handy.
- `sorted(..., key=lambda ...)` sorts by any rule; tuple keys give multi-level sorting.
- Classes with `__call__` package "layers" and "models".
- `random.seed` makes experiments reproducible; `random.choices` samples by probability; `math.exp` and `math.log` are everyday tools.
- Use `encoding="utf-8"` for files; NumPy makes numeric code much faster through vectorisation.

## 3.9 Exercises

<!-- exercise: top-words -->
### Exercise 1: Top three words

Read a line of lowercase English words separated by spaces. Print the 3 most frequent words with their counts, one per line, as `word count`. Break ties alphabetically.

```text
Sample input: to be or not to be that is the question to be
Sample output:
be 3
to 3
is 1
```

<details>
<summary>Hint 1</summary>

Count with a dictionary (or `Counter`) first.
</details>

<details>
<summary>Hint 2</summary>

`sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]`: by count, largest first, then by word.
</details>

<!-- exercise: vocab -->
### Exercise 2: Build a vocabulary

Read a line of text and build a **character** vocabulary: number all distinct characters in **sorted** order starting from 0, then print the number of each character of the original text, separated by spaces.

```text
Sample input: hello
Sample output: 1 0 2 2 3
```

(The sorted characters are `e h l o`, numbered 0, 1, 2, 3.)

<details>
<summary>Hint</summary>

`chars = sorted(set(text))`, `stoi = {ch: i for i, ch in enumerate(chars)}`.
</details>

<!-- exercise: grid-trap -->
### Exercise 3: The comprehension trap

What is `grid` after this code runs?

```python
grid = [[0] * 2] * 2
grid[0][0] = 9
print(grid)
```

<details>
<summary>Answer and explanation</summary>

`[[9, 0], [9, 0]]`. `* 2` copies a reference to the same inner list, so both rows are one list. Correct: `[[0] * 2 for _ in range(2)]`.
</details>

<!-- exercise: seed -->
### Exercise 4: Random seeds

Why set a random seed (e.g. `random.seed(42)`) in AI experiments?

- To make the program run faster
- To make results reproducible so you can compare the effect of changes
- To make random numbers more random
- To make the model more accurate

<details>
<summary>Answer and explanation</summary>

The same seed gives the same random sequence. With a fixed seed, differences in results come from your changes, not from luck.
</details>

## 3.10 Further reading

- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/): a clear NumPy chapter, free online.
- Next lesson: [Vectors and matrices](../04-vectors-matrices/en.md)
