# 3. Embeddings: meaning as a list of numbers

> 🎯 **Goals**: understand why we don't use token numbers or one-hot vectors directly; see that an **embedding** is just a "lookup table"; measure closeness of meaning with **cosine similarity** and do analogies like "king − man + woman ≈ queen" with vectors; learn word vectors automatically from "the company a word keeps"; train a Bengio-style neural language model with `nn.Embedding` in PyTorch and watch it group the vowels together on its own; meet positional encodings and sentence embeddings.
>
> ⏱ **Time**: 80 minutes　|　📋 **You need**: lessons 1–2 of this part; part 1 lesson 4 (vectors, dot products)

After tokenization every token has a number — say `cat` is 1234 and `dog` is 5678. But the numbers themselves mean nothing: 1234 and 1235 are no more "alike" than 1234 and 5678. In this lesson we give every token a **vector**, so that tokens with similar meanings are also close together in space. These are **word vectors**, also called **embeddings**, and they form the first layer of every language model.

## 3.1 The trouble with numbers and one-hot vectors

At the end of last lesson we represented characters as one-hot vectors: 27 characters means 27 dimensions with a single 1 in each vector. This has two big problems:

1. **Too big**: with a vocabulary of 150,000 tokens, each one-hot vector has 150,000 dimensions, almost all zeros;
2. **No notion of "similar"**: any two different one-hot vectors are perpendicular (dot product 0), so `cat` is exactly as far from `dog` as from `aeroplane`.

What we want: each token represented by a **much shorter dense vector** (say 768 dimensions, each a decimal number), with similar meanings giving similar vectors.

## 3.2 An embedding is a lookup table

An **embedding layer** is just a matrix `E` of shape "vocabulary size × vector dimension". Row i is the vector of token i. "Turning a token into a vector" means **taking out the row with that number**:

```text
               dim 0    dim 1    dim 2    dim 3
token 0  [    0.12,   -0.40,    0.88,    0.05 ]
token 1  [   -0.71,    0.33,    0.10,   -0.26 ]
token 2  [    0.45,    0.91,   -0.37,    0.64 ]     ← input number 2 takes out this row
…
```

Mathematically, "take row i" equals "multiply a one-hot vector by E" — so an embedding layer is a linear layer computed faster by table lookup. The `W[a]` in last lesson's neural bigram was exactly such a table. The embedding matrix starts out random and is **learned during training** together with the rest of the model.

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
emb = nn.Embedding(num_embeddings=10, embedding_dim=4)   # 10 tokens in the vocabulary, 4 dimensions each
ids = torch.tensor([3, 1, 3])
print(emb(ids))                                          # rows 3, 1 and 3 (shape 3×4)
print(torch.equal(emb(ids)[0], emb.weight[3]))            # True: it's a table lookup
onehot = nn.functional.one_hot(ids, 10).float()          # a 3×10 one-hot matrix
print(torch.allclose(onehot @ emb.weight, emb(ids)))      # True: same as one-hot times the matrix
```

## 3.3 Meaning as vectors: similarity and analogies

Suppose (for this demo we design them **by hand**) each word vector has 4 dimensions meaning "royal", "male", "female" and "young":

```python
import math

E = {                    # royal  male  female  young
    "king":     [1.0, 1.0, 0.0, 0.0],
    "queen":    [1.0, 0.0, 1.0, 0.0],
    "man":      [0.0, 1.0, 0.0, 0.0],
    "woman":    [0.0, 0.0, 1.0, 0.0],
    "prince":   [1.0, 1.0, 0.0, 1.0],
    "princess": [1.0, 0.0, 1.0, 1.0],
    "boy":      [0.0, 1.0, 0.0, 1.0],
    "girl":     [0.0, 0.0, 1.0, 1.0],
}

def cosine(a, b):         # cosine similarity: closer to 1 the more the directions agree
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

print("king vs queen:", round(cosine(E["king"], E["queen"]), 3))
print("king vs girl: ", round(cosine(E["king"], E["girl"]), 3))

def analogy(a, b, c):     # a is to b as c is to ?   → find the word closest to b − a + c
    target = [vb - va + vc for va, vb, vc in zip(E[a], E[b], E[c])]
    return max((w for w in E if w not in (a, b, c)), key=lambda w: cosine(target, E[w]))

print("man : king = woman : ?    →", analogy("man", "king", "woman"))
print("man : boy  = woman : ?    →", analogy("man", "boy", "woman"))
print("king : prince = queen : ? →", analogy("king", "prince", "queen"))
```

`king − man + woman`: take "male" away from "king" and add "female", and you land right on "queen". **Directions in the vector space encode relationships of meaning** (gender, royalty, age…).

**Cosine similarity** is the most common way to compare two embeddings: the smaller the angle between two vectors, the closer the cosine is to 1; perpendicular gives 0; opposite directions give −1. It looks at direction only, not length.

> [!NOTE]
> Real models' embeddings have hundreds to thousands of dimensions, and the individual dimensions are **not** human-readable features like "gender" or "royalty" — meaning is spread over combinations of many dimensions. But in 2013 Google's **word2vec** really did find regularities like `king − man + woman ≈ queen` in automatically learned word vectors, which caused a sensation.

## 3.4 Learning word vectors from context

Real word vectors are of course not designed by hand. The linguist J. R. Firth famously said:

> "You shall know a word by the company it keeps."

This is the **distributional hypothesis**: words that appear in similar contexts have similar meanings. `cat` and `dog` both often appear near "feed", "pet" and "cute", so their meanings are close.

The most direct approach: count which words appear around each word (a **co-occurrence matrix**); each word's row is its vector. Then use **PPMI** (positive pointwise mutual information) to reduce the influence of words like "the" and "a" that appear everywhere:

```python
import math

CORPUS = """the cat drinks milk . the dog drinks water . a cat chases a mouse . a dog chases a cat .
the cat sleeps on the sofa . the dog sleeps on the floor . my cat is cute . my dog is cute .
i eat an apple . i eat a banana . she eats an orange . he eats a banana . an apple is sweet .
a banana is sweet . an orange is sweet . i drink milk . she drinks water . he drinks juice .
paris is a city in france . berlin is a city in germany . rome is a city in italy .
i live in paris . she lives in berlin . he lives in rome . we visit paris . they visit rome ."""
words = CORPUS.split()
vocab = sorted(set(words))
idx = {w: i for i, w in enumerate(vocab)}

# co-occurrence matrix: counts[w][c] = how often word c appears within 2 positions of word w
window = 2
counts = [[0] * len(vocab) for _ in vocab]
for i, w in enumerate(words):
    for j in range(max(0, i - window), min(len(words), i + window + 1)):
        if j != i:
            counts[idx[w]][idx[words[j]]] += 1

# PPMI: words are related only if they appear together more often than by chance
total = sum(map(sum, counts))
row = [sum(r) for r in counts]
col = [sum(counts[i][j] for i in range(len(vocab))) for j in range(len(vocab))]
vec = [[max(0.0, math.log(counts[i][j] * total / (row[i] * col[j]))) if counts[i][j] else 0.0
        for j in range(len(vocab))] for i in range(len(vocab))]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)

def nearest(word, k=3):
    v = vec[idx[word]]
    skip = {".", "the", "a", "an", "is", "in"}
    scores = [(cosine(v, vec[idx[w]]), w) for w in vocab if w != word and w not in skip]
    return [(w, round(s, 2)) for s, w in sorted(scores, reverse=True)[:k]]

print("vocabulary size:", len(vocab), "- each word vector has", len(vocab), "dimensions")
for w in ["cat", "apple", "paris", "mouse"]:
    print(f"{w:6s} closest words:", nearest(w))
```

With only 30 sentences the program discovers on its own that `cat` and `dog`, `apple` and `banana`, `paris` and `rome` are similar — we never told it what any of these words mean.

This "count the co-occurrences" approach has two drawbacks: the vector dimension equals the vocabulary size (too big), and the vectors are sparse. Methods such as **word2vec** (2013) and **GloVe** (2014) instead **train** short dense vectors to predict the surrounding words. In modern large models, word vectors are no longer trained separately: they are the model's first layer, learned **together with the whole model** for the goal of predicting the next token.

## 3.5 Hands-on: a neural language model learns its own embeddings

In 2003 Yoshua Bengio and colleagues proposed the first successful neural language model:

1. look up the embedding vector of each of the previous n tokens;
2. **concatenate** these vectors and feed them into an MLP;
3. output the probabilities of the next token (softmax).

It beats the bigram model because it sees a longer context (3 characters here), and the embeddings let similar characters share knowledge. We make the embeddings just **2-dimensional** so we can look directly at what they learn:

```python requires=torch
import torch
from torch import nn

NAMES = """emma olivia ava isabella sophia mia charlotte amelia harper evelyn abigail emily
elizabeth sofia avery ella madison scarlett victoria aria grace chloe camila penelope riley
layla lillian nora zoey mila aubrey hannah lily addison eleanor natalie luna savannah leah
stella hazel ellie audrey violet claire bella aurora lucy anna samantha caroline sarah maya
liam noah william james oliver benjamin elijah lucas mason logan alexander ethan jacob
michael daniel henry jackson sebastian aiden matthew samuel david joseph carter owen wyatt
john jack luke dylan levi isaac gabriel julian mateo anthony lincoln joshua andrew caleb
ryan nathan thomas leo isaiah charles hudson hunter connor eli ezra aaron adrian nolan""".split()

chars = list(".abcdefghijklmnopqrstuvwxyz")
stoi = {c: i for i, c in enumerate(chars)}
block = 3                                   # context length: look at the previous 3 characters
X, Y = [], []
for name in NAMES:
    context = [0] * block                   # pad the start with "."
    for ch in name + ".":
        X.append(context)
        Y.append(stoi[ch])
        context = context[1:] + [stoi[ch]]
X, Y = torch.tensor(X), torch.tensor(Y)
print("training examples:", tuple(X.shape), " e.g.", [chars[i] for i in X[7]], "→", chars[Y[7]])

torch.manual_seed(0)
class NameMLP(nn.Module):                   # a scaled-down version of Bengio et al.'s 2003 neural language model
    def __init__(self, vocab=27, d=2, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)   # one d-dimensional vector per character
        self.fc1 = nn.Linear(block * d, hidden)
        self.fc2 = nn.Linear(hidden, vocab)
    def forward(self, x):                   # x: (batch, 3) character numbers
        e = self.emb(x)                     # (batch, 3, d)
        h = torch.tanh(self.fc1(e.flatten(1)))   # concatenate the 3 vectors into (batch, 3d)
        return self.fc2(h)                  # (batch, 27) logits

model = NameMLP()
opt = torch.optim.AdamW(model.parameters(), lr=0.01)
for step in range(2001):
    ix = torch.randint(0, len(X), (64,))
    loss = nn.functional.cross_entropy(model(X[ix]), Y[ix])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        with torch.no_grad():
            full = nn.functional.cross_entropy(model(X), Y).item()
        print(f"step {step:4d}  loss {full:.3f}")

E = model.emb.weight.detach()
vowels = [stoi[c] for c in "aeiou"]
others = [stoi[c] for c in "bcdfghjklmnprstvwz"]
def avg_dist(A, B):
    d = [torch.dist(E[a], E[b]).item() for a in A for b in B if a != b]
    return sum(d) / len(d)
print(f"average distance between vowels {avg_dist(vowels, vowels):.2f}, between vowels and consonants {avg_dist(vowels, others):.2f}")

torch.manual_seed(1)                        # generate names with the trained model
names = []
for _ in range(8):
    context, out = [0] * block, ""
    while len(out) < 20:
        probs = torch.softmax(model(torch.tensor([context])), dim=1)
        ix = torch.multinomial(probs, 1).item()
        if ix == 0:
            break
        out += chars[ix]
        context = context[1:] + [ix]
    names.append(out)
print("generated names:", names)
```

Some observations:

- The loss is much lower than the bigram's (about 2.1) and the generated names look much better — the model can see the previous 3 characters.
- We never told the model what a vowel is, yet in the learned 2-dimensional embeddings **the vowels are closer to each other**: they play similar roles in names (they can often replace one another). This is how embeddings learn "meaning".
- The training data is only about a hundred names, so the model also "recites" some names from the training set (the overfitting from part 2 lesson 6). Real models train on billions or trillions of tokens.

## 3.6 Positions need embeddings too

The MLP above **concatenates** the 3 vectors in order, so it knows which came first. But **attention**, next lesson's topic, processes all tokens together "on an equal footing" and **doesn't know the order** by itself — yet "dog bites man" and "man bites dog" mean very different things.

So Transformers also give each **position** a vector:

- **learned position embeddings** (GPT-2): another "position × dimension" table, with one vector for position 0, one for position 1… **added** to the token embedding;
- **sinusoidal positional encoding** (the original Transformer): generated from sin/cos functions of different frequencies, nothing to learn;
- **RoPE, rotary position embedding** (Llama, Qwen, DeepSeek and other modern models): not added to the input but applied inside attention by "rotating" vectors according to their position. More next lesson.

```formula
input vector = token embedding[token number] + position embedding[position]
```

## 3.7 Embeddings in large models

| Model | Vocabulary | Embedding dimension | Embedding parameters |
|-------|------------|---------------------|----------------------|
| GPT-2 small | 50,257 | 768 | about 38.6 million (about 31% of its 124 million) |
| Llama 3 8B | 128,256 | 4,096 | about 525 million |
| Qwen3 family (larger models) | about 151,000 | 4,096–5,120 etc. | hundreds of millions |

A model's last layer turns vectors back into "a score for every token" (logits), and its weight matrix has the same shape as the embedding matrix. Many models **share one matrix** for both (**weight tying**), which saves parameters and keeps the "input meaning" and "output meaning" consistent.

## 3.8 Sentence embeddings: one vector for a whole passage

Turning a whole sentence or passage into one vector lets you compare whether two passages mean similar things. This is the basis of **semantic search** and of **RAG** (retrieval-augmented generation, part 7): turn documents and questions into vectors and find the documents with the highest cosine similarity. Models built for this are called **embedding models**; many are multilingual, so a question in one language can find documents in another:

```python norun
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")      # supports 100+ languages; also try Qwen/Qwen3-Embedding-0.6B
docs = ["The cat drinks milk", "Кошка пьёт молоко", "Paris is the capital of France", "巴黎是法国的首都"]
vecs = model.encode(docs, normalize_embeddings=True)
query = model.encode(["What do kittens like to drink?"], normalize_embeddings=True)
print(query @ vecs.T)                           # highest for the two sentences about cats, whatever the language
```

## 3.9 Summary

- Token numbers and one-hot vectors can't express "similar"; an **embedding** gives every token a short dense vector so that similar meanings get similar vectors.
- An embedding layer is a **lookup table** (vocabulary × dimension), equivalent to a one-hot vector times a matrix; it is trained with the model.
- **Cosine similarity** measures how closely directions agree; differences between vectors can encode relationships (`king − man + woman ≈ queen`).
- **Distributional hypothesis**: words with similar contexts have similar meanings; word vectors can be learned from co-occurrence counts or by predicting surrounding words.
- Neural language model (Bengio 2003): embeddings → concatenate → MLP → softmax; the embeddings automatically group tokens that play similar roles.
- Attention doesn't know order by itself, so **positional encodings** are needed (learned, sinusoidal, RoPE); **sentence embeddings** are the basis of semantic search and RAG.

## 3.10 Exercises

<!-- exercise: nearest-word -->
### Exercise 1: The closest word

The first line is an integer `n`; each of the next `n` lines contains a word and its vector (numbers separated by spaces); the last line is the word to look up. Print the **other** word with the **highest cosine similarity** to the query, and the similarity (4 decimal places), separated by a space.

```text
Sample input:
4
cat 0.9 0.8 0.1
dog 0.8 0.9 0.2
car 0.1 0.2 0.9
bus 0.2 0.1 0.8
cat
Sample output: dog 0.9899
```

<details>
<summary>Hint</summary>

Split each line with `parts = line.split()`: `parts[0]` is the word and `parts[1:]` converted to floats is the vector. Use `max(..., key=...)` to find the most similar word.
</details>

<!-- exercise: analogy -->
### Exercise 2: Word analogies

The starter code contains the hand-designed table from section 3.3. Read a line with three words `a b c` and print the word closest to `b − a + c` (it may not be `a`, `b` or `c` themselves).

```text
Sample input: man king woman
Sample output: queen
```

<details>
<summary>Hint</summary>

`target = [vb - va + vc for va, vb, vc in zip(E[a], E[b], E[c])]`, then find the other word with the highest cosine similarity.
</details>

<!-- exercise: embedding-params -->
### Exercise 3: How many parameters in the embeddings?

Read a line with three integers `V d tied`: the vocabulary size, the embedding dimension, and whether the input embedding and the output layer share one matrix (1 = shared, 0 = separate). Print the total number of parameters in these two parts (ignoring biases).

```text
Sample input: 50257 768 1
Sample output: 38597376
```

<details>
<summary>Hint</summary>

One matrix has `V × d` parameters; without sharing there are two.
</details>

<!-- exercise: why-not-onehot -->
### Exercise 4: The trouble with one-hot

What is the main problem with representing words as one-hot vectors?

- computers can't store 0s and 1s
- any two different words are perpendicular and equally far apart, so "similar meaning" can't be expressed, and the dimension equals the vocabulary size
- one-hot vectors can't be fed into neural networks
- one-hot vectors only work for English

<details>
<summary>Answer and explanation</summary>

Different one-hot vectors always have a dot product of 0, so they are all "equally far" apart, and their dimension is as large as the vocabulary (over a hundred thousand). Embeddings solve both problems with short dense vectors.
</details>

<!-- exercise: why-position -->
### Exercise 5: Why do we need positional encodings?

- so the model knows how big its vocabulary is
- because attention by itself doesn't distinguish the order of tokens, yet "dog bites man" and "man bites dog" mean different things
- to reduce the number of parameters
- so the model supports Chinese

<details>
<summary>Answer and explanation</summary>

Attention computes over all tokens together; without positional information, shuffling the order wouldn't change the result. Positional encodings (learned, sinusoidal or RoPE) tell the model "which position" each token is in.
</details>

## 3.11 Further reading

- Mikolov et al.: "Efficient Estimation of Word Representations in Vector Space" (2013, word2vec)
- Bengio et al.: "A Neural Probabilistic Language Model" (2003)
- Andrej Karpathy: [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) (the inspiration for section 3.5)
- Jay Alammar: [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)
- Next lesson: [Attention: letting every token "see" its context](../04-attention/en.md)
