# 1. Pre-training data: what models read

> 🎯 **Goals**: know how much data large models are pre-trained on and where it comes from; understand a complete data pipeline — text extraction, language identification, heuristic filtering, **deduplication** (exact and MinHash near-duplicate), quality classification, removing personal information, and **decontamination**; understand data mixtures, repeated epochs, the annealing phase and synthetic data; know how documents are **packed** into fixed-length sequences before training.
>
> ⏱ **Time**: 80 minutes　|　📋 **You need**: Part 3 (especially lessons 1, 2 and 6)

In Part 3 we trained a small GPT on a few thousand characters. Real large models do exactly the same thing — predict the next token — on billions of times more text. As people in the field say, **data sets the ceiling on what a model can do**. Many teams spend more effort on data than on model architecture.

## 1.1 How much data?

| Model | Year | Pre-training tokens |
|-------|------|---------------------|
| GPT-3 | 2020 | about 300 billion (0.3T) |
| Llama 2 | 2023 | about 2 trillion (2T) |
| Llama 3 | 2024 | over 15 trillion (15T) |
| DeepSeek-V3 | 2024 | about 14.8 trillion |
| Qwen3 | 2025 | about 36 trillion |

How much is 15 trillion tokens? At roughly 4 characters per English token and about 500,000 characters per book, it's around **120 million books**. Reading one book a day would take over 300,000 years.

## 1.2 Where the data comes from

- **Web pages**: most data comes from the web, and the most important public source is **Common Crawl** — a non-profit that has crawled the internet every month or two since 2008, billions of pages each time, freely available.
- **Code**: open-source repositories on GitHub and elsewhere. Code doesn't just teach models to program; it's also thought to improve reasoning.
- **Books, papers, encyclopaedias**: Wikipedia, arXiv papers, public-domain books — high quality but small in quantity.
- **Maths and science**: maths web pages, textbooks, worked solutions, which have become more and more important.
- **Multilingual data**: there's far less Chinese, Russian, etc. on the web than English, so it's usually collected specially and given extra weight.
- **Synthetic data**: text generated or rewritten by existing large models (section 1.6).

Where data comes from also raises questions of **copyright and licensing**: which data may be used for training, and whether permission is needed, has been a hot topic for lawsuits and legislation around the world since 2023. The open-source community has therefore built many datasets with clear sources and licences, such as Hugging Face's **FineWeb** (released 2024: about 15 trillion tokens of English web text processed from 96 Common Crawl snapshots, with all the processing code published).

## 1.3 The data pipeline

Raw web pages are full of navigation bars, ads, garbled text, duplicates and low-quality writing. A typical pipeline looks like this:

```text
raw web pages (HTML)
   │ ① extract the main text: drop HTML tags, navigation, footers
   ▼
plain text ── ② language identification: which language is this?
   │ ③ heuristic filtering: drop text that's too short, garbled, too symbol-heavy, too repetitive
   │ ④ deduplication: keep one copy of identical and near-identical documents
   │ ⑤ quality classification: a model scores "educational value", "writing quality"
   │ ⑥ remove personal information: emails, phone numbers, ID numbers…
   │ ⑦ decontamination: delete text that overlaps with benchmark questions
   ▼
text ready for training (often only a few percent of the raw data)
```

### Heuristic filtering

First, simple rules quickly throw out obvious junk. DeepMind collected a set of such rules (document length, mean word length, symbol ratio, repetition and so on) while training Gopher (2021), and they're now widely used. Here's a simplified version:

```python
def check(doc):
    words = doc.split()
    if not 5 <= len(words) <= 100_000:          # Gopher requires at least 50 words; relaxed to 5 for this demo
        return "too short or long"
    mean_len = sum(len(w) for w in words) / len(words)
    if not 3 <= mean_len <= 10:
        return "odd word length"
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8:
        return "too few real words"
    if len({w.lower() for w in words}) / len(words) < 0.3:
        return "too repetitive"
    return "keep"

docs = [
    "Photosynthesis is the process by which green plants turn sunlight, water and carbon dioxide into sugar.",
    "Home | About | Login",
    "$$$ 100% !!! ### 2026 *** >>> <<< ### $$$ 100%",
    "click here click here click here click here click here click here",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb c d e",
]
for d in docs:
    print(f"{d[:40]:<42}→ {check(d)}")
```

These rules are crude but very fast. Note that they're designed **for a particular language**: Chinese has no spaces between words, so `doc.split()` is almost useless for it; Chinese text is measured in characters, or segmented into words first.

### Exact deduplication

The same article gets copied onto hundreds or thousands of websites. Duplicates waste compute and make the model **memorise** that text. The simplest deduplication: normalise the text (same case, collapsed whitespace), compute a **hash**, and keep only the first document for each hash:

```python
import hashlib
import re

def key(doc):
    normalized = re.sub(r"\s+", " ", doc.lower()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()

docs = [
    "Water boils at 100 degrees Celsius at sea level.",
    "Paris is the capital of France.",
    "water  boils at 100 degrees celsius at   sea level.",
    "Water boils at 100 degrees Celsius at sea level!",
]
seen, unique = set(), []
for d in docs:
    k = key(d)
    if k not in seen:
        seen.add(k)
        unique.append(d)
print(f"{len(docs)} documents → {len(unique)}")
for d in unique:
    print(" ", d)
```

The third document differs only in case and spacing, so it's removed; but the fourth has one extra exclamation mark and a completely different hash. The web is full of "near duplicates" — a few words changed, a date added, a different footer — which need **near-duplicate detection**.

### Near-duplicate detection: MinHash

A common measure of how alike two documents are is the **Jaccard similarity**: cut each document into small pieces (**shingles**, say every 5 consecutive characters), and divide the size of the intersection of the two sets of shingles A and B by the size of their union:

```formula
J(A, B) = |A ∩ B| / |A ∪ B|
```

But comparing billions of documents in pairs is impossible. The clever part of **MinHash**: score every shingle with a hash function and take the smallest; the probability that two documents have the same "smallest shingle" is **exactly** their Jaccard similarity. With 128 different hash functions you get a "signature" of 128 numbers, and the fraction of positions where two signatures agree estimates the Jaccard similarity:

```python
import hashlib

def shingles(text, k=5):
    t = " ".join(text.lower().split())
    return {t[i:i + k] for i in range(len(t) - k + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b)

def h(s, seed):                       # the seed-th hash function
    digest = hashlib.md5(f"{seed}:{s}".encode()).digest()
    return int.from_bytes(digest[:8], "big")

def minhash(sh, num=128):             # the signature: the minimum under each hash function
    return [min(h(s, seed) for s in sh) for seed in range(num)]

def estimate(sig1, sig2):
    return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)

docs = {
    "A": "The quick brown fox jumps over the lazy dog near the old river bank.",
    "B": "The quick brown fox jumped over the lazy dog near the old river bank!",
    "C": "Large language models are trained on trillions of tokens of text.",
}
sh = {k: shingles(v) for k, v in docs.items()}
sig = {k: minhash(v) for k, v in sh.items()}
for x, y in [("A", "B"), ("A", "C")]:
    print(f"{x}-{y}: true Jaccard {jaccard(sh[x], sh[y]):.2f}, MinHash estimate {estimate(sig[x], sig[y]):.2f}")
```

A signature of 128 numbers is far shorter than the text. Add **locality-sensitive hashing** (LSH): cut each signature into b bands of r numbers, and whenever any band matches exactly, list the two documents as "candidates" for a careful comparison. Two documents with similarity s become candidates with probability `1 - (1 - s^r)^b`. With b = 16 and r = 8:

| Similarity s | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
|--------------|-----|-----|-----|-----|-----|
| Probability of becoming candidates | 0.001 | 0.06 | 0.61 | 0.95 | 1.00 |

Nearly all similar pairs are found and nearly all dissimilar pairs are never compared — so deduplicating billions of documents takes only linear time.

### Quality classifiers

Rules only remove obvious junk. The next step is to **train a classifier to score documents**. FineWeb-Edu (2024) is a typical example: a large model (Llama 3 70B) first rated hundreds of thousands of web pages for "educational value" from 0 to 5; those ratings trained a small, fast classifier that then scored every page, and only high scorers were kept. Models trained on just these 1.3 trillion tokens of "educational" pages did clearly better on knowledge and reasoning tests than models trained on all the pages.

### Removing personal information, and decontamination

- **Personal information**: rules and models find email addresses, phone numbers, IP addresses, ID numbers and so on, which are replaced or deleted.
- **Decontamination**: if benchmark questions (and their answers) appear in the training data, a model's high benchmark score may just be "memorised". So documents with long overlaps with benchmarks (say, 13 consecutive identical words) are deleted. Lesson 5 returns to this.

## 1.4 Mixtures, repetition and annealing

How data from different sources is mixed is the **data mixture**. It has a huge effect on the model and is usually decided after many experiments with small models. A key issue: high-quality data is often scarce, so the mixture may require **reading it several times**:

```python
budget = 15_000                # train on 15 trillion tokens in total (unit: billions)
sources = {                    # name: (how much there is, share of the mix) — illustrative numbers
    "web":   (30_000, 0.70),
    "code":  (2_000, 0.15),
    "maths": (500, 0.07),
    "books and papers": (400, 0.05),
    "multilingual": (3_000, 0.03),
}
for name, (size, share) in sources.items():
    used = budget * share
    print(f"{used / size:4.2f} epochs ({used:>6,.0f}B tokens)  {name}")
```

Research (Muennighoff et al., 2023) shows that repeating the same data **up to about 4 times** works about as well as fresh data; beyond that, the benefit drops quickly.

Many models end pre-training with a phase where the learning rate goes very low and the share of high-quality data (maths, code, textbooks, selected web pages) goes up. This is **annealing**, and it improves the model noticeably with relatively few tokens.

## 1.5 Packing into training sequences

The model processes sequences of a fixed length (say 4096 or 8192 tokens), but documents vary in length. The usual approach is to join all documents end to end, put an **end-of-sequence token** (EOS) after each one, and cut the stream into equal blocks:

```python
docs = [[5, 8, 2], [7, 7], [1, 2, 3, 4, 5, 6], [9]]   # documents already tokenised (token ids)
EOS = 0
L = 4                                                 # length of each training sequence

stream = []
for d in docs:
    stream += d + [EOS]                               # EOS between documents
print("joined:", stream)

for i in range(0, len(stream) - L, L):
    x = stream[i:i + L]                               # input
    y = stream[i + 1:i + L + 1]                       # target: the "next token" at every position
    print("input", x, "→ target", y)
```

- The target is the input shifted by one: every position predicts the next token, exactly as in Part 3, lesson 6.
- After packing, one sequence may contain several unrelated documents. Simple implementations let them "see" each other; more careful ones change the attention mask so each token only sees earlier tokens **of the same document**.
- Long documents get split across sequences. A dedicated "long context" training phase uses longer sequences and more long documents (lesson 4).

## 1.6 Synthetic data

High-quality human text is running out, so **synthetic data** keeps growing in importance:

- **Rewriting**: a model rewrites mediocre web pages into clear question-and-answer or textbook-style text;
- **Generation**: Microsoft's Phi models (from 2023) used lots of model-generated "textbook" text, giving small models good reasoning ability;
- **Verifiable data**: maths problems and code can be checked automatically, so they can be generated in bulk and filtered (Part 5 shows how reasoning models exploit this).

Synthetic data has risks too: if generation after generation of models is trained only on the previous generation's output, the output grows ever more uniform and rare knowledge is lost — **model collapse**. So synthetic data is generally mixed with real data and carefully filtered.

## 1.7 Summary

- Pre-training data is measured in trillions of tokens: over 15T for Llama 3, about 36T for Qwen3.
- The main source is the web (Common Crawl), plus code, books, papers, maths and multilingual data.
- The pipeline: extract text → identify language → heuristic filters → deduplicate → quality classifier → remove personal data → decontaminate; often only a few percent survives.
- Exact deduplication: normalise + hash; near-duplicates: shingles + MinHash (fraction of matching positions ≈ Jaccard) + LSH bands.
- The mixture sets each kind of data's share; high-quality data can be repeated a few times; the final annealing phase uses more high-quality data.
- Documents are joined end to end with EOS between them and cut into equal sequences; the target is the input shifted by one.
- Synthetic data is increasingly important, but model collapse must be avoided.

## 1.8 Exercises

<!-- exercise: jaccard -->
### Exercise 1: Jaccard similarity

Read two lines of English. Lower-case each, split it into words on whitespace, and take every pair of **adjacent words** as a shingle (`the cat sat` has the shingles `the cat` and `cat sat`). Print the Jaccard similarity of the two sets of shingles to 3 decimal places. (Each line has at least 2 words.)

```text
Sample input:
the cat sat on the mat
the cat sat on a mat
Sample output: 0.429
```

<details>
<summary>Hint</summary>

`words = line.lower().split()`, and the shingle set is `{(words[i], words[i + 1]) for i in range(len(words) - 1)}`. Jaccard = size of the intersection / size of the union.
</details>

<!-- exercise: gopher-filter -->
### Exercise 2: Heuristic filtering

The first line holds the number of documents `n`; each of the next `n` lines is one document. Check each document against the four rules below and print `keep` if it passes them all, `drop` otherwise:

1. the number of words (split on whitespace) is between 5 and 1000;
2. the mean word length is between 3 and 10 (inclusive);
3. at least 80% of the words contain a letter;
4. distinct words (after lower-casing) make up at least 0.3 of all words.

```text
Sample input:
2
Photosynthesis turns sunlight water and carbon dioxide into sugar.
click here click here click here click here
Sample output:
keep
drop
```

<details>
<summary>Hint</summary>

Follow the `check` function in section 1.3, with an upper limit of 1000 words. Contains a letter: `any(c.isalpha() for c in w)`.
</details>

<!-- exercise: exact-dedup -->
### Exercise 3: Exact deduplication

The first line holds the number of documents `n`, and the next `n` lines are the documents. Two documents are duplicates if they're identical after "lower-casing, collapsing runs of whitespace into one space and trimming the ends". Keep only the first document of each group of duplicates: print the number kept, then the kept documents in their original order (print the **original** text).

```text
Sample input:
3
Hello World
hello   world
Goodbye
Sample output:
2
Hello World
Goodbye
```

<details>
<summary>Hint</summary>

Normalise with `" ".join(doc.lower().split())`, and remember the normalised forms you've seen in a set.
</details>

<!-- exercise: mixture-epochs -->
### Exercise 4: How many epochs of each source?

The first line is the total number of training tokens `T` (in billions). Each following line describes one source as `name size share` (size also in billions of tokens), until the end of the input. As in section 1.4, print how many times each source is read, as `name epochs` with 2 decimal places.

```text
Sample input:
1000
web 2000 0.8
code 100 0.15
math 20 0.05
Sample output:
web 0.40
code 1.50
math 2.50
```

<details>
<summary>Hint</summary>

Epochs = `T * share / size`. To read until the end of input: `import sys`, `for line in sys.stdin:`.
</details>

<!-- exercise: decontam -->
### Exercise 5: Why decontaminate?

What is "decontamination" in pre-training data processing for?

- deleting pages containing viruses and malicious code
- keeping benchmark questions out of the training data so that scores aren't inflated
- deleting text containing personal information
- removing duplicate web pages

<details>
<summary>Answer and explanation</summary>

If a model has seen benchmark questions and answers during training, its high score may only mean it memorised them, not that it's truly capable. So training documents with long overlaps with benchmarks are deleted. Personal information and deduplication are other steps of the pipeline.
</details>

<!-- exercise: minhash-why -->
### Exercise 6: What is MinHash for?

In data processing, what is MinHash mainly used for?

- scoring the quality of documents
- identifying a document's language
- efficiently finding nearly identical documents
- splitting documents into tokens

<details>
<summary>Answer and explanation</summary>

MinHash estimates the Jaccard similarity of two documents from a short signature, and with LSH bands it quickly finds near-duplicates among huge numbers of documents. Quality scoring uses classifiers, language identification uses dedicated models, and tokenisers split text into tokens.
</details>

## 1.9 Further reading

- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (2024) — every step of the data processing in detail, with code and data published
- Rae et al., *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (2021) — the quality filtering rules are in an appendix
- Muennighoff et al., *Scaling Data-Constrained Language Models* (2023) — repeated data is still useful, up to a point
- Shumailov et al., *AI models collapse when trained on recursively generated data* (Nature, 2024)
- Next lesson: [Scaling laws and compute](../02-scaling-compute/en.md)
