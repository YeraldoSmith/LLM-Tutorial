# 2. Tokenization: how text becomes tokens

> 🎯 **Goals**: understand why models need "tokenization" and the trade-offs of splitting by character, word or subword; understand Unicode and **UTF-8 bytes**; implement the **BPE (byte pair encoding)** tokenizer used by the GPT family yourself — training, encoding, decoding; see how a tokenizer's training data affects Chinese, Russian and other non-English users; meet special tokens and chat templates; understand why LLMs "can't count letters" and similar oddities.
>
> ⏱ **Time**: 80 minutes　|　📋 **You need**: lesson 1 of this part

Last lesson each character was a token. Real large models don't do that: they use a **tokenizer** to cut text into **subword** pieces and replace each piece with a number. The tokenizer looks like a mere "preprocessing" step, but it deeply affects a model's abilities, speed and cost — especially for people who don't speak English.

> [!NOTE]
> The BPE implementation in this lesson follows Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe) (MIT licence) and his video [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE).

## 2.1 Why tokenize? Three ways to cut

Neural networks only understand numbers, so text must first be cut into pieces and each piece given a number. How to cut is a trade-off:

| Method | Vocabulary size | Sequence length | Problems |
|--------|-----------------|-----------------|----------|
| **by character** | small (dozens for English; tens of thousands with Chinese characters) | very long | sequences are too long and expensive; a single character means almost nothing |
| **by word** | huge (hundreds of thousands or more) | short | new words, typos and names can't be represented ("out-of-vocabulary"); Chinese needs word segmentation first |
| **by subword** (BPE etc.) | moderate (30,000–250,000) | moderate | common words are one token, rare words split into a few pieces — the best of both |

Almost every modern LLM uses **subword** tokenization. For example, `unbelievable` might become `un` + `believ` + `able`, and even a word the model has never seen can always be assembled from existing pieces.

## 2.2 From characters to bytes: Unicode and UTF-8

Every character in a computer has a **Unicode number** (code point): `A` is 65, `你` is 20320. For storage and transmission, the most common encoding, **UTF-8**, turns a code point into 1–4 **bytes** (numbers from 0 to 255): an English letter takes 1 byte, a Cyrillic letter 2, a Chinese character 3, an emoji 4.

```python
for ch in ["A", "é", "Ж", "你", "😀"]:
    b = ch.encode("utf-8")
    print(f"{ch!r}: code point {ord(ch):6d}  UTF-8 bytes {list(b)} ({len(b)})")

text = "Hi 你好 Привет"
print(len(text), "characters,", len(text.encode("utf-8")), "bytes")
print(bytes([228, 189, 160]).decode("utf-8"))     # bytes back to text
```

Since GPT-2, most tokenizers start from **bytes**: any text (any language, emoji, even garbage) can be written as a byte sequence, so there is **never a character that can't be represented**. The first 256 tokens of the vocabulary are these 256 byte values; every later token is **merged** from them.

## 2.3 BPE: byte pair encoding

Training **BPE** (Byte Pair Encoding) is remarkably simple:

1. turn the training text into a byte sequence (each byte is a token, numbered 0–255);
2. count how often every **pair of neighbouring tokens** occurs;
3. merge **the most frequent pair** into a new token (number 256, 257…) and record this "merge rule";
4. repeat steps 2–3 until the vocabulary reaches the desired size.

To **encode** new text, apply the learned rules **in the order they were learned**; to **decode**, join the bytes of every token and decode them as UTF-8.

```python
EN = """A language model reads text as a sequence of tokens and learns to predict the next token.
The model never sees letters or words directly: it only sees numbers. Before training, we must
decide how to cut the text into pieces. If the pieces are single characters, the sequences become
very long. If the pieces are whole words, the vocabulary becomes huge and new words cannot be
represented. Byte pair encoding finds a middle way: it starts from bytes and repeatedly merges the
most frequent pair of neighbouring tokens into a new token. Frequent words become single tokens,
while rare words are split into smaller pieces. The same tokenizer is used for training the model
and for every request the model receives later, so the tokenizer is part of the model."""

def get_stats(ids):                       # count how often each neighbouring pair occurs
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):             # replace every occurrence of pair in ids with new_id
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text, vocab_size):
    ids = list(text.encode("utf-8"))      # start from bytes: 0–255
    merges = {}                           # (a, b) → number of the new token
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)  # the most frequent pair
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))   # earliest-learned rule first
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

merges, vocab = train(EN, 256 + 100)      # learn 100 merge rules
print("first 10 learned tokens:", [vocab[i].decode("utf-8", "replace") for i in range(256, 266)])
sentence = "the model predicts the next token"
ids = encode(sentence, merges)
print("encoded:", ids)
print("pieces: ", [vocab[i].decode("utf-8", "replace") for i in ids])
print(f"{len(sentence)} characters → {len(ids)} tokens; decodes back: {decode(ids, vocab) == sentence}")
```

The first merges are extremely common fragments such as `e ` (a word-final e plus a space), ` t` and `he `; whole words such as ` the `, `model ` and ` token` soon follow. Common words become 1 token; less common ones (like `predicts`) are split into pieces.

> [!NOTE]
> Real tokenizers add a **pre-tokenization** step before BPE: a regular expression first splits the text into words, numbers, punctuation and spaces, and BPE only merges inside each chunk. This prevents `dog.`, `dog!` and `dog?` from being learned as three different tokens.

## 2.4 Whom does the tokenizer favour? The "token tax" on non-English users

A tokenizer is learned from its **training data**. If that data is mostly English, English gets compressed well while other languages are left as scattered bytes. Below we train one tokenizer on English only and another on English + Chinese + Russian, then compare how many tokens sentences with the same meaning need:

```python
EN = """A language model reads text as a sequence of tokens and learns to predict the next token.
The model never sees letters or words directly: it only sees numbers. Before training, we must
decide how to cut the text into pieces. If the pieces are single characters, the sequences become
very long. If the pieces are whole words, the vocabulary becomes huge and new words cannot be
represented. Byte pair encoding finds a middle way: it starts from bytes and repeatedly merges the
most frequent pair of neighbouring tokens into a new token. Frequent words become single tokens,
while rare words are split into smaller pieces. The same tokenizer is used for training the model
and for every request the model receives later, so the tokenizer is part of the model."""
ZH = """语言模型把文本看成一串词元，学习预测下一个词元。模型看不到汉字本身，只能看到数字。
训练之前，我们必须决定怎样把文本切成小块。如果每个汉字都是一块，序列就会很长；
如果每个词都是一块，词表就会非常大，新词也无法表示。字节对编码从字节出发，
不断把最常见的相邻两块合并成新的一块，于是常见的词变成一个词元，少见的词被拆成更小的部分。
同一个分词器既用于训练模型，也用于模型以后收到的每一个请求，所以分词器也是模型的一部分。"""
RU = """Языковая модель читает текст как последовательность токенов и учится предсказывать
следующий токен. Модель не видит букв и слов, она видит только числа. Перед обучением нужно
решить, как разрезать текст на кусочки. Если кусочки — отдельные символы, последовательности
становятся очень длинными. Если кусочки — целые слова, словарь становится огромным, а новые
слова нельзя представить. Кодирование пар байтов находит середину: оно начинает с байтов и
снова и снова объединяет самую частую пару соседних токенов в новый токен."""

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text, vocab_size):
    ids, merges = list(text.encode("utf-8")), {}
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

for name, data in [("trained on English only", EN), ("trained on English + Chinese + Russian", EN + ZH + RU)]:
    merges, vocab = train(data, 256 + 300)
    print(name)
    for s in ["the model predicts the next token", "模型预测下一个词元", "модель предсказывает токен"]:
        ids = encode(s, merges)
        pieces = [vocab[i].decode("utf-8", "replace") for i in ids]
        print(f"   {len(s):2d} characters → {len(ids):2d} tokens  ", pieces[:6])
```

Trained on English only, the 9-character Chinese sentence becomes 27 tokens (3 bytes per character, nothing merged), and Russian is almost one token per byte; with Chinese and Russian added to the training data, the same sentences need half as many tokens or fewer. `�` means the token is only **some of the bytes** of a character and can't be displayed on its own.

This matters in practice because:

- **you pay per token**: APIs charge by the number of tokens, so the same meaning costs more in a token-hungry language;
- **the context is shorter**: a model can only read a limited number of tokens at once, so more tokens means less text fits;
- **generation is slower**: models generate one token at a time, so more tokens take longer.

The early GPT-2 tokenizer was very unfriendly to Chinese. Today's tokenizers (GPT-4o's `o200k_base`, Llama 3, Qwen, DeepSeek and others) are trained on lots of multilingual data with bigger vocabularies, and are much more efficient for Chinese and Russian. **The tokenizers of Chinese models such as Qwen and DeepSeek are especially efficient for Chinese.**

| Model / tokenizer | Vocabulary size |
|-------------------|-----------------|
| GPT-2 | 50,257 |
| GPT-4 (`cl100k_base`) | about 100,000 |
| GPT-4o (`o200k_base`) | about 200,000 |
| Llama 3 | about 128,000 |
| Qwen family | about 150,000 |
| DeepSeek-V3 | about 129,000 |

## 2.5 Using real tokenizers

In real projects you don't write BPE yourself; you use a library, such as OpenAI's `tiktoken` or Hugging Face `transformers` (which loads the tokenizer of any open model):

```python norun
# pip install tiktoken
import tiktoken
enc = tiktoken.get_encoding("o200k_base")          # the tokenizer used by GPT-4o
ids = enc.encode("Large language models are fun!")
print(len(ids), ids)
print([enc.decode([i]) for i in ids])               # see what each token is

# pip install transformers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")   # downloads from Hugging Face on first run
print(tok.tokenize("Large language models are fun!"))
```

> [!TIP]
> To see visually how text is split, try an online tool such as OpenAI's [Tokenizer](https://platform.openai.com/tokenizer) page or [The Tokenizer Playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground) on Hugging Face.

## 2.6 Special tokens and chat templates

Besides ordinary text pieces, the vocabulary contains **special tokens** that don't correspond to any text but act as "markers":

- `<|endoftext|>`: the end of a document (separates documents during pre-training);
- **role markers** for chat models: "what follows is the system prompt / the user / the assistant".

The back-and-forth you see in a chat interface is actually joined into one long text with special tokens before it goes into the model. This format is the **chat template**. The format used by Qwen and others looks roughly like this:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 1 + 1?<|im_end|>
<|im_start|>assistant
1 + 1 is 2.<|im_end|>
```

The model has learned that after `<|im_start|>assistant` it should generate an answer and stop at `<|im_end|>`. Different models use different templates, and the wrong template makes a model perform worse — Hugging Face's `tokenizer.apply_chat_template(...)` applies the right one automatically. We'll use it again when fine-tuning in part 5.

## 2.7 Oddities caused by tokenization

Many puzzling LLM behaviours come down to tokenization:

| Behaviour | Cause |
|-----------|-------|
| can't count the `r`s in `strawberry` | the model sees tokens like `str` + `aw` + `berry`, not individual letters |
| bad at spelling words backwards or manipulating letters | same: letters are hidden inside tokens |
| arithmetic on big numbers goes wrong | numbers are cut into irregular chunks (e.g. `12345` → `123` + `45`), so digits don't line up |
| a trailing space makes answers worse | `" token"` (with a leading space) and `"token"` are different tokens, and a lone space token is rare in training |
| Chinese and less common languages "cost more" | they are rare in the tokenizer's training data, so compression is worse (section 2.4) |

> [!NOTE]
> Newer reasoning models "think it through" (for example, writing a word out letter by letter before counting), so these problems are much less severe now, but the root cause is still tokenization. Some research even tries to remove the tokenizer completely and model bytes directly.

## 2.8 Summary

- Models only understand numbers: the **tokenizer** cuts text into tokens and numbers them. Subword tokenization balances vocabulary size against sequence length.
- Text → Unicode code points → **UTF-8 bytes** (1 byte for English letters, 2 for Cyrillic, 3 for Chinese); starting from bytes, any text can be represented.
- **BPE**: start from bytes and repeatedly merge the most frequent neighbouring pair; encoding applies the merges in learned order, decoding joins the bytes back together.
- A tokenizer's training data decides how well each language compresses: more tokens means higher cost, less text per context and slower generation. Modern multilingual tokenizers are far better.
- **Special tokens** and **chat templates** turn a multi-turn conversation into one text; oddities like letter counting and arithmetic errors often stem from tokenization.

## 2.9 Exercises

<!-- exercise: utf8-bytes -->
### Exercise 1: Characters and bytes

Read a line of text and print its **number of characters** and its **number of UTF-8 bytes**, separated by a space.

```text
Sample input: Hi 你好
Sample output: 5 9
```

<details>
<summary>Hint</summary>

`len(s)` is the number of characters; `len(s.encode("utf-8"))` is the number of bytes.
</details>

<!-- exercise: bpe-step -->
### Exercise 2: One BPE merge

Read a string without spaces and treat it as a sequence of characters. Find the **most frequent pair of neighbouring characters** (count overlapping pairs, so `aa` occurs twice in `aaa`; on a tie take the pair that **appears first**) and print it on the first line. Then merge every occurrence of that pair into a single token **from left to right** (without overlapping) and print the resulting token sequence on the second line, separated by spaces.

```text
Sample input: aaabdaaabac
Sample output:
aa
aa a b d aa a b a c
```

<details>
<summary>Hint</summary>

Count `zip(seq, seq[1:])` in a dict, which keeps insertion order; on a tie `max(counts, key=counts.get)` returns the pair added first. For the merge you can copy this lesson's `merge` function, using the two characters joined together as the new token instead of a number.
</details>

<!-- exercise: decode-bytes -->
### Exercise 3: Predict the output

```python
tokens = [72, 105, 32, 228, 189, 160]
print(bytes(tokens).decode("utf-8"))
print(len(tokens))
```

<details>
<summary>Answer and explanation</summary>

`72, 105, 32` are `H`, `i` and a space; the 3 bytes `228, 189, 160` together are the UTF-8 encoding of the Chinese character 你. So the output is `Hi 你` followed by `6` (6 bytes, but only 4 characters).
</details>

<!-- exercise: strawberry -->
### Exercise 4: Counting letters

Early LLMs often got "how many r's are in strawberry" wrong. What is the root cause?

- the model never learned English
- the model sees tokens (chunks of several letters), not individual letters
- the model isn't good enough at computation
- the word isn't in the training data

<details>
<summary>Answer and explanation</summary>

The word is split into chunks like `str`, `aw` and `berry`, and to the model each chunk is just a number; the letters inside a chunk aren't directly visible.
</details>

<!-- exercise: vocab-size -->
### Exercise 5: Is a bigger vocabulary better?

If a tokenizer's vocabulary grows from 30,000 to 200,000, which statement is **correct**?

- the same text becomes more tokens
- the same text becomes fewer tokens, but the embedding and output layers get more parameters
- it has no effect on the model
- the model can no longer handle unseen words

<details>
<summary>Answer and explanation</summary>

A bigger vocabulary can merge more pieces, so sequences get shorter (faster, more text per context). The cost: every token needs its own embedding row and the output layer must score every token, so parameters and computation grow. A byte-level BPE can represent any text whatever its vocabulary size.
</details>

## 2.10 Further reading

- Andrej Karpathy: [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (video) and [minbpe](https://github.com/karpathy/minbpe)
- Sennrich et al.: "Neural Machine Translation of Rare Words with Subword Units" (2016, the paper that brought BPE to NLP)
- Hugging Face LLM Course: [Tokenizers](https://huggingface.co/learn/llm-course/chapter2/4)
- Next lesson: [Embeddings: meaning as a list of numbers](../03-embeddings/en.md)
