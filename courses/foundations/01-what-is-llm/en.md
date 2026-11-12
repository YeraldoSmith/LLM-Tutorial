# 1. What exactly is a large language model?

> 🎯 **Goals**: explain in one sentence what a large language model (LLM) does; write a 10-line "language model" yourself; understand tokens, parameters and context windows; know how an LLM is built step by step, and what it is and isn't good at.
>
> ⏱ **Time**: 40 minutes　|　📋 **You need**: a little Python (variables, lists, dictionaries, loops). None at all? Do the first 8 Python lessons of [Our-tutorial](https://github.com/YeraldoSmith/Our-tutorial) first.

## 1.1 In one sentence: predict the next word

You've used your phone keyboard's suggestions: type "see you" and it offers "soon", "tomorrow", "later". It is **guessing the most likely next word from the words before it**.

A large language model (**L**arge **L**anguage **M**odel, LLM) — the thing behind ChatGPT, Claude, Gemini, DeepSeek, Qwen, Kimi and Llama — does **essentially the same thing**, just far, far better:

```text
input:  "Twinkle, twinkle, little"
model:  next word is "star" with probability 97%, "bat" 1%, "light" 0.5% ...
pick "star" and append it: "Twinkle, twinkle, little star"
predict again: "Twinkle, twinkle, little star," → "how" ...
repeat, and a whole passage gets "written".
```

This is **autoregressive generation**: predict one piece, append it to the input, predict the next. That's why chatbot answers appear a word at a time.

> [!IMPORTANT]
> Remember this sentence; everything else in the course is about doing it well: **a large language model is a function that, given the text so far, computes a probability distribution over what the next token will be.**

## 1.2 Hands on: a "language model" in 10 lines

The simplest method is **counting**: in some text, count which word follows each word and how often; to predict, pick the most frequent follower.

```python
text = "I love learning . I love coding . I love learning AI . you love learning ?"
words = text.split()

# count which word follows each word, and how often
follow = {}
for current, nxt in zip(words, words[1:]):
    follow.setdefault(current, {})
    follow[current][nxt] = follow[current].get(nxt, 0) + 1

print(follow["I"])      # {'love': 3}
print(follow["love"])   # {'learning': 3, 'coding': 1}

# generate: start from "you" and always take the most frequent follower
word = "you"
sentence = [word]
for _ in range(4):
    candidates = follow[word]
    word = max(candidates, key=candidates.get)
    sentence.append(word)
print(" ".join(sentence))
```

Output:

```text
{'love': 3}
{'learning': 3, 'coding': 1}
you love learning . I
```

Congratulations — you've written a (very small) language model! It looks only at the **previous word**, which makes it a **bigram model**. Part 3, lesson 1 builds a proper version: probabilities instead of counts, random sampling, and a way to measure how good its guesses are.

Its weaknesses are obvious:

- it sees only 1 word, so it **can't remember context**;
- word pairs it hasn't seen are **impossible** for it;
- it understands nothing; it just memorises statistics.

Real LLMs replace this table with a **neural network** that can take **hundreds of thousands of words** of context into account and that has learned meaning, grammar, facts and even patterns of reasoning. Getting from the table to the network is the journey of parts 2 and 3.

## 1.3 What makes it "large"?

| Dimension | our tiny model | modern LLMs (typical in 2026) |
|-----------|----------------|-------------------------------|
| **parameters** (adjustable numbers) | a dozen counts | billions to over a trillion (7B, 70B, 600B+; B = billion) |
| **training data** | one sentence | tens of trillions of tokens of web pages, books, code, papers |
| **context it can see** | 1 word | tens of thousands to millions of tokens (the context window) |
| **training cost** | 0.001 s | thousands of GPUs for weeks or months |

**Parameters** (also called **weights**) are the numbers inside a neural network. Training means **adjusting those numbers again and again** so the model predicts the next word better and better. A "70B model" has 70 billion parameters; at 2 bytes each, just storing them takes about 140 GB.

The surprising part: **just pushing "predict the next word" to the extreme on enormous data** forces the model to learn a lot — to predict the next word accurately it has to follow grammar, remember facts, track logic, even model the writer's intent.

## 1.4 Tokens: the model's "words"

The smallest unit a model handles isn't a character or a word but a **token**. A token may be a whole common word, part of a word, or punctuation:

```text
"unbelievable"   →  ["un", "believ", "able"]       3 tokens
"Привет, мир"    →  ["При", "вет", ",", " мир"]      about 3–4 tokens (depends on the tokenizer)
"2026"           →  ["202", "6"] or ["2026"]        depends on the tokenizer
```

The tool that splits text into tokens is a **tokenizer**; you'll build one in part 3, lesson 2. For now:

- roughly, 1 token ≈ 0.75 English words (other languages often need more tokens per word);
- price, speed and maximum length are all **measured in tokens**;
- the maximum number of tokens a model can see at once is its **context window**.

## 1.5 How an LLM is made

```text
 ① Pre-training                 ② Instruction tuning (SFT)    ③ Preference alignment        ④ Reasoning RL
 predict the next word on    →  train on "question → good  →  people/AI compare answers  →  reward correct answers on
 huge amounts of text           answer" demonstrations        (RLHF / DPO) to make them     checkable tasks (maths, code);
 → a "base model"               → it can chat                 more helpful and safer        it learns to "think first"
 (continues text, can't chat)
```

1. **Pre-training**: train "predict the next word" on tens of trillions of tokens. The resulting **base model** knows a lot but only continues text — ask it a question and it may continue with more questions. (Part 4)
2. **Supervised fine-tuning** (SFT): continue training on many "user question → ideal answer" demonstrations so the model learns to act as an assistant. (Part 5)
3. **Preference alignment**: have people (or AI) compare two answers and train on those preferences (RLHF, DPO…), making the model more helpful, honest and safe. (Part 5)
4. **Reasoning reinforcement learning**: a major recent advance. Reinforcement learning on tasks whose **answers can be checked automatically** — maths problems, programming tasks — teaches the model to write a long chain of thought before answering, greatly improving reasoning. "Reasoning models" such as OpenAI's o series and DeepSeek-R1 use this approach. (Part 5)

Then comes **deployment** — running the model efficiently for many users (part 6) — and **building applications** on top of it (part 7).

## 1.6 Training versus inference

- **Training**: adjusting the parameters. Very expensive, done once (or a few times).
- **Inference**: using the trained model to produce answers. Happens every time someone asks something.

> [!NOTE]
> Don't confuse *inference* (running a model) with *reasoning* (logical thinking, as in "reasoning models").

## 1.7 What LLMs are good and bad at

✅ Good at: writing and rewriting, translation, summarising, general knowledge, writing and explaining code, brainstorming, extracting information from text, multi-turn conversation…

⚠️ Watch out for:

- **Hallucination**: a model can **confidently make up** facts, papers, URLs or functions that don't exist. Its goal is "write a plausible next token", not "say only true things". Always verify important information.
- **Knowledge cut-off**: it only knows what was in its training data up to some date. For current information it needs search or documents you provide (RAG, part 7).
- **Exact computation**: multiplying big numbers or counting letters often goes wrong (partly because of how text is split into tokens). Letting it call a calculator or write code is far more reliable (tool calling, part 7).
- **Sensitivity to phrasing**: rephrase a question and the answer may change (prompt engineering, part 7).

## 1.8 How to use this course

```text
Part 1 Foundations ──► Part 2 Neural networks ──► Part 3 Build a GPT ──► Part 4 Training
                                                          │
        Part 8 Frontier ◄── Part 7 Apps ◄── Part 6 Deployment ◄── Part 5 Alignment
```

- To **really understand** how it works: go in order; parts 1–3 are the core — do the exercises.
- To **build AI apps quickly**: after part 1 jump to part 7 and come back for the theory.
- All code runs in the learning app. Most code in parts 1–3 is plain Python and **needs no GPU**.

## 1.9 Summary

- An LLM **computes the probability of the next token from the text so far**; repeating this generates text (autoregression).
- The simplest language model is just counting; LLMs use neural networks that see long contexts and learn meaning and knowledge.
- Parameters are a model's adjustable numbers and training adjusts them; tokens are the units of text; the context window is how many tokens fit at once.
- The pipeline: pre-training → instruction tuning → preference alignment → reasoning RL → deployment and applications.
- LLMs hallucinate, have a knowledge cut-off and are weak at exact computation — verify what matters.

## 1.10 Exercises

<!-- exercise: core-idea -->
### Exercise 1: The core task

What is the central step when an LLM generates text?

- Searching a database for a ready-made answer
- Computing the probability of the next token from the preceding text
- Translating the question into English first
- Combining common sentences at random

<details>
<summary>Answer and explanation</summary>

An LLM does one thing at a time: produce a probability distribution over the next token, pick one, append it, repeat. It doesn't look answers up in a database.
</details>

<!-- exercise: next-word -->
### Exercise 2: The most common next word

The starter code contains a text `text`. The program reads a word and prints the word that **most often directly follows it** in `text`. If there's a tie, print the one that **appears first** in the text.

```text
Sample input: the
Sample output: cat
```

(`the` is followed by cat twice, mat once and fish once.)

<details>
<summary>Hint 1</summary>

Like in 1.2: loop over neighbouring pairs with `zip(words, words[1:])` and count with a dictionary.
</details>

<details>
<summary>Hint 2</summary>

`max(counts, key=counts.get)` returns the key with the highest count; on a tie it returns the key **inserted first**, which is exactly "appears first".
</details>

<!-- exercise: hallucination -->
### Exercise 3: Hallucination

You ask an LLM for 3 papers on an obscure topic; it gives titles, authors and years that look very professional. What's the sensible thing to do?

- Cite them directly; LLMs don't make mistakes
- Search for each one and check that it really exists
- Ask it to write a longer answer
- Display them in a different font

<details>
<summary>Answer and explanation</summary>

LLMs can invent realistic-looking papers (hallucination). Facts, citations and data must be checked against reliable sources.
</details>

<!-- exercise: stages -->
### Exercise 4: Which stage?

"Train the model to predict the next word on huge amounts of web pages, books and code, producing a knowledgeable model that can only continue text." Which stage is this?

- Pre-training
- Instruction tuning
- Preference alignment
- Deployment

<details>
<summary>Answer and explanation</summary>

Pre-training. It produces a base model; instruction tuning and alignment then turn it into a useful assistant.
</details>

## 1.11 Further reading

- Andrej Karpathy: [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — a one-hour overview, perfect for beginners.
- 3Blue1Brown: [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs)
- Next lesson: [Your learning environment](../02-setup/en.md)
