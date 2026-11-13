# 5. Evaluation: how do you know a model is good?

> 🎯 **Goals**: measure language-modelling ability with perplexity and **bits per byte**; understand the two ways of scoring multiple-choice benchmarks and their traps; evaluate maths with **exact match** and code with **pass@k** (implementing the unbiased estimator); understand Chatbot Arena's **Elo ratings** and the biases of "LLM as a judge"; know the benchmarks common in 2026 (MMLU-Pro, GPQA, AIME, SWE-bench, HLE…); watch out for contamination, saturation, prompt sensitivity and **statistical noise** (and compute confidence intervals); see that the best evaluation is one designed for your own task.
>
> ⏱ **Time**: 70 minutes　|　📋 **You need**: Part 3, lessons 1 and 7; lessons 1 and 2 of this part

You've trained a model — how do you know whether it's good? How much better than the last version? Than someone else's model? It's much harder than it sounds: language models can do so many things that any single score shows only one side. This lesson covers the common evaluation methods and the traps in reading their scores.

## 5.1 Language modelling itself: perplexity and bits per byte

The most direct measure is the model's **loss** on unseen text (the average negative log-probability per token), or its exponential, the **perplexity** (Part 3, lesson 1):

```python
import math

probs = [0.25, 0.6, 0.05, 0.9, 0.4]               # the model's probability for each true next token
nll = [-math.log(p) for p in probs]               # negative log-probability per token (in nats)
loss = sum(nll) / len(nll)
print(f"average loss {loss:.3f} nats/token, perplexity {math.exp(loss):.2f}")

text = "Nice day."                                # suppose these 5 tokens spell this text
n_bytes = len(text.encode("utf-8"))
bpb = sum(nll) / math.log(2) / n_bytes            # total information in bits, divided by the bytes
print(f"{n_bytes} bytes, {bpb:.3f} bits per byte")
```

Perplexity has a problem: it depends on the **tokeniser**. A tokeniser with coarser tokens (more text per token) makes each token harder to predict and the perplexity higher, without the model being any worse. So to compare models with different tokenisers, **bits per byte** (BPB) is common: convert the total information to bits and divide by the number of bytes of the original text, which no longer depends on how it was tokenised.

Loss is the most important metric during training — smooth, stable and tied directly to the scaling laws (lesson 2). But users care whether a model answers questions correctly, writes working code and follows instructions, which needs dedicated benchmarks.

## 5.2 Multiple choice: even scoring takes care

**MMLU** (2020) is the best-known multiple-choice benchmark: about 14,000 four-option questions across 57 subjects (maths, history, law, medicine…). Its successor **MMLU-Pro** (2024) has ten options and more reasoning questions; **GPQA** (2023) has graduate-level physics, chemistry and biology questions designed so that even web searches hardly help ("GPQA Diamond" is its hardest 198 questions).

There are two ways to score a model on multiple choice:

1. **Generating the answer**: list the options in the prompt, let the model output "A", "B"…, and check the output. Suited to chat models.
2. **Comparing probabilities**: append each option's text to the question, compute the **log-probability** the model gives that text, and pick the highest. The model needn't follow an answer format, so this suits base models that haven't been instruction-tuned.

The second way has a trap: longer options contain more tokens, so their summed log-probability is lower. So scores are often **averaged** over length — which can in turn favour long options:

```python
# log-probabilities of each option's tokens (options of different lengths)
options = {
    "A": [-1.2, -0.8, -0.9],
    "B": [-2.5],
    "C": [-1.0, -1.1],
    "D": [-3.0, -0.2, -0.3, -0.4],
}
by_sum = max(options, key=lambda k: sum(options[k]))
by_mean = max(options, key=lambda k: sum(options[k]) / len(options[k]))
for k, lp in options.items():
    print(f"{k}: {len(lp)} tokens, sum {sum(lp):5.2f}, mean {sum(lp) / len(lp):6.3f}")
print("chosen by sum:", by_sum, "; chosen by mean:", by_mean)
```

Same model, same question, different scoring — different answer. So **when comparing scores, every detail of the evaluation must match**: the same prompt format, the same number of examples (0-shot or 5-shot), the same scoring method. That's why the "MMLU score" of one model often differs by several points between papers.

## 5.3 Generation tasks: maths and code

### Maths: exact match

Benchmarks such as **GSM8K** (grade-school word problems), **MATH** (competition maths) and **AIME** (the American Invitational Mathematics Examination, 30 problems a year, widely used for reasoning models since 2024) let the model write out its working, then extract the answer and compare it with the reference. "Extracting" sounds easy but is error-prone: the model may write `1,000`, `1000.0`, `$1000` or `the answer is 1000 dollars`… Answers are usually **normalised** before comparison, or the model is asked to put its final answer in a fixed format (such as `\boxed{1000}`).

### Code: pass@k

Benchmarks such as **HumanEval** (OpenAI, 2021, 164 Python function problems) and MBPP run the model's code in a sandbox and judge it with unit tests. The usual metric is **pass@k**: the model generates k answers per problem, and the problem counts as solved if **any one** passes the tests.

How do you estimate pass@k? The direct way — really generating k answers and checking for a correct one — is very noisy. Better: generate n (n ≥ k) answers per problem, count the c correct ones, and use combinations to compute the probability that "k answers drawn at random from the n include at least one correct one":

```formula
pass@k = 1 − C(n − c, k) / C(n, k)
```

```python
from math import comb

def pass_at_k(n, c, k):
    if n - c < k:                      # fewer than k wrong answers: any k chosen include a correct one
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)

def naive(n, c, k):                    # a common mistake: treating c/n as each attempt's success rate
    return 1 - (1 - c / n) ** k

for n, c, k in [(10, 3, 1), (10, 3, 5), (200, 10, 10), (200, 10, 100)]:
    print(f"n={n:3d} c={c:2d} k={k:3d}: unbiased {pass_at_k(n, c, k):.4f}, naive {naive(n, c, k):.4f}")
```

`1 − (1 − c/n)^k` looks reasonable, but it assumes drawing "with replacement" and systematically **underestimates** pass@k. The HumanEval paper introduced the unbiased estimator above.

Code benchmarks are getting ever closer to real work: **LiveCodeBench** keeps collecting newly published contest problems so that they can't have "contaminated" the training data; **SWE-bench Verified** (2024) is 500 issues from real GitHub projects, where the model must read the whole codebase and change the code so that the project's existing tests pass — which is already evaluating **agents** (Part 7).

## 5.4 Human preference: the arena and Elo

Many qualities — whether an answer helps, whether writing is good, whether the tone is right — have no reference answer. **Chatbot Arena** (LMArena, since 2023) works like this: a user types a question, two anonymous models each answer, and the user votes for the better one. Millions of votes rank all the models.

The ranking uses **Elo ratings** from chess (and the more rigorous Bradley–Terry model): the rating difference sets the "expected score", and a result better than expected gains points while a worse one loses them:

```formula
expected score E_A = 1 / (1 + 10^((R_B − R_A) / 400))　　new rating R_A' = R_A + K · (actual score − E_A)
```

```python
def elo(ra, rb, score_a, k=32):          # score_a: 1 if A wins, 0.5 for a draw, 0 if A loses
    expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))
    change = k * (score_a - expected_a)
    return ra + change, rb - change

print("two models at 1500, A wins:", elo(1500, 1500, 1))
print("A at 1600 loses to B at 1400:", [round(v, 1) for v in elo(1600, 1400, 0)])
print("A at 1600 beats B at 1400:", [round(v, 1) for v in elo(1600, 1400, 1)])
print("expected win rate for a 400-point gap:", round(1 / (1 + 10 ** (-400 / 400)), 3))
```

An upset (the stronger losing to the weaker) moves ratings a lot; an expected result moves them only a little.

### LLMs as judges

Human votes are slow and costly, so a strong model often acts as the judge (**LLM-as-a-judge**), as in MT-Bench and AlpacaEval. It's convenient, but it has known biases:

- **position bias**: a preference for the answer shown first (or last). Remedy: judge again with the order swapped, and only accept agreeing verdicts.
- **length bias**: a preference for longer, more detailed answers even when they're no better. AlpacaEval 2.0 introduced a "length-controlled" win rate for this reason.
- **self-preference**: a preference for answers written by itself (or its own model family).
- and judges make mistakes too: on specialist questions they may know no more than the models being judged.

## 5.5 Common benchmarks in 2026

| Area | Typical benchmarks |
|------|--------------------|
| knowledge and reasoning | MMLU-Pro, GPQA Diamond, **HLE** (Humanity's Last Exam, 2025: about 2,500 hard questions written by experts across many fields) |
| maths | MATH-500, AIME 2024/2025 |
| code | LiveCodeBench, SWE-bench Verified |
| following instructions | **IFEval** (instructions that can be checked automatically, such as "at most 100 words" or "no commas") |
| long context | needle in a haystack, RULER |
| agents | SWE-bench, τ-bench (customer-service tasks using tools), web browsing and computer use benchmarks |
| abstract reasoning | ARC-AGI (visual puzzles testing the ability to learn new rules) |
| human preference | LMArena |
| safety | dedicated evaluations of refusing harmful requests, jailbreaks, bias and more |

Benchmarks have ever shorter "lifetimes": when MMLU was released the best model scored in the low 40s; a few years later frontier models scored around 90 and it was **saturated**, no longer telling them apart. GSM8K went the same way. So ever harder benchmarks keep appearing, and HLE's name ("Humanity's Last Exam") reflects that race.

## 5.6 Traps in reading scores

- **Contamination**: if benchmark questions appear in the training data, scores are inflated (lesson 1 covered decontamination). New benchmarks (such as LiveCodeBench, or each year's AIME) partly avoid this.
- **Prompt sensitivity**: rewording a prompt, changing the examples or shuffling the options can move a score by several points.
- **Goodhart's law**: "when a measure becomes a target, it ceases to be a good measure." Optimising for a particular leaderboard can make a model good on that benchmark without improving it anywhere else.
- **Statistical noise**: a benchmark is finite, so the score itself has random error. With accuracy p on n questions, the 95% confidence interval is roughly `p ± 1.96 × √(p(1 − p) / n)`:

```python
import math

def accuracy_ci(correct, total):
    p = correct / total
    half = 1.96 * math.sqrt(p * (1 - p) / total)
    return p, p - half, p + half

for name, correct in [("model A", 360), ("model B", 375)]:
    p, low, high = accuracy_ci(correct, 500)
    print(f"{name}: {correct} of 500 correct, accuracy {p:.1%}, 95% confidence interval [{low:.1%}, {high:.1%}]")
```

72% and 75% look 3 points apart, but the two intervals overlap heavily — with only 500 questions, the difference **could well be luck**. Beware especially of benchmarks with few questions (AIME has only 30 a year!): the usual practice is to sample each question several times and report the average accuracy.

### The best evaluation is your own

Public benchmarks measure general ability. If you want a model for a particular job — sorting customer emails, extracting clauses from contracts, answering questions about your company's products — the most reliable approach is to **collect a few dozen to a few hundred real examples**, write reference answers or scoring rules, and build your own evaluation set. Run it every time you change models or edit a prompt. Part 7 goes into this in detail.

## 5.7 Summary

- Perplexity (or loss) measures language modelling but depends on the tokeniser; bits per byte doesn't.
- Multiple choice can be scored by generating an answer or comparing options' log-probabilities; the scoring method, prompt format and number of examples all affect scores and must match when comparing.
- Maths uses exact match after normalising answers; code uses unit tests and pass@k, with the unbiased estimator `1 − C(n−c, k)/C(n, k)`.
- Without reference answers, use human preference (Arena + Elo) or LLM judges, which have position, length and self-preference biases.
- Benchmarks saturate, get contaminated and are prompt-sensitive; scores have statistical error, so look at confidence intervals.
- Build a dedicated evaluation set for your own task.

## 5.8 Exercises

<!-- exercise: perplexity -->
### Exercise 1: Perplexity

Read one line of numbers: the model's probability for each true token. Print the average loss (in nats) and the perplexity, both to 3 decimal places, separated by a space.

```text
Sample input: 0.25 0.6 0.05 0.9 0.4
Sample output: 1.183 3.264
```

<details>
<summary>Hint</summary>

Loss = `-sum(math.log(p) for p in probs) / len(probs)`, perplexity = `math.exp(loss)`.
</details>

<!-- exercise: pass-at-k -->
### Exercise 2: pass@k

Read n, c and k: n answers were generated per problem, and c of them passed the tests. Compute pass@k with the unbiased estimator, to 4 decimal places.

```text
Sample input: 10 3 5
Sample output: 0.9167
```

<details>
<summary>Hint</summary>

`from math import comb`; the result is 1 when `n - c < k`, otherwise `1 - comb(n - c, k) / comb(n, k)`.
</details>

<!-- exercise: elo -->
### Exercise 3: Elo ratings

Read one line: A's rating, B's rating, A's result (1 win, 0.5 draw, 0 loss) and K. Print A's and B's new ratings after the game, each to 1 decimal place, separated by a space.

```text
Sample input: 1600 1400 0 32
Sample output: 1575.7 1424.3
```

<details>
<summary>Hint</summary>

`expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))`, `change = k * (score - expected_a)`; add change to A and subtract it from B.
</details>

<!-- exercise: accuracy-ci -->
### Exercise 4: A confidence interval for accuracy

Read the number of correct answers and the number of questions, and print the accuracy and the lower and upper limits of its 95% confidence interval (using `p ± 1.96 × √(p(1 − p)/n)`), all to 3 decimal places, separated by spaces.

```text
Sample input: 360 500
Sample output: 0.720 0.681 0.759
```

<details>
<summary>Hint</summary>

`p = correct / total`, `half = 1.96 * math.sqrt(p * (1 - p) / total)`; print `p`, `p - half` and `p + half`.
</details>

<!-- exercise: small-diff -->
### Exercise 5: What do 3 points mean?

On a 200-question benchmark, model X scores 70% and model Y 73%. Which conclusion is most reasonable?

- Y is definitely stronger than X
- X is definitely stronger than Y
- the gap is within statistical error, so we can't yet tell which is stronger; more questions or more samples are needed
- the benchmark is too small, so the scores mean nothing

<details>
<summary>Answer and explanation</summary>

With 200 questions and an accuracy around 70%, the 95% confidence interval's half-width is about 6.4 percentage points; the two models' intervals overlap heavily, and a 3-point gap may well be random noise. The scores aren't meaningless — they just can't separate these two models.
</details>

<!-- exercise: judge-bias -->
### Exercise 6: LLM judges

When asking an LLM which of two answers is better, which practice reduces its **position bias**?

- only show it the longer answer
- judge again with the two answers swapped, and only accept agreeing verdicts
- let the model being evaluated act as the judge
- set the temperature to 2

<details>
<summary>Answer and explanation</summary>

Position bias means the judge favours the answer in a particular position. Judging once in each order reveals and cancels it. Letting a model judge itself introduces self-preference; temperature and answer length have nothing to do with position bias.
</details>

## 5.9 Further reading

- Hendrycks et al., *Measuring Massive Multitask Language Understanding* (2020, MMLU)
- Chen et al., *Evaluating Large Language Models Trained on Code* (2021, HumanEval and the unbiased pass@k estimator)
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023)
- Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* (2023)
- Phan et al., *Humanity's Last Exam* (2025)
- EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the most widely used open-source evaluation tool
- That's the end of Part 4! Next: [Part 5 · Post-training and alignment](../../alignment/01-sft/en.md) — instruction tuning, LoRA, RLHF, DPO and reasoning models.
