# 3. Reward models and RLHF

> 🎯 **Goals**: see why "comparing two answers" is easier than "writing the perfect answer", and what preference data looks like; train a **reward model** with the **Bradley–Terry model** (implemented yourself in plain Python); understand the RLHF objective — maximise reward while a **KL penalty** keeps the model on a leash — and derive its optimal solution; do some "reinforcement learning" yourself with the **REINFORCE** policy gradient; learn **PPO**'s clipped objective and its costs; recognise **reward hacking**, length bias and sycophancy, and meet RLAIF and "Constitutional AI".
>
> ⏱ **Time**: 90 minutes　|　📋 **You need**: lesson 1 of this part; Part 2, lessons 2 and 4 (softmax, gradients)

SFT taught the model to imitate demonstrations. But demonstrations have a ceiling: the answers annotators write aren't necessarily the best, and many qualities — more helpful, more honest, safer — are hard to pin down by example. People found that **comparing which of two answers is better** is much easier than **writing a perfect answer yourself**. **Reinforcement Learning from Human Feedback** (RLHF) uses such comparisons to improve the model.

## 3.1 Preference data

RLHF's data is a series of **preferences**:

```text
Prompt: Explain what photosynthesis is.
Answer A: Photosynthesis is the process by which plants use light energy to turn carbon dioxide and water into sugar and oxygen... (clear, accurate)
Answer B: Photosynthesis is plants sunbathing. (too brief)
Label: A is better than B
```

Usually the SFT model generates several answers to the same prompt, and annotators (or, as we'll see, another model) say which is better. Each example is recorded as **(prompt, chosen answer, rejected answer)**. InstructGPT used tens of thousands of such comparisons.

## 3.2 Reward models: scoring answers

We need a **reward model** (RM) that can score any answer automatically. It's usually a language model whose output layer is replaced by a "scoring head" that outputs a single number. How do you learn scores from comparisons like "A is better than B"? With the **Bradley–Terry model** (the same one behind Elo in Part 4, lesson 5):

```formula
P(A better than B) = σ(r(A) − r(B))　　　loss = −log σ(r(chosen) − r(rejected))
```

σ is the sigmoid function. The bigger the gap between two answers' rewards, the surer the model is which is better; the loss pushes the chosen answer's reward above the rejected one's.

Below we train a minimal reward model in plain Python. A real reward model learns its own features from the answer's text; to keep things visible, here each answer is described by 4 hand-made features, and the reward is their weighted sum:

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# each answer is described by 4 features: [answers the question, polite, length (hundreds of words), contains a factual error]
pairs = [                                  # (chosen answer, rejected answer)
    ([1, 1, 0.5, 0], [0, 1, 0.4, 0]),
    ([1, 0, 0.3, 0], [1, 0, 0.3, 1]),
    ([1, 1, 0.8, 0], [1, 0, 0.6, 0]),
    ([1, 0, 1.2, 0], [0, 0, 1.5, 0]),
    ([1, 1, 0.4, 0], [1, 1, 0.9, 1]),
    ([1, 0, 0.6, 0], [0, 1, 0.2, 0]),
]

def reward(w, f):
    return sum(wi * fi for wi, fi in zip(w, f))

w = [0.0] * 4
for epoch in range(200):                   # gradient descent minimising −log σ(r_chosen − r_rejected)
    for c, r in pairs:
        diff = reward(w, c) - reward(w, r)
        g = sigmoid(diff) - 1              # derivative of the loss with respect to diff
        w = [wi - 0.5 * g * (ci - ri) for wi, ci, ri in zip(w, c, r)]

names = ["answers the question", "polite", "length", "factual error"]
print("learned weights:", {n: round(x, 2) for n, x in zip(names, w)})
acc = sum(reward(w, c) > reward(w, r) for c, r in pairs) / len(pairs)
print(f"accuracy on the training data: {acc:.0%}")

short = [1, 0, 0.3, 0]                     # short and correct
padded = [1, 0, 2.0, 0]                    # the same content, padded with filler
print(f"short correct answer: {reward(w, short):.2f}, padded answer: {reward(w, padded):.2f}")
```

The reward model learned to reward answering the question and to penalise factual errors heavily — very sensible. But it **also learned "longer is better"**: in the training data, chosen answers happened to be longer more often. So an answer padded with filler scores higher. Remember this; section 3.6 comes back to it.

## 3.3 The RLHF objective: maximise reward, but don't wander off

With a reward model, the next step is to adjust the language model (in reinforcement learning, the **policy**, written π) so its answers score higher. But if it chases **only** a high score, the model will find ways to "game the reward model" — producing text the reward model rates highly but that is actually bad. So the RLHF objective adds a constraint: **don't stray too far from the original SFT model (the reference model π_ref)**:

```formula
maximise　E[r(answer)] − β · KL(π ‖ π_ref)
```

The **KL divergence** `KL(π‖π_ref) = Σ π(y) · log(π(y) / π_ref(y))` measures how different two probability distributions are: 0 when they're identical, larger the more they differ (it's a close relative of cross-entropy from Part 1, lesson 6: cross-entropy = entropy + KL divergence). β sets how tight the leash is. This objective has a beautiful optimal solution:

```formula
π*(answer) ∝ π_ref(answer) · exp(r(answer) / β)
```

In other words, the optimal policy **reweights** the reference model by the exponential of the reward. A "toy" problem with only 4 candidate answers shows what β does:

```python
import math

answers = ["correct, concise", "correct, wordy", "wrong but confident", "refuses to answer"]
ref = [0.3, 0.3, 0.2, 0.2]                 # the SFT (reference) model's probability of each answer
reward = [1.0, 0.6, -1.0, -0.5]            # the reward model's scores

for beta in (10, 1, 0.3, 0.05):
    weights = [p * math.exp(r / beta) for p, r in zip(ref, reward)]
    pi = [w / sum(weights) for w in weights]
    kl = sum(p * math.log(p / q) for p, q in zip(pi, ref))
    avg_r = sum(p * r for p, r in zip(pi, reward))
    print(f"β = {beta:<5} optimal policy {[round(p, 3) for p in pi]}  average reward {avg_r:.3f}  KL {kl:.3f}")
```

With a large β the policy barely moves (KL ≈ 0); the smaller β, the more boldly it chases reward and the further it drifts from the reference; as β approaches 0 it bets everything on the highest-reward answer. In reality the reward model isn't perfect, and betting too hard falls into the reward-hacking trap — so β can't be too small.

## 3.4 Policy gradients: REINFORCE

A real language model has an astronomical number of possible answers, so the optimum can't be computed by listing them as above. We can only **sample**: generate an answer, score it, and adjust the parameters according to the score. The most basic method is **REINFORCE** (the policy gradient):

```formula
∇ E[r] = E[ (r − b) · ∇ log π(answer) ]
```

Intuitively: having generated an answer, if its reward is above "the average" b (the **baseline**), raise its probability; if below, lower it. The baseline doesn't change the gradient's expectation but greatly reduces its randomness. Let's train the 4-answer policy above with REINFORCE (folding the KL penalty into the reward) and see whether it finds the β = 0.3 optimum by itself:

```python
import math
import random

random.seed(0)
ref = [0.3, 0.3, 0.2, 0.2]
reward = [1.0, 0.6, -1.0, -0.5]
beta, lr = 0.3, 0.1

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

logits = [math.log(p) for p in ref]                 # start from the reference model
for step in range(3001):
    pi = softmax(logits)
    i = random.choices(range(4), weights=pi)[0]     # sample an answer
    score = reward[i] - beta * math.log(pi[i] / ref[i])      # reward minus the KL penalty
    baseline = sum(p * (r - beta * math.log(p / q)) for p, r, q in zip(pi, reward, ref))
    advantage = score - baseline
    for j in range(4):                              # derivative of log softmax with respect to logit j
        logits[j] += lr * advantage * ((1 if j == i else 0) - pi[j])
    if step % 1000 == 0:
        print(f"step {step:4d}: {[round(p, 3) for p in softmax(logits)]}")
```

Purely by "trying and looking at the score", the policy converges close to the optimum `[0.788, 0.208, 0.001, 0.004]` computed in section 3.3. Real RLHF works the same way, except that an "answer" is a whole passage and log π(answer) is the sum of its tokens' log-probabilities.

## 3.5 PPO: steadier policy updates

REINFORCE is simple, but one step too large can suddenly ruin the policy beyond recovery. InstructGPT and early ChatGPT used **PPO** (Proximal Policy Optimization, OpenAI 2017), whose core idea is to limit how far each step can go:

```formula
ratio = π_new(answer) / π_old(answer)　　objective = min( ratio · A, clip(ratio, 1 − ε, 1 + ε) · A )
```

A is the **advantage** (like `r − b` above), and ε is usually 0.2. Once the new policy has raised a good answer's probability by more than 20%, raising it further earns nothing extra — so every update is a small, quick step:

```python
def ppo_objective(ratio, advantage, eps=0.2):
    clipped = max(1 - eps, min(1 + eps, ratio))
    return min(ratio * advantage, clipped * advantage)

for ratio, adv in [(1.0, 1.0), (1.5, 1.0), (0.5, 1.0), (1.5, -1.0), (0.5, -1.0)]:
    print(f"ratio {ratio}, advantage {adv:+}: objective {ppo_objective(ratio, adv):+.2f}")
```

PPO works well, but it's **expensive** and **complicated**: training keeps 4 large models in memory at once — the policy being trained, the frozen reference model, the reward model, and a **value model** (critic) for estimating advantages — plus a pile of hyperparameters needing careful tuning. This pushed people towards simpler methods: next lesson's **DPO** drops reinforcement learning altogether, and lesson 5's **GRPO** drops the value model.

## 3.6 Reward hacking

A reward model is only an **approximation** of human preferences. The harder a policy optimises it, the more easily it finds its loopholes — **reward hacking**, another case of Goodhart's law (Part 4, lesson 5):

- **Length bias**: as in section 3.2's example, the reward model favours long answers, so the policy learns to write long, empty ones.
- **Sycophancy**: people tend to rate answers that agree with them more highly, so the model learns to go along with the user even when the user is wrong.
- **Format gaming**: heavy use of lists, bold text and emoji, because they look "more thorough".
- Gao et al. (2022) studied "over-optimisation" systematically: as the policy moves further from the reference, the reward model's score keeps rising, but the **true** quality first rises and then falls.

Remedies include the KL penalty, bigger and better reward models, controlling for length in the preference data, regularly collecting fresh preference data to retrain the reward model, and averaging several reward models.

## 3.7 Letting AI give the feedback

Human preference labels are slow and expensive. Hence **RLAIF** (Reinforcement Learning from AI Feedback): a strong model compares answers instead of people. Anthropic's **Constitutional AI** (2022) is the best-known example: first write down a set of principles (the "constitution"), such as "choose the more honest, less harmful answer"; then have the model critique and revise its own answers and compare pairs of answers according to those principles, and train on the AI-generated preferences. This makes the alignment rules more transparent (they're written in the principles) and greatly reduces cost.

## 3.8 Summary

- Comparing is easier than demonstrating; preference data comes as (prompt, chosen, rejected) triples.
- Reward models are trained with the Bradley–Terry loss −log σ(r_chosen − r_rejected); they pick up biases in the data (such as favouring long answers).
- The RLHF objective: maximise E[r] − β·KL(π‖π_ref); the optimum is π* ∝ π_ref·exp(r/β); β sets how far from the reference the policy may go.
- REINFORCE: ∇E[r] = E[(r − b)∇log π], with a baseline to reduce variance; PPO limits each update with a clipped probability ratio, but needs 4 models and is complex and costly.
- Reward hacking: length, sycophancy, formatting; countered with KL, better reward models, length control and more.
- RLAIF and Constitutional AI have models give feedback according to principles.

## 3.9 Exercises

<!-- exercise: bt-prob -->
### Exercise 1: The Bradley–Terry probability

Read the rewards `r_a` and `r_b` of two answers, and print the probability that "A is better than B", σ(r_a − r_b), and the loss when A is the chosen answer, −log σ(r_a − r_b), each to 4 decimal places, separated by a space.

```text
Sample input: 2.0 0.5
Sample output: 0.8176 0.2014
```

<details>
<summary>Hint</summary>

`p = 1 / (1 + math.exp(-(ra - rb)))`, and the loss is `-math.log(p)`.
</details>

<!-- exercise: rm-eval -->
### Exercise 2: Evaluating a reward model

The first line holds the number of preference pairs `n`; each of the next `n` lines holds the reward model's scores for the chosen answer and the rejected answer. Print the average Bradley–Terry loss (4 decimal places) and the accuracy (the fraction where the chosen score is **strictly higher** than the rejected one, as a percentage with 1 decimal place), separated by a space.

```text
Sample input:
3
2.0 0.5
0.1 0.3
1.0 -1.0
Sample output: 0.3755 66.7%
```

<details>
<summary>Hint</summary>

Each pair's loss is `-math.log(sigmoid(rc - rr))`; print the accuracy with `f"{acc:.1%}"`.
</details>

<!-- exercise: kl-optimal -->
### Exercise 3: The optimal policy under a KL constraint

The first line gives the reference model's probability of each answer, the second each answer's reward, and the third β. Print the optimal policy π* ∝ π_ref · exp(r/β), each probability to 4 decimal places, separated by spaces.

```text
Sample input:
0.3 0.3 0.2 0.2
1.0 0.6 -1.0 -0.5
1
Sample output: 0.5238 0.3511 0.0473 0.0779
```

<details>
<summary>Hint</summary>

Compute `ref[i] * math.exp(reward[i] / beta)` for each answer, then divide by their sum.
</details>

<!-- exercise: ppo-clip -->
### Exercise 4: PPO's clipped objective

The first line is ε; each following line is a pair `ratio advantage`, until the end of the input. For each line print PPO's objective `min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)` to 4 decimal places.

```text
Sample input:
0.2
1.0 1.0
1.5 1.0
0.5 -1.0
Sample output:
1.0000
1.2000
-0.8000
```

<details>
<summary>Hint</summary>

`clipped = max(1 - eps, min(1 + eps, ratio))`, then print `min(ratio * adv, clipped * adv)`.
</details>

<!-- exercise: why-kl -->
### Exercise 5: Why the KL penalty?

Why does the RLHF objective include the term β·KL(π‖π_ref)?

- to make training faster
- to stop the policy, in chasing the reward model's high scores, from straying far from the original model and producing content the reward model rates highly but that is actually bad
- to make the model's answers shorter
- because the gradient can't be computed without it

<details>
<summary>Answer and explanation</summary>

A reward model only approximates human preferences and has loopholes. The KL penalty keeps the policy near the original SFT model, preserving fluent language and existing abilities while reducing reward hacking.
</details>

<!-- exercise: hacking -->
### Exercise 6: Reward hacking

Which of these is typical "reward hacking"?

- the model learns to get maths problems right
- the reward model favours long answers, so the policy learns to stuff its answers with repetitive, useless content
- the training data contains duplicate web pages
- the model's context length is too short

<details>
<summary>Answer and explanation</summary>

Reward hacking means the policy exploits flaws in the reward model to score highly while real quality doesn't improve, or even falls. Length bias is the most common example.
</details>

## 3.10 Further reading

- Christiano et al., *Deep Reinforcement Learning from Human Preferences* (2017)
- Ouyang et al., *Training language models to follow instructions with human feedback* (2022, InstructGPT)
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
- Gao et al., *Scaling Laws for Reward Model Overoptimization* (2022)
- Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (2022)
- Nathan Lambert, *The RLHF Book* (free online at rlhfbook.com)
- Next lesson: [DPO and friends](../04-dpo/en.md)
