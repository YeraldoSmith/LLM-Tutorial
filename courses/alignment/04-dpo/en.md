# 4. DPO and friends: preference optimisation without reinforcement learning

> 🎯 **Goals**: starting from last lesson's RLHF optimum, **derive the DPO loss** and understand the phrase "your language model is secretly a reward model"; compute the DPO loss and **implicit rewards**; train with DPO on a toy problem and check that it pursues the same optimum as RLHF; know how DPO is done in practice (data, β, the reference model, learning rate) and its common problems; meet the ideas behind variants such as IPO, KTO, ORPO and SimPO.
>
> ⏱ **Time**: 70 minutes　|　📋 **You need**: lessons 1 and 3 of this part

Last lesson's RLHF first trains a reward model and then runs reinforcement learning with PPO: 4 models at once, constant generation during training, and many hyperparameters — very hard to tune. In 2023, Rafailov et al. at Stanford proposed **DPO** (Direct Preference Optimization): **train the language model directly on preference data**, with no separate reward model and no reinforcement learning. Simple and stable, it quickly became the most common preference-optimisation method for open models.

## 4.1 Starting from the RLHF optimum

Last lesson gave the optimal policy under a KL constraint:

```formula
π*(y) = π_ref(y) · exp(r(y) / β) / Z
```

Z is the normalising constant that makes the probabilities add up to 1 (a fixed number for a given prompt, but it requires summing over every possible answer, so it can't be computed). Turn this equation **around** to express the reward in terms of the policy:

```formula
r(y) = β · log( π*(y) / π_ref(y) ) + β · log Z
```

This says: **once you know the optimal policy and the reference model, the reward is determined** (up to a constant). Now substitute it into the reward model's Bradley–Terry formula. For the same prompt, with chosen answer y_w and rejected answer y_l:

```formula
P(y_w better than y_l) = σ( r(y_w) − r(y_l) ) = σ( β · log(π(y_w)/π_ref(y_w)) − β · log(π(y_l)/π_ref(y_l)) )
```

The uncomputable β·log Z **cancels out exactly**! So the likelihood of the preferences can be written directly in terms of the language model π's probabilities, and maximising it (minimising its negative log) gives the **DPO loss**:

```formula
L_DPO = −log σ( β · [ log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l)) ] )
```

The DPO paper's subtitle is "Your Language Model is Secretly a Reward Model": `β · log(π(y)/π_ref(y))` is the model's **implicit reward**. Training with DPO trains this implicit reward model to fit the preferences — and the policy itself is then the very optimal policy RLHF was after.

## 4.2 Computing the DPO loss

DPO needs only four numbers: the log-probabilities the policy model and the reference model give to the chosen and rejected answers (each answer's log-probability is the sum over its tokens):

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def dpo(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    reward_w = beta * (pi_w - ref_w)           # the chosen answer's implicit reward
    reward_l = beta * (pi_l - ref_l)           # the rejected answer's implicit reward
    margin = reward_w - reward_l
    return -math.log(sigmoid(margin)), reward_w, reward_l

# the reference model's log-probabilities for the two answers (fixed during training)
ref_w, ref_l = -42.0, -40.0
for label, pi_w, pi_l in [("start of training (π = π_ref)", -42.0, -40.0),
                          ("chosen answer more likely", -38.0, -40.0),
                          ("rejected answer less likely", -42.0, -50.0),
                          ("both fall, rejected falls more", -45.0, -60.0)]:
    loss, rw, rl = dpo(pi_w, pi_l, ref_w, ref_l)
    print(f"loss {loss:.4f}, implicit rewards: chosen {rw:+.2f}, rejected {rl:+.2f}  ← {label}")
```

Some observations:

- At the start, π = π_ref, both implicit rewards are 0, and the loss is log 2 ≈ 0.693.
- The loss only cares about the **difference**: raising the chosen answer's probability or lowering the rejected one's both reduce it.
- The last line is common: **both** answers became less likely, the rejected one more so. The loss went down, yet the good chosen answer also became less likely — a known problem with DPO (section 4.5).
- Taking the gradient for one preference pair shows it's weighted by `σ(−margin)`: pairs the model already separates well (large margin) contribute almost nothing, so learning focuses on the pairs it **hasn't learned yet**.

## 4.3 Checking it on a toy problem

Back to last lesson's 4-answer problem. This time we **don't tell** the model the rewards; we give it only 4000 preferences, generated at random from the "true rewards" with Bradley–Terry probabilities (simulating human labels). Running gradient descent on the DPO loss, where does the policy go?

```python
import math
import random

random.seed(1)

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

ref = [0.3, 0.3, 0.2, 0.2]
true_reward = [1.0, 0.6, -1.0, -0.5]           # the model never sees this, only preferences generated from it
beta = 0.3

data = []                                      # (chosen, rejected)
for _ in range(4000):
    i, j = random.sample(range(4), 2)
    if random.random() < sigmoid(true_reward[i] - true_reward[j]):
        data.append((i, j))
    else:
        data.append((j, i))

theta = [math.log(p) for p in ref]             # the policy's logits, starting from the reference model
for step in range(601):
    pi = softmax(theta)
    grad, loss = [0.0] * 4, 0.0
    for w, l in data:
        z = beta * (math.log(pi[w] / ref[w]) - math.log(pi[l] / ref[l]))
        loss += -math.log(sigmoid(z))
        g = (1 - sigmoid(z)) * beta            # this pair's gradient weight σ(−z)·β
        grad[w] += g
        grad[l] -= g
    theta = [t + 5.0 * g / len(data) for t, g in zip(theta, grad)]
    if step % 200 == 0:
        print(f"step {step:3d}: loss {loss / len(data):.4f}, policy {[round(p, 3) for p in softmax(theta)]}")

best = [p * math.exp(r / beta) for p, r in zip(ref, true_reward)]
print("RLHF's theoretical optimum:", [round(b / sum(best), 3) for b in best])
```

No reward model, no sampling, just preference data — and DPO arrives near RLHF's theoretical optimum (the preferences are a finite random sample, so the result differs slightly from the theory). That's exactly what section 4.1's derivation promised.

(In this particular set-up the gradient with respect to the logits is exactly `g·(e_w − e_l)`: the π-dependent terms of log softmax cancel between the chosen and rejected answers.)

## 4.4 DPO in practice

- **Data**: as for reward models, (prompt, chosen, rejected) triples. Common public datasets include UltraFeedback (several models' answers scored by GPT-4).
- **Reference model**: usually the SFT model. Its log-probabilities don't change during training, so they can be computed in advance and stored, leaving only one model in GPU memory during training.
- **β**: around 0.1 is common. The larger β, the less the model dares to stray from the reference.
- **Learning rate**: very small, around 5e-7 for full training; 1–3 epochs.
- With the TRL library (needs a GPU; check the official documentation for the current API):

```python norun
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")   # prompt / chosen / rejected
trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",                          # usually SFT'd first
    train_dataset=dataset,
    args=DPOConfig(output_dir="qwen3-dpo", beta=0.1),
)
trainer.train()
```

## 4.5 DPO's problems

- **The chosen answer's probability can fall too**: as in the last line of section 4.2, DPO only asks for "chosen − rejected" to grow, and both may fall together, making the model's outputs odd. A common remedy adds a little SFT loss on the chosen answers.
- **Offline data**: preference data is usually collected in advance, generated by other (or earlier) models. Once training changes the model's answers, the old data fits less well. **Online / iterative DPO** — generating new answers with the current model, labelling fresh preferences and training again — works better and comes closer to RLHF.
- **Length**: if the chosen answers in the data tend to be longer, DPO also learns "longer is better" (just like reward models, section 3.6).
- **Overfitting**: preference data is noisy (people don't always agree), and DPO may push the gap on some pairs towards infinity.

Which is better, DPO or PPO, is an ongoing debate. The general view: DPO is simple, stable and cheap, fine for most situations; a carefully tuned online RL setup has a higher ceiling. Frontier models usually use both, and on tasks with reliable automatic checking, such as maths and code, they increasingly use next lesson's reinforcement learning methods.

## 4.6 The DPO family

Many variants followed DPO, each changing the loss function:

| Method | Year | Main idea |
|--------|------|-----------|
| **IPO** | 2023 | replace −log σ with a squared loss so the implicit reward gap settles near a target value, preventing overfitting |
| **KTO** | 2024 | no pairs needed: each answer is just labelled "good" or "bad"; inspired by prospect theory from behavioural economics |
| **ORPO** | 2024 | no reference model: add an "odds ratio" penalty to the SFT loss, doing SFT and preference optimisation in one step |
| **SimPO** | 2024 | no reference model: use the **average** per-token log-probability as the implicit reward (removing length bias), and require a gap of at least γ |

SimPO's loss, for example:

```formula
L_SimPO = −log σ( β · [ log π(y_w) / |y_w| − log π(y_l) / |y_l| ] − γ )
```

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def simpo(logp_w, len_w, logp_l, len_l, beta=2.0, gamma=1.0):
    margin = beta * (logp_w / len_w - logp_l / len_l) - gamma
    return -math.log(sigmoid(margin))

# the chosen answer is long (so its total log-probability is lower), the rejected one short
print("comparing totals (as DPO does, without dividing by length): chosen", -120.0, "rejected", -30.0)
print("comparing per-token averages: chosen", -120.0 / 100, "rejected", -30.0 / 20)
print(f"SimPO loss: {simpo(-120.0, 100, -30.0, 20):.4f}")
```

A long answer naturally has a lower total log-probability; dividing by length compares "how natural each token is on average", so length no longer helps or hurts.

## 4.7 Summary

- Solving RLHF's optimum π* ∝ π_ref·exp(r/β) for the reward gives r = β·log(π/π_ref) + constant; substituted into Bradley–Terry, the constant cancels and the DPO loss appears.
- L_DPO = −log σ(β·[log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l))]); β·log(π/π_ref) is the implicit reward.
- DPO needs no reward model and no RL sampling — only preference pairs and a reference model (whose log-probabilities can be precomputed); on a toy problem it heads for RLHF's optimum.
- In practice: β ≈ 0.1 and a very small learning rate; problems: the chosen answer's probability may fall, offline data fits poorly, length bias, overfitting; online / iterative DPO works better.
- Variants: IPO (squared loss), KTO (single good/bad labels), ORPO and SimPO (no reference model; SimPO normalises by length).

## 4.8 Exercises

<!-- exercise: dpo-loss -->
### Exercise 1: The DPO loss

Read one line of 5 numbers: the policy's log-probability for the chosen answer and for the rejected answer, the reference model's log-probability for the chosen answer and for the rejected answer, and β. Print the DPO loss and the implicit reward margin (chosen − rejected), each to 4 decimal places, separated by a space.

```text
Sample input: -38 -40 -42 -40 0.1
Sample output: 0.5130 0.4000
```

<details>
<summary>Hint</summary>

`margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))`, and the loss is `-math.log(1 / (1 + math.exp(-margin)))`.
</details>

<!-- exercise: dpo-batch -->
### Exercise 2: DPO on a batch

The first line holds the number of pairs `n` and β; each of the next `n` lines holds 4 numbers (in exercise 1's order: policy chosen, policy rejected, reference chosen, reference rejected). Print the average DPO loss (4 decimal places) and the fraction of pairs whose implicit reward margin is greater than 0 (as a percentage with 1 decimal place), separated by a space.

```text
Sample input:
3 0.1
-38 -40 -42 -40
-42 -40 -42 -40
-45 -60 -42 -50
Sample output: 0.5364 66.7%
```

<details>
<summary>Hint</summary>

For each pair compute the margin and the loss; add up the losses and count the pairs with `margin > 0`.
</details>

<!-- exercise: simpo-loss -->
### Exercise 3: The SimPO loss

Read one line of 6 numbers: the chosen answer's total log-probability and token count, the rejected answer's total log-probability and token count, β and γ. Print the SimPO loss to 4 decimal places.

```text
Sample input: -120 100 -30 20 2 1
Sample output: 0.9130
```

<details>
<summary>Hint</summary>

`margin = beta * (lw / nw - ll / nl) - gamma`, and the loss is `-math.log(sigmoid(margin))`.
</details>

<!-- exercise: implicit-reward -->
### Exercise 4: Ranking by implicit reward

The first line is β; each following line describes one answer as `name policy_logprob reference_logprob`, until the end of the input. Print each answer's name and implicit reward β·(policy − reference) (3 decimal places), **from highest to lowest** reward; break ties by name in dictionary order.

```text
Sample input:
0.5
a -10 -12
b -8 -8
c -20 -15
Sample output:
a 1.000
b 0.000
c -2.500
```

<details>
<summary>Hint</summary>

Put (name, reward) pairs in a list and sort with `sorted(items, key=lambda t: (-t[1], t[0]))`.
</details>

<!-- exercise: dpo-needs -->
### Exercise 5: What doesn't DPO need?

Compared with PPO-based RLHF, which of these does DPO not need?

- preference data (pairs of chosen / rejected answers)
- a reference model (or its precomputed log-probabilities)
- a separately trained reward model and reinforcement-learning sampling during training
- a language model to train

<details>
<summary>Answer and explanation</summary>

DPO "hides" the reward model inside the policy itself (the implicit reward) and trains directly on preference pairs in a supervised way, with no separate reward model and no generation during training. It still needs preference data and a reference model.
</details>

<!-- exercise: dpo-beta -->
### Exercise 6: What β does

In DPO, what does a larger β mean?

- the model strays from the reference more boldly
- the implicit rewards are on a larger scale, so a smaller change in probabilities already opens a gap and the model needn't stray as far from the reference — like a stronger KL constraint
- a larger learning rate
- more training data

<details>
<summary>Answer and explanation</summary>

β comes from the KL penalty coefficient in the RLHF objective: the larger β, the stronger the constraint and the closer the optimal policy stays to the reference. In the DPO loss, β scales up the log-probability ratios, so very small changes are enough to reduce the loss.
</details>

## 4.9 Further reading

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023)
- Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences* (2023, IPO)
- Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization* (2024)
- Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model* (2024)
- Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward* (2024)
- Next lesson: [Reasoning models and GRPO](../05-reasoning-rl/en.md)
