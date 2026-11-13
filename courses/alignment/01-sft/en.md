# 1. Instruction tuning: from "text continuer" to "assistant"

> 🎯 **Goals**: tell **base models** and **chat models** apart; see the overall post-training pipeline (SFT → preference optimisation → reinforcement learning); understand where instruction-tuning (**SFT**) data comes from and why "few but good" beats "many but messy"; read and implement **chat templates** (ChatML, Llama 3) yourself; understand why training computes the loss only on **the assistant's replies**, and implement the loss mask; know SFT's usual practices, its risks, and "distillation".
>
> ⏱ **Time**: 70 minutes　|　📋 **You need**: Part 3; Part 4, lessons 1 and 5

A model fresh from pre-training is a **base model**. It has read the internet and knows a great deal, but it does only one thing: **continue text**. Ask it "What is the capital of France?" and it may carry on with "What is the capital of Germany? What is the capital of Italy?" — because that's what quiz lists on the web look like. Turning it into an **assistant** that answers questions and follows instructions takes **post-training**.

## 1.1 Base models and chat models

Typical reactions of the two kinds of model to the same question:

```text
Input: How do I make a soft-boiled egg?

Base model (continuing):
How do I make a soft-boiled egg? How do I fry the perfect egg? How do I make egg fried rice? This week we've collected 10 everyday egg recipes for you...

Chat model (answering):
1. Bring water to the boil and gently lower a fridge-cold egg in with a spoon;
2. keep it at a gentle simmer for 6 minutes 30 seconds;
3. move it straight into ice water to cool, then peel.
```

The base model isn't ignorant of eggs — all its knowledge is there; it just doesn't know that "now it should answer the question". What post-training teaches is mainly **format and behaviour**: when to answer, how to organise an answer, how to refuse harmful requests, how to stay consistent across a multi-turn conversation.

## 1.2 The post-training pipeline

OpenAI's **InstructGPT** (2022) established the classic three steps from which today's pipelines grew:

```text
base model
  │ ① supervised fine-tuning (SFT): train on demonstrations of "instruction → good answer"   ← this lesson
  ▼
SFT model
  │ ② preference optimisation: bring the model closer to human preferences (RLHF lesson 3, DPO lesson 4)
  ▼
aligned chat model
  │ ③ reinforcement learning with verifiable rewards: practise reasoning on tasks that can be checked automatically, such as maths and code (lesson 5)
  ▼
reasoning model
```

InstructGPT's famous result: a 1.3-billion-parameter model put through this pipeline was preferred by human raters over the original 175-billion-parameter GPT-3. **Post-training uses a tiny fraction of pre-training's compute, yet it decides what using the model feels like.**

## 1.3 SFT data: few but good

**Supervised fine-tuning** (SFT) trains exactly like pre-training — predicting the next token — only the data is now **demonstration conversations**:

```text
User: Translate "天气真好" into English.
Assistant: The weather is really nice.
```

Where does this data come from?

- **Written by people**: annotators write questions and high-quality answers. The best quality, but expensive.
- **Distilled from stronger models**: a powerful model writes the answers. Stanford's Alpaca (2023) spent a few hundred dollars generating 52,000 instructions with an OpenAI model and turned Llama 7B into a respectable assistant. (Note: many commercial models' terms of use forbid using their outputs to train competing models.)
- **Existing data rewritten as conversations**: Q&A sites, textbook exercises and so on.

An important finding came from Meta's **LIMA** (2023): just **1000** carefully chosen demonstrations made a 65-billion-parameter base model into a rather good assistant. The researchers proposed the "**superficial alignment hypothesis**": a model's knowledge and abilities come almost entirely from pre-training, and SFT mainly teaches it **which format and style** to use with people. So the **quality and diversity** of SFT data matter far more than its quantity — and a wrong or sloppy demonstration will be learned just as faithfully.

## 1.4 Chat templates

A conversation has several turns and roles (system, user, assistant), but the model only sees a sequence of tokens. A **chat template** specifies how to "flatten" a conversation into text, marking where each part starts and ends with **special tokens**. Two common templates:

```python
def chatml(messages, add_generation_prompt=True):
    """The ChatML format (used by Qwen and others)."""
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"          # tells the model: the assistant speaks next
    return text

def llama3(messages, add_generation_prompt=True):
    """Llama 3's format."""
    text = "<|begin_of_text|>"
    for m in messages:
        text += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    if add_generation_prompt:
        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text

conversation = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is 1 + 1?"},
]
print(chatml(conversation))
print("------")
print(llama3(conversation))
```

- `<|im_start|>`, `<|eot_id|>` and the like are **special tokens** reserved in the tokeniser; ordinary text is never split into them, so the model can reliably tell who is speaking.
- When generating, the model sees `<|im_start|>assistant\n` and knows it's time to answer; when it's done it generates `<|im_end|>` (or `<|eot_id|>`), and the program stops generating on that token.
- **Using the wrong template is one of the most common mistakes**: a model trained with template A and chatted with template B performs much worse, or even talks nonsense. Hugging Face tokenisers store the model's own template; call `tokenizer.apply_chat_template(messages)` rather than assembling it yourself.
- The "system" message sets the assistant's identity, tone and rules; Part 7 covers how to write a good one.

## 1.5 Computing the loss only on the answers

In training we want the model to learn **how to answer**, not **how to ask** — the user's words are given and needn't be "predicted". So when computing the loss, the system and user parts are **masked**, and only the assistant's replies count. The way to do it: set the labels at positions that shouldn't count to a special value (in PyTorch, by convention, **-100**), which cross-entropy skips automatically:

```python
# the tokens of a conversation (words stand in for tokens here), and who each token belongs to
tokens = ["<|im_start|>", "user", "Hi", "<|im_end|>",
          "<|im_start|>", "assistant", "Hello!", "How", "can", "I", "help?", "<|im_end|>"]
roles  = ["user"] * 6 + ["assistant"] * 6           # the reply starts at "Hello!" and includes the final end token
ids = list(range(100, 100 + len(tokens)))           # pretend token ids

labels = [i if r == "assistant" else -100 for i, r in zip(ids, roles)]
for t, l in zip(tokens, labels):
    print(f"label {l:>5}   {t}")
print("tokens that count towards the loss:", sum(l != -100 for l in labels), "/", len(labels))
```

Note that the end token `<|im_end|>` counts too: the model must learn **to stop when its answer is finished**, or it will go on writing forever.

In PyTorch, `cross_entropy`'s `ignore_index` defaults to -100:

```python requires=torch
import torch
import torch.nn.functional as F

torch.manual_seed(0)
vocab, T = 50, 8
logits = torch.randn(T, vocab)                         # the model's predictions at 8 positions
labels = torch.tensor([-100, -100, -100, 7, 12, 3, 9, -100])   # only 4 positions are the assistant's reply

masked = F.cross_entropy(logits, labels)               # skips -100 automatically
manual = F.cross_entropy(logits[3:7], labels[3:7])     # take only those 4 positions by hand
print(f"masked loss {masked.item():.4f}, reply-only loss {manual.item():.4f}")
```

## 1.6 SFT in practice

With Hugging Face's **TRL** library, SFT takes only a few lines (it needs a GPU and an internet connection to download the model; the API may change between versions, so check the official documentation):

```python norun
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")   # each example is a "messages" conversation
trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",                                 # a small base model
    train_dataset=dataset,
    args=SFTConfig(output_dir="qwen3-sft"),
)
trainer.train()
```

Some lessons from experience:

- **The learning rate is much smaller than in pre-training** (around 1e-5 is common for full fine-tuning), for just 1–3 epochs.
- **Packing**: as in pre-training, several short conversations can share one sequence to keep the GPU busy (Part 4, lesson 1).
- **Catastrophic forgetting**: training too long on narrow data makes the model forget its other abilities. The usual remedy is to mix in some general conversation data and check a range of benchmarks (Part 4, lesson 5).
- **Too much training "overfits" to style**: answers become formulaic, or the model even invents content to imitate the confident tone of its training data.
- **Distilling reasoning**: when DeepSeek-R1 was released (2025), the team used about 800,000 examples with long reasoning traces generated by R1 to fine-tune smaller Qwen and Llama models, greatly improving their maths and coding. For small models, "copying the homework" of a strong model is often more effective than finding their own way with reinforcement learning.

When there are too many parameters for your GPU memory, you can train only a tiny fraction of them — that's **LoRA**, the next lesson.

## 1.7 Summary

- Base models only continue text; post-training teaches the format and behaviour of an assistant.
- The post-training pipeline: SFT → preference optimisation (RLHF, DPO) → RL with verifiable rewards — far less compute than pre-training.
- SFT trains on demonstration conversations exactly as in pre-training; data comes from people, distillation from strong models and rewriting; quality and diversity matter more than quantity (LIMA: 1000 examples).
- Chat templates mark roles and boundaries with special tokens; always use the model's own template (`apply_chat_template`).
- Compute the loss only on the assistant's replies (end token included): label every other position -100.
- SFT uses a small learning rate and few epochs; watch out for catastrophic forgetting; fine-tuning small models on a strong model's reasoning (distillation) works well.

## 1.8 Exercises

<!-- exercise: chatml -->
### Exercise 1: The ChatML template

The first line holds the number of messages `n`; each of the next `n` lines has the form `role: content` (a colon and one space between them). Print the whole text in section 1.4's ChatML format, ending with `<|im_start|>assistant` (the generation prompt).

```text
Sample input:
2
system: Be brief.
user: Hi!
Sample output:
<|im_start|>system
Be brief.<|im_end|>
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
```

<details>
<summary>Hint</summary>

`role, content = line.split(": ", 1)`. For each message print `<|im_start|>{role}\n{content}<|im_end|>`; finish with a line `<|im_start|>assistant`.
</details>

<!-- exercise: masked-loss -->
### Exercise 2: The loss on replies only

The first line gives each token's role as a letter (`U` for user, `A` for assistant), separated by spaces; the second gives each token's loss. Print the average loss over the assistant tokens only, to 4 decimal places. (There's at least one assistant token.)

```text
Sample input:
U U U A A A A
5.1 4.8 6.0 1.2 0.8 2.0 0.4
Sample output: 1.1000
```

<details>
<summary>Hint</summary>

Walk through the roles and losses together with `zip`, adding up only the losses whose role is `A`, then divide by the number of assistant tokens.
</details>

<!-- exercise: pack-sft -->
### Exercise 3: Packing conversations

The first line is the maximum sequence length `L`; the second gives each conversation's number of tokens (none longer than L). Pack them **in order**: whenever the current sequence can't fit the next conversation, start a new sequence. Print the number of sequences needed and the average fill rate of all sequences (total conversation tokens ÷ (sequences × L)) as a percentage with 1 decimal place.

```text
Sample input:
10
4 5 3 8 2 6
Sample output: 4 70.0%
```

<details>
<summary>Hint</summary>

Keep `used`, how much of the current sequence is taken; when `used + n > L`, count one more sequence and set `used = n`, otherwise `used += n`. Remember that the first conversation starts a sequence too.
</details>

<!-- exercise: why-mask -->
### Exercise 4: Why mask the user's words?

Why does SFT usually compute the loss only on the assistant's replies?

- because the user's words are too short to compute a loss on
- because we're training the model how to answer, not how to imitate users asking; the user's words are given input
- because computing the loss on the user's words makes training 10 times slower
- because the user's words contain no special tokens

<details>
<summary>Answer and explanation</summary>

The user's and system's words are the "conditions" of the conversation; the model doesn't need to learn to generate them. Computing the loss on them too would make the model spend effort imitating users, and it might start "speaking for the user" in its replies.
</details>

<!-- exercise: lima -->
### Exercise 5: What LIMA shows

LIMA (2023) turned a base model into a good assistant with just 1000 selected examples. What does this show?

- pre-training doesn't really matter
- a model's knowledge and abilities come mainly from pre-training, and SFT mainly teaches the format and style of communication, so data quality matters more than quantity
- 1000 examples suit any task, and more are useless
- SFT doesn't need to compute a loss

<details>
<summary>Answer and explanation</summary>

This is the "superficial alignment hypothesis": a base model already has the knowledge and abilities, and a few high-quality demonstrations teach it to express them as an assistant. It doesn't mean more data is never useful — learning new skills (such as long reasoning) still takes a lot of data.
</details>

## 1.9 Further reading

- Ouyang et al., *Training language models to follow instructions with human feedback* (2022, InstructGPT)
- Zhou et al., *LIMA: Less Is More for Alignment* (2023)
- Taori et al., Stanford Alpaca (2023)
- Hugging Face: [chat templates documentation](https://huggingface.co/docs/transformers/chat_templating), [TRL documentation](https://huggingface.co/docs/trl)
- Next lesson: [LoRA and QLoRA](../02-lora/en.md)
