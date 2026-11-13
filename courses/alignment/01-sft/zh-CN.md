# 1. 指令微调：从「续写机器」到「助手」

> 🎯 **本课目标**：分清**基础模型**和**对话模型**；了解后训练的整体流程（SFT → 偏好优化 → 强化学习）；理解指令微调（**SFT**）的数据从哪里来，为什么「少而精」比「多而杂」更好；看懂并亲手实现**对话模板**（ChatML、Llama 3）；理解为什么训练时只在**助手的回答**上计算损失，并实现损失掩码；知道 SFT 的常见做法、风险和「蒸馏」。
>
> ⏱ **预计用时**：70 分钟　｜　📋 **前置知识**：第三部分；第四部分第 1、5 课

预训练完的模型，叫**基础模型**（base model）。它读遍了互联网，知识渊博，但它只会做一件事：**续写**。你问它「法国的首都是哪里？」，它可能接着写「德国的首都是哪里？意大利的首都是哪里？」——因为网上的题目列表就长这样。要把它变成一个会回答问题、听从指令的**助手**，还需要**后训练**（post-training）。

## 1.1 基础模型和对话模型

同一个问题，两种模型的典型反应：

```text
输入：怎样煮一个溏心蛋？

基础模型（续写）：
怎样煮一个溏心蛋？怎样煎出完美的荷包蛋？怎样做蛋炒饭？本期我们为大家整理了 10 个鸡蛋的家常做法……

对话模型（回答）：
1. 水烧开后，用勺子把冷藏的鸡蛋轻轻放进锅里；
2. 保持小火微沸，煮 6 分 30 秒；
3. 立刻捞出放进冰水里冷却，剥壳即可。
```

基础模型并非不懂怎么煮蛋——它的知识都在，只是不知道「现在该回答问题」。后训练要教给它的，主要是**格式和行为**：什么时候该回答、怎样组织答案、怎样拒绝有害的请求、怎样在多轮对话里保持一致。

## 1.2 后训练的流程

2022 年 OpenAI 的 **InstructGPT** 确立了经典的三步，今天的流程在此基础上发展而来：

```text
基础模型
  │ ① 监督微调（SFT）：用「指令 → 好回答」的示范数据训练        ← 本课
  ▼
SFT 模型
  │ ② 偏好优化：让模型更符合人的偏好（RLHF 第 3 课、DPO 第 4 课）
  ▼
对齐的对话模型
  │ ③ 可验证奖励的强化学习：在数学、代码等能自动判对错的任务上练习推理（第 5 课）
  ▼
推理模型
```

InstructGPT 有个著名的结果：经过这套流程的 13 亿参数模型，在人类评价中比 1750 亿参数的原始 GPT-3 还受欢迎。**后训练用的计算量只是预训练的零头，却决定了模型用起来的感觉。**

## 1.3 SFT 数据：少而精

**监督微调**（Supervised Fine-Tuning，SFT）的训练方法和预训练完全一样——预测下一个 token——只是数据换成了**示范对话**：

```text
用户：把「天气真好」翻译成英文。
助手：The weather is really nice.
```

这些数据从哪里来？

- **人工编写**：请标注员写问题和高质量的回答。质量最高，但很贵。
- **从更强的模型蒸馏**：让强大的模型来写回答。斯坦福的 Alpaca（2023）只花了几百美元，用 OpenAI 的模型生成了 5.2 万条指令数据，就把 Llama 7B 变成了一个像样的助手。（注意：很多商业模型的使用条款禁止用它们的输出训练竞争模型。）
- **把已有的数据改写成对话**：问答网站、教科书习题等。

一个重要的发现来自 Meta 的 **LIMA**（2023）：只用 **1000 条**精心挑选的示范，就能让一个 650 亿参数的基础模型成为相当不错的助手。研究者提出了「**表面对齐假说**」：模型的知识和能力几乎全部来自预训练，SFT 主要是教会它**用什么格式、以什么风格**与人交流。所以 SFT 数据的**质量和多样性**远比数量重要——一条错误或敷衍的示范，模型也会认真学下来。

## 1.4 对话模板

对话由多轮、多个角色（系统、用户、助手）组成，而模型只认识一串 token。**对话模板**（chat template）规定了怎样把对话「压平」成一段文本，并用**特殊 token** 标出每一段的开始和结束。下面是两种常见的模板：

```python
def chatml(messages, add_generation_prompt=True):
    """ChatML 格式（Qwen 等模型使用）。"""
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"          # 提示模型：轮到助手说话了
    return text

def llama3(messages, add_generation_prompt=True):
    """Llama 3 的格式。"""
    text = "<|begin_of_text|>"
    for m in messages:
        text += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    if add_generation_prompt:
        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text

conversation = [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "1 + 1 等于几？"},
]
print(chatml(conversation))
print("------")
print(llama3(conversation))
```

- `<|im_start|>`、`<|eot_id|>` 等是分词器里专门保留的**特殊 token**，普通文字永远不会被切成它们，所以模型能可靠地分辨「谁在说话」。
- 模型生成时，看到 `<|im_start|>assistant\n` 就知道该写回答了；写完后它会生成 `<|im_end|>`（或 `<|eot_id|>`），程序看到这个 token 就停止生成。
- **用错模板是最常见的错误之一**：用 A 模板训练的模型，换成 B 模板来对话，效果会大打折扣，甚至胡言乱语。Hugging Face 的分词器里保存了模型自己的模板，用 `tokenizer.apply_chat_template(messages)` 调用即可，不要自己拼。
- 「系统」消息用来设定助手的身份、语气和规则，第七部分会详细讲怎样写好它。

## 1.5 只在回答上计算损失

训练时，我们希望模型学会**怎样回答**，而不是学会**怎样提问**——用户的话是给定的，不需要模型来「预测」。所以计算损失时，要把系统消息和用户消息部分**遮住**，只在助手的回答上计算。实现方法是：把不需要算损失的位置的标签设成一个特殊值（PyTorch 的约定是 **-100**），交叉熵会自动跳过这些位置：

```python
# 一段对话切成的 token（这里用词代替 token），以及每个 token 属于谁
tokens = ["<|im_start|>", "user", "你好", "<|im_end|>",
          "<|im_start|>", "assistant", "你好！", "有什么", "可以", "帮你？", "<|im_end|>"]
roles  = ["user"] * 6 + ["assistant"] * 5           # 助手的回答从「你好！」开始，包括最后的结束符
ids = list(range(100, 100 + len(tokens)))           # 假装的 token id

labels = [i if r == "assistant" else -100 for i, r in zip(ids, roles)]
for t, l in zip(tokens, labels):
    print(f"标签 {l:>5}   {t}")
print("参与计算损失的 token：", sum(l != -100 for l in labels), "/", len(labels))
```

注意结束符 `<|im_end|>` 也要计算损失：模型必须学会**在回答完时停下来**，否则它会一直写下去。

在 PyTorch 里，`cross_entropy` 的 `ignore_index` 参数默认就是 -100：

```python requires=torch
import torch
import torch.nn.functional as F

torch.manual_seed(0)
vocab, T = 50, 8
logits = torch.randn(T, vocab)                         # 模型对 8 个位置的预测
labels = torch.tensor([-100, -100, -100, 7, 12, 3, 9, -100])   # 只有 4 个位置是助手的回答

masked = F.cross_entropy(logits, labels)               # 自动跳过 -100
manual = F.cross_entropy(logits[3:7], labels[3:7])     # 手动只取那 4 个位置
print(f"带掩码的损失 {masked.item():.4f}，只算回答部分 {manual.item():.4f}")
```

## 1.6 SFT 的实际做法

用 Hugging Face 的 **TRL** 库，SFT 只需要几行代码（需要 GPU 和联网下载模型，API 可能随版本变化，以官方文档为准）：

```python norun
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")   # 每条数据是一段 messages 对话
trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",                                 # 一个小的基础模型
    train_dataset=dataset,
    args=SFTConfig(output_dir="qwen3-sft"),
)
trainer.train()
```

一些经验：

- **学习率比预训练小得多**（全参数微调常用 1e-5 左右），只训练 1～3 个 epoch。
- **打包**：和预训练一样，可以把多条短对话拼进一个序列里，提高 GPU 利用率（第四部分第 1 课）。
- **灾难性遗忘**：在一个窄领域的数据上训练太久，模型会忘掉其他能力。常见的对策是混入一部分通用的对话数据，并在多种评测上检查（第四部分第 5 课）。
- **训练太多会「过拟合」到风格上**：回答开始千篇一律，甚至编造内容来模仿训练数据中自信的语气。
- **蒸馏推理能力**：DeepSeek-R1（2025）发布时，团队用 R1 生成的约 80 万条带有长推理过程的数据，对 Qwen 和 Llama 的较小模型做了 SFT，小模型的数学和代码能力大幅提升。对小模型来说，从强模型「抄作业」往往比自己用强化学习摸索更有效。

参数太多、显存不够时，可以只训练很少的一部分参数——这就是下一课的 **LoRA**。

## 1.7 小结

- 基础模型只会续写；后训练教会它作为助手的格式和行为。
- 后训练流程：SFT → 偏好优化（RLHF、DPO）→ 可验证奖励的强化学习；计算量远小于预训练。
- SFT 用示范对话训练，方法和预训练相同；数据来自人工、强模型蒸馏、改写；质量和多样性比数量重要（LIMA：1000 条）。
- 对话模板用特殊 token 标出角色和边界；必须使用模型自己的模板（`apply_chat_template`）。
- 只在助手的回答（包括结束符）上计算损失：其余位置的标签设为 -100。
- SFT 学习率小、轮数少，注意灾难性遗忘；用强模型的推理过程做 SFT（蒸馏）对小模型很有效。

## 1.8 练习

<!-- exercise: chatml -->
### 练习 1：ChatML 模板

第一行输入消息条数 `n`，接下来 `n` 行，每行格式是 `角色: 内容`（角色和内容之间是一个冒号加一个空格）。按 1.4 节的 ChatML 格式输出整段文本，并在最后加上 `<|im_start|>assistant`（生成提示）。

```text
样例输入：
2
system: Be brief.
user: Hi!
样例输出：
<|im_start|>system
Be brief.<|im_end|>
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
```

<details>
<summary>提示</summary>

`role, content = line.split(": ", 1)`。每条消息输出 `<|im_start|>{role}\n{content}<|im_end|>\n`；最后输出 `<|im_start|>assistant` 一行。用 `print(..., end="")` 可以避免多出空行。
</details>

<!-- exercise: masked-loss -->
### 练习 2：只算回答的损失

第一行是每个 token 的角色，用字母表示（`U` 是用户，`A` 是助手），用空格隔开；第二行是每个 token 的损失。输出只在助手 token 上计算的平均损失，保留 4 位小数。（至少有一个助手 token。）

```text
样例输入：
U U U A A A A
5.1 4.8 6.0 1.2 0.8 2.0 0.4
样例输出：1.1000
```

<details>
<summary>提示</summary>

用 `zip` 同时遍历角色和损失，只累加角色为 `A` 的损失，再除以助手 token 的个数。
</details>

<!-- exercise: pack-sft -->
### 练习 3：打包对话

第一行是序列的最大长度 `L`，第二行是每条对话的 token 数（每条都不超过 L）。**按顺序**打包：当前序列放不下下一条对话时，就开启一个新序列。输出需要的序列数，以及所有序列的平均填充率（对话的总 token 数 ÷ (序列数 × L)），填充率用百分数表示、保留 1 位小数。

```text
样例输入：
10
4 5 3 8 2 6
样例输出：4 70.0%
```

<details>
<summary>提示</summary>

用一个变量 `used` 记录当前序列已经用了多少；`used + n > L` 时序列数加 1，`used = n`，否则 `used += n`。别忘了第一条对话也要开启一个序列。
</details>

<!-- exercise: why-mask -->
### 练习 4：为什么要遮住用户的话？

SFT 时，为什么通常只在助手的回答上计算损失？

- 因为用户的话太短，算不出损失
- 因为我们要训练模型怎样回答，而不是怎样模仿用户提问；用户的话是给定的输入
- 因为对用户的话计算损失会让训练变慢 10 倍
- 因为用户的话里没有特殊 token

<details>
<summary>答案与解析</summary>

用户和系统的话是对话的「条件」，模型不需要学会生成它们。如果也在上面计算损失，模型会花力气去学习模仿用户，并且可能在回答时自己「替用户说话」。
</details>

<!-- exercise: lima -->
### 练习 5：LIMA 的启示

LIMA（2023）只用 1000 条精选数据就把基础模型训练成了不错的助手。这说明了什么？

- 预训练其实不重要
- 模型的知识和能力主要来自预训练，SFT 主要教会它交流的格式和风格，所以数据质量比数量更重要
- 1000 条数据适用于任何任务，再多就没有用了
- SFT 不需要计算损失

<details>
<summary>答案与解析</summary>

这就是「表面对齐假说」：基础模型已经具备了知识和能力，少量高质量的示范就能教会它如何以助手的方式表达。这并不意味着更多数据永远没用——学习新的技能（比如长推理）仍然需要大量数据。
</details>

## 1.9 延伸阅读

- Ouyang 等：《Training language models to follow instructions with human feedback》（2022，InstructGPT）
- Zhou 等：《LIMA: Less Is More for Alignment》（2023）
- Taori 等：Stanford Alpaca（2023）
- Hugging Face：[聊天模板文档](https://huggingface.co/docs/transformers/chat_templating)、[TRL 文档](https://huggingface.co/docs/trl)
- 下一课：[LoRA 与 QLoRA](../02-lora/zh-CN.md)
