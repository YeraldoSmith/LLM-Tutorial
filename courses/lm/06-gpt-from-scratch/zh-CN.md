# 6. 亲手训练一个 GPT

> 🎯 **本课目标**：把前五课的零件组装成一个**完整的 GPT**（约 100 行代码）；理解语言模型的训练数据是怎样构造的——**错一位**的输入和目标、**每个位置都在预测**；在普通电脑上用不到半分钟训练它学会两位数加法，并在**从没见过的题目**上验证它真的学会了；再学会用同一份代码在你自己的文本（比如中文小说）上训练；最后看清从「迷你 GPT」到真正的大模型还差什么。
>
> ⏱ **预计用时**：90 分钟　｜　📋 **前置知识**：本部分第 1～5 课；第二部分第 5 课（PyTorch）；需要安装 PyTorch

这一课是第三部分的高潮。我们不再只看零件，而是从头到尾训练一个真正的 Transformer 语言模型，和 GPT 的结构完全相同，只是小得多。

> [!NOTE]
> 本课的代码结构参考了 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 [minGPT](https://github.com/karpathy/minGPT)（MIT 许可证，minGPT 里也有一个加法的例子）。读完本课再去读 nanoGPT 的 `model.py`，你会发现几乎每一行都看得懂。

## 6.1 任务：让 GPT 学会做加法

为什么选加法，而不是一上来就用小说或网页？

- **不需要下载任何数据**：题目用代码生成，任何人、任何语言的读者都能直接运行；
- **能检验「真的学会了」还是「背下来了」**：两位数加法一共 100 × 100 = 10000 道题，我们**留出 1000 道**完全不给模型看，训练后再拿它们来考；
- **它就是语言建模**：把每道题写成一串字符，比如 `37+48=580`，模型只是在「预测下一个字符」——和 GPT 预测下一个 token 完全一样。

你可能注意到了：37 + 48 = 85，可我们写的是 `580`。这是一个小技巧：**答案补成 3 位后倒过来写**（`085` → `580`）。我们做加法是从个位开始算、向高位进位的；而模型是从左到右一个一个地生成。倒过来写，模型就能先生成个位、再生成十位和百位，和人做竖式的顺序一致，学起来容易得多。

## 6.2 训练数据：错一位，每个位置都在预测

语言模型的训练样本构造方式非常简单：**输入是一段文本，目标是同一段文本往后错一位**。

```text
完整序列：   3   7   +   4   8   =   5   8   0
输入 x：     3   7   +   4   8   =   5   8
目标 y：     7   +   4   8   =   5   8   0
             ↑ 看到「3」时预测「7」；看到「37」时预测「+」；……看到「37+48=58」时预测「0」
```

因为有因果掩码，位置 t 只能看到前 t 个字符，所以**一次前向传播就同时训练了每个位置的预测**——一个长度为 T 的序列提供 T 个训练信号。这正是 Transformer 训练效率高的原因之一（RNN 必须一步一步地算）。

在这个任务里，题目本身（两个随机数）是无法预测的，我们只想让模型学会**根据题目给出答案**。所以把题目部分的目标设成 `-1`，再告诉损失函数 `ignore_index=-1`，这些位置就不计入损失。（对话模型的微调也是这样做的：只在「助手的回答」上计算损失，第五部分会讲。）

## 6.3 完整代码

下面就是全部代码：数据、模型、训练、评估。模型的每个部分你都在前几课见过：token 嵌入 + 位置嵌入（第 3 课）、多头因果自注意力（第 4 课）、Block 与残差（第 5 课）、权重绑定（第 3 课）、AdamW（第二部分第 6 课）。

```python requires=torch timeout=300
import random
import time
import torch
from torch import nn
import torch.nn.functional as F

# ---------------- 1. 数据：两位数加法，例如 "37+48=" 后面跟着答案 85（补成 3 位并倒过来写："580"）
random.seed(0)
torch.manual_seed(0)
pairs = [(a, b) for a in range(100) for b in range(100)]
random.shuffle(pairs)
test_pairs, train_pairs = pairs[:1000], pairs[1000:]       # 1000 道题留作测试，模型训练时从没见过

def to_text(a, b):
    return f"{a:02d}+{b:02d}={str(a + b).zfill(3)[::-1]}"    # 结果补成 3 位再倒过来："085" → "580"

chars = list("0123456789+=")                               # 词表只有 12 个字符
stoi = {c: i for i, c in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]

block_size = 9                                             # "37+48=580" 正好 9 个字符
def get_batch(batch_size=128):
    batch = random.sample(train_pairs, batch_size)
    data = torch.tensor([encode(to_text(a, b)) for a, b in batch])
    x, y = data[:, :-1], data[:, 1:].clone()               # y 是 x 往后错一位：每个位置预测下一个字符
    y[:, :5] = -1                                          # 题目部分不算损失，只学习「=」之后的答案
    return x, y

# ---------------- 2. 模型：一个迷你 GPT
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab, block_size, d=64, n_layer=2, n_heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)                  # token 嵌入
        self.pos = nn.Embedding(block_size, d)             # 位置嵌入
        self.blocks = nn.Sequential(*[Block(d, n_heads) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight                 # 权重绑定：输出层与嵌入共用矩阵
    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        logits = self.head(self.ln(self.blocks(x)))        # (B, T, vocab)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new):                        # 贪心生成：每次取概率最大的 token
        for _ in range(n_new):
            logits = self(idx[:, -block_size:])
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

model = GPT(len(chars), block_size)
print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

def accuracy(pairs):
    model.eval()
    prompts = torch.tensor([encode(f"{a:02d}+{b:02d}=") for a, b in pairs])
    out = model.generate(prompts, 3)[:, 6:]                # 生成 3 个字符的（倒序）答案
    model.train()
    right = 0
    for (a, b), row in zip(pairs, out.tolist()):
        answer = int("".join(chars[i] for i in row)[::-1])
        right += answer == a + b
    return right / len(pairs)

# ---------------- 3. 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.1)
start = time.time()
for step in range(1, 1001):
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss.item():.3f}  测试集准确率 {accuracy(test_pairs):.1%}  ({time.time() - start:.0f}s)")

print("几道模型从没见过的题：")
for a, b in test_pairs[:6]:
    out = model.generate(torch.tensor([encode(f"{a:02d}+{b:02d}=")]), 3)[0, 6:].tolist()
    reply = int("".join(chars[i] for i in out)[::-1])
    print(f"  {a:2d} + {b:2d} = {reply:3d}   {'✓' if reply == a + b else '✗（正确答案 ' + str(a + b) + '）'}")
```

在普通笔记本电脑的 CPU 上，这段程序大约运行 10～30 秒。

## 6.4 读懂结果

运行后你会看到类似这样的输出（具体数字因电脑而略有不同）：

```text
参数量：101,440
step  200  loss 1.515  测试集准确率 2.7%
step  400  loss 0.287  测试集准确率 66.4%
step  600  loss 0.072  测试集准确率 98.2%
step  800  loss 0.097  测试集准确率 93.4%
step 1000  loss 0.002  测试集准确率 100.0%
```

- **一开始完全不会**：准确率只有百分之几，和瞎猜差不多。
- **然后突然「开窍」**：几百步之间准确率从个位数跳到 90% 以上。神经网络学习规律时常常有这种「突变」。
- **测试题也全对**：这 1000 道题模型**一次都没见过**，说明它学到的不是「37+48 的答案是 85」这样的死记硬背，而是**加法和进位的规则**。
- 中间偶尔有起伏（比如 800 步时降到 93%），这是正常的训练噪声。

一个只有 10 万个参数的模型，在不到一分钟里，仅仅通过「预测下一个字符」就学会了加法。把同样的方法放大一百万倍，就是今天的大语言模型。

> [!TIP]
> 动手试试：把 `n_layer` 改成 1、把 `d` 改成 32，或者去掉「倒着写」的技巧（把 `[::-1]` 去掉，同时改掉 `accuracy` 里的倒序），看看准确率和学习速度怎么变化。这就是研究人员每天在做的**消融实验**。

## 6.5 在你自己的文本上训练

同一个模型，换一份数据，就能学习任何语言的文字。下面是一个**字符级**语言模型的训练脚本：准备一个文本文件 `input.txt`（任何语言都行，比如一部中文小说、一本诗集），它会学习文字的风格，然后「续写」。相比加法，主要的改动是：词表来自文本本身、训练样本是文本中随机截取的片段、生成时按概率**采样**而不是总取最大值（下一课详细讲采样）。

```python norun
# 需要先准备好 input.txt（建议至少几十万字；文本越多效果越好），并把 6.3 节的 Block、GPT 类复制到这里
import torch

text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))                            # 文本里出现过的所有字符就是词表
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]            # 后 10% 做验证集，监控是否过拟合

block_size, batch_size = 128, 32
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])       # 同样是错一位，这次每个位置都算损失
    return x, y

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT(len(chars), block_size, d=256, n_layer=6, n_heads=8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(5000):
    x, y = get_batch("train")
    _, loss = model(x.to(device), y.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(step, loss.item())

# 生成：从一个开头出发，按概率采样下一个字符
idx = torch.tensor([[stoi[c] for c in "从前"]], device=device)    # 开头必须是 input.txt 里出现过的字
for _ in range(200):
    logits = model(idx[:, -block_size:])[:, -1] / 0.8   # 0.8 是「温度」，越低越保守（下一课讲）
    nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
    idx = torch.cat([idx, nxt], dim=1)
print("".join(chars[i] for i in idx[0].tolist()))
```

> [!NOTE]
> 字符级模型最适合入门：不需要分词器，而且对中文特别直观（一个汉字就是一个 token）。中文常用字有几千个，词表比英文大，所以需要更多的数据和训练步数。有 GPU 的话用 Google Colab 等免费的云端 GPU 会快很多（第一部分第 2 课）。英文的经典练习数据是 Karpathy 整理的 [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)（约 1 MB 的莎士比亚剧本）。

## 6.6 从迷你 GPT 到真正的大模型

我们的模型和真正的大模型，**结构上没有本质区别**。差别在于规模和工程：

| | 本课的加法 GPT | GPT-2 small（2019） | GPT-3（2020） | 今天的前沿模型 |
|---|---|---|---|---|
| 参数 | 10 万 | 1.24 亿 | 1750 亿 | 数千亿到上万亿（常用 MoE） |
| 层数 × 维度 | 2 × 64 | 12 × 768 | 96 × 12288 | 几十到上百层 × 数千维 |
| 词表 | 12 个字符 | 5 万（BPE） | 5 万（BPE） | 10 万～25 万 |
| 上下文 | 9 | 1024 | 2048 | 12.8 万～数百万 |
| 训练数据 | 9000 道加法题 | 约 100 亿 token | 约 3000 亿 token | 十几万亿到几十万亿 token |
| 训练硬件 | 一台笔记本的 CPU，几十秒 | 几十块 GPU | 上千块 GPU | 上万到十万块 GPU，数月 |

要从这里走到那里，需要：**更好的数据**（大规模清洗、去重、配比）、**更多的算力和分布式训练**（把模型和数据切到成千上万块 GPU 上）、**缩放定律**（预先算出多大的模型配多少数据最划算）、以及上一部分讲过的各种训练技巧。这些是第四部分的内容。

而且，预训练出来的模型只会「续写」，还不会「听指令、好好回答问题」。把它变成一个有用、安全的助手，还需要**微调和对齐**（第五部分）。

## 6.7 小结

- 一个完整的 GPT = token 嵌入 + 位置嵌入 → 若干个 Block（注意力 + MLP + 残差 + 归一化）→ 归一化 → 输出层；约 100 行 PyTorch 代码。
- 训练数据：**目标 = 输入错一位**；因果掩码让一次前向传播同时训练序列中的每个位置；不想学的位置用 `ignore_index` 排除。
- 留出**测试集**才能判断模型是学会了规律还是背下了答案；把答案倒着写这样的「数据表示」技巧会大大影响学习难度。
- 同一份代码换成任何文本就能训练字符级语言模型；生成时按概率采样。
- 迷你 GPT 和真正的大模型结构相同，区别在于数据、规模、算力和工程，以及之后的微调与对齐。

## 6.8 练习

<!-- exercise: shift-pairs -->
### 练习 1：错一位

输入一个字符串，输出语言模型用它做训练时的所有「输入 → 目标」对：对第 i 个字符（i 从 0 开始，到倒数第二个字符为止），输出一行，包含这个字符和它后面的那个字符，中间用一个空格隔开。

```text
样例输入：hello
样例输出：
h e
e l
l l
l o
```

<details>
<summary>提示</summary>

`for a, b in zip(s, s[1:]): print(a, b)`
</details>

<!-- exercise: format-problem -->
### 练习 2：把题目写成训练文本

输入两个 0～99 之间的整数 `a b`，按本课的格式输出训练文本：两个数都补成 2 位，答案补成 3 位后倒过来写。

```text
样例输入：37 48
样例输出：37+48=580
```

<details>
<summary>提示</summary>

`f"{a:02d}+{b:02d}={str(a + b).zfill(3)[::-1]}"`
</details>

<!-- exercise: read-answer -->
### 练习 3：读出模型的回答

模型生成的是本课格式的完整文本，例如 `37+48=580`。输入这样一行，输出正常写法的算式：答案正过来，所有数都去掉前面多余的 0（例如 `04+21=520` 应输出 `4+21=25`）。

```text
样例输入：37+48=580
样例输出：37+48=85
```

<details>
<summary>提示</summary>

用 `split("=")` 分开题目和答案；倒序答案再倒回来：`int(ans[::-1])`。题目部分里的数可以用 `int()` 去掉前面的 0。
</details>

<!-- exercise: how-many-signals -->
### 练习 4：一次能学多少

一个 batch 里有 32 个序列，每个序列长 128 个 token（每个位置都计算损失）。一次前向传播一共提供了多少个「预测下一个 token」的训练信号？

- 32
- 128
- 4096
- 160

<details>
<summary>答案与解析</summary>

每个序列的每个位置都在预测下一个 token：32 × 128 = 4096。这就是 Transformer 训练高效的原因之一。
</details>

<!-- exercise: why-holdout -->
### 练习 5：为什么要留出 1000 道题？

- 为了让训练更快
- 为了检验模型学到的是加法规则，而不只是背下了见过的题
- 因为模型容量不够，装不下 10000 道题
- 为了让损失降得更低

<details>
<summary>答案与解析</summary>

只在训练过的题上测试，无法区分「学会了」和「背下来了」。在从没见过的题上全对，才说明模型学到了可以推广的规律——这就是**泛化**。
</details>

## 6.9 延伸阅读

- Andrej Karpathy：[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)（视频，用 tinyshakespeare 训练一个字符级 GPT）
- [nanoGPT](https://github.com/karpathy/nanoGPT)、[nanochat](https://github.com/karpathy/nanochat)（从分词器到对话界面，约 100 美元训练一个小型 ChatGPT）
- Sebastian Raschka：[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（第 4、5 章：实现并预训练 GPT）
- 下一课：[解码：模型怎样选出下一个词](../07-decoding/zh-CN.md)
