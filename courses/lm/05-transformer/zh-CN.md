# 5. Transformer：把积木搭起来

> 🎯 **本课目标**：看懂一个 GPT 式 Transformer 的完整结构图；理解一个 **Block** 里的两件事——注意力（token 之间交流）和 MLP（每个 token 各自计算），以及**残差流**和 **Pre-Norm**；用 PyTorch 写出 Block；亲手算出 GPT-2 的 1.24 亿个参数从哪里来；分清编码器（BERT）、解码器（GPT）和编码器-解码器（T5）；认识 Llama 式的现代改进（RMSNorm、SwiGLU、RoPE、GQA）和**混合专家**（MoE）。
>
> ⏱ **预计用时**：90 分钟　｜　📋 **前置知识**：本部分第 3、4 课；第二部分第 3、6 课

我们已经有了所有零件：嵌入（第 3 课）、注意力（第 4 课）、MLP（第二部分第 3 课）、归一化和残差连接（第二部分第 6 课）。这一课把它们组装成 **Transformer**——更准确地说，是 GPT 系列使用的**仅解码器**（decoder-only）Transformer。下一课我们就用它训练一个真正的小 GPT。

## 5.1 全景图

```text
输入文本  "今天天气"
   │  分词器
   ▼
token 编号  [1123, 452, 452, 9981]                    （T 个 token）
   │  token 嵌入 + 位置信息
   ▼
向量  (T × d)  ─────────────────────────┐
   │                                     │  残差流（residual stream）
   │   ┌──────────── Block 1 ──────────┐ │  每个 Block 读取它、
   │   │ x = x + Attention(Norm(x))    │ │  再把自己的结果「加」回去
   │   │ x = x + MLP(Norm(x))          │ │
   │   └───────────────────────────────┘ │
   │            …… 重复 N 次 ……           │
   │   ┌──────────── Block N ──────────┐ │
   │   └───────────────────────────────┘ │
   ▼                                     │
最后的 Norm  ◄───────────────────────────┘
   │  输出层（d → 词表大小）
   ▼
logits  (T × V)  →  softmax  →  每个位置上「下一个 token」的概率
```

几个关键点：

- 整个过程中，每个 token 始终对应一个 d 维向量，形状一直是 `T × d`，直到最后才变成 `T × V` 的 logits。
- 第 t 个位置的输出预测的是**第 t+1 个** token。生成时只用最后一个位置的预测；**训练时每个位置都在预测**，一次前向传播就得到 T 个训练信号（下一课细讲）。
- 模型的「深度」就是 Block 的层数 N：GPT-2 small 有 12 层，Llama 3 70B 有 80 层。

## 5.2 Block：先交流，再思考

每个 Block 由两个子层组成，每个子层都是「**归一化 → 计算 → 加回残差**」：

```formula
x = x + Attention(Norm(x))     ← token 之间交流：从上下文收集信息
x = x + MLP(Norm(x))           ← 每个 token 各自思考：处理收集到的信息
```

1. **注意力**是唯一让不同位置的 token **互相交换信息**的地方（上一课）。
2. **MLP**（也叫 FFN，前馈网络）对**每个 token 独立地**做同样的计算：`d → 4d → d`，中间用 GELU 等激活函数。研究认为大量「知识」（比如「巴黎是法国的首都」）储存在 MLP 的权重里。
3. **残差连接**：每个子层只计算一个「修改量」加到 x 上，而不是替换 x。可以把 x 想象成一条贯穿全部层的**残差流**，每一层都从中读取、再往里写入。这让梯度能畅通无阻地传到最前面（第二部分第 6 课），上百层的网络也能训练。
4. **Pre-Norm**：归一化放在每个子层的**输入**处（而不是原始 Transformer 那样放在相加之后），训练更稳定，是 GPT-2 以来的标准做法。

## 5.3 用 PyTorch 写一个 Block

```python requires=torch
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):           # 上一课的多头因果自注意力（用 PyTorch 自带的高效实现）
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

class MLP(nn.Module):                           # d → 4d → d
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 4 * d)
        self.proj = nn.Linear(4 * d, d)
    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # ① token 之间交流信息
        x = x + self.mlp(self.ln2(x))           # ② 每个 token 各自「思考」
        return x

torch.manual_seed(0)
block = Block(d=64, n_heads=4)
x = torch.randn(2, 10, 64)                      # 2 个序列，各 10 个 token，每个 64 维
print("输入", tuple(x.shape), "→ 输出", tuple(block(x).shape))   # 形状不变，可以一直叠下去
print("一个 Block 的参数：", sum(p.numel() for p in block.parameters()), "= 12d² + 13d =", 12 * 64**2 + 13 * 64)
```

输入和输出的形状完全一样，所以可以把任意多个 Block 叠在一起。

## 5.4 GPT-2 的 1.24 亿个参数从哪里来？

一个 Block 的参数（d 是模型维度）：

| 部分 | 参数 |
|------|------|
| 注意力：QKV 投影 `d × 3d` + 偏置 `3d` | 3d² + 3d |
| 注意力：输出投影 `d × d` + 偏置 `d` | d² + d |
| MLP：`d × 4d` + `4d`，`4d × d` + `d` | 8d² + 5d |
| 两个 LayerNorm（各有 γ、β） | 4d |
| **合计** | **12d² + 13d** |

再加上 token 嵌入（`V × d`）、位置嵌入（`上下文长度 × d`）和最后一个 LayerNorm（`2d`）；输出层与 token 嵌入**共用**同一个矩阵（权重绑定），不额外算参数：

```python
def gpt2_params(n_layer, d, vocab=50257, ctx=1024):
    emb = vocab * d + ctx * d                       # token 嵌入 + 位置嵌入
    attn = (d * 3 * d + 3 * d) + (d * d + d)        # QKV 投影 + 输出投影（含偏置）
    mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)     # d → 4d → d（含偏置）
    norms = 2 * (2 * d)                             # 两个 LayerNorm，各有 γ 和 β
    per_layer = attn + mlp + norms
    final = 2 * d                                   # 最后一个 LayerNorm；输出层与 token 嵌入共用
    return per_layer, emb + n_layer * per_layer + final

for name, n_layer, d in [("GPT-2 small", 12, 768), ("GPT-2 medium", 24, 1024),
                         ("GPT-2 large", 36, 1280), ("GPT-2 XL", 48, 1600)]:
    per_layer, total = gpt2_params(n_layer, d)
    print(f"{name:13s} {n_layer:2d} 层  d={d:4d}  每层 {per_layer / 1e6:5.1f}M  总计 {total / 1e6:7.1f}M")
```

算出来正好是 OpenAI 公布的 124M、355M、774M、1.5B。可以看到：**参数主要来自 `12d²`**——维度翻倍，每层参数约变成 4 倍。

## 5.5 三种 Transformer

原始的 Transformer（2017）是为机器翻译设计的，有**编码器**和**解码器**两部分。后来它分化成三个家族：

| 家族 | 代表 | 注意力 | 训练目标 | 擅长 |
|------|------|--------|----------|------|
| **仅编码器** | BERT（2018） | **双向**：每个 token 能看到整句话 | 完形填空：遮住一些词让模型猜 | 理解：分类、检索、句向量 |
| **编码器-解码器** | 原始 Transformer、T5 | 编码器双向，解码器因果，再通过「交叉注意力」读取编码器 | 把输入序列变成输出序列 | 翻译、摘要 |
| **仅解码器** | GPT 系列、Llama、Qwen、DeepSeek、Claude…… | **因果**：只能看前文 | 预测下一个 token | 生成；规模足够大后几乎什么都能做 |

今天的大语言模型几乎全部是**仅解码器**结构：它最简单、最容易扩展，而且「预测下一个 token」这个目标可以用互联网上**任何文本**来训练，不需要人工标注。

## 5.6 现代的改进：Llama 式 Transformer

GPT-2 之后，Transformer 的整体结构没有大变，但很多细节都改进了。Llama（Meta，2023）的组合成为了事实标准，Qwen、Mistral、DeepSeek 等都大量采用：

| 部件 | GPT-2（2019） | Llama 式（2023 至今） | 为什么 |
|------|---------------|------------------------|--------|
| 归一化 | LayerNorm | **RMSNorm** | 更省计算，效果相当 |
| 位置信息 | 可学习的位置嵌入（加在输入上） | **RoPE**（在注意力里旋转 q、k） | 表示相对位置，更容易扩展上下文 |
| MLP | GELU，`d → 4d → d` | **SwiGLU** 门控 MLP | 同样参数量下效果更好 |
| K、V 头 | 和 query 头一样多 | **GQA**：几个 query 头共用一组 K、V | KV 缓存小好几倍，推理更快 |
| 偏置 | 线性层都有偏置 | 通常**不要偏置** | 几乎没影响，更简单 |
| 输出层 | 与嵌入共用 | 大模型通常**不共用** | 参数充足时效果略好 |

**SwiGLU** 是一种「门控」MLP：用一条支路算出「门」（经过 SiLU 激活），去乘另一条支路的结果，再投影回 d 维。它有三个矩阵，所以中间维度通常取 `8d/3` 左右，让参数量和普通 MLP 差不多：

```python requires=torch
import torch
from torch import nn
import torch.nn.functional as F

class SwiGLU(nn.Module):                        # Llama、Qwen、DeepSeek 等使用的 MLP
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))   # 「门」逐元素乘以「内容」

mlp = SwiGLU(d=64, hidden=172)                  # 172 ≈ 8 × 64 / 3
print(tuple(mlp(torch.randn(2, 10, 64)).shape), "参数：", sum(p.numel() for p in mlp.parameters()))
```

用同样的方法可以算出 Llama 3 的参数量。Llama 3 8B：32 层，d = 4096，32 个 query 头、8 个 KV 头，SwiGLU 中间维度 14336，词表 128256：

```python
def llama_params(n_layer, d, n_heads, n_kv_heads, ffn_hidden, vocab):
    head_dim = d // n_heads
    attn = d * d + 2 * d * (n_kv_heads * head_dim) + d * d      # W_Q、W_K、W_V（GQA 让 K、V 变小）、W_O
    mlp = 3 * d * ffn_hidden                                    # SwiGLU 的三个矩阵
    norms = 2 * d                                               # 两个 RMSNorm（只有 γ）
    per_layer = attn + mlp + norms
    return n_layer * per_layer + 2 * vocab * d + d              # 输入嵌入、输出层不共用；最后一个 RMSNorm

print(f"Llama 3 8B ： {llama_params(32, 4096, 32, 8, 14336, 128256) / 1e9:.2f}B")
print(f"Llama 3 70B： {llama_params(80, 8192, 64, 8, 28672, 128256) / 1e9:.2f}B")
```

结果与 Meta 公布的 8.03B 和 70.6B 一致。可以看到，在 Llama 式模型里，**MLP 占了参数的大头**（8B 模型每层约 1.76 亿个 MLP 参数，注意力只有约 4200 万）。

## 5.7 混合专家（MoE）：参数很多，每次只用一部分

模型越大越聪明，但每个 token 的计算量也越大。**混合专家**（Mixture of Experts，MoE）的思路是：把一个大 MLP 换成**很多个小 MLP（专家）**，再加一个**路由器**（router）。每个 token 只被送到得分最高的少数几个专家那里：

```python
import math

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

# 4 个「专家」（真实模型里每个都是一个 MLP；这里用简单的函数代替）
experts = [lambda x: [2 * v for v in x],         # 专家 0
           lambda x: [v + 1 for v in x],         # 专家 1
           lambda x: [-v for v in x],            # 专家 2
           lambda x: [v * v for v in x]]         # 专家 3
router = [[0.5, -1.0], [1.0, 0.2], [-0.3, 0.8], [0.1, 0.1]]     # 路由器：每个专家一行权重

def moe(x, top_k=2):
    scores = [sum(w * v for w, v in zip(row, x)) for row in router]
    chosen = sorted(range(len(experts)), key=lambda i: -scores[i])[:top_k]   # 只选得分最高的 k 个
    gates = softmax([scores[i] for i in chosen])                            # 选中专家的权重
    outs = [experts[i](x) for i in chosen]                                  # 只有被选中的专家要计算
    y = [sum(g * o[j] for g, o in zip(gates, outs)) for j in range(len(x))]
    return chosen, [round(g, 3) for g in gates], [round(v, 3) for v in y]

for x in ([1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]):
    print("输入", x, "→ 选中的专家、权重、输出：", moe(x))
```

不同的输入被路由到不同的专家。这样，模型的**总参数**可以非常大（储存更多知识），而每个 token **实际参与计算的参数**（激活参数）只是一小部分：

| 模型 | 总参数 | 每个 token 激活的参数 |
|------|--------|------------------------|
| Mixtral 8x7B | 约 470 亿 | 约 130 亿（8 个专家选 2 个） |
| DeepSeek-V3 / R1 | 6710 亿 | 约 370 亿 |
| Qwen3-235B-A22B | 2350 亿 | 约 220 亿 |

MoE 已经是最前沿的大模型的主流选择。它的难点在于：要让各个专家「分工均衡」（不能所有 token 都挤到同一个专家），而且全部专家的参数都要装进显存。

## 5.8 小结

- GPT 式 Transformer：token 嵌入（+ 位置）→ N 个 Block → 最后的归一化 → 输出层 → 每个位置下一个 token 的 logits。
- Block = `x + Attention(Norm(x))`，再 `x + MLP(Norm(x))`：注意力负责 token 之间**交流**，MLP 负责每个 token 各自**计算**；残差流贯穿全部层，Pre-Norm 让训练稳定。
- 每层参数约 **12d²**；GPT-2 small = 12 层 × d=768 ≈ 1.24 亿参数。
- 三个家族：仅编码器（BERT，双向，理解）、编码器-解码器（T5，翻译）、**仅解码器**（GPT、Llama、Qwen……，今天的 LLM）。
- 现代改进：RMSNorm、RoPE、SwiGLU、GQA、去掉偏置；**MoE** 用路由器让每个 token 只激活少数专家，总参数大而计算量小。

## 5.9 练习

<!-- exercise: gpt2-params -->
### 练习 1：数一数 GPT-2 式模型的参数

输入一行四个整数 `n_layer d vocab ctx`（层数、维度、词表大小、上下文长度），按 5.4 节的 GPT-2 结构（每层 `12d² + 13d`，token 嵌入 + 位置嵌入，最后一个 LayerNorm `2d`，输出层与嵌入共用）输出总参数量。

```text
样例输入：12 768 50257 1024
样例输出：124439808
```

<details>
<summary>提示</summary>

`vocab * d + ctx * d + n_layer * (12 * d * d + 13 * d) + 2 * d`
</details>

<!-- exercise: llama-layer -->
### 练习 2：Llama 式的一层

输入一行四个整数 `d n_heads n_kv_heads ffn_hidden`，按 5.6 节的 Llama 结构（无偏置；`W_Q` 和 `W_O` 各 `d × d`；`W_K` 和 `W_V` 各 `d × (n_kv_heads × d / n_heads)`；SwiGLU 三个矩阵 `3 × d × ffn_hidden`；两个 RMSNorm 共 `2d`）输出**一层**的参数量。

```text
样例输入：4096 32 8 14336
样例输出：218112000
```

<details>
<summary>提示</summary>

先算 `head_dim = d // n_heads`，K、V 每个是 `d * n_kv_heads * head_dim`。
</details>

<!-- exercise: residual-role -->
### 练习 3：残差连接的作用

在 Transformer 里，每个子层写成 `x = x + f(x)` 而不是 `x = f(x)`，主要好处是？

- 减少参数量
- 梯度可以沿着「加法」这条路直接传到前面的层，很深的网络也能训练；每层只需学习一个「修改量」
- 让注意力看不到未来
- 让输出变成概率

<details>
<summary>答案与解析</summary>

`x + f(x)` 的导数是 `1 + f'(x)`，那个 1 让梯度至少能原样穿过每一层；所有层共同读写一条「残差流」。
</details>

<!-- exercise: which-family -->
### 练习 4：哪种结构？

GPT、Llama、Qwen、DeepSeek 这些对话大模型用的是哪种 Transformer？

- 仅编码器（像 BERT）
- 编码器-解码器（像 T5）
- 仅解码器（因果注意力，预测下一个 token）
- 不是 Transformer

<details>
<summary>答案与解析</summary>

它们都是仅解码器结构：因果注意力 + 预测下一个 token。这种结构最简单、最容易扩大规模，也能用任何文本训练。
</details>

<!-- exercise: moe-why -->
### 练习 5：为什么用 MoE？

DeepSeek-V3 总共有 6710 亿个参数，但每个 token 只激活约 370 亿个。MoE 的主要好处是？

- 模型文件更小
- 总参数（知识容量）很大，但每个 token 的计算量只相当于一个小得多的模型
- 不需要训练路由器
- 不需要 GPU

<details>
<summary>答案与解析</summary>

每个 token 只经过少数几个专家，计算量由激活参数决定，而知识可以储存在全部专家里。代价是所有专家的参数仍然要放进内存。
</details>

## 5.10 延伸阅读

- Jay Alammar：[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- Anthropic：[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)（「残差流」视角的来源）
- Sebastian Raschka：《从零构建大模型》（*Build a Large Language Model (From Scratch)*，有中文版）及其开源代码 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- Meta：《The Llama 3 Herd of Models》（2024）；DeepSeek-AI：《DeepSeek-V3 Technical Report》（2024）
- 下一课：[亲手训练一个 GPT](../06-gpt-from-scratch/zh-CN.md)
