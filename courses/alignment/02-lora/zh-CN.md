# 2. LoRA 与 QLoRA：只训练一小部分参数

> 🎯 **本课目标**：明白全参数微调为什么昂贵；理解**参数高效微调**（PEFT）的思路；掌握 **LoRA** 的原理——冻结原来的权重，只训练两个小矩阵 B、A，权重的变化是 `(α/r)·B·A`；理解为什么低秩就够用，为什么 B 要初始化为 0，训练完怎样**合并**权重；亲手用 PyTorch 实现一个 LoRA 层；理解 **QLoRA** 怎样把基础模型压到 4 位，让一张显卡就能微调大模型；会用 PEFT 库，知道 LoRA 的常见设置和局限。
>
> ⏱ **预计用时**：70 分钟　｜　📋 **前置知识**：本部分第 1 课；第二部分第 5 课；第四部分第 2、3 课

上一课的 SFT 默认要更新模型的**全部**参数。第四部分第 2 课算过：用 Adam 训练，每个参数要 16 字节显存——70 亿参数的模型要 112 GB，远超一张消费级显卡。而且，如果你要为 10 个不同的任务各微调一个版本，就要保存 10 份完整的模型。

## 2.1 参数高效微调

**参数高效微调**（Parameter-Efficient Fine-Tuning，PEFT）的思路是：**冻结**原来的模型，只训练少量新增的参数。这样：

- 梯度和优化器状态只需要为少量参数保存，显存大大减少；
- 每个任务只需保存那一小部分参数（几 MB 到几百 MB），而不是整个模型；
- 原来的模型完好无损，可以在不同任务之间随时切换。

各种 PEFT 方法中，最流行的是微软 2021 年提出的 **LoRA**（Low-Rank Adaptation，低秩适配）。

## 2.2 LoRA 的原理

微调一个线性层，本质上是把它的权重 W 改成 W + ΔW。LoRA 的关键假设是：**ΔW 不需要是任意的矩阵，一个「低秩」矩阵就够了**。于是把 ΔW 写成两个瘦长矩阵的乘积：

```formula
W' = W + (α / r) · B · A　　　（W：d_out × d_in，冻结；B：d_out × r；A：r × d_in；r 远小于 d）
```

- **r** 叫**秩**（rank），通常取 8～64；
- **α** 是一个缩放系数（常取 r 的 1～2 倍），让调整 r 时不必重新调学习率；
- 前向传播时，`W'x = Wx + (α/r)·B(Ax)`：先把 x 压到 r 维，再放大回去，计算量很小。

可训练的参数从 d_out × d_in 个，变成了 r × (d_in + d_out) 个：

```python
def lora_params(d_in, d_out, r):
    return r * (d_in + d_out)

for d_in, d_out, r in [(4096, 4096, 8), (4096, 4096, 16), (4096, 14336, 16), (8192, 8192, 64)]:
    full = d_in * d_out
    lora = lora_params(d_in, d_out, r)
    print(f"{d_out:>5}×{d_in:<5} 的矩阵，r={r:>2}：完整 {full:>11,} 个参数，LoRA {lora:>9,} 个（{lora / full:.2%}）")
```

用纯 Python 看看前向传播：

```python
def matvec(M, x):
    return [sum(m * v for m, v in zip(row, x)) for row in M]

W = [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]      # 冻结的原权重，2×3
A = [[0.5, -0.5, 1.0]]                       # r = 1，A 是 1×3
B = [[2.0], [-1.0]]                          # B 是 2×1
alpha, r = 2, 1
x = [1.0, 2.0, 3.0]

base = matvec(W, x)
delta = [(alpha / r) * v for v in matvec(B, matvec(A, x))]
print("原模型输出：", base)
print("LoRA 的修正：", delta)
print("微调后输出：", [b + d for b, d in zip(base, delta)])

merged = [[W[i][j] + (alpha / r) * B[i][0] * A[0][j] for j in range(3)] for i in range(2)]
print("合并权重后直接算：", matvec(merged, x))
```

两点要注意：

- **B 初始化为 0**（A 用小的随机数）。这样训练刚开始时 B·A = 0，模型和原来**完全一样**，微调从原模型平稳地出发。
- **训练完可以合并**：把 `(α/r)·B·A` 直接加进 W，得到一个普通的模型，推理时没有任何额外开销。也可以不合并，在运行时根据请求切换不同的 LoRA。

## 2.3 为什么低秩就够了？

一个 4096 × 4096 的矩阵有 1600 多万个数，r = 8 的 LoRA 只有 6.5 万个可训练参数，怎么够用？研究发现，微调时权重的变化 ΔW 往往集中在**少数几个方向**上（它的「内在维度」很低）——毕竟微调只是在已有能力的基础上做一些调整。

用奇异值分解（SVD）可以看出一个矩阵「有几个主要方向」。下面构造一个「真正的变化只有 4 个方向、再加一点噪声」的矩阵：

```python requires=numpy
import numpy as np

rng = np.random.default_rng(0)
d = 256
delta = rng.normal(size=(d, 4)) @ rng.normal(size=(4, d)) + 0.05 * rng.normal(size=(d, d))

U, S, Vt = np.linalg.svd(delta)
print("最大的 8 个奇异值：", np.round(S[:8], 1))
for r in (1, 2, 4, 8, 32):
    approx = (U[:, :r] * S[:r]) @ Vt[:r]                  # 只保留前 r 个方向
    err = np.linalg.norm(delta - approx) / np.linalg.norm(delta)
    print(f"r = {r:>2}：相对误差 {err:.3f}")
```

前 4 个奇异值很大，之后骤然变小；r = 4 时误差就只剩噪声的水平，再加大 r 几乎没有帮助。LoRA 赌的就是：真实的微调也是这样。

## 2.4 亲手实现 LoRA

用 PyTorch 写一个 LoRA 层：包住一个冻结的 `nn.Linear`，只训练 A 和 B。我们故意让目标权重和原权重之间只差一个**秩为 1** 的变化，看看 r = 4 的 LoRA 能不能学会：

```python requires=torch timeout=120
import torch
from torch import nn

torch.manual_seed(0)
d = 64

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                                   # 冻结原权重
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))      # B 从 0 开始
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

base = nn.Linear(d, d, bias=False)
u, v = torch.randn(d, 1), torch.randn(1, d)
target_W = base.weight.detach() + 0.5 * (u @ v) / d ** 0.5           # 目标 = 原权重 + 一个秩 1 的变化
x = torch.randn(2048, d)
y = x @ target_W.T

layer = LoRALinear(base, r=4, alpha=8)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f"可训练参数 {trainable} / 总参数 {total}")

opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
for step in range(301):
    loss = nn.functional.mse_loss(layer(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"第 {step:3d} 步：损失 {loss.item():.6f}")

with torch.no_grad():
    merged = layer.base.weight + layer.scale * layer.B @ layer.A      # 合并成一个普通的权重
    print(f"合并后的权重与目标的最大误差：{(merged - target_W).abs().max().item():.5f}")
```

只训练了 512 个参数（原矩阵有 4096 个），损失就降到了几乎为 0。

## 2.5 用 PEFT 库给大模型加 LoRA

实际中用 Hugging Face 的 **PEFT** 库，几行代码就能给模型的指定层加上 LoRA（需要下载模型）：

```python norun
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
config = LoraConfig(
    r=16,                                    # 秩
    lora_alpha=32,                           # α
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # 注意力和 MLP 的线性层
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()           # 通常只有百分之一左右可训练
```

常见的经验设置：

- **加在哪些层**：最初的论文只加在注意力的 q、v 上；后来发现加在**所有线性层**（包括 MLP）上效果更好。
- **r 和 α**：r = 8～64，α = r 或 2r。任务越难、和原模型差别越大，r 越大。
- **学习率**：比全参数微调大一个数量级，常用 1e-4～2e-4。
- **得到的文件很小**：一个 LoRA 适配器通常只有几十 MB，可以分享、组合，推理服务（如 vLLM）还能同时加载很多个 LoRA，按请求切换。

## 2.6 QLoRA：一张显卡微调大模型

LoRA 减少了梯度和优化器状态，但冻结的基础模型本身仍要放进显存：BF16 下每个参数 2 字节，700 亿参数就是 140 GB。**QLoRA**（Dettmers 等，2023）进一步把冻结的基础模型**量化成 4 位**：

- **NF4**（4 位 NormalFloat）：一种专门为「近似正态分布的权重」设计的 4 位格式，比普通的 4 位整数更精确；
- **双重量化**：连量化用的缩放系数也再量化一次，进一步省显存；
- **分页优化器**：显存不够时把优化器状态暂时挪到内存里，防止显存峰值导致崩溃；
- LoRA 的 A、B 仍用 BF16 训练，计算时把 4 位权重临时还原成 BF16。

```python
def finetune_memory_gb(N, method, lora_fraction=0.01):
    if method == "全参数":
        return 16 * N                         # BF16 参数 + 梯度 + FP32 主参数和 Adam（第四部分第 2 课）
    base_bytes = {"LoRA": 2, "QLoRA": 0.5}[method]   # 冻结的基础模型：BF16 或 4 位
    return base_bytes * N + 16 * N * lora_fraction   # 再加上 LoRA 参数自己的训练状态

for N in (7, 70):                              # 单位：十亿参数，结果单位：GB（不含激活值）
    row = [f"{m} {finetune_memory_gb(N, m):6.1f} GB" for m in ("全参数", "LoRA", "QLoRA")]
    print(f"{N:>2}B 模型：" + "，".join(row))
```

QLoRA 的论文在**一张 48 GB 的 GPU** 上微调了 650 亿参数的模型，效果和 16 位的全参数微调相当。这让个人和小团队也能定制大模型。代价是速度：每次计算都要把 4 位权重还原，训练比 LoRA 慢一些。

## 2.7 LoRA 的局限和变体

- **学得少，也忘得少**：Biderman 等（2024）发现，在需要学习大量新知识的任务上（比如在大量代码或数学数据上继续训练），LoRA 明显不如全参数微调；但 LoRA 也更少「忘记」原来的能力。所以 LoRA 很适合调整格式、风格和特定任务，不太适合大规模地注入新知识。
- **变体**：DoRA（2024）把权重分解成「大小」和「方向」分别调整；还有各种改进初始化和学习率的方法。原理都和 LoRA 一样：冻结大的、训练小的。

## 2.8 小结

- 全参数微调显存大、每个任务都要存一份完整模型；PEFT 冻结原模型，只训练少量新参数。
- LoRA：W' = W + (α/r)·B·A，可训练参数从 d_out·d_in 降到 r·(d_in + d_out)；B 初始化为 0，训练开始时模型不变。
- 微调的权重变化通常是低秩的，所以小的 r 就够用；训练完可以把 BA 合并进 W，推理没有额外开销。
- 常见设置：加在所有线性层，r = 8～64，α = r～2r，学习率 1e-4 左右；适配器文件很小，可以随时切换。
- QLoRA：基础模型量化为 4 位 NF4，加上双重量化和分页优化器，一张显卡就能微调几百亿参数的模型。
- LoRA 适合调整格式、风格和任务，大量注入新知识时不如全参数微调。

## 2.9 练习

<!-- exercise: lora-params -->
### 练习 1：LoRA 有多少参数

输入一个线性层的 `d_in`、`d_out` 和 LoRA 的秩 `r`。输出完整矩阵的参数量、LoRA 的可训练参数量，以及后者占前者的百分比（保留 2 位小数），用空格隔开。

```text
样例输入：4096 4096 8
样例输出：16777216 65536 0.39%
```

<details>
<summary>提示</summary>

完整：`d_in * d_out`；LoRA：`r * (d_in + d_out)`；百分比：`f"{lora / full:.2%}"`。
</details>

<!-- exercise: lora-forward -->
### 练习 2：LoRA 的前向传播

第一行是 `d_out d_in r`；接下来 `d_out` 行是冻结的权重 W；再接下来 `r` 行是 A（每行 `d_in` 个数）；再接下来 `d_out` 行是 B（每行 `r` 个数）；然后一行是 α；最后一行是输入向量 x。输出 `W x + (α/r)·B(A x)`，每个数保留 3 位小数，用空格隔开。

```text
样例输入：
2 3 1
1 0 2
0 1 -1
0.5 -0.5 1
2
-1
2
1 2 3
样例输出：17.000 -6.000
```

<details>
<summary>提示</summary>

写一个矩阵乘向量的函数 `matvec(M, x)`。先算 `Ax`（r 个数），再算 `B(Ax)`（d_out 个数），乘以 `alpha / r` 后加到 `Wx` 上。
</details>

<!-- exercise: model-lora -->
### 练习 3：整个模型的 LoRA 参数

一个模型有 `L` 层，模型维度为 `d`。假设每层有 `m` 个需要加 LoRA 的 `d × d` 线性层，秩为 `r`。输出 LoRA 的总参数量，以百万为单位，保留 2 位小数，后面加 `M`。

```text
样例输入：32 4096 4 16
样例输出：16.78M
```

<details>
<summary>提示</summary>

每个 d × d 的层有 `r * (d + d)` 个 LoRA 参数，一共有 `L * m` 个这样的层。
</details>

<!-- exercise: finetune-memory -->
### 练习 4：三种微调要多少显存

输入模型的参数量 N（十亿）。按 2.6 节的估算（全参数 16 字节 / 参数；LoRA 的基础模型 2 字节 / 参数、QLoRA 的基础模型 0.5 字节 / 参数，两者都再加上 1% 参数 × 16 字节的 LoRA 训练状态），输出三种方法需要的显存（GB），各保留 1 位小数，用空格隔开。

```text
样例输入：7
样例输出：112.0 15.1 4.6
```

<details>
<summary>提示</summary>

LoRA：`2 * N + 16 * N * 0.01`；QLoRA：`0.5 * N + 16 * N * 0.01`。N 以十亿为单位时结果就是 GB。
</details>

<!-- exercise: b-zero -->
### 练习 5：B 为什么从 0 开始？

LoRA 中矩阵 B 初始化为 0、A 初始化为小的随机数。这样做的主要目的是什么？

- 让训练速度加倍
- 让训练开始时 B·A = 0，模型的输出和原模型完全一样，从原模型平稳地出发
- 让 LoRA 的参数量减半
- 因为 B 不需要训练

<details>
<summary>答案与解析</summary>

B·A = 0 意味着一开始的权重就是 W 本身，微调不会一上来就破坏原模型。A 不能也是 0：那样 B 和 A 的梯度都是 0，训练无法开始。
</details>

<!-- exercise: qlora-what -->
### 练习 6：QLoRA 量化了什么？

QLoRA 把什么量化成了 4 位？

- LoRA 的 A、B 矩阵
- 冻结的基础模型的权重；LoRA 的 A、B 仍用 16 位训练
- 训练数据
- 梯度

<details>
<summary>答案与解析</summary>

QLoRA 把占显存最多、又不需要更新的基础模型压成 4 位（NF4），可训练的 LoRA 参数仍是 BF16，计算时临时把 4 位权重还原。
</details>

## 2.10 延伸阅读

- Hu 等：《LoRA: Low-Rank Adaptation of Large Language Models》（2021）
- Dettmers 等：《QLoRA: Efficient Finetuning of Quantized LLMs》（2023）
- Biderman 等：《LoRA Learns Less and Forgets Less》（2024）
- Liu 等：《DoRA: Weight-Decomposed Low-Rank Adaptation》（2024）
- Hugging Face [PEFT 文档](https://huggingface.co/docs/peft)
- 下一课：[奖励模型与 RLHF](../03-rlhf/zh-CN.md)
