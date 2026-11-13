# 3. 分布式训练与混合精度

> 🎯 **本课目标**：认识 FP32、FP16、BF16、FP8 等数字格式，理解**混合精度训练**为什么要保留 FP32 的主参数，FP16 为什么需要损失缩放；理解**梯度累积**；理解**数据并行**和 **all-reduce**（亲手模拟环形 all-reduce）；理解 **ZeRO / FSDP** 怎样把优化器状态、梯度和参数切分到多张 GPU 上；理解**张量并行**和**流水线并行**（以及流水线的「气泡」）；了解激活重计算和真实的大规模训练是什么样子。
>
> ⏱ **预计用时**：90 分钟　｜　📋 **前置知识**：本部分第 2 课；第二部分第 4、5 课

上一课算过：训练一个 700 亿参数的模型，光是参数和优化器状态就要 1 TB 多的显存，计算量要几千张 GPU 算上一两个月。这一课讲两件事：**怎样让每张 GPU 算得更快、更省显存**（低精度），以及**怎样让成千上万张 GPU 一起训练一个模型**（并行）。

## 3.1 数字格式：位数越少，越快越省

计算机里的浮点数由三部分组成：符号、**指数**（决定能表示多大、多小的数，即**范围**）和**尾数**（决定有多少位有效数字，即**精度**）。

| 格式 | 总位数 | 指数位 | 尾数位 | 最大值 | 大约几位有效数字 |
|------|--------|--------|--------|--------|------------------|
| FP32 | 32 | 8 | 23 | 3.4 × 10³⁸ | 7 |
| FP16 | 16 | 5 | 10 | 65504 | 3 |
| BF16 | 16 | 8 | 7 | 3.4 × 10³⁸ | 2～3 |
| FP8（E4M3） | 8 | 4 | 3 | 448 | 1 |

位数减半，显存和数据传输减半，GPU 的矩阵运算速度通常翻倍（H100 的 BF16 约 989 TFLOPS，FP8 约 1979 TFLOPS）。代价是精度和范围。用 Python 的 `struct` 模块模拟一下：

```python
import struct

def to_fp32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

def to_fp16(x):
    try:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    except OverflowError:                  # 超出 FP16 的范围
        return float("inf")

def to_bf16(x):                            # BF16 就是 FP32 的高 16 位（四舍五入）
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]

print(f"{'x':>13} {'FP32':>14} {'FP16':>14} {'BF16':>14}")
for x in (1 / 3, 3.14159265, 1.001, 70000.0, 1e-8):
    print(f"{x:>13.6g} {to_fp32(x):>14.8g} {to_fp16(x):>14.8g} {to_bf16(x):>14.8g}")
```

- **FP16**：精度还行，但**范围很小**——超过 65504 就变成无穷大，小于大约 6 × 10⁻⁸ 就变成 0。训练时很小的梯度会直接「消失」。
- **BF16**（Google 为深度学习设计的格式）：和 FP32 一样有 8 位指数，**范围相同**，不会溢出；但只有 7 位尾数，精度很低——`1.001` 变成了 `1`。
- 现在训练大模型的标准选择是 **BF16**。

## 3.2 混合精度训练

BF16 精度这么低，为什么还能训练？关键是**混合精度**：

- **前向和反向传播**中的矩阵乘法用 BF16 计算（占绝大部分时间），速度快、省显存；
- **参数的更新**用 FP32 的「主参数」完成。

为什么更新一定要用 FP32？看一个例子：参数是 1.0，每一步加上一个很小的更新 0.0001：

```python
import struct

def to_bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]

w_bf16, w_fp32 = 1.0, 1.0
for step in range(1000):
    w_bf16 = to_bf16(w_bf16 + 0.0001)      # 每次加完都存回 BF16
    w_fp32 = w_fp32 + 0.0001
print("只用 BF16 存参数：", w_bf16)        # 一直是 1.0！
print("用 FP32 存主参数：", round(w_fp32, 4))
```

BF16 在 1.0 附近能表示的最小间隔是 2⁻⁷ ≈ 0.0078，`1.0 + 0.0001` 四舍五入后还是 1.0——更新**完全丢失了**。训练后期的更新往往就是这么小，所以参数必须用 FP32 保存和更新（这正是上一课「每个参数 16 字节」里那 4 字节主参数的来历）。

**FP16 的损失缩放**：如果用 FP16，还有一个问题——很小的梯度会下溢成 0。解决办法是先把损失乘以一个大数（比如 65536），梯度也跟着放大，算完再除回去：

```python
import struct

def to_fp16(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]

g = 1e-8                                   # 一个很小的梯度
print("直接存成 FP16：", to_fp16(g))
scale = 65536
print("先放大 65536 倍，存成 FP16，再缩小：", to_fp16(g * scale) / scale)
```

BF16 的范围和 FP32 一样，不需要损失缩放，这是它取代 FP16 的主要原因。

在 PyTorch 里，混合精度只需要一个 `torch.autocast`：

```python requires=torch timeout=120
import torch

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(512, 64, device=device)
y = 2 * x[:, :1] + 0.5

for step in range(201):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):   # 这里面的矩阵乘法用 BF16
        pred = model(x)
    loss = torch.nn.functional.mse_loss(pred.float(), y)
    opt.zero_grad()
    loss.backward()
    opt.step()                                                       # 参数和梯度仍是 FP32
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  输出 {pred.dtype}  "
              f"参数 {model[0].weight.dtype}  梯度 {model[0].weight.grad.dtype}")
```

**FP8 训练**：DeepSeek-V3（2024）把大部分矩阵乘法换成了 FP8，并用「细粒度缩放」（每一小块数据配一个缩放因子）控制误差，是第一批在超大规模上成功使用 FP8 训练的模型之一。

## 3.3 梯度累积

批大小（batch size）越大，梯度越稳定。大模型训练的全局批大小通常是**几百万个 token**，一张 GPU 一次根本放不下。办法之一是**梯度累积**：一次只算一小批（微批次），把梯度累加起来，攒够了再更新一次参数：

```python
data = [(1.0, 2.1), (2.0, 3.9), (3.0, 6.2), (4.0, 7.8), (5.0, 10.1), (6.0, 12.2), (7.0, 13.8), (8.0, 16.1)]
w = 0.5

def grad(batch, w):                         # 均方误差 mean((w·x - y)²) 对 w 的梯度
    return sum(2 * (w * x - y) * x for x, y in batch) / len(batch)

full = grad(data, w)                        # 一次算完整个批次

micro = [data[i:i + 2] for i in range(0, len(data), 2)]   # 4 个微批次，每个 2 条数据
acc = 0.0
for mb in micro:
    acc += grad(mb, w) / len(micro)         # 累加，并除以累积的步数
print(f"整批计算：{full:.4f}，梯度累积：{acc:.4f}")
```

两者完全相等。梯度累积用**时间**换**显存**：多算几次前向和反向，才更新一次参数。

## 3.4 数据并行和 all-reduce

**数据并行**（Data Parallelism, DP）是最常用的并行方式：每张 GPU 上都放一份**完整的模型**，各自处理**不同的数据**，算出各自的梯度；然后所有 GPU 把梯度**求平均**，每张 GPU 都用这个平均梯度更新参数，所以模型始终保持一致。这和梯度累积在数学上是一回事——只不过微批次分给了不同的 GPU 同时计算。

「每张 GPU 都得到所有梯度之和」这个操作叫 **all-reduce**。最朴素的做法是全都发给一张 GPU 加起来再发回去，但那张 GPU 会成为瓶颈。实际中常用**环形 all-reduce**：GPU 排成一个环，每张只和下一张通信。把数据切成 n 块：

1. **reduce-scatter**：转 n−1 圈，每一圈每张 GPU 把一块数据传给下一张，下一张把它加到自己的那块上。结束时，每张 GPU 各拥有**一块**完整的总和。
2. **all-gather**：再转 n−1 圈，把这些完整的块传遍整个环。

```python
def ring_allreduce(data):
    n = len(data)                              # GPU 数
    c = len(data[0]) // n                      # 每块的长度
    buf = [list(d) for d in data]
    sent = [0] * n                             # 每张 GPU 发送了多少个数
    for step in range(n - 1):                  # ① reduce-scatter
        msgs = []
        for i in range(n):
            k = (i - step) % n                 # 这一步 GPU i 发送第 k 块
            msgs.append(((i + 1) % n, k, buf[i][k * c:(k + 1) * c]))
            sent[i] += c
        for dst, k, chunk in msgs:
            for j in range(c):
                buf[dst][k * c + j] += chunk[j]
    for step in range(n - 1):                  # ② all-gather
        msgs = []
        for i in range(n):
            k = (i + 1 - step) % n
            msgs.append(((i + 1) % n, k, buf[i][k * c:(k + 1) * c]))
            sent[i] += c
        for dst, k, chunk in msgs:
            buf[dst][k * c:(k + 1) * c] = chunk
    return buf, sent

grads = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400], [1000, 2000, 3000, 4000]]
result, sent = ring_allreduce(grads)
for i, r in enumerate(result):
    print(f"GPU {i}：{r}")
print("每张 GPU 发送的数据量：", sent, "（梯度一共 4 个数）")
```

每张 GPU 只发送了 `2(n−1)/n` 倍的梯度量——即使有 1000 张 GPU，每张发送的数据也不到梯度总量的 2 倍，而且所有连接同时在工作。NVIDIA 的 **NCCL** 库实现了这些「集合通信」操作，PyTorch 的 `DistributedDataParallel`（DDP）就用它来同步梯度。

## 3.5 ZeRO 和 FSDP：把冗余切掉

数据并行有一个浪费：每张 GPU 都存着**完整的**参数、梯度和优化器状态（每参数 16 字节）。微软的 **ZeRO**（2019）想到：既然大家存的一样，不如**切开分着存**，需要时再临时交换：

| 阶段 | 切分什么 | 每张 GPU 的显存（N 个参数、n 张 GPU） |
|------|----------|----------------------------------------|
| 普通 DP | 什么都不切 | 16N |
| ZeRO-1 | 优化器状态（12 字节） | 4N + 12N / n |
| ZeRO-2 | 再加上梯度 | 2N + 14N / n |
| ZeRO-3 | 再加上参数本身 | 16N / n |

```python
def zero_gb(N, n, stage):
    per_param = {0: 16, 1: 4 + 12 / n, 2: 2 + 14 / n, 3: 16 / n}[stage]
    return N * per_param / 1e9

N, n = 70e9, 64
for stage in range(4):
    print(f"ZeRO-{stage}：每张 GPU {zero_gb(N, n, stage):7.1f} GB")
```

ZeRO-3 时，每张 GPU 只存 1/n 的参数；计算某一层之前，先把这一层的参数从其他 GPU 收集过来（all-gather），用完就扔掉。PyTorch 自带的 **FSDP**（Fully Sharded Data Parallel）就是 ZeRO-3 的实现，DeepSpeed 库也提供了完整的 ZeRO。

## 3.6 张量并行和流水线并行

当一层的矩阵太大、或者希望减少每张 GPU 上的计算量时，就要把**模型本身**拆开。

**张量并行**（Tensor Parallelism, TP，来自 NVIDIA 的 Megatron-LM）：把一个矩阵乘法拆到几张 GPU 上。以 MLP 为例，第一个矩阵**按列**切开，第二个矩阵**按行**切开，每张 GPU 独立算完自己的部分，最后只需要一次 all-reduce 把结果加起来：

```python
def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

def relu(M):
    return [[max(0, v) for v in row] for row in M]

x = [[1, 2], [3, -4]]                       # 2 个 token，每个 2 维
W = [[1, 0, 2, -1], [0, 1, 1, 3]]           # 2 → 4（MLP 第一层）
V = [[1, 2], [0, 1], [1, 0], [2, 1]]        # 4 → 2（MLP 第二层）
full = matmul(relu(matmul(x, W)), V)        # 一张 GPU 的完整计算

W0, W1 = [r[:2] for r in W], [r[2:] for r in W]   # GPU 0 拿前两列，GPU 1 拿后两列
V0, V1 = V[:2], V[2:]                              # 对应地，V 按行切开
part0 = matmul(relu(matmul(x, W0)), V0)            # 两张 GPU 各算各的，中间不用通信
part1 = matmul(relu(matmul(x, W1)), V1)
summed = [[a + b for a, b in zip(r0, r1)] for r0, r1 in zip(part0, part1)]   # 一次 all-reduce
print("完整计算：", full)
print("张量并行：", summed, "相同" if full == summed else "不同")
```

（ReLU 是逐个元素计算的，所以按列切开后各自做也完全正确。注意力也可以按「头」切开。）张量并行每一层都要通信，需要 GPU 之间非常快的连接，所以通常只在**同一台机器**的 8 张 GPU 之间使用（它们用 NVLink 相连）。

**流水线并行**（Pipeline Parallelism, PP）：把模型**按层**切开，比如 80 层分给 8 张 GPU，每张 10 层，数据像流水线一样依次经过。问题是：第 1 张 GPU 算完第一批数据，第 8 张 GPU 才刚开始；其他 GPU 在等待，这些空闲叫**气泡**（bubble）。办法是把一个批次切成很多个微批次，让它们首尾相接地流过：

```python
def bubble(p, m):              # p 段流水线，m 个微批次
    return (p - 1) / (m + p - 1)

for m in (1, 4, 16, 64):
    print(f"8 段流水线、{m:>2} 个微批次：GPU 空闲 {bubble(8, m):.0%}")
```

## 3.7 激活重计算，以及把它们组合起来

- **激活重计算**（activation checkpointing）：前向传播时不保存所有中间结果，只保存少数「检查点」；反向传播需要时再从检查点**重新算一遍**。多花约 1/3 的计算，换来显存的大幅节省。
- **组合使用**：真实的大规模训练会同时使用好几种并行。比如 Meta 训练 Llama 3 405B 时，在 1.6 万张 H100 上组合了张量并行（8 张，同一台机器内）、流水线并行（16 段）、数据并行（FSDP），以及为长序列准备的**上下文并行**（把一条长序列切到多张 GPU 上）——被称为「4D 并行」。混合专家模型（第 4 课）还有**专家并行**：不同的专家放在不同的 GPU 上。
- **故障**：几万张 GPU 一起工作好几个月，硬件故障是家常便饭。Llama 3 的论文提到，在 54 天里遇到了 400 多次意外中断。所以训练程序要经常保存**检查点**（checkpoint），出故障时能从最近的检查点恢复。

## 3.8 小结

- FP32 / FP16 / BF16 / FP8：位数越少越快越省，代价是精度和范围；BF16 和 FP32 范围相同，是现在的标准选择。
- 混合精度：矩阵乘法用 BF16，参数用 FP32 主副本更新（小的更新在 BF16 里会丢失）；FP16 需要损失缩放。
- 梯度累积：多个微批次的梯度累加后再更新，用时间换显存。
- 数据并行：每张 GPU 一份完整模型、不同数据，用 all-reduce 平均梯度；环形 all-reduce 每张 GPU 只发送 2(n−1)/n 倍的数据。
- ZeRO / FSDP：把优化器状态、梯度、参数切分到各 GPU，ZeRO-3 每张只存 1/n。
- 张量并行：一层的矩阵按行列切开，通信多，用在机器内；流水线并行：按层切开，用微批次减少气泡。
- 激活重计算用计算换显存；大规模训练组合多种并行，并经常保存检查点。

## 3.9 练习

<!-- exercise: zero-memory -->
### 练习 1：ZeRO 的显存

输入参数量 N（十亿）、GPU 数量 n 和 ZeRO 阶段（0～3）。按 3.5 节的表格，输出每张 GPU 需要的显存（GB），保留 1 位小数（不算激活值）。

```text
样例输入：70 64 3
样例输出：17.5
```

<details>
<summary>提示</summary>

每参数字节数：阶段 0 是 16，阶段 1 是 `4 + 12 / n`，阶段 2 是 `2 + 14 / n`，阶段 3 是 `16 / n`。N 以十亿为单位时，`N * 字节数` 正好是 GB。
</details>

<!-- exercise: pipeline-bubble -->
### 练习 2：流水线的气泡

输入流水线的段数 `p` 和微批次数 `m`，输出 GPU 空闲时间所占的比例 `(p - 1) / (m + p - 1)`，保留 3 位小数。

```text
样例输入：8 16
样例输出：0.304
```

<details>
<summary>提示</summary>

直接套公式：`print(f"{(p - 1) / (m + p - 1):.3f}")`。
</details>

<!-- exercise: ring-traffic -->
### 练习 3：all-reduce 的通信量

输入梯度的大小 S（GB）和 GPU 数量 n，输出用环形 all-reduce 时每张 GPU 需要发送的数据量（GB），即 `2(n−1)/n × S`，保留 2 位小数。

```text
样例输入：16 8
样例输出：28.00
```

<details>
<summary>提示</summary>

`print(f"{2 * (n - 1) / n * S:.2f}")`。注意 n 很大时结果接近 `2S`。
</details>

<!-- exercise: global-batch -->
### 练习 4：全局批大小

输入序列长度、每张 GPU 的微批次大小（条）、梯度累积步数和数据并行的 GPU 数。每次参数更新一共处理多少个 token？再换算成百万（M），保留 2 位小数。两个数用空格隔开。

```text
样例输入：8192 2 4 128
样例输出：8388608 8.39
```

<details>
<summary>提示</summary>

token 数 = 序列长度 × 微批次大小 × 累积步数 × GPU 数。百万：除以 `1e6`。
</details>

<!-- exercise: why-bf16 -->
### 练习 5：为什么用 BF16？

训练大模型时，BF16 取代 FP16 成为首选，主要原因是什么？

- BF16 的精度比 FP16 高
- BF16 的指数位和 FP32 一样多，范围相同，不容易上溢或下溢，不需要损失缩放
- BF16 只占 8 位
- BF16 不需要 FP32 的主参数

<details>
<summary>答案与解析</summary>

BF16 有 8 位指数，范围和 FP32 相同；FP16 只有 5 位指数，最大 65504，很小的梯度又会下溢成 0。BF16 的精度其实比 FP16 **低**（7 位尾数对 10 位），所以参数更新仍要用 FP32 主参数。
</details>

<!-- exercise: zero3 -->
### 练习 6：ZeRO-3 切分什么

ZeRO-3（或 FSDP）在数据并行的基础上切分了哪些内容？

- 只切分数据
- 只切分优化器状态
- 优化器状态、梯度和参数都切分到各张 GPU 上
- 按层把模型切分到不同 GPU 上

<details>
<summary>答案与解析</summary>

ZeRO-1 切分优化器状态，ZeRO-2 再加梯度，ZeRO-3 连参数也切分，每张 GPU 只存 1/n，用到时再临时收集。按层切分是流水线并行。
</details>

## 3.10 延伸阅读

- Micikevicius 等：《Mixed Precision Training》（2018）
- Rajbhandari 等：《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》（2019）
- Shoeybi 等：《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》（2019）
- Hugging Face：《The Ultra-Scale Playbook: Training LLMs on GPU Clusters》（2025，图文并茂的并行训练指南）
- 下一课：[现代架构](../04-architectures/zh-CN.md)
