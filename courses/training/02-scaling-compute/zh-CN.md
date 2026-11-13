# 2. 规模定律与算力

> 🎯 **本课目标**：会数一个 Transformer 有多少参数；会用 **C ≈ 6ND** 估算训练要多少计算量、要多少张 GPU 训练多少天；理解**规模定律**——损失随参数量、数据量、计算量按幂律下降；理解 **Chinchilla** 的结论（每个参数约 20 个 token）以及为什么今天的模型常常「过度训练」；会估算训练和推理需要多少显存；了解「测试时计算」这条新的扩展路线。
>
> ⏱ **预计用时**：70 分钟　｜　📋 **前置知识**：本部分第 1 课；第三部分第 5、6 课

训练一个大模型要花几百万到上亿美元。在花钱之前，研究者需要回答：模型做多大？用多少数据？要多少张 GPU、训练多久？训练完大概能有多好？这一课的工具让你能用纸笔（和几行 Python）回答这些问题。

## 2.1 数参数

第三部分第 5 课的 Transformer 里，参数主要在两个地方：注意力和前馈网络（MLP）。设模型维度为 d、层数为 L：

- 注意力：Q、K、V、输出四个 d × d 的矩阵，共 **4d²**；
- MLP：d → 4d → d 两个矩阵，共 **8d²**；
- 所以每层大约 **12d²**，再加上词表大小 V × d 的嵌入矩阵。

现代的 Llama 式模型稍有不同：K、V 用**分组查询注意力**（GQA，第 4 课）只占较少的头；MLP 用 SwiGLU，有三个矩阵；输入嵌入和输出层通常不共享。我们两种都算一下：

```python
def gpt2_params(L, d, V, ctx):
    per_layer = 12 * d * d + 13 * d          # 12d² 的矩阵，加上偏置和 LayerNorm
    return L * per_layer + V * d + ctx * d + 2 * d

def llama_params(L, d, n_heads, n_kv, ffn, V):
    head_dim = d // n_heads
    attn = 2 * d * d + 2 * d * n_kv * head_dim    # Q 和输出 d×d；K、V 只有 n_kv 个头
    mlp = 3 * d * ffn                             # SwiGLU：门控、上投影、下投影
    per_layer = attn + mlp + 2 * d                # 加两个 RMSNorm
    return L * per_layer + 2 * V * d + d          # 输入嵌入 + 输出层（不共享）+ 最后的 RMSNorm

print(f"GPT-2 small : {gpt2_params(12, 768, 50257, 1024) / 1e6:.0f}M")
print(f"GPT-2 XL    : {gpt2_params(48, 1600, 50257, 1024) / 1e9:.2f}B")
print(f"Llama 3 8B  : {llama_params(32, 4096, 32, 8, 14336, 128256) / 1e9:.2f}B")
print(f"Llama 3 70B : {llama_params(80, 8192, 64, 8, 28672, 128256) / 1e9:.2f}B")
```

算出的数字和官方公布的参数量（1.24 亿、15 亿、80 亿、706 亿）完全吻合。可以看出：模型越大，参数越集中在每层的矩阵里（和 d² 成正比），嵌入层所占的比例越小。

## 2.2 计算量：C ≈ 6ND

训练一个 N 个参数的模型、读 D 个 token，总的计算量（浮点运算次数，FLOPs）大约是：

```formula
C ≈ 6 · N · D
```

为什么是 6？每个参数在处理每个 token 时：

- **前向传播**做一次乘法和一次加法 → 2 次运算；
- **反向传播**要算对输入的梯度和对参数的梯度，大约是前向的两倍 → 4 次运算。

（注意力本身的计算和序列长度有关，在序列不太长时只占一小部分，这里忽略。）同理，**推理**时每生成一个 token 大约需要 **2N** 次运算。

有了这个公式，就能估算训练时间：

```python
def train_days(N, D, gpus, peak_tflops, mfu):
    flops = 6 * N * D
    per_second = gpus * peak_tflops * 1e12 * mfu      # 整个集群每秒实际完成的运算
    return flops / per_second / 86400

H100 = 989            # H100 的 BF16 稠密矩阵运算峰值，约 989 TFLOPS
print(f"80 亿参数 × 15 万亿 token：{6 * 8e9 * 15e12:.1e} FLOPs")
for N, D, gpus in [(8e9, 15e12, 1024), (70e9, 15e12, 4096), (405e9, 15e12, 16384)]:
    days = train_days(N, D, gpus, H100, 0.40)
    print(f"{N / 1e9:>4.0f}B 参数、{D / 1e12:.0f}T token、{gpus:>5} 张 H100：约 {days:5.1f} 天")
```

- **MFU**（Model FLOPs Utilization，模型算力利用率）：GPU 的峰值算力只是理论值，实际训练中要等数据、等通信、做各种杂事，能用上 **30%～50%** 就算不错了。
- Meta 训练 Llama 3 405B 时用了 **1.6 万张 H100**；DeepSeek 报告 DeepSeek-V3 的完整训练用了约 **279 万 H800 GPU 小时**。这些数字和我们的估算处在同一个量级。

## 2.3 规模定律

2020 年，OpenAI 的 Kaplan 等人发现了一个惊人的规律：在很大的范围内（跨越好几个数量级），语言模型的测试损失随参数量 N、数据量 D、计算量 C 都按**幂律**平滑地下降。也就是说，在对数坐标里几乎是一条直线——只要加大规模，效果就**可以预测地**变好。这就是**规模定律**（scaling laws），也是各家公司敢于投入巨资训练更大模型的底气。

2022 年，DeepMind 的 Hoffmann 等人（**Chinchilla** 论文）用 400 多个模型做实验，把损失写成参数量和数据量的函数：

```formula
L(N, D) = E + A / N^α + B / D^β
```

- E 是**不可消除的损失**：语言本身的随机性，再大的模型也消除不了；
- A / N^α：模型**太小**带来的损失；
- B / D^β：数据**太少**带来的损失。

下面用 Besiroglu 等人（2024）重新拟合 Chinchilla 数据得到的系数，看看在**计算量固定**时，参数和数据该怎么分配：

```python
E, A, B, alpha, beta = 1.82, 482.01, 2085.43, 0.3478, 0.3658

def loss(N, D):
    return E + A / N**alpha + B / D**beta

def best_split(C):                         # 计算量 C 固定：试遍各种 N，D 由 C = 6ND 决定
    candidates = [10 ** (k / 100) for k in range(700, 1300)]
    return min(candidates, key=lambda N: loss(N, C / (6 * N)))

print("计算量 C      最优参数 N   最优数据 D   D/N   预测损失")
for C in (1e21, 1e22, 1e23, 5.76e23, 1e24, 1e25):
    N = best_split(C)
    D = C / (6 * N)
    print(f"{C:9.1e}   {N / 1e9:8.1f}B   {D / 1e9:8.0f}B  {D / N:5.1f}   {loss(N, D):.3f}")

print()
print(f"Gopher     280B 参数 × 0.3T token：损失 {loss(280e9, 0.3e12):.3f}")
print(f"Chinchilla  70B 参数 × 1.4T token：损失 {loss(70e9, 1.4e12):.3f}")
```

结论：

- 计算量增加时，**参数量和数据量应该大致同比例增加**；最优点上，每个参数大约对应 **20 个 token**。
- 在此之前，大家主要在增加参数量。DeepMind 自己的 Gopher（2800 亿参数，3000 亿 token）其实**训练得不够**。用同样的计算量（表里 5.8e+23 那一行），训练一个小 4 倍的 Chinchilla（700 亿参数，1.4 万亿 token），效果反而更好。

## 2.4 为什么今天的模型「过度训练」

Chinchilla 的最优只考虑了**训练**的花费。但模型训练一次，却要被使用千千万万次，**推理**的花费和参数量成正比。所以现在的常见做法是：训练一个比「最优」更小的模型，喂给它远多于 20 倍的数据——训练时多花一些，换来以后每次推理都更便宜。

```python
E, A, B, alpha, beta = 1.82, 482.01, 2085.43, 0.3478, 0.3658

def loss(N, D):
    return E + A / N**alpha + B / D**beta

N = 8e9
print("80 亿参数的模型，读越来越多的数据：")
for D in (160e9, 1e12, 5e12, 15e12, 36e12):
    print(f"  {D / 1e12:5.2f}T token（每参数 {D / N:5.0f} 个）：预测损失 {loss(N, D):.3f}")
```

Llama 3 8B 读了 15 万亿 token，是「最优」的近 100 倍，损失仍然在下降。（这里的数字是用公式**外推**的，只能看趋势：拟合公式时用的模型比这小得多。）

## 2.5 显存：为什么一张 GPU 训练不了大模型

训练时，每个参数要占多少显存？以常见的 **BF16 混合精度 + Adam 优化器**为例：

| 内容 | 每个参数的字节数 |
|------|------------------|
| BF16 的参数（前向、反向用） | 2 |
| BF16 的梯度 | 2 |
| FP32 的「主参数」（更新时用，保证精度） | 4 |
| Adam 的一阶动量 m（FP32） | 4 |
| Adam 的二阶动量 v（FP32） | 4 |
| **合计** | **16** |

```python
import math

def train_memory_gb(N):
    return 16 * N / 1e9                      # 不含激活值

def infer_memory_gb(N, bytes_per_param=2):
    return bytes_per_param * N / 1e9         # 只要存参数（不含 KV 缓存，第六部分）

for N in (1e9, 8e9, 70e9, 405e9):
    need = train_memory_gb(N)
    print(f"{N / 1e9:>4.0f}B：训练约 {need:>6.0f} GB（至少 {math.ceil(need / 80):>3} 张 80GB GPU），"
          f"BF16 推理约 {infer_memory_gb(N):>4.0f} GB")
```

这还没算**激活值**——前向传播的中间结果，反向传播时要用，和批大小、序列长度成正比，往往比参数还占地方。所以 80 亿参数的模型，一张 80GB 的 GPU 都放不下它的训练状态；700 亿参数的模型至少要十几张。下一课讲怎样把模型和数据分到成百上千张 GPU 上。

## 2.6 涌现、测试时计算和新的扩展方向

- **涌现能力**：有些能力（比如多位数算术）似乎在模型达到某个规模时「突然出现」。但 Schaeffer 等人（2023）指出，这很大程度上和评测指标有关：用「完全答对才算分」的指标，能力看起来是突变；换成连续的指标（比如每一位数字的正确率），提升其实是平滑的。
- **测试时计算**（test-time compute）：2024 年起，OpenAI o1、DeepSeek-R1 等**推理模型**展示了另一条扩展路线：不只加大训练，还让模型在回答前「多想一会儿」——生成长长的思考过程。思考得越多，难题的正确率越高，形成了新的规模定律（第五部分会详细讲）。
- **数据瓶颈**：高质量的人类文本是有限的，按现在的速度，几年内可能被用完。这推动了合成数据、多模态数据和更高效训练方法的研究。

## 2.7 小结

- 每层参数约 12d²（GPT 式）；Llama 式模型要考虑 GQA、SwiGLU 和不共享的嵌入层。
- 训练计算量 C ≈ 6ND，推理每个 token ≈ 2N；训练天数 = 6ND ÷（GPU 数 × 峰值算力 × MFU）。
- 规模定律：损失随 N、D、C 按幂律平滑下降；L(N, D) = E + A/N^α + B/D^β。
- Chinchilla：计算量固定时，N 和 D 同比例增加，约 20 个 token / 参数；为了降低推理成本，今天的模型常常远超这个比例。
- 训练显存约 16 字节 / 参数（BF16 + Adam），另加激活值；BF16 推理约 2 字节 / 参数。
- 新方向：测试时计算、合成数据。

## 2.8 练习

<!-- exercise: param-count -->
### 练习 1：数参数

输入层数 `L`、模型维度 `d` 和词表大小 `V`。按「每层 12d²，加上嵌入层 V × d」估算参数量（忽略偏置、LayerNorm 和位置嵌入），以百万为单位输出，保留 1 位小数，后面加 `M`。

```text
样例输入：12 768 50257
样例输出：123.5M
```

<details>
<summary>提示</summary>

`params = 12 * L * d * d + V * d`，输出 `f"{params / 1e6:.1f}M"`。
</details>

<!-- exercise: train-days -->
### 练习 2：要训练多少天？

输入一行：参数量 N（十亿）、训练 token 数 D（十亿）、GPU 数量、每张 GPU 的峰值算力（TFLOPS）、MFU（0～1 之间的小数）。用 C = 6ND 估算训练需要多少天，保留 1 位小数。

```text
样例输入：8 15000 1024 989 0.4
样例输出：20.6
```

<details>
<summary>提示</summary>

先把单位换算好：N 和 D 都乘以 `1e9`，算力乘以 `1e12`。天数 = `6 * N * D / (gpus * tflops * 1e12 * mfu) / 86400`。
</details>

<!-- exercise: chinchilla-opt -->
### 练习 3：Chinchilla 最优

输入计算量 C（FLOPs，可能写成 `1e23` 这样的形式）。按「C = 6ND」和「D = 20N」两条规则，求出最优的参数量 N 和数据量 D，以十亿为单位输出，各保留 1 位小数，用空格隔开。

```text
样例输入：1e23
样例输出：28.9 577.4
```

<details>
<summary>提示</summary>

把 D = 20N 代入 C = 6ND，得到 C = 120N²，所以 `N = (C / 120) ** 0.5`。`float("1e23")` 能直接读入这种写法。
</details>

<!-- exercise: adam-memory -->
### 练习 4：训练要几张 GPU？

输入参数量 N（十亿）和每张 GPU 的显存（GB）。按每个参数 16 字节（BF16 混合精度 + Adam，不算激活值）计算训练状态需要多少 GB 显存，以及至少需要几张 GPU，用空格隔开输出（显存取整数）。

```text
样例输入：70 80
样例输出：1120 14
```

<details>
<summary>提示</summary>

`need = 16 * N`（N 以十亿为单位时，结果正好是 GB）。GPU 数用 `math.ceil(need / mem)` 向上取整。
</details>

<!-- exercise: six-nd -->
### 练习 5：6ND 里的 6

训练计算量公式 C ≈ 6ND 中的「6」是怎么来的？

- 6 是 Transformer 每层的矩阵个数
- 每个参数每处理一个 token：前向传播约 2 次运算，反向传播约 4 次运算
- 6 是经验数字，没有道理
- Adam 优化器每个参数要存 6 个数

<details>
<summary>答案与解析</summary>

前向传播里每个参数参与一次乘法和一次加法（2 次）；反向传播既要算对输入的梯度、又要算对参数的梯度，大约是前向的两倍（4 次）。合计 6 次。推理只有前向，所以是每 token 约 2N。
</details>

<!-- exercise: chinchilla-lesson -->
### 练习 6：Chinchilla 的结论

在计算量固定的情况下，Chinchilla 论文得出了什么结论？

- 应该把几乎全部计算量用来增大模型
- 应该用尽可能小的模型、尽可能多的数据
- 参数量和训练 token 数应该大致同比例增加，此前的大模型训练数据太少了
- 数据量对损失没有影响

<details>
<summary>答案与解析</summary>

Chinchilla 发现最优的做法是 N 和 D 同比例增加（约 20 个 token / 参数）。Gopher 等之前的模型参数很多、数据偏少；同样的计算量，更小的模型配更多的数据效果更好。
</details>

## 2.9 延伸阅读

- Kaplan 等：《Scaling Laws for Neural Language Models》（2020）
- Hoffmann 等：《Training Compute-Optimal Large Language Models》（2022，Chinchilla）
- Besiroglu 等：《Chinchilla Scaling: A replication attempt》（2024）
- Schaeffer 等：《Are Emergent Abilities of Large Language Models a Mirage?》（2023）
- 下一课：[分布式训练与混合精度](../03-distributed/zh-CN.md)
