# 4. 现代架构：更快、更省、更长

> 🎯 **本课目标**：在第三部分第 5 课（Llama 式 Transformer、MoE 入门）的基础上更进一步：理解 Pre-Norm、RMSNorm 和 QK-Norm 怎样让训练更稳定；比较 MHA、MQA、GQA 和 DeepSeek 的 **MLA**，学会计算 **KV 缓存**有多大；理解长上下文的技术（RoPE 插值、YaRN、滑动窗口、注意力汇聚点）；亲手实现 **FlashAttention** 的核心——**在线 softmax**；理解 MoE 的**负载均衡**；了解多 token 预测、线性注意力和混合架构等新方向。
>
> ⏱ **预计用时**：90 分钟　｜　📋 **前置知识**：第三部分第 4、5 课；本部分第 2、3 课

2017 年的 Transformer 和 2026 年最强的模型，整体结构几乎一样：嵌入 → 很多个「注意力 + MLP」Block → 输出层。但几乎每个零件都被改进过，目标都是三件事：**训练更稳定**、**推理更快更省**、**能处理更长的上下文**。

## 4.1 归一化：Pre-Norm、RMSNorm 和 QK-Norm

原始的 Transformer 把 LayerNorm 放在残差相加**之后**（Post-Norm）。模型一深，训练就容易发散。GPT-2 起改成了放在每个子层**之前**（Pre-Norm）：

```text
Post-Norm：x = Norm(x + Attention(x))        Pre-Norm：x = x + Attention(Norm(x))
```

Pre-Norm 让残差连接成为一条「干净的高速公路」，梯度可以畅通无阻地传到底层，所以几百层的模型也能稳定训练。

**RMSNorm**（Llama 起成为标准）比 LayerNorm 更简单：不减去平均值，只除以均方根，再乘以可学习的缩放 γ：

```python
import math

def layer_norm(x, eps=1e-6):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    return [(v - mean) / math.sqrt(var + eps) for v in x]

def rms_norm(x, eps=1e-6):
    rms = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v / rms for v in x]

x = [2.0, -1.0, 4.0, 3.0]
print("LayerNorm：", [round(v, 3) for v in layer_norm(x)])
print("RMSNorm：  ", [round(v, 3) for v in rms_norm(x)])
print("RMSNorm 后的均方根：", round(math.sqrt(sum(v * v for v in rms_norm(x)) / len(x)), 3))
```

少算一个平均值看似微不足道，但归一化在每层都要做好几次，省下的时间很可观，效果却几乎一样。

**QK-Norm**：训练很大的模型时，注意力的 logits（q·k）有时会越来越大，softmax 变得极端，训练就崩了。在算点积之前，先对 q 和 k 各做一次 RMSNorm，就能把它们控制住。Gemma 3、Qwen3、OLMo 2 等模型都用了 QK-Norm。

## 4.2 注意力的变体和 KV 缓存

生成文本时，模型每生成一个 token，都要用到前面所有 token 的 K 和 V。为了不重复计算，它们会被保存下来，叫做 **KV 缓存**（第六部分会详细讲）。KV 缓存的大小是：

```formula
KV 缓存（每个 token）= 2 × 层数 × KV 头数 × 每个头的维度 × 每个数的字节数
```

上下文很长、同时服务的用户很多时，KV 缓存往往比模型参数本身还占显存。于是出现了几种减少 K、V 的方法：

- **MHA**（多头注意力）：每个 query 头都有自己的 K、V 头。
- **MQA**（多查询注意力，2019）：所有 query 头**共用一组** K、V。缓存最小，但效果有所下降。
- **GQA**（分组查询注意力，2023）：query 头分成几组，每组共用一组 K、V。Llama 3 70B 有 64 个 query 头、8 个 KV 头。效果接近 MHA，缓存小 8 倍——现在最常用。
- **MLA**（多头潜在注意力，DeepSeek-V2，2024）：不直接缓存 K、V，而是缓存一个压缩过的**低维「潜在」向量**（DeepSeek-V3 每层 512 维，加上 64 维给 RoPE 用），用的时候再还原成各个头的 K、V。

```python
def kv_cache_gb(layers, values_per_layer, tokens, bytes_per_value=2):
    return layers * values_per_layer * tokens * bytes_per_value / 1e9

T = 128_000                                       # 一条 12.8 万 token 的上下文，BF16
configs = [
    ("MHA, 64 KV heads", 80, 2 * 64 * 128),       # 80 层，K 和 V 各 64 个头 × 128 维
    ("GQA, 8 KV heads", 80, 2 * 8 * 128),         # Llama 3 70B 的设置
    ("MQA, 1 KV head", 80, 2 * 1 * 128),
    ("MLA (DeepSeek-V3)", 61, 512 + 64),          # 61 层，每层缓存 576 个数
]
for name, layers, per_layer in configs:
    kb = layers * per_layer * 2 / 1024
    print(f"{name:<18} 每个 token {kb:6.1f} KB，12.8 万 token 共 {kv_cache_gb(layers, per_layer, T):6.1f} GB")
```

从 MHA 到 GQA，缓存缩小了 8 倍；MLA 则在 6710 亿参数的模型上把它压到了每个 token 不到 70 KB。

## 4.3 长上下文

2023 年初，大多数模型只能处理 2K～4K 个 token；到 2025 年，128K 已是标配，不少模型支持 100 万 token 以上。这需要几方面的技术。

### RoPE 的外推问题和插值

第三部分第 4 课讲过，RoPE 让第 i 组维度按 `θᵢ = base^(−2i/d)` 的速度旋转。转得最慢的那一组，在训练长度以内只转过很小的角度。如果直接用到更长的上下文，这一组会转到训练时**从没见过的角度**，模型就不知所措了：

```python
import math

d, base = 128, 10_000
theta_slow = base ** (-(d - 2) / d)          # 转得最慢的一组
train_len, target_len = 8_192, 32_768

print(f"训练长度 {train_len}：最慢的一组最多转到 {train_len * theta_slow:.2f} 弧度")
print(f"直接用到 {target_len}：会转到 {target_len * theta_slow:.2f} 弧度（训练时没见过）")

scale = train_len / target_len               # 位置插值：把位置「压缩」回训练范围
print(f"位置插值后：最多 {target_len * scale * theta_slow:.2f} 弧度")

new_base = base * (target_len / train_len) ** (d / (d - 2))   # 「NTK」式：改大 base，让慢的组转得更慢
theta_new = new_base ** (-(d - 2) / d)
print(f"把 base 改成 {new_base:,.0f} 后：最多 {target_len * theta_new:.2f} 弧度")
```

- **位置插值**（Position Interpolation，2023）：把位置乘以一个缩放系数，让角度回到训练过的范围，再用少量长文本微调。
- **NTK 缩放、YaRN**（2023）：改大 base 或者对不同频率的组区别对待——转得快的组（负责区分相邻 token）保持不变，只调整转得慢的组。现在的模型常用 YaRN 把上下文扩展 4～32 倍。
- 训练上，通常先用较短的序列（比如 8K）预训练，最后再专门用长文档训练一个「长上下文阶段」（Llama 3 就是这样扩展到 128K 的）。

### 滑动窗口和注意力汇聚点

- **滑动窗口注意力**：每个 token 只看最近的 W 个 token（比如 4096 个），计算和缓存都不再随上下文无限增长。Mistral 7B 用过这种方法；Gemma 3 把「局部」（滑动窗口）层和「全局」层按 5:1 交替排列——大多数层只看附近，少数层看全局，KV 缓存因此小了很多。
- **注意力汇聚点**（attention sink，2023）：研究者发现，模型会把大量的注意力分给**最开头的几个 token**，哪怕它们没什么意义——这是 softmax 必须让权重加起来等于 1 的副作用。如果滑动窗口把开头的 token 丢掉了，模型就会崩溃；只要一直保留开头几个 token，就能稳定地处理很长的文本流。

能「放得下」长上下文，不等于能「用好」它。有研究发现，模型对放在长上下文**中间**的信息利用得最差（「迷失在中间」，lost in the middle）。第 5 课会介绍怎样评测长上下文能力。

## 4.4 FlashAttention：精确，但更省

注意力要算一个 n × n 的分数矩阵。n = 128K 时，这个矩阵有 160 亿个元素。直接算的话，要把它写进 GPU 的显存（HBM）再读出来——GPU 的计算其实很快，**真正慢的是搬运数据**。

**FlashAttention**（Tri Dao 等，2022）的思路是：把 Q、K、V 切成小块，放进 GPU 芯片上又小又快的缓存（SRAM）里，**一块一块地**算，完全不把 n × n 的矩阵写出来。难点在于 softmax 需要知道**所有**分数的最大值和总和。答案是**在线 softmax**：一边读入新的分数块，一边更新「目前的最大值」和「目前的总和」，旧的结果用一个修正系数调整：

```python
import math

scores = [2.0, -1.0, 0.5, 3.0, 1.0, -0.5, 2.5, 0.0]       # 一个 query 对 8 个 key 的分数
values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]  # 每个 key 对应的 value（这里是一维的）

# 普通做法：先算完所有分数，再 softmax，再加权平均
m = max(scores)
w = [math.exp(s - m) for s in scores]
standard = sum(wi * v for wi, v in zip(w, values)) / sum(w)

# 在线做法：每次只看 3 个分数，从不同时拿着全部分数
run_max, run_sum, run_out = float("-inf"), 0.0, 0.0
for start in range(0, len(scores), 3):
    block_s, block_v = scores[start:start + 3], values[start:start + 3]
    new_max = max(run_max, max(block_s))
    fix = math.exp(run_max - new_max)                     # 最大值变了：旧的累计结果要按比例缩小
    run_sum = run_sum * fix + sum(math.exp(s - new_max) for s in block_s)
    run_out = run_out * fix + sum(math.exp(s - new_max) * v for s, v in zip(block_s, block_v))
    run_max = new_max
    print(f"读完前 {start + len(block_s)} 个：当前最大值 {run_max}，当前结果 {run_out / run_sum:.4f}")

print(f"普通做法：{standard:.6f}，在线做法：{run_out / run_sum:.6f}")
```

两种做法的结果**完全相同**——FlashAttention 不是近似，而是同一个计算的更聪明的顺序。它让注意力需要的额外显存从 O(n²) 降到 O(n)，速度也快了好几倍。FlashAttention-2、-3 针对新的 GPU 进一步优化，现在几乎所有训练和推理框架都在用它（PyTorch 的 `scaled_dot_product_attention` 会自动调用）。

## 4.5 MoE 的负载均衡

第三部分第 5 课介绍了 MoE：路由器为每个 token 选出得分最高的几个专家。训练 MoE 最大的麻烦是**负载不均衡**：一开始某几个专家稍好一点，路由器就把更多 token 发给它们，它们越学越好，得到的 token 越来越多……最后大部分专家闲着，模型退化成了小模型，而且负责热门专家的 GPU 忙不过来。

常见的解决办法是加一个**辅助损失**（Switch Transformer，2021）：设 fᵢ 是被分给专家 i 的 token 比例，Pᵢ 是路由器给专家 i 的平均概率，E 是专家数：

```formula
辅助损失 = E · Σᵢ fᵢ · Pᵢ
```

完全均衡时它等于 1；越不均衡，它越大。把它加进总损失，模型就会被推向均衡：

```python
import math
import random

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

random.seed(0)
E, T = 4, 1000                                       # 4 个专家，1000 个 token
for name, bias in [("偏向专家 0", [2.0, 0, 0, 0]), ("均衡", [0, 0, 0, 0])]:
    counts, prob_sum = [0] * E, [0.0] * E
    for _ in range(T):
        p = softmax([random.gauss(0, 1) + b for b in bias])   # 路由器给这个 token 的概率
        counts[p.index(max(p))] += 1                          # top-1：分给概率最大的专家
        prob_sum = [s + pi for s, pi in zip(prob_sum, p)]
    f = [c / T for c in counts]
    P = [s / T for s in prob_sum]
    aux = E * sum(fi * pi for fi, pi in zip(f, P))
    print(f"{name}：各专家的 token 比例 {[round(v, 2) for v in f]}，辅助损失 {aux:.3f}")
```

DeepSeek 在此基础上做了几项改进：

- **细粒度专家 + 共享专家**（DeepSeekMoE，2024）：把专家切得更小、更多（DeepSeek-V3 每层 256 个路由专家，每个 token 选 8 个），再加一个所有 token 都会经过的**共享专家**，负责通用知识；
- **无辅助损失的均衡**（DeepSeek-V3）：辅助损失会干扰主要的训练目标，于是改为给每个专家的路由分数加一个偏置：某个专家太忙就调低它的偏置，太闲就调高，偏置只影响「选谁」，不影响输出的加权。

## 4.6 更多新方向

- **多 token 预测**（MTP）：DeepSeek-V3 在训练时，每个位置不仅预测下一个 token，还用额外的小模块预测再下一个 token。这给模型提供了更密集的训练信号，推理时这些模块还能用来加速（投机解码，第六部分）。
- **线性注意力和状态空间模型**：普通注意力的计算随上下文长度平方增长。Mamba（2023）等**状态空间模型**和各种**线性注意力**把它降到了线性，生成时只需要一个固定大小的「状态」，不需要越来越大的 KV 缓存。但它们在需要「精确回忆」上下文的任务上通常不如普通注意力。
- **混合架构**：于是很多新模型把两者混合：大部分层用线性注意力或状态空间层，每隔几层放一层普通注意力。Jamba（2024）、MiniMax-01（2025）、Qwen3-Next（2025）等都采用了这样的设计，在超长上下文上兼顾了速度和效果。

## 4.7 小结

- Pre-Norm 让深层模型稳定；RMSNorm 更省计算；QK-Norm 防止注意力 logits 爆炸。
- KV 缓存（每 token）= 2 × 层数 × KV 头数 × 头维度 × 字节数；GQA 让多个 query 头共用 K、V，MLA 缓存一个压缩的潜在向量。
- 长上下文：位置插值、NTK、YaRN 调整 RoPE；先短后长地训练；滑动窗口和局部 / 全局交替减少开销；要保留注意力汇聚点。
- FlashAttention：分块计算 + 在线 softmax，结果精确，显存从 O(n²) 降到 O(n)，速度更快。
- MoE 负载均衡：辅助损失 E·Σfᵢ·Pᵢ，或 DeepSeek 的偏置调节；细粒度专家 + 共享专家。
- 新方向：多 token 预测、线性注意力和状态空间模型、混合架构。

## 4.8 练习

<!-- exercise: rmsnorm -->
### 练习 1：RMSNorm

输入一行数（一个向量），输出它经过 RMSNorm（γ 全为 1，eps = 1e-6）之后的结果，每个数保留 4 位小数，用空格隔开。

```text
样例输入：2 -1 4 3
样例输出：0.7303 -0.3651 1.4606 1.0954
```

<details>
<summary>提示</summary>

`rms = math.sqrt(sum(v * v for v in x) / len(x) + 1e-6)`，然后每个数除以 `rms`。
</details>

<!-- exercise: kv-cache -->
### 练习 2：KV 缓存有多大

输入一行：层数、KV 头数、每个头的维度、上下文的 token 数、每个数的字节数。输出这条上下文的 KV 缓存大小（GB，1 GB = 10⁹ 字节），保留 2 位小数。

```text
样例输入：32 8 128 8192 2
样例输出：1.07
```

<details>
<summary>提示</summary>

`2 * 层数 * KV头数 * 头维度 * token数 * 字节数 / 1e9`，前面的 2 是因为 K 和 V 各存一份。
</details>

<!-- exercise: online-softmax -->
### 练习 3：在线 softmax

第一行是一个 query 对各个 key 的注意力分数，第二行是对应的 value（每个 value 是一个数），第三行是块的大小 B。按 4.4 节的方法，每次只处理 B 个分数，用在线 softmax 算出注意力的输出（加权平均），保留 6 位小数。

```text
样例输入：
2 -1 0.5 3 1 -0.5 2.5 0
10 20 30 40 50 60 70 80
3
样例输出：44.332136
```

<details>
<summary>提示</summary>

维护 `run_max`、`run_sum`、`run_out` 三个量。每读入一块：先求新的最大值，再用 `math.exp(run_max - new_max)` 修正旧的 `run_sum` 和 `run_out`。注意一开始 `run_max` 是负无穷，`math.exp(float("-inf"))` 等于 0，正好可以用。
</details>

<!-- exercise: moe-balance -->
### 练习 4：MoE 的辅助损失

第一行是 token 数 T 和专家数 E，接下来 T 行，每行是路由器给这个 token 的 E 个概率。每个 token 分给概率最大的专家（top-1；概率相同时选编号小的）。按 4.5 节的公式输出辅助损失 `E · Σ fᵢ · Pᵢ`，保留 4 位小数。

```text
样例输入：
4 2
0.9 0.1
0.8 0.2
0.6 0.4
0.3 0.7
样例输出：1.1500
```

<details>
<summary>提示</summary>

fᵢ = 分给专家 i 的 token 数 / T；Pᵢ = 所有 token 在第 i 个概率上的平均值。`p.index(max(p))` 就是概率最大（并且编号最小）的专家。
</details>

<!-- exercise: gqa-why -->
### 练习 5：为什么用 GQA？

和标准的多头注意力（MHA）相比，GQA 的主要好处是什么？

- 让每个 query 头看到更多的 token
- 几个 query 头共用一组 K、V，KV 缓存小了好几倍，推理更快，效果几乎不变
- 不再需要位置编码
- 让注意力的计算从 O(n²) 变成 O(n)

<details>
<summary>答案与解析</summary>

GQA 把 query 头分组，每组共用一组 K、V，所以需要缓存的 K、V 少了很多（Llama 3 70B 少了 8 倍）。它不改变注意力的 O(n²) 计算，也不影响位置编码。
</details>

<!-- exercise: flash-exact -->
### 练习 6：FlashAttention 是近似吗？

关于 FlashAttention，下面哪个说法是对的？

- 它只计算一部分注意力分数，所以结果是近似的
- 它算出的结果和普通注意力完全相同，只是通过分块和在线 softmax 减少了显存读写
- 它把 softmax 换成了另一个更快的函数
- 它只能在推理时使用，不能用于训练

<details>
<summary>答案与解析</summary>

FlashAttention 是精确的：它改变的是计算顺序，不再把 n × n 的分数矩阵写进显存，而是分块计算并用在线 softmax 合并结果。训练和推理都在用。
</details>

## 4.9 延伸阅读

- Ainslie 等：《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》（2023）
- DeepSeek-AI：《DeepSeek-V2》（2024，MLA 和 DeepSeekMoE）、《DeepSeek-V3 Technical Report》（2024）
- Dao 等：《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》（2022）
- Peng 等：《YaRN: Efficient Context Window Extension of Large Language Models》（2023）
- Xiao 等：《Efficient Streaming Language Models with Attention Sinks》（2023）
- Gu、Dao：《Mamba: Linear-Time Sequence Modeling with Selective State Spaces》（2023）
- 下一课：[评测](../05-evaluation/zh-CN.md)
