# 4. DPO 及其家族：不用强化学习的偏好优化

> 🎯 **本课目标**：从上一课 RLHF 的最优解出发，**推导出 DPO 的损失函数**，理解「你的语言模型其实就是一个奖励模型」这句话；会计算 DPO 损失和**隐式奖励**；在一个玩具问题上用 DPO 训练，验证它和 RLHF 追求的是同一个最优解；知道 DPO 的实际做法（数据、β、参考模型、学习率）和常见问题；了解 IPO、KTO、ORPO、SimPO 等变体的思路。
>
> ⏱ **预计用时**：70 分钟　｜　📋 **前置知识**：本部分第 1、3 课

上一课的 RLHF 要先训练一个奖励模型，再用 PPO 做强化学习：同时要 4 个模型，要在训练中不断生成回答，超参数又多，非常难调。2023 年，斯坦福的 Rafailov 等人提出了 **DPO**（Direct Preference Optimization，直接偏好优化）：**直接在偏好数据上训练语言模型**，不需要单独的奖励模型，也不需要强化学习。它简单、稳定，很快成为开源模型最常用的偏好优化方法。

## 4.1 从 RLHF 的最优解出发

上一课得到了 KL 约束下的最优策略：

```formula
π*(y) = π_ref(y) · exp(r(y) / β) / Z
```

Z 是让所有概率加起来等于 1 的归一化常数（对某个提示是固定的数，但要对所有可能的回答求和，根本算不出来）。把这个式子**反过来**，用策略表示奖励：

```formula
r(y) = β · log( π*(y) / π_ref(y) ) + β · log Z
```

这说明：**只要知道最优策略和参考模型，奖励就确定了**（差一个常数）。现在把它代入奖励模型的 Bradley–Terry 公式。对同一个提示，被选中的回答 y_w、被拒绝的回答 y_l：

```formula
P(y_w 好于 y_l) = σ( r(y_w) − r(y_l) ) = σ( β · log(π(y_w)/π_ref(y_w)) − β · log(π(y_l)/π_ref(y_l)) )
```

算不出来的 β·log Z **正好消掉了**！于是可以直接用语言模型 π 的概率来写偏好的似然，最大化它（最小化负对数）就得到了 **DPO 损失**：

```formula
L_DPO = −log σ( β · [ log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l)) ] )
```

DPO 论文的副标题是「你的语言模型其实就是一个奖励模型」：`β · log(π(y)/π_ref(y))` 就是模型**隐含的奖励**。训练 DPO，就是在训练这个隐式奖励模型去拟合偏好——同时，策略本身也就是 RLHF 想要的那个最优策略。

## 4.2 计算 DPO 损失

DPO 需要的只是四个数：策略模型和参考模型分别给「被选中的回答」和「被拒绝的回答」的对数概率（每个回答是它所有 token 的对数概率之和）：

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def dpo(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    reward_w = beta * (pi_w - ref_w)           # 被选中回答的隐式奖励
    reward_l = beta * (pi_l - ref_l)           # 被拒绝回答的隐式奖励
    margin = reward_w - reward_l
    return -math.log(sigmoid(margin)), reward_w, reward_l

# 参考模型给两个回答的对数概率（训练过程中不变）
ref_w, ref_l = -42.0, -40.0
for label, pi_w, pi_l in [("训练开始（π = π_ref）", -42.0, -40.0),
                          ("选中的回答变得更可能", -38.0, -40.0),
                          ("拒绝的回答变得更不可能", -42.0, -50.0),
                          ("两者都下降，但拒绝的降得更多", -45.0, -60.0)]:
    loss, rw, rl = dpo(pi_w, pi_l, ref_w, ref_l)
    print(f"损失 {loss:.4f}，隐式奖励：选中 {rw:+.2f}，拒绝 {rl:+.2f}  ← {label}")
```

几点观察：

- 训练开始时 π = π_ref，两个隐式奖励都是 0，损失是 log 2 ≈ 0.693。
- 损失只关心**差值**：提高被选中回答的概率、或者降低被拒绝回答的概率，都能降低损失。
- 最后一行很常见：两个回答的概率**都下降了**，只是拒绝的降得更多。损失变小了，但被选中的好回答其实也变得不那么可能了——这是 DPO 的一个已知问题（4.5 节）。
- 对一个偏好对求梯度，会发现它的权重是 `σ(−margin)`：模型已经分得很清楚的对子（margin 很大）几乎不再贡献梯度，学习集中在**还没学会的**那些对子上。

## 4.3 在玩具问题上验证

回到上一课的 4 选 1 问题。这次我们**不告诉**模型奖励是多少，只给它 4000 条偏好数据——数据是按「真实奖励」和 Bradley–Terry 概率随机生成的（模拟人类标注）。用 DPO 损失做梯度下降，看看策略会去哪里：

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
true_reward = [1.0, 0.6, -1.0, -0.5]           # 模型看不到它，只看到据此生成的偏好
beta = 0.3

data = []                                      # (选中, 拒绝)
for _ in range(4000):
    i, j = random.sample(range(4), 2)
    if random.random() < sigmoid(true_reward[i] - true_reward[j]):
        data.append((i, j))
    else:
        data.append((j, i))

theta = [math.log(p) for p in ref]             # 策略的 logits，从参考模型出发
for step in range(601):
    pi = softmax(theta)
    grad, loss = [0.0] * 4, 0.0
    for w, l in data:
        z = beta * (math.log(pi[w] / ref[w]) - math.log(pi[l] / ref[l]))
        loss += -math.log(sigmoid(z))
        g = (1 - sigmoid(z)) * beta            # 这个对子的梯度权重 σ(−z)·β
        grad[w] += g
        grad[l] -= g
    theta = [t + 5.0 * g / len(data) for t, g in zip(theta, grad)]
    if step % 200 == 0:
        print(f"第 {step:3d} 步：损失 {loss / len(data):.4f}，策略 {[round(p, 3) for p in softmax(theta)]}")

best = [p * math.exp(r / beta) for p, r in zip(ref, true_reward)]
print("RLHF 的理论最优解：", [round(b / sum(best), 3) for b in best])
```

没有奖励模型，没有采样，只用偏好数据，DPO 就走到了 RLHF 理论最优解的附近（偏好数据是有限的随机样本，所以结果和理论值略有差别）。这正是 4.1 节推导的结论。

（在这个特殊的设定下，logits 的梯度恰好是 `g·(e_w − e_l)`：log softmax 中和 π 有关的项在「选中」和「拒绝」之间相互抵消了。）

## 4.4 DPO 的实际做法

- **数据**：和奖励模型一样，是 (提示, 选中, 拒绝) 三元组。常用的公开数据集有 UltraFeedback（用 GPT-4 给多个模型的回答打分）等。
- **参考模型**：通常就是 SFT 后的模型。参考模型的对数概率在训练中不变，可以事先算好存起来，训练时只需要一个模型在显存里。
- **β**：常用 0.1 左右。β 越大，越不敢离参考模型太远。
- **学习率**：很小，全参数训练常用 5e-7 左右；训练 1～3 个 epoch。
- 用 TRL 库（需要 GPU，API 以官方文档为准）：

```python norun
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")   # prompt / chosen / rejected
trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",                          # 一般先做过 SFT
    train_dataset=dataset,
    args=DPOConfig(output_dir="qwen3-dpo", beta=0.1),
)
trainer.train()
```

## 4.5 DPO 的问题

- **选中回答的概率也会下降**：就像 4.2 节最后一行，DPO 只要求「选中 − 拒绝」变大，两者可能一起下降，模型的输出可能变得奇怪。常见的补救是在损失里再加一点被选中回答的 SFT 损失。
- **离线数据**：偏好数据通常是事先收集好的，由别的模型（或早期的模型）生成。模型训练后的回答分布已经变了，旧数据就不那么「对口」了。**在线 / 迭代 DPO**——用当前的模型生成新回答、重新标注偏好、再训练——效果更好，也更接近 RLHF。
- **长度**：如果数据里被选中的回答往往更长，DPO 同样会学到「越长越好」（和奖励模型一样，3.6 节）。
- **过拟合**：偏好数据有噪声（人的判断并不总是一致），DPO 可能把某些对子的差距推到无穷大。

DPO 和 PPO 谁更好，是持续的争论。一般的看法是：DPO 简单、稳定、便宜，适合大多数场景；精心调好的在线 RL 上限更高。前沿的模型通常两者都用，而在数学、代码等有可靠的自动判分的任务上，越来越多地使用下一课的强化学习方法。

## 4.6 DPO 家族

DPO 之后出现了很多变体，都是在损失函数上做文章：

| 方法 | 年份 | 主要思路 |
|------|------|----------|
| **IPO** | 2023 | 把 −log σ 换成平方损失，让隐式奖励的差距停在一个目标值附近，防止过拟合 |
| **KTO** | 2024 | 不需要成对的数据，只要每个回答标上「好」或「不好」；灵感来自行为经济学的前景理论 |
| **ORPO** | 2024 | 不需要参考模型：在 SFT 损失上加一项「几率比」惩罚，一步同时完成 SFT 和偏好优化 |
| **SimPO** | 2024 | 不需要参考模型：用**平均**每个 token 的对数概率作为隐式奖励（消除长度偏差），并要求差距至少为 γ |

以 SimPO 为例，它的损失是：

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

# 选中的回答很长（总对数概率因此更低），拒绝的回答很短
print("按总和比较（像 DPO 不除以长度）：选中", -120.0, "拒绝", -30.0)
print("按每 token 平均比较：选中", -120.0 / 100, "拒绝", -30.0 / 20)
print(f"SimPO 损失：{simpo(-120.0, 100, -30.0, 20):.4f}")
```

长回答的总对数概率天然更低；除以长度后，比较的是「每个 token 平均有多自然」，长度就不再占便宜或吃亏了。

## 4.7 小结

- 从 RLHF 的最优解 π* ∝ π_ref·exp(r/β) 反解出奖励 r = β·log(π/π_ref) + 常数，代入 Bradley–Terry，常数消去，得到 DPO 损失。
- L_DPO = −log σ(β·[log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l))])；β·log(π/π_ref) 是隐式奖励。
- DPO 不需要奖励模型和强化学习采样，只需要偏好对和一个（可以预先算好的）参考模型；在玩具问题上它走向了 RLHF 的最优解。
- 实践：β≈0.1，学习率很小；问题：选中回答的概率也可能下降、离线数据不对口、长度偏差、过拟合；在线 / 迭代 DPO 更好。
- 变体：IPO（平方损失）、KTO（单个好坏标签）、ORPO 和 SimPO（不需要参考模型，SimPO 按长度归一化）。

## 4.8 练习

<!-- exercise: dpo-loss -->
### 练习 1：DPO 损失

输入一行 5 个数：策略模型给选中回答的对数概率、给拒绝回答的对数概率，参考模型给选中回答的对数概率、给拒绝回答的对数概率，以及 β。输出 DPO 损失和隐式奖励的差值（选中 − 拒绝），各保留 4 位小数，用空格隔开。

```text
样例输入：-38 -40 -42 -40 0.1
样例输出：0.5130 0.4000
```

<details>
<summary>提示</summary>

`margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))`，损失 `-math.log(1 / (1 + math.exp(-margin)))`。
</details>

<!-- exercise: dpo-batch -->
### 练习 2：一批数据的 DPO

第一行是对子的数量 `n` 和 β，接下来 `n` 行，每行 4 个数（顺序同练习 1：策略选中、策略拒绝、参考选中、参考拒绝）。输出平均 DPO 损失（4 位小数）和「隐式奖励差值大于 0」的对子所占的比例（百分数，1 位小数），用空格隔开。

```text
样例输入：
3 0.1
-38 -40 -42 -40
-42 -40 -42 -40
-45 -60 -42 -50
样例输出：0.5364 66.7%
```

<details>
<summary>提示</summary>

对每一对算出 margin 和损失，累加损失、统计 `margin > 0` 的个数。
</details>

<!-- exercise: simpo-loss -->
### 练习 3：SimPO 损失

输入一行 6 个数：选中回答的总对数概率和 token 数、拒绝回答的总对数概率和 token 数、β、γ。输出 SimPO 损失，保留 4 位小数。

```text
样例输入：-120 100 -30 20 2 1
样例输出：0.9130
```

<details>
<summary>提示</summary>

`margin = beta * (lw / nw - ll / nl) - gamma`，损失 `-math.log(sigmoid(margin))`。
</details>

<!-- exercise: implicit-reward -->
### 练习 4：隐式奖励排序

第一行是 β，接下来每行是一个回答：`名字 策略对数概率 参考对数概率`，一直到输入结束。按隐式奖励 β·(策略 − 参考) **从高到低**输出每个回答的名字和隐式奖励（保留 3 位小数）；奖励相同时按名字的字典序。

```text
样例输入：
0.5
a -10 -12
b -8 -8
c -20 -15
样例输出：
a 1.000
b 0.000
c -2.500
```

<details>
<summary>提示</summary>

把 (名字, 奖励) 放进列表，用 `sorted(items, key=lambda t: (-t[1], t[0]))` 排序。
</details>

<!-- exercise: dpo-needs -->
### 练习 5：DPO 不需要什么？

和 PPO 版的 RLHF 相比，DPO 不需要下面哪一样？

- 偏好数据（选中 / 拒绝的回答对）
- 参考模型（或它预先算好的对数概率）
- 单独训练的奖励模型和训练中的强化学习采样
- 一个要训练的语言模型

<details>
<summary>答案与解析</summary>

DPO 把奖励模型「藏」进了策略本身（隐式奖励），直接在偏好对上做监督式的训练，不需要单独的奖励模型，也不需要在训练中生成回答。它仍然需要偏好数据和参考模型。
</details>

<!-- exercise: dpo-beta -->
### 练习 6：β 的作用

在 DPO 中，β 越大意味着什么？

- 模型越大胆地偏离参考模型
- 隐式奖励的尺度越大，同样的概率变化就足以拉开差距，模型越不需要偏离参考模型——相当于 KL 约束更强
- 学习率越大
- 训练数据越多

<details>
<summary>答案与解析</summary>

β 来自 RLHF 目标中的 KL 惩罚系数：β 越大，约束越强，最优策略离参考模型越近。在 DPO 损失里，β 放大了对数概率比，模型只需很小的改变就能让损失下降。
</details>

## 4.9 延伸阅读

- Rafailov 等：《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》（2023）
- Azar 等：《A General Theoretical Paradigm to Understand Learning from Human Preferences》（2023，IPO）
- Ethayarajh 等：《KTO: Model Alignment as Prospect Theoretic Optimization》（2024）
- Hong 等：《ORPO: Monolithic Preference Optimization without Reference Model》（2024）
- Meng 等：《SimPO: Simple Preference Optimization with a Reference-Free Reward》（2024）
- 下一课：[推理模型与 GRPO](../05-reasoning-rl/zh-CN.md)
