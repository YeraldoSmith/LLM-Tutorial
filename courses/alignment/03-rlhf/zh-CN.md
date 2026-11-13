# 3. 奖励模型与 RLHF

> 🎯 **本课目标**：明白为什么「比较两个回答」比「写出完美回答」更容易，偏好数据长什么样；用 **Bradley–Terry 模型**训练一个**奖励模型**（亲手用纯 Python 实现）；理解 RLHF 的目标——最大化奖励，同时用 **KL 惩罚**拴住模型——并推导出它的最优解；用 **REINFORCE** 策略梯度亲手做一次「强化学习」；了解 **PPO** 的裁剪目标和它的代价；认识**奖励欺骗**、长度偏差、谄媚等问题，以及 RLAIF 和「宪法式 AI」。
>
> ⏱ **预计用时**：90 分钟　｜　📋 **前置知识**：本部分第 1 课；第二部分第 2、4 课（softmax、梯度）

SFT 让模型学会了模仿示范。但示范有上限：标注员写出的回答不一定是最好的，而且很多品质——更有帮助、更诚实、更安全——很难用示范说清楚。人们发现，**比较两个回答哪个更好**，比**自己写一个完美的回答**容易得多。**基于人类反馈的强化学习**（Reinforcement Learning from Human Feedback，RLHF）就是利用这种比较来改进模型。

## 3.1 偏好数据

RLHF 的数据是一条条**偏好**：

```text
提示：给我解释一下什么是光合作用。
回答 A：光合作用是植物利用光能，把二氧化碳和水变成糖和氧气的过程……（清楚、准确）
回答 B：光合作用就是植物晒太阳。（太简略）
标注：A 好于 B
```

通常的做法是：让 SFT 模型对同一个提示生成几个回答，请标注员（或者，后面会讲，请另一个模型）指出哪个更好。每条数据记成 **(提示, 被选中的回答, 被拒绝的回答)**。InstructGPT 用了几万条这样的比较。

## 3.2 奖励模型：给回答打分

我们需要一个能自动给任何回答打分的**奖励模型**（reward model，RM）。它通常就是一个语言模型，把最后的输出层换成一个输出单个数字的「打分头」。怎样从「A 好于 B」这种比较中学出分数？用 **Bradley–Terry 模型**（第四部分第 5 课的 Elo 也是它）：

```formula
P(A 好于 B) = σ(r(A) − r(B))　　　损失 = −log σ(r(被选中) − r(被拒绝))
```

σ 是 sigmoid 函数。两个回答分差越大，模型越确信哪个更好；损失鼓励被选中的回答比被拒绝的分数高。

下面用纯 Python 训练一个极简的奖励模型。真实的奖励模型从回答的文字里自己学习特征，这里为了看清楚，直接用 4 个手工特征来描述一个回答，奖励是它们的加权和：

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# 每个回答用 4 个特征描述：[是否回答了问题, 是否有礼貌, 长度（百字）, 是否有事实错误]
pairs = [                                  # (被选中的回答, 被拒绝的回答)
    ([1, 1, 0.5, 0], [0, 1, 0.4, 0]),
    ([1, 0, 0.3, 0], [1, 0, 0.3, 1]),
    ([1, 1, 0.8, 0], [1, 0, 0.6, 0]),
    ([1, 0, 1.2, 0], [0, 0, 1.5, 0]),
    ([1, 1, 0.4, 0], [1, 1, 0.9, 1]),
    ([1, 0, 0.6, 0], [0, 1, 0.2, 0]),
]

def reward(w, f):
    return sum(wi * fi for wi, fi in zip(w, f))

w = [0.0] * 4
for epoch in range(200):                   # 梯度下降，最小化 −log σ(r_选中 − r_拒绝)
    for c, r in pairs:
        diff = reward(w, c) - reward(w, r)
        g = sigmoid(diff) - 1              # 损失对 diff 的导数
        w = [wi - 0.5 * g * (ci - ri) for wi, ci, ri in zip(w, c, r)]

names = ["回答了问题", "有礼貌", "长度", "有事实错误"]
print("学到的权重：", {n: round(x, 2) for n, x in zip(names, w)})
acc = sum(reward(w, c) > reward(w, r) for c, r in pairs) / len(pairs)
print(f"训练数据上的准确率：{acc:.0%}")

short = [1, 0, 0.3, 0]                     # 简短、正确
padded = [1, 0, 2.0, 0]                    # 内容一样，只是加了很多废话
print(f"简短的正确回答 {reward(w, short):.2f} 分，注水的回答 {reward(w, padded):.2f} 分")
```

奖励模型学到了：回答问题加分，事实错误重重扣分——很合理。但它**也学到了「越长越好」**：训练数据里被选中的回答碰巧往往更长。于是，一个只是加了大量废话的回答，得分反而更高。记住这个现象，3.6 节会再讨论。

## 3.3 RLHF 的目标：最大化奖励，但别跑太远

有了奖励模型，下一步是调整语言模型（在强化学习里叫**策略**，policy，记作 π），让它生成的回答得分更高。但如果**只**追求高分，模型会想方设法「钻奖励模型的空子」——生成一些奖励模型打高分、实际上却很糟糕的文字。所以 RLHF 的目标还要加上一个约束：**不要偏离原来的 SFT 模型（参考模型 π_ref）太远**：

```formula
最大化　E[r(回答)] − β · KL(π ‖ π_ref)
```

**KL 散度** `KL(π‖π_ref) = Σ π(y) · log(π(y) / π_ref(y))` 衡量两个概率分布的差别：两个分布完全相同时为 0，差别越大它越大（它和第一部分第 6 课的交叉熵是近亲：交叉熵 = 熵 + KL 散度）。β 控制拴得多紧。这个目标有一个漂亮的最优解：

```formula
π*(回答) ∝ π_ref(回答) · exp(r(回答) / β)
```

也就是说，最优的策略是在参考模型的基础上，按奖励的指数**重新加权**。用一个只有 4 个候选回答的「玩具」问题看看 β 的作用：

```python
import math

answers = ["正确简洁", "正确啰嗦", "错误但自信", "拒绝回答"]
ref = [0.3, 0.3, 0.2, 0.2]                 # SFT 模型（参考模型）生成每个回答的概率
reward = [1.0, 0.6, -1.0, -0.5]            # 奖励模型的打分

for beta in (10, 1, 0.3, 0.05):
    weights = [p * math.exp(r / beta) for p, r in zip(ref, reward)]
    pi = [w / sum(weights) for w in weights]
    kl = sum(p * math.log(p / q) for p, q in zip(pi, ref))
    avg_r = sum(p * r for p, r in zip(pi, reward))
    print(f"β = {beta:<5} 最优策略 {[round(p, 3) for p in pi]}  平均奖励 {avg_r:.3f}  KL {kl:.3f}")
```

β 很大时，策略几乎不动（KL ≈ 0）；β 越小，策略越大胆地追逐奖励，离参考模型越远；β 趋近 0 时，它会把全部概率押在奖励最高的回答上。现实中，奖励模型并不完美，押得太狠就会掉进奖励欺骗的陷阱，所以 β 不能太小。

## 3.4 策略梯度：REINFORCE

真实的语言模型有天文数字般多的可能回答，不可能像上面那样枚举出来算最优解。只能**采样**：生成回答、打分、根据分数调整参数。最基本的方法是 **REINFORCE**（策略梯度）：

```formula
∇ E[r] = E[ (r − b) · ∇ log π(回答) ]
```

直观地说：生成了一个回答，如果它的奖励比「平均水平」b（**基线**）高，就提高生成它的概率；比平均低，就降低。基线不改变梯度的期望，但能大大减小随机性。下面用 REINFORCE 训练上面那个 4 选 1 的策略（把 KL 惩罚并入奖励中），看它能不能自己找到 β = 0.3 时的最优解：

```python
import math
import random

random.seed(0)
ref = [0.3, 0.3, 0.2, 0.2]
reward = [1.0, 0.6, -1.0, -0.5]
beta, lr = 0.3, 0.1

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

logits = [math.log(p) for p in ref]                 # 从参考模型出发
for step in range(3001):
    pi = softmax(logits)
    i = random.choices(range(4), weights=pi)[0]     # 采样一个回答
    score = reward[i] - beta * math.log(pi[i] / ref[i])      # 奖励减去 KL 惩罚
    baseline = sum(p * (r - beta * math.log(p / q)) for p, r, q in zip(pi, reward, ref))
    advantage = score - baseline
    for j in range(4):                              # ∇ log softmax 对第 j 个 logit 的导数
        logits[j] += lr * advantage * ((1 if j == i else 0) - pi[j])
    if step % 1000 == 0:
        print(f"第 {step:4d} 步：{[round(p, 3) for p in softmax(logits)]}")
```

只靠「试一试、看分数」，策略就收敛到了 3.3 节算出的最优解 `[0.788, 0.208, 0.001, 0.004]` 附近。真实的 RLHF 是同样的道理，只是「回答」是一整段文字，log π(回答) 是其中每个 token 的对数概率之和。

## 3.5 PPO：更稳定的策略更新

REINFORCE 简单，但一步走太大，策略就可能突然变坏、再也回不来。InstructGPT 和早期的 ChatGPT 用的是 **PPO**（Proximal Policy Optimization，近端策略优化，OpenAI 2017）。它的核心是限制每一步的更新幅度：

```formula
ratio = π_新(回答) / π_旧(回答)　　目标 = min( ratio · A, clip(ratio, 1 − ε, 1 + ε) · A )
```

A 是**优势**（advantage，和上面的 `r − b` 类似），ε 通常取 0.2。如果新策略已经把某个好回答的概率提高了 20% 以上，再提高也不会有额外的奖励——这样每次更新都是「小步快跑」：

```python
def ppo_objective(ratio, advantage, eps=0.2):
    clipped = max(1 - eps, min(1 + eps, ratio))
    return min(ratio * advantage, clipped * advantage)

for ratio, adv in [(1.0, 1.0), (1.5, 1.0), (0.5, 1.0), (1.5, -1.0), (0.5, -1.0)]:
    print(f"ratio {ratio}，优势 {adv:+}：目标 {ppo_objective(ratio, adv):+.2f}")
```

PPO 很有效，但很**昂贵**和**复杂**：训练时要同时放 4 个大模型——正在训练的策略、冻结的参考模型、奖励模型，以及估计优势用的**价值模型**（critic）；还有一大堆需要仔细调整的超参数。这推动了更简单的方法：下一课的 **DPO** 完全不用强化学习，第 5 课的 **GRPO** 则去掉了价值模型。

## 3.6 奖励欺骗

奖励模型只是人类偏好的一个**近似**。策略越用力地优化它，越容易找到它的漏洞——这叫**奖励欺骗**（reward hacking），是古德哈特定律（第四部分第 5 课）的又一个例子：

- **长度偏差**：就像 3.2 节的例子，奖励模型偏爱长回答，策略就学会了写得又长又空。
- **谄媚**（sycophancy）：人们倾向于给赞同自己的回答更高的评价，模型就学会了附和用户，哪怕用户是错的。
- **格式欺骗**：大量使用列表、加粗、表情符号，因为看起来「更用心」。
- Gao 等（2022）系统地研究了「过度优化」：随着策略离参考模型越来越远，奖励模型给的分数一直上升，但**真实的**质量先升后降。

应对的办法包括：KL 惩罚、更大更好的奖励模型、在偏好数据里专门控制长度、定期收集新的偏好数据重新训练奖励模型、以及用多个奖励模型取平均。

## 3.7 让 AI 来提供反馈

人工标注偏好又慢又贵。于是出现了 **RLAIF**（Reinforcement Learning from AI Feedback）：让一个强大的模型代替人来比较回答。Anthropic 的**宪法式 AI**（Constitutional AI，2022）是代表：先写下一组原则（「宪法」），比如「选择更诚实、更无害的回答」，再让模型依据这些原则批评和修改自己的回答、比较两个回答的好坏，用这些 AI 生成的偏好来训练。这让对齐的规则更透明（写在原则里），也大大降低了成本。

## 3.8 小结

- 比较比示范容易；偏好数据是 (提示, 选中, 拒绝) 三元组。
- 奖励模型用 Bradley–Terry 损失 −log σ(r_选中 − r_拒绝) 训练；它会学到数据里的偏差（比如偏爱长回答）。
- RLHF 目标：最大化 E[r] − β·KL(π‖π_ref)；最优解 π* ∝ π_ref·exp(r/β)；β 控制离参考模型多远。
- REINFORCE：∇E[r] = E[(r − b)∇log π]，基线减小方差；PPO 用裁剪的概率比限制每步的更新，但要 4 个模型，复杂又昂贵。
- 奖励欺骗：长度、谄媚、格式；用 KL、更好的奖励模型、长度控制等应对。
- RLAIF 和宪法式 AI 用模型按原则提供反馈。

## 3.9 练习

<!-- exercise: bt-prob -->
### 练习 1：Bradley–Terry 概率

输入两个回答的奖励 `r_a` 和 `r_b`，输出「A 好于 B」的概率 σ(r_a − r_b)，以及「A 被选中」时的损失 −log σ(r_a − r_b)，各保留 4 位小数，用空格隔开。

```text
样例输入：2.0 0.5
样例输出：0.8176 0.2014
```

<details>
<summary>提示</summary>

`p = 1 / (1 + math.exp(-(ra - rb)))`，损失是 `-math.log(p)`。
</details>

<!-- exercise: rm-eval -->
### 练习 2：评估奖励模型

第一行是偏好对的数量 `n`，接下来 `n` 行，每行是奖励模型给「被选中的回答」和「被拒绝的回答」的分数。输出平均的 Bradley–Terry 损失（保留 4 位小数）和准确率（被选中的分数**严格高于**被拒绝的比例，用百分数表示，保留 1 位小数），用空格隔开。

```text
样例输入：
3
2.0 0.5
0.1 0.3
1.0 -1.0
样例输出：0.3755 66.7%
```

<details>
<summary>提示</summary>

每一对的损失是 `-math.log(sigmoid(rc - rr))`；准确率用 `f"{acc:.1%}"` 输出。
</details>

<!-- exercise: kl-optimal -->
### 练习 3：KL 约束下的最优策略

第一行是参考模型给各个回答的概率，第二行是各个回答的奖励，第三行是 β。按 π* ∝ π_ref · exp(r/β) 输出最优策略，每个概率保留 4 位小数，用空格隔开。

```text
样例输入：
0.3 0.3 0.2 0.2
1.0 0.6 -1.0 -0.5
1
样例输出：0.5238 0.3511 0.0473 0.0779
```

<details>
<summary>提示</summary>

先算出每个回答的 `ref[i] * math.exp(reward[i] / beta)`，再除以它们的总和。
</details>

<!-- exercise: ppo-clip -->
### 练习 4：PPO 的裁剪目标

第一行是 ε，接下来每行是一对 `ratio advantage`，一直到输入结束。对每一行输出 PPO 的目标 `min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)`，保留 4 位小数。

```text
样例输入：
0.2
1.0 1.0
1.5 1.0
0.5 -1.0
样例输出：
1.0000
1.2000
-0.8000
```

<details>
<summary>提示</summary>

`clipped = max(1 - eps, min(1 + eps, ratio))`，输出 `min(ratio * adv, clipped * adv)`。
</details>

<!-- exercise: why-kl -->
### 练习 5：为什么要 KL 惩罚？

RLHF 的目标中为什么要加上 β·KL(π‖π_ref) 这一项？

- 为了让训练速度更快
- 为了防止策略为了追逐奖励模型的高分而偏离原模型太远，生成奖励模型打高分但实际很糟的内容
- 为了让模型的回答更短
- 因为没有它就无法计算梯度

<details>
<summary>答案与解析</summary>

奖励模型只是人类偏好的近似，有漏洞。KL 惩罚把策略拴在原来的 SFT 模型附近，既保留了语言的流畅和已有的能力，也减少了奖励欺骗。
</details>

<!-- exercise: hacking -->
### 练习 6：奖励欺骗

下面哪个是典型的「奖励欺骗」？

- 模型学会了把数学题算对
- 奖励模型偏爱长回答，策略就学会在回答里堆砌大量重复、无用的内容
- 训练数据里有重复的网页
- 模型的上下文长度不够

<details>
<summary>答案与解析</summary>

奖励欺骗是指策略利用奖励模型的缺陷得到高分，而实际质量并没有提高，甚至下降。长度偏差就是最常见的例子。
</details>

## 3.10 延伸阅读

- Christiano 等：《Deep Reinforcement Learning from Human Preferences》（2017）
- Ouyang 等：《Training language models to follow instructions with human feedback》（2022，InstructGPT）
- Schulman 等：《Proximal Policy Optimization Algorithms》（2017）
- Gao 等：《Scaling Laws for Reward Model Overoptimization》（2022）
- Bai 等：《Constitutional AI: Harmlessness from AI Feedback》（2022）
- Nathan Lambert：《RLHF Book》（在线免费，rlhfbook.com）
- 下一课：[DPO 及其家族](../04-dpo/zh-CN.md)
