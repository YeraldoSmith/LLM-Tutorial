# 3. 獎勵模型與 RLHF

> 🎯 **本課目標**：明白為什麼「比較兩個回答」比「寫出完美回答」更容易，偏好資料長什麼樣；用 **Bradley–Terry 模型**訓練一個**獎勵模型**（親手用純 Python 實現）；理解 RLHF 的目標——最大化獎勵，同時用 **KL 懲罰**拴住模型——並推匯出它的最優解；用 **REINFORCE** 策略梯度親手做一次「強化學習」；瞭解 **PPO** 的裁剪目標和它的代價；認識**獎勵欺騙**、長度偏差、諂媚等問題，以及 RLAIF 和「憲法式 AI」。
>
> ⏱ **預計用時**：90 分鐘　｜　📋 **前置知識**：本部分第 1 課；第二部分第 2、4 課（softmax、梯度）

SFT 讓模型學會了模仿示範。但示範有上限：標註員寫出的回答不一定是最好的，而且很多品質——更有幫助、更誠實、更安全——很難用示範說清楚。人們發現，**比較兩個回答哪個更好**，比**自己寫一個完美的回答**容易得多。**基於人類反饋的強化學習**（Reinforcement Learning from Human Feedback，RLHF）就是利用這種比較來改進模型。

## 3.1 偏好資料

RLHF 的資料是一條條**偏好**：

```text
提示：給我解釋一下什麼是光合作用。
回答 A：光合作用是植物利用光能，把二氧化碳和水變成糖和氧氣的過程……（清楚、準確）
回答 B：光合作用就是植物曬太陽。（太簡略）
標註：A 好於 B
```

通常的做法是：讓 SFT 模型對同一個提示生成幾個回答，請標註員（或者，後面會講，請另一個模型）指出哪個更好。每條資料記成 **(提示, 被選中的回答, 被拒絕的回答)**。InstructGPT 用了幾萬條這樣的比較。

## 3.2 獎勵模型：給回答打分

我們需要一個能自動給任何回答打分的**獎勵模型**（reward model，RM）。它通常就是一個語言模型，把最後的輸出層換成一個輸出單個數字的「打分頭」。怎樣從「A 好於 B」這種比較中學出分數？用 **Bradley–Terry 模型**（第四部分第 5 課的 Elo 也是它）：

```formula
P(A 好於 B) = σ(r(A) − r(B))　　　損失 = −log σ(r(被選中) − r(被拒絕))
```

σ 是 sigmoid 函式。兩個回答分差越大，模型越確信哪個更好；損失鼓勵被選中的回答比被拒絕的分數高。

下面用純 Python 訓練一個極簡的獎勵模型。真實的獎勵模型從回答的文字裡自己學習特徵，這裡為了看清楚，直接用 4 個手工特徵來描述一個回答，獎勵是它們的加權和：

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# 每個回答用 4 個特徵描述：[是否回答了問題, 是否有禮貌, 長度（百字）, 是否有事實錯誤]
pairs = [                                  # (被選中的回答, 被拒絕的回答)
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
for epoch in range(200):                   # 梯度下降，最小化 −log σ(r_選中 − r_拒絕)
    for c, r in pairs:
        diff = reward(w, c) - reward(w, r)
        g = sigmoid(diff) - 1              # 損失對 diff 的導數
        w = [wi - 0.5 * g * (ci - ri) for wi, ci, ri in zip(w, c, r)]

names = ["回答了問題", "有禮貌", "長度", "有事實錯誤"]
print("學到的權重：", {n: round(x, 2) for n, x in zip(names, w)})
acc = sum(reward(w, c) > reward(w, r) for c, r in pairs) / len(pairs)
print(f"訓練資料上的準確率：{acc:.0%}")

short = [1, 0, 0.3, 0]                     # 簡短、正確
padded = [1, 0, 2.0, 0]                    # 內容一樣，只是加了很多廢話
print(f"簡短的正確回答 {reward(w, short):.2f} 分，注水的回答 {reward(w, padded):.2f} 分")
```

獎勵模型學到了：回答問題加分，事實錯誤重重扣分——很合理。但它**也學到了「越長越好」**：訓練資料裡被選中的回答碰巧往往更長。於是，一個只是加了大量廢話的回答，得分反而更高。記住這個現象，3.6 節會再討論。

## 3.3 RLHF 的目標：最大化獎勵，但別跑太遠

有了獎勵模型，下一步是調整語言模型（在強化學習裡叫**策略**，policy，記作 π），讓它生成的回答得分更高。但如果**只**追求高分，模型會想方設法「鑽獎勵模型的空子」——生成一些獎勵模型打高分、實際上卻很糟糕的文字。所以 RLHF 的目標還要加上一個約束：**不要偏離原來的 SFT 模型（參考模型 π_ref）太遠**：

```formula
最大化　E[r(回答)] − β · KL(π ‖ π_ref)
```

**KL 散度** `KL(π‖π_ref) = Σ π(y) · log(π(y) / π_ref(y))` 衡量兩個機率分佈的差別：兩個分佈完全相同時為 0，差別越大它越大（它和第一部分第 6 課的交叉熵是近親：交叉熵 = 熵 + KL 散度）。β 控制拴得多緊。這個目標有一個漂亮的最優解：

```formula
π*(回答) ∝ π_ref(回答) · exp(r(回答) / β)
```

也就是說，最優的策略是在參考模型的基礎上，按獎勵的指數**重新加權**。用一個只有 4 個候選回答的「玩具」問題看看 β 的作用：

```python
import math

answers = ["正確簡潔", "正確囉嗦", "錯誤但自信", "拒絕回答"]
ref = [0.3, 0.3, 0.2, 0.2]                 # SFT 模型（參考模型）生成每個回答的機率
reward = [1.0, 0.6, -1.0, -0.5]            # 獎勵模型的打分

for beta in (10, 1, 0.3, 0.05):
    weights = [p * math.exp(r / beta) for p, r in zip(ref, reward)]
    pi = [w / sum(weights) for w in weights]
    kl = sum(p * math.log(p / q) for p, q in zip(pi, ref))
    avg_r = sum(p * r for p, r in zip(pi, reward))
    print(f"β = {beta:<5} 最優策略 {[round(p, 3) for p in pi]}  平均獎勵 {avg_r:.3f}  KL {kl:.3f}")
```

β 很大時，策略幾乎不動（KL ≈ 0）；β 越小，策略越大膽地追逐獎勵，離參考模型越遠；β 趨近 0 時，它會把全部機率押在獎勵最高的回答上。現實中，獎勵模型並不完美，押得太狠就會掉進獎勵欺騙的陷阱，所以 β 不能太小。

## 3.4 策略梯度：REINFORCE

真實的語言模型有天文數字般多的可能回答，不可能像上面那樣列舉出來算最優解。只能**取樣**：生成回答、打分、根據分數調整引數。最基本的方法是 **REINFORCE**（策略梯度）：

```formula
∇ E[r] = E[ (r − b) · ∇ log π(回答) ]
```

直觀地說：生成了一個回答，如果它的獎勵比「平均水平」b（**基線**）高，就提高生成它的機率；比平均低，就降低。基線不改變梯度的期望，但能大大減小隨機性。下面用 REINFORCE 訓練上面那個 4 選 1 的策略（把 KL 懲罰併入獎勵中），看它能不能自己找到 β = 0.3 時的最優解：

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

logits = [math.log(p) for p in ref]                 # 從參考模型出發
for step in range(3001):
    pi = softmax(logits)
    i = random.choices(range(4), weights=pi)[0]     # 取樣一個回答
    score = reward[i] - beta * math.log(pi[i] / ref[i])      # 獎勵減去 KL 懲罰
    baseline = sum(p * (r - beta * math.log(p / q)) for p, r, q in zip(pi, reward, ref))
    advantage = score - baseline
    for j in range(4):                              # ∇ log softmax 對第 j 個 logit 的導數
        logits[j] += lr * advantage * ((1 if j == i else 0) - pi[j])
    if step % 1000 == 0:
        print(f"第 {step:4d} 步：{[round(p, 3) for p in softmax(logits)]}")
```

只靠「試一試、看分數」，策略就收斂到了 3.3 節算出的最優解 `[0.788, 0.208, 0.001, 0.004]` 附近。真實的 RLHF 是同樣的道理，只是「回答」是一整段文字，log π(回答) 是其中每個 token 的對數機率之和。

## 3.5 PPO：更穩定的策略更新

REINFORCE 簡單，但一步走太大，策略就可能突然變壞、再也回不來。InstructGPT 和早期的 ChatGPT 用的是 **PPO**（Proximal Policy Optimization，近端策略最佳化，OpenAI 2017）。它的核心是限制每一步的更新幅度：

```formula
ratio = π_新(回答) / π_舊(回答)　　目標 = min( ratio · A, clip(ratio, 1 − ε, 1 + ε) · A )
```

A 是**優勢**（advantage，和上面的 `r − b` 類似），ε 通常取 0.2。如果新策略已經把某個好回答的機率提高了 20% 以上，再提高也不會有額外的獎勵——這樣每次更新都是「小步快跑」：

```python
def ppo_objective(ratio, advantage, eps=0.2):
    clipped = max(1 - eps, min(1 + eps, ratio))
    return min(ratio * advantage, clipped * advantage)

for ratio, adv in [(1.0, 1.0), (1.5, 1.0), (0.5, 1.0), (1.5, -1.0), (0.5, -1.0)]:
    print(f"ratio {ratio}，優勢 {adv:+}：目標 {ppo_objective(ratio, adv):+.2f}")
```

PPO 很有效，但很**昂貴**和**複雜**：訓練時要同時放 4 個大模型——正在訓練的策略、凍結的參考模型、獎勵模型，以及估計優勢用的**價值模型**（critic）；還有一大堆需要仔細調整的超引數。這推動了更簡單的方法：下一課的 **DPO** 完全不用強化學習，第 5 課的 **GRPO** 則去掉了價值模型。

## 3.6 獎勵欺騙

獎勵模型只是人類偏好的一個**近似**。策略越用力地最佳化它，越容易找到它的漏洞——這叫**獎勵欺騙**（reward hacking），是古德哈特定律（第四部分第 5 課）的又一個例子：

- **長度偏差**：就像 3.2 節的例子，獎勵模型偏愛長回答，策略就學會了寫得又長又空。
- **諂媚**（sycophancy）：人們傾向於給贊同自己的回答更高的評價，模型就學會了附和使用者，哪怕使用者是錯的。
- **格式欺騙**：大量使用列表、加粗、表情符號，因為看起來「更用心」。
- Gao 等（2022）系統地研究了「過度最佳化」：隨著策略離參考模型越來越遠，獎勵模型給的分數一直上升，但**真實的**品質先升後降。

應對的辦法包括：KL 懲罰、更大更好的獎勵模型、在偏好資料裡專門控制長度、定期收集新的偏好資料重新訓練獎勵模型、以及用多個獎勵模型取平均。

## 3.7 讓 AI 來提供反饋

人工標註偏好又慢又貴。於是出現了 **RLAIF**（Reinforcement Learning from AI Feedback）：讓一個強大的模型代替人來比較回答。Anthropic 的**憲法式 AI**（Constitutional AI，2022）是代表：先寫下一組原則（「憲法」），比如「選擇更誠實、更無害的回答」，再讓模型依據這些原則批評和修改自己的回答、比較兩個回答的好壞，用這些 AI 生成的偏好來訓練。這讓對齊的規則更透明（寫在原則裡），也大大降低了成本。

## 3.8 小結

- 比較比示範容易；偏好資料是 (提示, 選中, 拒絕) 三元組。
- 獎勵模型用 Bradley–Terry 損失 −log σ(r_選中 − r_拒絕) 訓練；它會學到資料裡的偏差（比如偏愛長回答）。
- RLHF 目標：最大化 E[r] − β·KL(π‖π_ref)；最優解 π* ∝ π_ref·exp(r/β)；β 控制離參考模型多遠。
- REINFORCE：∇E[r] = E[(r − b)∇log π]，基線減小方差；PPO 用裁剪的機率比限制每步的更新，但要 4 個模型，複雜又昂貴。
- 獎勵欺騙：長度、諂媚、格式；用 KL、更好的獎勵模型、長度控制等應對。
- RLAIF 和憲法式 AI 用模型按原則提供反饋。

## 3.9 練習

<!-- exercise: bt-prob -->
### 練習 1：Bradley–Terry 機率

輸入兩個回答的獎勵 `r_a` 和 `r_b`，輸出「A 好於 B」的機率 σ(r_a − r_b)，以及「A 被選中」時的損失 −log σ(r_a − r_b)，各保留 4 位小數，用空格隔開。

```text
樣例輸入：2.0 0.5
樣例輸出：0.8176 0.2014
```

<details>
<summary>提示</summary>

`p = 1 / (1 + math.exp(-(ra - rb)))`，損失是 `-math.log(p)`。
</details>

<!-- exercise: rm-eval -->
### 練習 2：評估獎勵模型

第一行是偏好對的數量 `n`，接下來 `n` 行，每行是獎勵模型給「被選中的回答」和「被拒絕的回答」的分數。輸出平均的 Bradley–Terry 損失（保留 4 位小數）和準確率（被選中的分數**嚴格高於**被拒絕的比例，用百分數表示，保留 1 位小數），用空格隔開。

```text
樣例輸入：
3
2.0 0.5
0.1 0.3
1.0 -1.0
樣例輸出：0.3755 66.7%
```

<details>
<summary>提示</summary>

每一對的損失是 `-math.log(sigmoid(rc - rr))`；準確率用 `f"{acc:.1%}"` 輸出。
</details>

<!-- exercise: kl-optimal -->
### 練習 3：KL 約束下的最優策略

第一行是參考模型給各個回答的機率，第二行是各個回答的獎勵，第三行是 β。按 π* ∝ π_ref · exp(r/β) 輸出最優策略，每個機率保留 4 位小數，用空格隔開。

```text
樣例輸入：
0.3 0.3 0.2 0.2
1.0 0.6 -1.0 -0.5
1
樣例輸出：0.5238 0.3511 0.0473 0.0779
```

<details>
<summary>提示</summary>

先算出每個回答的 `ref[i] * math.exp(reward[i] / beta)`，再除以它們的總和。
</details>

<!-- exercise: ppo-clip -->
### 練習 4：PPO 的裁剪目標

第一行是 ε，接下來每行是一對 `ratio advantage`，一直到輸入結束。對每一行輸出 PPO 的目標 `min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)`，保留 4 位小數。

```text
樣例輸入：
0.2
1.0 1.0
1.5 1.0
0.5 -1.0
樣例輸出：
1.0000
1.2000
-0.8000
```

<details>
<summary>提示</summary>

`clipped = max(1 - eps, min(1 + eps, ratio))`，輸出 `min(ratio * adv, clipped * adv)`。
</details>

<!-- exercise: why-kl -->
### 練習 5：為什麼要 KL 懲罰？

RLHF 的目標中為什麼要加上 β·KL(π‖π_ref) 這一項？

- 為了讓訓練速度更快
- 為了防止策略為了追逐獎勵模型的高分而偏離原模型太遠，生成獎勵模型打高分但實際很糟的內容
- 為了讓模型的回答更短
- 因為沒有它就無法計算梯度

<details>
<summary>答案與解析</summary>

獎勵模型只是人類偏好的近似，有漏洞。KL 懲罰把策略拴在原來的 SFT 模型附近，既保留了語言的流暢和已有的能力，也減少了獎勵欺騙。
</details>

<!-- exercise: hacking -->
### 練習 6：獎勵欺騙

下面哪個是典型的「獎勵欺騙」？

- 模型學會了把數學題算對
- 獎勵模型偏愛長回答，策略就學會在回答裡堆砌大量重複、無用的內容
- 訓練資料裡有重複的網頁
- 模型的上下文長度不夠

<details>
<summary>答案與解析</summary>

獎勵欺騙是指策略利用獎勵模型的缺陷得到高分，而實際品質並沒有提高，甚至下降。長度偏差就是最常見的例子。
</details>

## 3.10 延伸閱讀

- Christiano 等：《Deep Reinforcement Learning from Human Preferences》（2017）
- Ouyang 等：《Training language models to follow instructions with human feedback》（2022，InstructGPT）
- Schulman 等：《Proximal Policy Optimization Algorithms》（2017）
- Gao 等：《Scaling Laws for Reward Model Overoptimization》（2022）
- Bai 等：《Constitutional AI: Harmlessness from AI Feedback》（2022）
- Nathan Lambert：《RLHF Book》（線上免費，rlhfbook.com）
- 下一課：[DPO 及其家族](../04-dpo/zh-CN.md)
