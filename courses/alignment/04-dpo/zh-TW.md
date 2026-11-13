# 4. DPO 及其家族：不用強化學習的偏好最佳化

> 🎯 **本課目標**：從上一課 RLHF 的最優解出發，**推匯出 DPO 的損失函式**，理解「你的語言模型其實就是一個獎勵模型」這句話；會計算 DPO 損失和**隱式獎勵**；在一個玩具問題上用 DPO 訓練，驗證它和 RLHF 追求的是同一個最優解；知道 DPO 的實際做法（資料、β、參考模型、學習率）和常見問題；瞭解 IPO、KTO、ORPO、SimPO 等變體的思路。
>
> ⏱ **預計用時**：70 分鐘　｜　📋 **前置知識**：本部分第 1、3 課

上一課的 RLHF 要先訓練一個獎勵模型，再用 PPO 做強化學習：同時要 4 個模型，要在訓練中不斷生成回答，超引數又多，非常難調。2023 年，斯坦福的 Rafailov 等人提出了 **DPO**（Direct Preference Optimization，直接偏好最佳化）：**直接在偏好資料上訓練語言模型**，不需要單獨的獎勵模型，也不需要強化學習。它簡單、穩定，很快成為開源模型最常用的偏好最佳化方法。

## 4.1 從 RLHF 的最優解出發

上一課得到了 KL 約束下的最優策略：

```formula
π*(y) = π_ref(y) · exp(r(y) / β) / Z
```

Z 是讓所有機率加起來等於 1 的歸一化常數（對某個提示是固定的數，但要對所有可能的回答求和，根本算不出來）。把這個式子**反過來**，用策略表示獎勵：

```formula
r(y) = β · log( π*(y) / π_ref(y) ) + β · log Z
```

這說明：**只要知道最優策略和參考模型，獎勵就確定了**（差一個常數）。現在把它代入獎勵模型的 Bradley–Terry 公式。對同一個提示，被選中的回答 y_w、被拒絕的回答 y_l：

```formula
P(y_w 好於 y_l) = σ( r(y_w) − r(y_l) ) = σ( β · log(π(y_w)/π_ref(y_w)) − β · log(π(y_l)/π_ref(y_l)) )
```

算不出來的 β·log Z **正好消掉了**！於是可以直接用語言模型 π 的機率來寫偏好的似然，最大化它（最小化負對數）就得到了 **DPO 損失**：

```formula
L_DPO = −log σ( β · [ log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l)) ] )
```

DPO 論文的副標題是「你的語言模型其實就是一個獎勵模型」：`β · log(π(y)/π_ref(y))` 就是模型**隱含的獎勵**。訓練 DPO，就是在訓練這個隱式獎勵模型去擬合偏好——同時，策略本身也就是 RLHF 想要的那個最優策略。

## 4.2 計算 DPO 損失

DPO 需要的只是四個數：策略模型和參考模型分別給「被選中的回答」和「被拒絕的回答」的對數機率（每個回答是它所有 token 的對數機率之和）：

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def dpo(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    reward_w = beta * (pi_w - ref_w)           # 被選中回答的隱式獎勵
    reward_l = beta * (pi_l - ref_l)           # 被拒絕回答的隱式獎勵
    margin = reward_w - reward_l
    return -math.log(sigmoid(margin)), reward_w, reward_l

# 參考模型給兩個回答的對數機率（訓練過程中不變）
ref_w, ref_l = -42.0, -40.0
for label, pi_w, pi_l in [("訓練開始（π = π_ref）", -42.0, -40.0),
                          ("選中的回答變得更可能", -38.0, -40.0),
                          ("拒絕的回答變得更不可能", -42.0, -50.0),
                          ("兩者都下降，但拒絕的降得更多", -45.0, -60.0)]:
    loss, rw, rl = dpo(pi_w, pi_l, ref_w, ref_l)
    print(f"損失 {loss:.4f}，隱式獎勵：選中 {rw:+.2f}，拒絕 {rl:+.2f}  ← {label}")
```

幾點觀察：

- 訓練開始時 π = π_ref，兩個隱式獎勵都是 0，損失是 log 2 ≈ 0.693。
- 損失只關心**差值**：提高被選中回答的機率、或者降低被拒絕回答的機率，都能降低損失。
- 最後一行很常見：兩個回答的機率**都下降了**，只是拒絕的降得更多。損失變小了，但被選中的好回答其實也變得不那麼可能了——這是 DPO 的一個已知問題（4.5 節）。
- 對一個偏好對求梯度，會發現它的權重是 `σ(−margin)`：模型已經分得很清楚的對子（margin 很大）幾乎不再貢獻梯度，學習集中在**還沒學會的**那些對子上。

## 4.3 在玩具問題上驗證

回到上一課的 4 選 1 問題。這次我們**不告訴**模型獎勵是多少，只給它 4000 條偏好資料——資料是按「真實獎勵」和 Bradley–Terry 機率隨機生成的（模擬人類標註）。用 DPO 損失做梯度下降，看看策略會去哪裡：

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
true_reward = [1.0, 0.6, -1.0, -0.5]           # 模型看不到它，只看到據此生成的偏好
beta = 0.3

data = []                                      # (選中, 拒絕)
for _ in range(4000):
    i, j = random.sample(range(4), 2)
    if random.random() < sigmoid(true_reward[i] - true_reward[j]):
        data.append((i, j))
    else:
        data.append((j, i))

theta = [math.log(p) for p in ref]             # 策略的 logits，從參考模型出發
for step in range(601):
    pi = softmax(theta)
    grad, loss = [0.0] * 4, 0.0
    for w, l in data:
        z = beta * (math.log(pi[w] / ref[w]) - math.log(pi[l] / ref[l]))
        loss += -math.log(sigmoid(z))
        g = (1 - sigmoid(z)) * beta            # 這個對子的梯度權重 σ(−z)·β
        grad[w] += g
        grad[l] -= g
    theta = [t + 5.0 * g / len(data) for t, g in zip(theta, grad)]
    if step % 200 == 0:
        print(f"第 {step:3d} 步：損失 {loss / len(data):.4f}，策略 {[round(p, 3) for p in softmax(theta)]}")

best = [p * math.exp(r / beta) for p, r in zip(ref, true_reward)]
print("RLHF 的理論最優解：", [round(b / sum(best), 3) for b in best])
```

沒有獎勵模型，沒有采樣，只用偏好資料，DPO 就走到了 RLHF 理論最優解的附近（偏好資料是有限的隨機樣本，所以結果和理論值略有差別）。這正是 4.1 節推導的結論。

（在這個特殊的設定下，logits 的梯度恰好是 `g·(e_w − e_l)`：log softmax 中和 π 有關的項在「選中」和「拒絕」之間相互抵消了。）

## 4.4 DPO 的實際做法

- **資料**：和獎勵模型一樣，是 (提示, 選中, 拒絕) 三元組。常用的公開資料集有 UltraFeedback（用 GPT-4 給多個模型的回答打分）等。
- **參考模型**：通常就是 SFT 後的模型。參考模型的對數機率在訓練中不變，可以事先算好存起來，訓練時只需要一個模型在視訊記憶體裡。
- **β**：常用 0.1 左右。β 越大，越不敢離參考模型太遠。
- **學習率**：很小，全引數訓練常用 5e-7 左右；訓練 1～3 個 epoch。
- 用 TRL 庫（需要 GPU，API 以官方文件為準）：

```python norun
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")   # prompt / chosen / rejected
trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",                          # 一般先做過 SFT
    train_dataset=dataset,
    args=DPOConfig(output_dir="qwen3-dpo", beta=0.1),
)
trainer.train()
```

## 4.5 DPO 的問題

- **選中回答的機率也會下降**：就像 4.2 節最後一行，DPO 只要求「選中 − 拒絕」變大，兩者可能一起下降，模型的輸出可能變得奇怪。常見的補救是在損失裡再加一點被選中回答的 SFT 損失。
- **離線資料**：偏好資料通常是事先收集好的，由別的模型（或早期的模型）生成。模型訓練後的回答分佈已經變了，舊資料就不那麼「對口」了。**線上 / 迭代 DPO**——用當前的模型生成新回答、重新標註偏好、再訓練——效果更好，也更接近 RLHF。
- **長度**：如果資料裡被選中的回答往往更長，DPO 同樣會學到「越長越好」（和獎勵模型一樣，3.6 節）。
- **過擬合**：偏好資料有噪聲（人的判斷並不總是一致），DPO 可能把某些對子的差距推到無窮大。

DPO 和 PPO 誰更好，是持續的爭論。一般的看法是：DPO 簡單、穩定、便宜，適合大多數場景；精心調好的線上 RL 上限更高。前沿的模型通常兩者都用，而在數學、程式碼等有可靠的自動判分的任務上，越來越多地使用下一課的強化學習方法。

## 4.6 DPO 家族

DPO 之後出現了很多變體，都是在損失函式上做文章：

| 方法 | 年份 | 主要思路 |
|------|------|----------|
| **IPO** | 2023 | 把 −log σ 換成平方損失，讓隱式獎勵的差距停在一個目標值附近，防止過擬合 |
| **KTO** | 2024 | 不需要成對的資料，只要每個回答標上「好」或「不好」；靈感來自行為經濟學的前景理論 |
| **ORPO** | 2024 | 不需要參考模型：在 SFT 損失上加一項「機率比」懲罰，一步同時完成 SFT 和偏好最佳化 |
| **SimPO** | 2024 | 不需要參考模型：用**平均**每個 token 的對數機率作為隱式獎勵（消除長度偏差），並要求差距至少為 γ |

以 SimPO 為例，它的損失是：

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

# 選中的回答很長（總對數機率因此更低），拒絕的回答很短
print("按總和比較（像 DPO 不除以長度）：選中", -120.0, "拒絕", -30.0)
print("按每 token 平均比較：選中", -120.0 / 100, "拒絕", -30.0 / 20)
print(f"SimPO 損失：{simpo(-120.0, 100, -30.0, 20):.4f}")
```

長回答的總對數機率天然更低；除以長度後，比較的是「每個 token 平均有多自然」，長度就不再佔便宜或吃虧了。

## 4.7 小結

- 從 RLHF 的最優解 π* ∝ π_ref·exp(r/β) 反解出獎勵 r = β·log(π/π_ref) + 常數，代入 Bradley–Terry，常數消去，得到 DPO 損失。
- L_DPO = −log σ(β·[log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l))])；β·log(π/π_ref) 是隱式獎勵。
- DPO 不需要獎勵模型和強化學習取樣，只需要偏好對和一個（可以預先算好的）參考模型；在玩具問題上它走向了 RLHF 的最優解。
- 實踐：β≈0.1，學習率很小；問題：選中回答的機率也可能下降、離線資料不對口、長度偏差、過擬合；線上 / 迭代 DPO 更好。
- 變體：IPO（平方損失）、KTO（單個好壞標籤）、ORPO 和 SimPO（不需要參考模型，SimPO 按長度歸一化）。

## 4.8 練習

<!-- exercise: dpo-loss -->
### 練習 1：DPO 損失

輸入一行 5 個數：策略模型給選中回答的對數機率、給拒絕回答的對數機率，參考模型給選中回答的對數機率、給拒絕回答的對數機率，以及 β。輸出 DPO 損失和隱式獎勵的差值（選中 − 拒絕），各保留 4 位小數，用空格隔開。

```text
樣例輸入：-38 -40 -42 -40 0.1
樣例輸出：0.5130 0.4000
```

<details>
<summary>提示</summary>

`margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))`，損失 `-math.log(1 / (1 + math.exp(-margin)))`。
</details>

<!-- exercise: dpo-batch -->
### 練習 2：一批資料的 DPO

第一行是對子的數量 `n` 和 β，接下來 `n` 行，每行 4 個數（順序同練習 1：策略選中、策略拒絕、參考選中、參考拒絕）。輸出平均 DPO 損失（4 位小數）和「隱式獎勵差值大於 0」的對子所佔的比例（百分數，1 位小數），用空格隔開。

```text
樣例輸入：
3 0.1
-38 -40 -42 -40
-42 -40 -42 -40
-45 -60 -42 -50
樣例輸出：0.5364 66.7%
```

<details>
<summary>提示</summary>

對每一對算出 margin 和損失，累加損失、統計 `margin > 0` 的個數。
</details>

<!-- exercise: simpo-loss -->
### 練習 3：SimPO 損失

輸入一行 6 個數：選中回答的總對數機率和 token 數、拒絕回答的總對數機率和 token 數、β、γ。輸出 SimPO 損失，保留 4 位小數。

```text
樣例輸入：-120 100 -30 20 2 1
樣例輸出：0.9130
```

<details>
<summary>提示</summary>

`margin = beta * (lw / nw - ll / nl) - gamma`，損失 `-math.log(sigmoid(margin))`。
</details>

<!-- exercise: implicit-reward -->
### 練習 4：隱式獎勵排序

第一行是 β，接下來每行是一個回答：`名字 策略對數機率 參考對數機率`，一直到輸入結束。按隱式獎勵 β·(策略 − 參考) **從高到低**輸出每個回答的名字和隱式獎勵（保留 3 位小數）；獎勵相同時按名字的字典序。

```text
樣例輸入：
0.5
a -10 -12
b -8 -8
c -20 -15
樣例輸出：
a 1.000
b 0.000
c -2.500
```

<details>
<summary>提示</summary>

把 (名字, 獎勵) 放進列表，用 `sorted(items, key=lambda t: (-t[1], t[0]))` 排序。
</details>

<!-- exercise: dpo-needs -->
### 練習 5：DPO 不需要什麼？

和 PPO 版的 RLHF 相比，DPO 不需要下面哪一樣？

- 偏好資料（選中 / 拒絕的回答對）
- 參考模型（或它預先算好的對數機率）
- 單獨訓練的獎勵模型和訓練中的強化學習取樣
- 一個要訓練的語言模型

<details>
<summary>答案與解析</summary>

DPO 把獎勵模型「藏」進了策略本身（隱式獎勵），直接在偏好對上做監督式的訓練，不需要單獨的獎勵模型，也不需要在訓練中生成回答。它仍然需要偏好資料和參考模型。
</details>

<!-- exercise: dpo-beta -->
### 練習 6：β 的作用

在 DPO 中，β 越大意味著什麼？

- 模型越大膽地偏離參考模型
- 隱式獎勵的尺度越大，同樣的機率變化就足以拉開差距，模型越不需要偏離參考模型——相當於 KL 約束更強
- 學習率越大
- 訓練資料越多

<details>
<summary>答案與解析</summary>

β 來自 RLHF 目標中的 KL 懲罰係數：β 越大，約束越強，最優策略離參考模型越近。在 DPO 損失裡，β 放大了對數機率比，模型只需很小的改變就能讓損失下降。
</details>

## 4.9 延伸閱讀

- Rafailov 等：《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》（2023）
- Azar 等：《A General Theoretical Paradigm to Understand Learning from Human Preferences》（2023，IPO）
- Ethayarajh 等：《KTO: Model Alignment as Prospect Theoretic Optimization》（2024）
- Hong 等：《ORPO: Monolithic Preference Optimization without Reference Model》（2024）
- Meng 等：《SimPO: Simple Preference Optimization with a Reference-Free Reward》（2024）
- 下一課：[推理模型與 GRPO](../05-reasoning-rl/zh-CN.md)
