# 1. 指令微調：從「續寫機器」到「助手」

> 🎯 **本課目標**：分清**基礎模型**和**對話模型**；瞭解後訓練的整體流程（SFT → 偏好最佳化 → 強化學習）；理解指令微調（**SFT**）的資料從哪裡來，為什麼「少而精」比「多而雜」更好；看懂並親手實現**對話模板**（ChatML、Llama 3）；理解為什麼訓練時只在**助手的回答**上計算損失，並實現損失掩碼；知道 SFT 的常見做法、風險和「蒸餾」。
>
> ⏱ **預計用時**：70 分鐘　｜　📋 **前置知識**：第三部分；第四部分第 1、5 課

預訓練完的模型，叫**基礎模型**（base model）。它讀遍了網際網路，知識淵博，但它只會做一件事：**續寫**。你問它「法國的首都是哪裡？」，它可能接著寫「德國的首都是哪裡？義大利的首都是哪裡？」——因為網上的題目列表就長這樣。要把它變成一個會回答問題、聽從指令的**助手**，還需要**後訓練**（post-training）。

## 1.1 基礎模型和對話模型

同一個問題，兩種模型的典型反應：

```text
輸入：怎樣煮一個溏心蛋？

基礎模型（續寫）：
怎樣煮一個溏心蛋？怎樣煎出完美的荷包蛋？怎樣做蛋炒飯？本期我們為大家整理了 10 個雞蛋的家常做法……

對話模型（回答）：
1. 水燒開後，用勺子把冷藏的雞蛋輕輕放進鍋裡；
2. 保持小火微沸，煮 6 分 30 秒；
3. 立刻撈出放進冰水裡冷卻，剝殼即可。
```

基礎模型並非不懂怎麼煮蛋——它的知識都在，只是不知道「現在該回答問題」。後訓練要教給它的，主要是**格式和行為**：什麼時候該回答、怎樣組織答案、怎樣拒絕有害的請求、怎樣在多輪對話裡保持一致。

## 1.2 後訓練的流程

2022 年 OpenAI 的 **InstructGPT** 確立了經典的三步，今天的流程在此基礎上發展而來：

```text
基礎模型
  │ ① 監督微調（SFT）：用「指令 → 好回答」的示範資料訓練        ← 本課
  ▼
SFT 模型
  │ ② 偏好最佳化：讓模型更符合人的偏好（RLHF 第 3 課、DPO 第 4 課）
  ▼
對齊的對話模型
  │ ③ 可驗證獎勵的強化學習：在數學、程式碼等能自動判對錯的任務上練習推理（第 5 課）
  ▼
推理模型
```

InstructGPT 有個著名的結果：經過這套流程的 13 億引數模型，在人類評價中比 1750 億引數的原始 GPT-3 還受歡迎。**後訓練用的計算量只是預訓練的零頭，卻決定了模型用起來的感覺。**

## 1.3 SFT 資料：少而精

**監督微調**（Supervised Fine-Tuning，SFT）的訓練方法和預訓練完全一樣——預測下一個 token——只是資料換成了**示範對話**：

```text
使用者：把「天氣真好」翻譯成英文。
助手：The weather is really nice.
```

這些資料從哪裡來？

- **人工編寫**：請標註員寫問題和高品質的回答。品質最高，但很貴。
- **從更強的模型蒸餾**：讓強大的模型來寫回答。斯坦福的 Alpaca（2023）只花了幾百美元，用 OpenAI 的模型生成了 5.2 萬條指令資料，就把 Llama 7B 變成了一個像樣的助手。（注意：很多商業模型的使用條款禁止用它們的輸出訓練競爭模型。）
- **把已有的資料改寫成對話**：問答網站、教科書習題等。

一個重要的發現來自 Meta 的 **LIMA**（2023）：只用 **1000 條**精心挑選的示範，就能讓一個 650 億引數的基礎模型成為相當不錯的助手。研究者提出了「**表面對齊假說**」：模型的知識和能力幾乎全部來自預訓練，SFT 主要是教會它**用什麼格式、以什麼風格**與人交流。所以 SFT 資料的**品質和多樣性**遠比數量重要——一條錯誤或敷衍的示範，模型也會認真學下來。

## 1.4 對話模板

對話由多輪、多個角色（系統、使用者、助手）組成，而模型只認識一串 token。**對話模板**（chat template）規定了怎樣把對話「壓平」成一段文字，並用**特殊 token** 標出每一段的開始和結束。下面是兩種常見的模板：

```python
def chatml(messages, add_generation_prompt=True):
    """ChatML 格式（Qwen 等模型使用）。"""
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"          # 提示模型：輪到助手說話了
    return text

def llama3(messages, add_generation_prompt=True):
    """Llama 3 的格式。"""
    text = "<|begin_of_text|>"
    for m in messages:
        text += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    if add_generation_prompt:
        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text

conversation = [
    {"role": "system", "content": "你是一個簡潔的助手。"},
    {"role": "user", "content": "1 + 1 等於幾？"},
]
print(chatml(conversation))
print("------")
print(llama3(conversation))
```

- `<|im_start|>`、`<|eot_id|>` 等是分詞器裡專門保留的**特殊 token**，普通文字永遠不會被切成它們，所以模型能可靠地分辨「誰在說話」。
- 模型生成時，看到 `<|im_start|>assistant\n` 就知道該寫回答了；寫完後它會生成 `<|im_end|>`（或 `<|eot_id|>`），程式看到這個 token 就停止生成。
- **用錯模板是最常見的錯誤之一**：用 A 模板訓練的模型，換成 B 模板來對話，效果會大打折扣，甚至胡言亂語。Hugging Face 的分詞器裡儲存了模型自己的模板，用 `tokenizer.apply_chat_template(messages)` 呼叫即可，不要自己拼。
- 「系統」訊息用來設定助手的身份、語氣和規則，第七部分會詳細講怎樣寫好它。

## 1.5 只在回答上計算損失

訓練時，我們希望模型學會**怎樣回答**，而不是學會**怎樣提問**——使用者的話是給定的，不需要模型來「預測」。所以計算損失時，要把系統訊息和使用者訊息部分**遮住**，只在助手的回答上計算。實現方法是：把不需要算損失的位置的標籤設成一個特殊值（PyTorch 的約定是 **-100**），交叉熵會自動跳過這些位置：

```python
# 一段對話切成的 token（這裡用詞代替 token），以及每個 token 屬於誰
tokens = ["<|im_start|>", "user", "你好", "<|im_end|>",
          "<|im_start|>", "assistant", "你好！", "有什麼", "可以", "幫你？", "<|im_end|>"]
roles  = ["user"] * 6 + ["assistant"] * 5           # 助手的回答從「你好！」開始，包括最後的結束符
ids = list(range(100, 100 + len(tokens)))           # 假裝的 token id

labels = [i if r == "assistant" else -100 for i, r in zip(ids, roles)]
for t, l in zip(tokens, labels):
    print(f"標籤 {l:>5}   {t}")
print("參與計算損失的 token：", sum(l != -100 for l in labels), "/", len(labels))
```

注意結束符 `<|im_end|>` 也要計算損失：模型必須學會**在回答完時停下來**，否則它會一直寫下去。

在 PyTorch 裡，`cross_entropy` 的 `ignore_index` 引數預設就是 -100：

```python requires=torch
import torch
import torch.nn.functional as F

torch.manual_seed(0)
vocab, T = 50, 8
logits = torch.randn(T, vocab)                         # 模型對 8 個位置的預測
labels = torch.tensor([-100, -100, -100, 7, 12, 3, 9, -100])   # 只有 4 個位置是助手的回答

masked = F.cross_entropy(logits, labels)               # 自動跳過 -100
manual = F.cross_entropy(logits[3:7], labels[3:7])     # 手動只取那 4 個位置
print(f"帶掩碼的損失 {masked.item():.4f}，只算回答部分 {manual.item():.4f}")
```

## 1.6 SFT 的實際做法

用 Hugging Face 的 **TRL** 庫，SFT 只需要幾行程式碼（需要 GPU 和聯網下載模型，API 可能隨版本變化，以官方文件為準）：

```python norun
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")   # 每條資料是一段 messages 對話
trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",                                 # 一個小的基礎模型
    train_dataset=dataset,
    args=SFTConfig(output_dir="qwen3-sft"),
)
trainer.train()
```

一些經驗：

- **學習率比預訓練小得多**（全引數微調常用 1e-5 左右），只訓練 1～3 個 epoch。
- **打包**：和預訓練一樣，可以把多條短對話拼進一個序列裡，提高 GPU 利用率（第四部分第 1 課）。
- **災難性遺忘**：在一個窄領域的資料上訓練太久，模型會忘掉其他能力。常見的對策是混入一部分通用的對話資料，並在多種評測上檢查（第四部分第 5 課）。
- **訓練太多會「過擬合」到風格上**：回答開始千篇一律，甚至編造內容來模仿訓練資料中自信的語氣。
- **蒸餾推理能力**：DeepSeek-R1（2025）釋出時，團隊用 R1 生成的約 80 萬條帶有長推理過程的資料，對 Qwen 和 Llama 的較小模型做了 SFT，小模型的數學和程式碼能力大幅提升。對小模型來說，從強模型「抄作業」往往比自己用強化學習摸索更有效。

引數太多、視訊記憶體不夠時，可以只訓練很少的一部分引數——這就是下一課的 **LoRA**。

## 1.7 小結

- 基礎模型只會續寫；後訓練教會它作為助手的格式和行為。
- 後訓練流程：SFT → 偏好最佳化（RLHF、DPO）→ 可驗證獎勵的強化學習；計算量遠小於預訓練。
- SFT 用示範對話訓練，方法和預訓練相同；資料來自人工、強模型蒸餾、改寫；品質和多樣性比數量重要（LIMA：1000 條）。
- 對話模板用特殊 token 標出角色和邊界；必須使用模型自己的模板（`apply_chat_template`）。
- 只在助手的回答（包括結束符）上計算損失：其餘位置的標籤設為 -100。
- SFT 學習率小、輪數少，注意災難性遺忘；用強模型的推理過程做 SFT（蒸餾）對小模型很有效。

## 1.8 練習

<!-- exercise: chatml -->
### 練習 1：ChatML 模板

第一行輸入訊息條數 `n`，接下來 `n` 行，每行格式是 `角色: 內容`（角色和內容之間是一個冒號加一個空格）。按 1.4 節的 ChatML 格式輸出整段文字，並在最後加上 `<|im_start|>assistant`（生成提示）。

```text
樣例輸入：
2
system: Be brief.
user: Hi!
樣例輸出：
<|im_start|>system
Be brief.<|im_end|>
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
```

<details>
<summary>提示</summary>

`role, content = line.split(": ", 1)`。每條訊息輸出 `<|im_start|>{role}\n{content}<|im_end|>\n`；最後輸出 `<|im_start|>assistant` 一行。用 `print(..., end="")` 可以避免多出空行。
</details>

<!-- exercise: masked-loss -->
### 練習 2：只算回答的損失

第一行是每個 token 的角色，用字母表示（`U` 是使用者，`A` 是助手），用空格隔開；第二行是每個 token 的損失。輸出只在助手 token 上計算的平均損失，保留 4 位小數。（至少有一個助手 token。）

```text
樣例輸入：
U U U A A A A
5.1 4.8 6.0 1.2 0.8 2.0 0.4
樣例輸出：1.1000
```

<details>
<summary>提示</summary>

用 `zip` 同時遍歷角色和損失，只累加角色為 `A` 的損失，再除以助手 token 的個數。
</details>

<!-- exercise: pack-sft -->
### 練習 3：打包對話

第一行是序列的最大長度 `L`，第二行是每條對話的 token 數（每條都不超過 L）。**按順序**打包：當前序列放不下下一條對話時，就開啟一個新序列。輸出需要的序列數，以及所有序列的平均填充率（對話的總 token 數 ÷ (序列數 × L)），填充率用百分數表示、保留 1 位小數。

```text
樣例輸入：
10
4 5 3 8 2 6
樣例輸出：4 70.0%
```

<details>
<summary>提示</summary>

用一個變數 `used` 記錄當前序列已經用了多少；`used + n > L` 時序列數加 1，`used = n`，否則 `used += n`。別忘了第一條對話也要開啟一個序列。
</details>

<!-- exercise: why-mask -->
### 練習 4：為什麼要遮住使用者的話？

SFT 時，為什麼通常只在助手的回答上計算損失？

- 因為使用者的話太短，算不出損失
- 因為我們要訓練模型怎樣回答，而不是怎樣模仿使用者提問；使用者的話是給定的輸入
- 因為對使用者的話計算損失會讓訓練變慢 10 倍
- 因為使用者的話裡沒有特殊 token

<details>
<summary>答案與解析</summary>

使用者和系統的話是對話的「條件」，模型不需要學會生成它們。如果也在上面計算損失，模型會花力氣去學習模仿使用者，並且可能在回答時自己「替使用者說話」。
</details>

<!-- exercise: lima -->
### 練習 5：LIMA 的啟示

LIMA（2023）只用 1000 條精選資料就把基礎模型訓練成了不錯的助手。這說明了什麼？

- 預訓練其實不重要
- 模型的知識和能力主要來自預訓練，SFT 主要教會它交流的格式和風格，所以資料品質比數量更重要
- 1000 條資料適用於任何任務，再多就沒有用了
- SFT 不需要計算損失

<details>
<summary>答案與解析</summary>

這就是「表面對齊假說」：基礎模型已經具備了知識和能力，少量高品質的示範就能教會它如何以助手的方式表達。這並不意味著更多資料永遠沒用——學習新的技能（比如長推理）仍然需要大量資料。
</details>

## 1.9 延伸閱讀

- Ouyang 等：《Training language models to follow instructions with human feedback》（2022，InstructGPT）
- Zhou 等：《LIMA: Less Is More for Alignment》（2023）
- Taori 等：Stanford Alpaca（2023）
- Hugging Face：[聊天模板文件](https://huggingface.co/docs/transformers/chat_templating)、[TRL 文件](https://huggingface.co/docs/trl)
- 下一課：[LoRA 與 QLoRA](../02-lora/zh-CN.md)
