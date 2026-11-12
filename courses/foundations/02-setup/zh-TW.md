# 2. 學習環境與工具

> 🎯 **本課目標**：知道學這門課需要什麼樣的電腦；裝好 Python、uv 和常用庫；瞭解沒有顯示卡時怎麼用免費的雲端 GPU；學會估算一個模型需要多少記憶體；認識 Hugging Face 和國內映象。
>
> ⏱ **預計用時**：30 分鐘（安裝時間另計）　｜　📋 **前置知識**：第 1 課；會用終端（[Our-tutorial「程式設計起步」第 3 課](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/03-terminal)）

## 2.1 我需要什麼樣的電腦？

**好訊息：前三部分不需要顯示卡。** 本課程的核心內容——神經網路、自動求導、注意力、親手寫的小 GPT——都用純 Python 或很小的模型實現，普通膝上型電腦就能執行。

| 學到哪裡 | 需要什麼 |
|----------|----------|
| 第 1～3 部分 | 任何能執行 Python 的電腦。學習程式自帶 Python，大部分練習開箱即用 |
| 第 3 部分第 6 課（訓練小 GPT）、第 4～5 部分的實驗 | 安裝 PyTorch；有 NVIDIA 顯示卡或蘋果 M 系列晶片會快很多，沒有也可以用免費雲端 GPU |
| 第 6 部分（本地執行開源模型） | 8 GB 以上記憶體可以執行 1～4B 的小模型；16～32 GB 記憶體或視訊記憶體可以執行 7～14B 的量化模型 |
| 第 7 部分（開發應用） | 普通電腦 + 一個模型 API 金鑰（或本地模型） |

### GPU 為什麼重要？

神經網路的計算幾乎全是**大量的乘法和加法**。CPU 像幾個聰明的教授，一次處理一件事；GPU 像幾千個小學生，每人只會簡單的算術，但可以**同時**算。對神經網路來說，後者快得多——訓練和執行大模型基本都在 GPU 上進行。

- **NVIDIA 顯示卡**：生態最完整（CUDA），PyTorch 等框架首選。
- **蘋果 M 系列晶片**：PyTorch 透過 MPS 後端支援；統一記憶體讓 Mac 能執行比較大的模型。
- **AMD 顯示卡**：透過 ROCm 支援，Linux 上較完善。

## 2.2 安裝 Python 和 uv

如果你在學習程式裡學習，第 1～3 部分基本不用裝任何東西。想在自己的環境裡執行課程程式碼（推薦，後面會用到 numpy、PyTorch），請安裝：

1. **Python 3.12 或更新版本**：安裝方法見 [Our-tutorial「安裝開發工具」](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/04-install-tools)。
2. **uv**：一個非常快的 Python 包和環境管理工具，正在成為 Python 專案的新標準。

```console
# Windows
winget install -e --id astral-sh.uv
# macOS
brew install uv
# Linux / macOS 通用
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 為本課程建立一個專案

```console
mkdir llm-study
cd llm-study
uv init
uv add numpy matplotlib jupyter
uv run python -c "import numpy; print(numpy.__version__)"
```

`uv init` 新建專案，`uv add` 安裝庫並記錄到 `pyproject.toml`，`uv run` 在專案的**虛擬環境**中執行命令。

> [!NOTE]
> **虛擬環境**（virtual environment）是每個專案獨立的「庫的抽屜」：這個專案裝的庫不會影響別的專案，也不會弄亂系統裡的 Python。用 uv 時它會自動幫你建立和使用。如果你習慣 pip，也可以 `python -m venv .venv` 後用 `pip install`。

### 安裝 PyTorch（第 2 部分第 5 課開始需要）

```console
uv add torch
```

在有 NVIDIA 顯示卡的 Windows/Linux 電腦上，需要安裝帶 CUDA 的版本。到 <https://pytorch.org/get-started/locally/> 選擇你的系統，頁面會給出正確的安裝命令。驗證：

```python norun
import torch
print(torch.__version__)
print("CUDA 可用：", torch.cuda.is_available())          # NVIDIA 顯示卡
print("MPS 可用：", torch.backends.mps.is_available())   # 蘋果晶片
```

### 國內網路加速

在中國大陸，下載 Python 包可能很慢，可以使用映象：

```console
uv add numpy --default-index https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2.3 Jupyter：邊寫邊看的筆記本

很多 AI 教程和研究程式碼使用 **Jupyter Notebook**：程式碼分成一個個「單元格」，每個單元格可以單獨執行，結果（包括圖表）直接顯示在下面，非常適合做實驗。

```console
uv run jupyter lab
```

瀏覽器會開啟 Jupyter 介面。VS Code 安裝 Jupyter 擴充套件後也能直接開啟 `.ipynb` 檔案。本課程的程式碼在學習程式裡執行就足夠了；做自己的實驗時，Jupyter 會很方便。

## 2.4 沒有顯示卡？用免費的雲端 GPU

| 平臺 | 說明 |
|------|------|
| [Google Colab](https://colab.research.google.com/) | 瀏覽器裡的 Jupyter，免費版提供 GPU（有時長限制） |
| [Kaggle Notebooks](https://www.kaggle.com/code) | 每週有免費 GPU 時長 |
| [魔搭 ModelScope Notebook](https://modelscope.cn/my/mynotebook) | 國內可直接訪問，新使用者有免費 GPU 時長 |
| AutoDL、阿里雲 PAI、騰訊雲等 | 按小時租用 GPU，價格較低，適合國內使用者做較大的實驗 |

## 2.5 Hugging Face：模型的「GitHub」

[Hugging Face](https://huggingface.co/) 是開源模型和資料集最大的集散地：Llama、Qwen、DeepSeek、Mistral、Gemma 等開源模型都在上面釋出。它的 `transformers` 庫幾乎是使用開源模型的標準工具。

國內訪問較慢時：

- 設定映象：`export HF_ENDPOINT=https://hf-mirror.com`（Windows PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`）
- 或者使用 [魔搭社群 ModelScope](https://modelscope.cn/)，國內主流開源模型都會同步釋出。

## 2.6 動手：估算模型需要多少記憶體

執行模型時，**所有引數都要放進記憶體（或視訊記憶體）**。每個引數佔多少位元組取決於儲存精度：

| 精度 | 每個引數的位元組數 | 說明 |
|------|------------------|------|
| FP32（單精度浮點） | 4 | 訓練時的「標準」精度 |
| BF16 / FP16（半精度） | 2 | 現在訓練和推理的常用精度 |
| INT8 / FP8 | 1 | 8 位量化 |
| INT4 | 0.5 | 4 位量化，本地執行時最常用 |

所以：**記憶體（GB）≈ 引數量（十億）× 每個引數的位元組數**。例如 7B 模型用 BF16：7 × 2 = 14 GB；量化到 4 位：7 × 0.5 = 3.5 GB。（實際執行還需要額外的空間存放計算的中間結果和 KV 快取，第 6 部分會細講。）

```python
def model_memory_gb(params_billion, bytes_per_param):
    return params_billion * bytes_per_param

for name, size in [("小模型 1.5B", 1.5), ("7B", 7), ("70B", 70)]:
    print(f"{name}: BF16 約 {model_memory_gb(size, 2):.1f} GB，"
          f"4 位量化約 {model_memory_gb(size, 0.5):.1f} GB")
```

> [!TIP]
> 一個實用的經驗：想在自己電腦上流暢執行模型，量化後的模型大小最好不超過記憶體（或視訊記憶體）的 60%～70%。

## 2.7 本課程的學習方式

- **課文中的程式碼**：點程式碼框上方的「▶ 放進編輯器執行」，在右邊修改、執行、觀察。
- **練習**：程式設計題點「✏️ 在編輯器中做這道題」，寫完點「✔ 檢查答案」自動判題。
- **需要第三方庫的練習**會標註所需的包（如 `numpy`），學習程式會提示你用 `pip install` 安裝。
- 網站版（<https://yeraldo.com/learn>）可以在瀏覽器裡直接執行純 Python 程式碼。

## 2.8 小結

- 前三部分不需要顯示卡；訓練和執行較大的模型時，GPU（NVIDIA / 蘋果晶片）或雲端 GPU 會快很多。
- 推薦 Python 3.12+ 與 uv：`uv init`、`uv add`、`uv run`；虛擬環境讓每個專案的庫互不影響。
- Jupyter 適合做實驗；Colab、Kaggle、魔搭提供免費 GPU。
- Hugging Face 是開源模型集散地，國內可用 hf-mirror 或魔搭。
- 記憶體估算：**GB ≈ 引數量（B）× 每引數位元組數**（FP32=4，BF16=2，INT8=1，INT4=0.5）。

## 2.9 練習

<!-- exercise: need-gpu -->
### 練習 1：需要顯示卡嗎？

學習本課程第 1～3 部分（神經網路原理、注意力、手寫小 GPT），必須要有 NVIDIA 顯示卡嗎？

- 必須，沒有顯示卡無法學習
- 不需要，普通電腦就可以，大部分程式碼是純 Python
- 必須是蘋果電腦
- 必須租用雲伺服器

<details>
<summary>答案與解析</summary>

前三部分的程式碼都是為普通電腦設計的。之後訓練更大的模型時，可以使用免費的雲端 GPU。
</details>

<!-- exercise: memory -->
### 練習 2：記憶體估算

輸入一行兩個數：模型的引數量（單位：十億，可能是小數）和每個引數的位元組數。輸出模型引數佔用的記憶體（GB），保留 1 位小數。

```text
樣例輸入：7 2
樣例輸出：14.0
```

<details>
<summary>提示</summary>

`p, b = map(float, input().split())`，結果是 `p * b`，用 `f"{x:.1f}"` 格式化。
</details>

<!-- exercise: venv -->
### 練習 3：虛擬環境

為什麼要為每個專案使用虛擬環境？

- 讓程式碼執行得更快
- 讓不同專案安裝的庫互不干擾
- 為了能使用 GPU
- 為了把程式碼上傳到 GitHub

<details>
<summary>答案與解析</summary>

虛擬環境把每個專案的庫隔離開：A 專案需要舊版 numpy、B 專案需要新版，也不會衝突。
</details>

<!-- exercise: int4 -->
### 練習 4：4 位量化

一個 70B 引數的模型量化成 4 位（每個引數 0.5 位元組）後，引數大約佔多少記憶體？

- 7 GB
- 35 GB
- 70 GB
- 140 GB

<details>
<summary>答案與解析</summary>

70 × 0.5 = 35 GB。用 BF16 則需要 140 GB——這就是量化能讓大模型在普通硬體上執行的原因。
</details>

## 2.10 延伸閱讀

- [uv 官方文件](https://docs.astral.sh/uv/)
- [PyTorch 安裝頁](https://pytorch.org/get-started/locally/)
- [Hugging Face 中文入門](https://huggingface.co/learn/llm-course/zh-CN/chapter1/1)
- 下一課：[AI 程式設計需要的 Python](../03-python-for-ai/zh-CN.md)
