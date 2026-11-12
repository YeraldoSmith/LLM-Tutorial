# LLM Tutorial · 從零到精通大語言模型

[简体中文](README.md) · **繁體中文** · [English](README.en.md) · [Русский](README.ru.md)

一套**零基礎**也能學會的大語言模型（LLM）教程：從「什麼是大模型」講起，補齊需要的數學和 Python，親手實現神經網路、自動求導和一個小 GPT，再一路學到預訓練、對齊、推理部署、RAG、智慧體和 2026 年的前沿進展。

- 🧭 **真正從零開始**：數學只要求初中水平，用到的每個概念都從直覺講起，配可以執行的程式碼
- 🛠️ **動手實現，而不是隻調包**：分詞器（BPE）、注意力、Transformer、取樣、LoRA、DPO 損失、GRPO 優勢、量化、投機解碼、向量檢索……都先用純 Python 寫一遍
- ✍️ **每課都有練習**，程式設計題由學習程式**自動判題**
- 🌏 **4 種語言**：簡體中文、繁體中文、English、Русский；學習程式介面還支援日本語、한국어、Español、Français、Deutsch
- 🖥️ **獨立學習程式**（Windows、macOS、Linux），也可以在網站 [yeraldo.com/learn](https://yeraldo.com/learn) 上直接學
- 🆕 **與時俱進**：涵蓋 MoE、GQA/MLA、RoPE、FlashAttention、FP8、DPO、GRPO 與可驗證獎勵強化學習、推理模型、MCP、上下文工程等

## 課程地圖

| 部分 | 內容 | 適合 |
|------|------|------|
| [1 · 預備知識](courses/foundations/) | 大模型是什麼、學習環境、AI 用的 Python、向量與矩陣、導數與梯度、機率與資訊量 | 所有人從這裡開始 |
| [2 · 機器學習與神經網路](courses/ml/) | 梯度下降、分類與 softmax、神經網路、反向傳播與自動求導、PyTorch、訓練技巧 | |
| [3 · 從零構建語言模型](courses/lm/) | 語言模型、分詞與 BPE、詞向量、注意力、Transformer、手寫 GPT、解碼與取樣 | 想真正弄懂原理 |
| [4 · 訓練大模型](courses/training/) | 預訓練資料、規模定律與算力、分散式與混合精度、現代架構、評測 | |
| [5 · 後訓練與對齊](courses/alignment/) | 指令微調、LoRA/QLoRA、RLHF、DPO、推理模型與 GRPO、安全與可解釋性 | |
| [6 · 推理與部署](courses/inference/) | KV 快取、量化、vLLM/SGLang/llama.cpp/Ollama、投機解碼、本地執行模型 | 想把模型用起來 |
| [7 · 大模型應用開發](courses/apps/) | API 呼叫、提示詞與上下文工程、工具呼叫、RAG、智慧體與 MCP、評測與安全 | 想做 AI 應用 |
| [8 · 前沿與趨勢](courses/frontier/) | 2026 年的模型版圖、多模態、研究方向、如何持續學習 | |

> [!TIP]
> 完全沒寫過程式碼？先學姊妹倉庫 [Our-tutorial](https://github.com/YeraldoSmith/Our-tutorial) 的「程式設計起步」和「Python」前 8 課，再回來。

## 怎麼學

**學習程式（推薦）**：到 [Releases](https://github.com/YeraldoSmith/LLM-Tutorial/releases) 下載對應系統的壓縮包，解壓執行 `LLMTutorial`。程式自帶 Python，前三部分的絕大多數練習不需要安裝任何東西。部分課程會用到 `numpy`、`torch`，按課文提示用 `pip install` 安裝即可。

**從原始碼執行**（需要 Python 3.11+）：

```bash
git clone https://github.com/YeraldoSmith/LLM-Tutorial.git
cd LLM-Tutorial
pip install -r app/requirements.txt
python app/run.py
```

**在 GitHub 上閱讀**：每一課是一個資料夾，`zh-CN.md`、`zh-TW.md`、`en.md`、`ru.md` 分別是各語言課文，`exercises.toml` 裡是測試與參考答案。

**在網站上學**：<https://yeraldo.com/learn> —— Python 程式碼直接在瀏覽器裡執行。

## 致謝與參考

本教程站在許多優秀開源專案的肩膀上，課文中都註明了出處，強烈推薦配合閱讀：
Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)、[nanoGPT](https://github.com/karpathy/nanoGPT)、[minbpe](https://github.com/karpathy/minbpe)、[nanochat](https://github.com/karpathy/nanochat) 與 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)；
Sebastian Raschka 的 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)；
[動手學深度學習](https://zh.d2l.ai/)；Datawhale 的 [happy-llm](https://github.com/datawhalechina/happy-llm)；[MiniMind](https://github.com/jingyaogong/minimind)；
Maxime Labonne 的 [llm-course](https://github.com/mlabonne/llm-course)；[Hugging Face LLM Course](https://huggingface.co/learn/llm-course)；
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)；以及各篇原始論文。

## 參與貢獻

歡迎指出錯誤、補充例子、翻譯成更多語言，見 [CONTRIBUTING.md](CONTRIBUTING.md)。提交前執行 `python tools/check.py --examples`。

## 許可

課文與練習：[CC BY-SA 4.0](LICENSE-CONTENT.md)；學習程式與工具：[MIT](LICENSE)。
