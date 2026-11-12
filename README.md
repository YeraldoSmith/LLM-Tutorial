# LLM Tutorial · 从零到精通大语言模型

**简体中文** · [繁體中文](README.zh-TW.md) · [English](README.en.md) · [Русский](README.ru.md)

一套**零基础**也能学会的大语言模型（LLM）教程：从「什么是大模型」讲起，补齐需要的数学和 Python，亲手实现神经网络、自动求导和一个小 GPT，再一路学到预训练、对齐、推理部署、RAG、智能体和 2026 年的前沿进展。

- 🧭 **真正从零开始**：数学只要求初中水平，用到的每个概念都从直觉讲起，配可以运行的代码
- 🛠️ **动手实现，而不是只调包**：分词器（BPE）、注意力、Transformer、采样、LoRA、DPO 损失、GRPO 优势、量化、投机解码、向量检索……都先用纯 Python 写一遍
- ✍️ **每课都有练习**，编程题由学习程序**自动判题**
- 🌏 **4 种语言**：简体中文、繁體中文、English、Русский；学习程序界面还支持日本語、한국어、Español、Français、Deutsch
- 🖥️ **独立学习程序**（Windows、macOS、Linux），也可以在网站 [yeraldo.com/learn](https://yeraldo.com/learn) 上直接学
- 🆕 **与时俱进**：涵盖 MoE、GQA/MLA、RoPE、FlashAttention、FP8、DPO、GRPO 与可验证奖励强化学习、推理模型、MCP、上下文工程等

## 课程地图

| 部分 | 内容 | 适合 |
|------|------|------|
| [1 · 预备知识](courses/foundations/) | 大模型是什么、学习环境、AI 用的 Python、向量与矩阵、导数与梯度、概率与信息量 | 所有人从这里开始 |
| [2 · 机器学习与神经网络](courses/ml/) | 梯度下降、分类与 softmax、神经网络、反向传播与自动求导、PyTorch、训练技巧 | |
| [3 · 从零构建语言模型](courses/lm/) | 语言模型、分词与 BPE、词向量、注意力、Transformer、手写 GPT、解码与采样 | 想真正弄懂原理 |
| [4 · 训练大模型](courses/training/) | 预训练数据、规模定律与算力、分布式与混合精度、现代架构、评测 | |
| [5 · 后训练与对齐](courses/alignment/) | 指令微调、LoRA/QLoRA、RLHF、DPO、推理模型与 GRPO、安全与可解释性 | |
| [6 · 推理与部署](courses/inference/) | KV 缓存、量化、vLLM/SGLang/llama.cpp/Ollama、投机解码、本地运行模型 | 想把模型用起来 |
| [7 · 大模型应用开发](courses/apps/) | API 调用、提示词与上下文工程、工具调用、RAG、智能体与 MCP、评测与安全 | 想做 AI 应用 |
| [8 · 前沿与趋势](courses/frontier/) | 2026 年的模型版图、多模态、研究方向、如何持续学习 | |

> [!TIP]
> 完全没写过代码？先学姊妹仓库 [Our-tutorial](https://github.com/YeraldoSmith/Our-tutorial) 的「编程起步」和「Python」前 8 课，再回来。

## 怎么学

**学习程序（推荐）**：到 [Releases](https://github.com/YeraldoSmith/LLM-Tutorial/releases) 下载对应系统的压缩包，解压运行 `LLMTutorial`。程序自带 Python，前三部分的绝大多数练习不需要安装任何东西。部分课程会用到 `numpy`、`torch`，按课文提示用 `pip install` 安装即可。

**从源码运行**（需要 Python 3.11+）：

```bash
git clone https://github.com/YeraldoSmith/LLM-Tutorial.git
cd LLM-Tutorial
pip install -r app/requirements.txt
python app/run.py
```

**在 GitHub 上阅读**：每一课是一个文件夹，`zh-CN.md`、`zh-TW.md`、`en.md`、`ru.md` 分别是各语言课文，`exercises.toml` 里是测试与参考答案。

**在网站上学**：<https://yeraldo.com/learn> —— Python 代码直接在浏览器里运行。

## 致谢与参考

本教程站在许多优秀开源项目的肩膀上，课文中都注明了出处，强烈推荐配合阅读：
Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)、[nanoGPT](https://github.com/karpathy/nanoGPT)、[minbpe](https://github.com/karpathy/minbpe)、[nanochat](https://github.com/karpathy/nanochat) 与 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)；
Sebastian Raschka 的 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)；
[动手学深度学习](https://zh.d2l.ai/)；Datawhale 的 [happy-llm](https://github.com/datawhalechina/happy-llm)；[MiniMind](https://github.com/jingyaogong/minimind)；
Maxime Labonne 的 [llm-course](https://github.com/mlabonne/llm-course)；[Hugging Face LLM Course](https://huggingface.co/learn/llm-course)；
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)；以及各篇原始论文。

## 参与贡献

欢迎指出错误、补充例子、翻译成更多语言，见 [CONTRIBUTING.md](CONTRIBUTING.md)。提交前运行 `python tools/check.py --examples`。

## 许可

课文与练习：[CC BY-SA 4.0](LICENSE-CONTENT.md)；学习程序与工具：[MIT](LICENSE)。
