# LLM Tutorial · Large Language Models from Zero to Expert

[简体中文](README.md) · [繁體中文](README.zh-TW.md) · **English** · [Русский](README.ru.md)

A course on large language models that works **from absolute zero**: it starts with "what is an LLM?", fills in the maths and Python you need, has you build a neural network, an autograd engine and a small GPT by hand, and then goes all the way through pre-training, alignment, inference and deployment, RAG, agents and the state of the art in 2026.

- 🧭 **Really from zero**: school-level maths is enough; every concept starts from intuition and comes with runnable code
- 🛠️ **Build it, don't just call it**: tokenizer (BPE), attention, the Transformer, sampling, LoRA, the DPO loss, GRPO advantages, quantization, speculative decoding, vector search… written in plain Python first
- ✍️ **Exercises in every lesson**, with coding tasks **checked automatically** by the app
- 🌏 **4 languages**: Simplified Chinese, Traditional Chinese, English, Russian; the app's interface also speaks Japanese, Korean, Spanish, French and German
- 🖥️ **A desktop app** (Windows, macOS, Linux), or learn on the web at [yeraldo.com/learn](https://yeraldo.com/learn)
- 🆕 **Up to date**: MoE, GQA/MLA, RoPE, FlashAttention, FP8, DPO, GRPO and RL with verifiable rewards, reasoning models, MCP, context engineering and more

## Course map

| Part | Content | For |
|------|---------|-----|
| [1 · Foundations](courses/foundations/) | what an LLM is, your environment, Python for AI, vectors and matrices, derivatives and gradients, probability and information | everyone starts here |
| [2 · Machine learning and neural networks](courses/ml/) | gradient descent, classification and softmax, neural networks, backpropagation and autograd, PyTorch, training tricks | |
| [3 · Building a language model from scratch](courses/lm/) | language models, tokenization and BPE, embeddings, attention, the Transformer, a hand-written GPT, decoding and sampling | understanding how it really works |
| [4 · Training large models](courses/training/) | pre-training data, scaling laws and compute, distributed and mixed precision, modern architectures, evaluation | |
| [5 · Post-training and alignment](courses/alignment/) | instruction tuning, LoRA/QLoRA, RLHF, DPO, reasoning models and GRPO, safety and interpretability | |
| [6 · Inference and deployment](courses/inference/) | KV cache, quantization, vLLM/SGLang/llama.cpp/Ollama, speculative decoding, local models | putting models to work |
| [7 · Building with LLMs](courses/apps/) | APIs, prompt and context engineering, tool calling, RAG, agents and MCP, evaluation and security | building AI apps |
| [8 · The frontier](courses/frontier/) | the 2026 model landscape, multimodality, research directions, how to keep learning | |

> [!TIP]
> Never written code? Do "Start here" and the first 8 Python lessons of the sister repository [Our-tutorial](https://github.com/YeraldoSmith/Our-tutorial) first.

## Progress

Parts 1–4 (foundations, machine learning and neural networks, building a language model from scratch, training large models — 24 lessons) are complete: you train a small GPT yourself and learn how large models are trained and evaluated. Parts 5–8 are being written and will arrive lesson by lesson.

## How to learn

**The app (recommended)**: download the archive for your system from [Releases](https://github.com/YeraldoSmith/LLM-Tutorial/releases) and run `LLMTutorial`. It includes Python, so most exercises in parts 1–3 need nothing else. Some lessons use `numpy` or `torch`; install them with `pip install` as the lesson explains.

**From source** (Python 3.11+):

```bash
git clone https://github.com/YeraldoSmith/LLM-Tutorial.git
cd LLM-Tutorial
pip install -r app/requirements.txt
python app/run.py
```

**On GitHub**: each lesson is a folder with `zh-CN.md`, `zh-TW.md`, `en.md` and `ru.md`; tests and reference solutions are in `exercises.toml`.

**On the web**: <https://yeraldo.com/learn> — Python runs right in your browser.

## Credits

This course stands on the shoulders of many excellent open projects (credited in the lessons), all worth reading alongside it: Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), [nanoGPT](https://github.com/karpathy/nanoGPT), [minbpe](https://github.com/karpathy/minbpe), [nanochat](https://github.com/karpathy/nanochat) and [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html); Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch); [Dive into Deep Learning](https://d2l.ai/); Maxime Labonne's [llm-course](https://github.com/mlabonne/llm-course); the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course); [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/); and the original papers.

## Contributing

Corrections, examples and translations are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Run `python tools/check.py --examples` before sending changes.

## Licence

Lessons and exercises: [CC BY-SA 4.0](LICENSE-CONTENT.md); app and tools: [MIT](LICENSE).
