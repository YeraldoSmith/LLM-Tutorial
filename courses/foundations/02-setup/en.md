# 2. Your learning environment

> 🎯 **Goals**: know what kind of computer this course needs; install Python, uv and common libraries; use free cloud GPUs if you don't have one; estimate how much memory a model needs; meet Hugging Face.
>
> ⏱ **Time**: 30 minutes (plus installation)　|　📋 **You need**: lesson 1; basic terminal use ([Our-tutorial "Start here", lesson 3](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/03-terminal))

## 2.1 What computer do I need?

**Good news: parts 1–3 need no GPU.** The core of this course — neural networks, autograd, attention, a hand-written small GPT — is implemented in plain Python or with tiny models that run on an ordinary laptop.

| When | What you need |
|------|---------------|
| parts 1–3 | any computer that runs Python. The learning app includes Python; most exercises work out of the box |
| part 3 lesson 6 (training a small GPT), experiments in parts 4–5 | PyTorch; an NVIDIA GPU or Apple M-series chip helps a lot, otherwise use a free cloud GPU |
| part 6 (running open models locally) | 8 GB RAM for 1–4B models; 16–32 GB RAM or VRAM for quantized 7–14B models |
| part 7 (building apps) | an ordinary computer plus a model API key (or a local model) |

### Why do GPUs matter?

Neural-network computation is almost entirely **huge numbers of multiplications and additions**. A CPU is like a few brilliant professors working on one thing at a time; a GPU is like thousands of pupils who only know simple arithmetic but all work **at the same time**. For neural networks the second is far faster — training and running LLMs happens on GPUs.

- **NVIDIA**: the most complete ecosystem (CUDA), the first choice for PyTorch and friends.
- **Apple M-series**: supported by PyTorch through the MPS backend; unified memory lets Macs run fairly large models.
- **AMD**: supported via ROCm, best on Linux.

## 2.2 Install Python and uv

If you learn inside the app, parts 1–3 need almost nothing installed. To run the course code in your own environment (recommended — numpy and PyTorch come later), install:

1. **Python 3.12 or newer**: see [Our-tutorial "Installing developer tools"](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/04-install-tools).
2. **uv**: a very fast Python package and environment manager that is becoming the new standard for Python projects.

```console
# Windows
winget install -e --id astral-sh.uv
# macOS
brew install uv
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create a project for this course

```console
mkdir llm-study
cd llm-study
uv init
uv add numpy matplotlib jupyter
uv run python -c "import numpy; print(numpy.__version__)"
```

`uv init` creates a project, `uv add` installs libraries and records them in `pyproject.toml`, and `uv run` runs a command in the project's **virtual environment**.

> [!NOTE]
> A **virtual environment** is a separate "drawer of libraries" for each project: libraries installed for one project don't affect another or the system's Python. uv creates and uses it for you. If you prefer pip: `python -m venv .venv`, then `pip install`.

### Install PyTorch (from part 2, lesson 5)

```console
uv add torch
```

On Windows/Linux with an NVIDIA GPU you need a CUDA build: choose your system at <https://pytorch.org/get-started/locally/> to get the right command. Check:

```python norun
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())          # NVIDIA
print("MPS available:", torch.backends.mps.is_available())   # Apple silicon
```

## 2.3 Jupyter: notebooks for experiments

Many AI tutorials and research projects use **Jupyter notebooks**: code is split into "cells" you run one at a time, with results (including charts) shown right below — ideal for experiments.

```console
uv run jupyter lab
```

The Jupyter interface opens in your browser. VS Code with the Jupyter extension opens `.ipynb` files too. For this course the app is enough; for your own experiments Jupyter is handy.

## 2.4 No GPU? Use a free cloud GPU

| Platform | Notes |
|----------|-------|
| [Google Colab](https://colab.research.google.com/) | Jupyter in the browser; the free tier includes a GPU (with time limits) |
| [Kaggle Notebooks](https://www.kaggle.com/code) | free GPU hours every week |
| [Lightning AI](https://lightning.ai/), [Modal](https://modal.com/) | free monthly credits |
| RunPod, Lambda, Vast.ai and others | rent GPUs by the hour for bigger experiments |

## 2.5 Hugging Face: the "GitHub" of models

[Hugging Face](https://huggingface.co/) is the main hub for open models and datasets: Llama, Qwen, DeepSeek, Mistral, Gemma and many others are published there. Its `transformers` library is the de-facto standard for using open models.

## 2.6 Hands on: how much memory does a model need?

To run a model, **all its parameters must fit in memory (RAM or VRAM)**. Bytes per parameter depend on the precision:

| Precision | Bytes per parameter | Notes |
|-----------|---------------------|-------|
| FP32 (single precision) | 4 | the classic training precision |
| BF16 / FP16 (half precision) | 2 | the usual precision for training and inference today |
| INT8 / FP8 | 1 | 8-bit quantization |
| INT4 | 0.5 | 4-bit quantization, the most common for running locally |

So: **memory (GB) ≈ parameters (billions) × bytes per parameter**. A 7B model in BF16: 7 × 2 = 14 GB; quantized to 4 bits: 7 × 0.5 = 3.5 GB. (Running it also needs room for intermediate results and the KV cache — part 6 explains.)

```python
def model_memory_gb(params_billion, bytes_per_param):
    return params_billion * bytes_per_param

for name, size in [("small 1.5B", 1.5), ("7B", 7), ("70B", 70)]:
    print(f"{name}: BF16 about {model_memory_gb(size, 2):.1f} GB, "
          f"4-bit about {model_memory_gb(size, 0.5):.1f} GB")
```

> [!TIP]
> A practical rule: for smooth local use, keep the quantized model below about 60–70% of your RAM (or VRAM).

## 2.7 How to learn with this course

- **Code in lessons**: click "▶ Try it" above a code box, then change, run and observe it on the right.
- **Exercises**: for coding tasks click "✏️ Solve in the editor", then "✔ Check answer".
- **Exercises that need third-party libraries** list the packages (e.g. `numpy`); the app tells you to install them with `pip install`.
- The web version (<https://yeraldo.com/learn>) runs plain-Python code directly in your browser.

## 2.8 Summary

- Parts 1–3 need no GPU; for training and running bigger models a GPU (NVIDIA / Apple silicon) or a cloud GPU is much faster.
- Use Python 3.12+ and uv: `uv init`, `uv add`, `uv run`; virtual environments keep projects' libraries apart.
- Jupyter is great for experiments; Colab and Kaggle offer free GPUs.
- Hugging Face is the hub for open models.
- Memory estimate: **GB ≈ parameters (B) × bytes per parameter** (FP32=4, BF16=2, INT8=1, INT4=0.5).

## 2.9 Exercises

<!-- exercise: need-gpu -->
### Exercise 1: Do I need a GPU?

To study parts 1–3 (how neural networks work, attention, a hand-written small GPT), do you need an NVIDIA GPU?

- Yes, it's impossible without one
- No, an ordinary computer is enough; most code is plain Python
- You need a Mac
- You must rent a cloud server

<details>
<summary>Answer and explanation</summary>

Parts 1–3 are designed for ordinary computers. Later, for bigger models, you can use a free cloud GPU.
</details>

<!-- exercise: memory -->
### Exercise 2: Memory estimate

Read one line with two numbers: the parameter count (in billions, possibly a decimal) and bytes per parameter. Print the memory needed for the parameters in GB with 1 decimal place.

```text
Sample input: 7 2
Sample output: 14.0
```

<details>
<summary>Hint</summary>

`p, b = map(float, input().split())`; the result is `p * b`, formatted with `f"{x:.1f}"`.
</details>

<!-- exercise: venv -->
### Exercise 3: Virtual environments

Why use a virtual environment for each project?

- It makes code run faster
- It keeps the libraries of different projects from interfering
- It's required to use a GPU
- It's needed to upload code to GitHub

<details>
<summary>Answer and explanation</summary>

Virtual environments isolate each project's libraries: project A can use an old numpy and project B a new one without conflict.
</details>

<!-- exercise: int4 -->
### Exercise 4: 4-bit quantization

A 70B-parameter model quantized to 4 bits (0.5 bytes per parameter) needs roughly how much memory for its parameters?

- 7 GB
- 35 GB
- 70 GB
- 140 GB

<details>
<summary>Answer and explanation</summary>

70 × 0.5 = 35 GB; in BF16 it would be 140 GB — which is why quantization lets big models run on ordinary hardware.
</details>

## 2.10 Further reading

- [uv documentation](https://docs.astral.sh/uv/)
- [PyTorch "Get started"](https://pytorch.org/get-started/locally/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- Next lesson: [The Python you need for AI](../03-python-for-ai/en.md)
