# 2. 学习环境与工具

> 🎯 **本课目标**：知道学这门课需要什么样的电脑；装好 Python、uv 和常用库；了解没有显卡时怎么用免费的云端 GPU；学会估算一个模型需要多少内存；认识 Hugging Face 和国内镜像。
>
> ⏱ **预计用时**：30 分钟（安装时间另计）　｜　📋 **前置知识**：第 1 课；会用终端（[Our-tutorial「编程起步」第 3 课](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/03-terminal)）

## 2.1 我需要什么样的电脑？

**好消息：前三部分不需要显卡。** 本课程的核心内容——神经网络、自动求导、注意力、亲手写的小 GPT——都用纯 Python 或很小的模型实现，普通笔记本电脑就能运行。

| 学到哪里 | 需要什么 |
|----------|----------|
| 第 1～3 部分 | 任何能运行 Python 的电脑。学习程序自带 Python，大部分练习开箱即用 |
| 第 3 部分第 6 课（训练小 GPT）、第 4～5 部分的实验 | 安装 PyTorch；有 NVIDIA 显卡或苹果 M 系列芯片会快很多，没有也可以用免费云端 GPU |
| 第 6 部分（本地运行开源模型） | 8 GB 以上内存可以运行 1～4B 的小模型；16～32 GB 内存或显存可以运行 7～14B 的量化模型 |
| 第 7 部分（开发应用） | 普通电脑 + 一个模型 API 密钥（或本地模型） |

### GPU 为什么重要？

神经网络的计算几乎全是**大量的乘法和加法**。CPU 像几个聪明的教授，一次处理一件事；GPU 像几千个小学生，每人只会简单的算术，但可以**同时**算。对神经网络来说，后者快得多——训练和运行大模型基本都在 GPU 上进行。

- **NVIDIA 显卡**：生态最完整（CUDA），PyTorch 等框架首选。
- **苹果 M 系列芯片**：PyTorch 通过 MPS 后端支持；统一内存让 Mac 能运行比较大的模型。
- **AMD 显卡**：通过 ROCm 支持，Linux 上较完善。

## 2.2 安装 Python 和 uv

如果你在学习程序里学习，第 1～3 部分基本不用装任何东西。想在自己的环境里运行课程代码（推荐，后面会用到 numpy、PyTorch），请安装：

1. **Python 3.12 或更新版本**：安装方法见 [Our-tutorial「安装开发工具」](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/04-install-tools)。
2. **uv**：一个非常快的 Python 包和环境管理工具，正在成为 Python 项目的新标准。

```console
# Windows
winget install -e --id astral-sh.uv
# macOS
brew install uv
# Linux / macOS 通用
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 为本课程创建一个项目

```console
mkdir llm-study
cd llm-study
uv init
uv add numpy matplotlib jupyter
uv run python -c "import numpy; print(numpy.__version__)"
```

`uv init` 新建项目，`uv add` 安装库并记录到 `pyproject.toml`，`uv run` 在项目的**虚拟环境**中运行命令。

> [!NOTE]
> **虚拟环境**（virtual environment）是每个项目独立的「库的抽屉」：这个项目装的库不会影响别的项目，也不会弄乱系统里的 Python。用 uv 时它会自动帮你创建和使用。如果你习惯 pip，也可以 `python -m venv .venv` 后用 `pip install`。

### 安装 PyTorch（第 2 部分第 5 课开始需要）

```console
uv add torch
```

在有 NVIDIA 显卡的 Windows/Linux 电脑上，需要安装带 CUDA 的版本。到 <https://pytorch.org/get-started/locally/> 选择你的系统，页面会给出正确的安装命令。验证：

```python norun
import torch
print(torch.__version__)
print("CUDA 可用：", torch.cuda.is_available())          # NVIDIA 显卡
print("MPS 可用：", torch.backends.mps.is_available())   # 苹果芯片
```

### 国内网络加速

在中国大陆，下载 Python 包可能很慢，可以使用镜像：

```console
uv add numpy --default-index https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2.3 Jupyter：边写边看的笔记本

很多 AI 教程和研究代码使用 **Jupyter Notebook**：代码分成一个个「单元格」，每个单元格可以单独运行，结果（包括图表）直接显示在下面，非常适合做实验。

```console
uv run jupyter lab
```

浏览器会打开 Jupyter 界面。VS Code 安装 Jupyter 扩展后也能直接打开 `.ipynb` 文件。本课程的代码在学习程序里运行就足够了；做自己的实验时，Jupyter 会很方便。

## 2.4 没有显卡？用免费的云端 GPU

| 平台 | 说明 |
|------|------|
| [Google Colab](https://colab.research.google.com/) | 浏览器里的 Jupyter，免费版提供 GPU（有时长限制） |
| [Kaggle Notebooks](https://www.kaggle.com/code) | 每周有免费 GPU 时长 |
| [魔搭 ModelScope Notebook](https://modelscope.cn/my/mynotebook) | 国内可直接访问，新用户有免费 GPU 时长 |
| AutoDL、阿里云 PAI、腾讯云等 | 按小时租用 GPU，价格较低，适合国内用户做较大的实验 |

## 2.5 Hugging Face：模型的「GitHub」

[Hugging Face](https://huggingface.co/) 是开源模型和数据集最大的集散地：Llama、Qwen、DeepSeek、Mistral、Gemma 等开源模型都在上面发布。它的 `transformers` 库几乎是使用开源模型的标准工具。

国内访问较慢时：

- 设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`（Windows PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`）
- 或者使用 [魔搭社区 ModelScope](https://modelscope.cn/)，国内主流开源模型都会同步发布。

## 2.6 动手：估算模型需要多少内存

运行模型时，**所有参数都要放进内存（或显存）**。每个参数占多少字节取决于存储精度：

| 精度 | 每个参数的字节数 | 说明 |
|------|------------------|------|
| FP32（单精度浮点） | 4 | 训练时的「标准」精度 |
| BF16 / FP16（半精度） | 2 | 现在训练和推理的常用精度 |
| INT8 / FP8 | 1 | 8 位量化 |
| INT4 | 0.5 | 4 位量化，本地运行时最常用 |

所以：**内存（GB）≈ 参数量（十亿）× 每个参数的字节数**。例如 7B 模型用 BF16：7 × 2 = 14 GB；量化到 4 位：7 × 0.5 = 3.5 GB。（实际运行还需要额外的空间存放计算的中间结果和 KV 缓存，第 6 部分会细讲。）

```python
def model_memory_gb(params_billion, bytes_per_param):
    return params_billion * bytes_per_param

for name, size in [("小模型 1.5B", 1.5), ("7B", 7), ("70B", 70)]:
    print(f"{name}: BF16 约 {model_memory_gb(size, 2):.1f} GB，"
          f"4 位量化约 {model_memory_gb(size, 0.5):.1f} GB")
```

> [!TIP]
> 一个实用的经验：想在自己电脑上流畅运行模型，量化后的模型大小最好不超过内存（或显存）的 60%～70%。

## 2.7 本课程的学习方式

- **课文中的代码**：点代码框上方的「▶ 放进编辑器运行」，在右边修改、运行、观察。
- **练习**：编程题点「✏️ 在编辑器中做这道题」，写完点「✔ 检查答案」自动判题。
- **需要第三方库的练习**会标注所需的包（如 `numpy`），学习程序会提示你用 `pip install` 安装。
- 网站版（<https://yeraldo.com/learn>）可以在浏览器里直接运行纯 Python 代码。

## 2.8 小结

- 前三部分不需要显卡；训练和运行较大的模型时，GPU（NVIDIA / 苹果芯片）或云端 GPU 会快很多。
- 推荐 Python 3.12+ 与 uv：`uv init`、`uv add`、`uv run`；虚拟环境让每个项目的库互不影响。
- Jupyter 适合做实验；Colab、Kaggle、魔搭提供免费 GPU。
- Hugging Face 是开源模型集散地，国内可用 hf-mirror 或魔搭。
- 内存估算：**GB ≈ 参数量（B）× 每参数字节数**（FP32=4，BF16=2，INT8=1，INT4=0.5）。

## 2.9 练习

<!-- exercise: need-gpu -->
### 练习 1：需要显卡吗？

学习本课程第 1～3 部分（神经网络原理、注意力、手写小 GPT），必须要有 NVIDIA 显卡吗？

- 必须，没有显卡无法学习
- 不需要，普通电脑就可以，大部分代码是纯 Python
- 必须是苹果电脑
- 必须租用云服务器

<details>
<summary>答案与解析</summary>

前三部分的代码都是为普通电脑设计的。之后训练更大的模型时，可以使用免费的云端 GPU。
</details>

<!-- exercise: memory -->
### 练习 2：内存估算

输入一行两个数：模型的参数量（单位：十亿，可能是小数）和每个参数的字节数。输出模型参数占用的内存（GB），保留 1 位小数。

```text
样例输入：7 2
样例输出：14.0
```

<details>
<summary>提示</summary>

`p, b = map(float, input().split())`，结果是 `p * b`，用 `f"{x:.1f}"` 格式化。
</details>

<!-- exercise: venv -->
### 练习 3：虚拟环境

为什么要为每个项目使用虚拟环境？

- 让代码运行得更快
- 让不同项目安装的库互不干扰
- 为了能使用 GPU
- 为了把代码上传到 GitHub

<details>
<summary>答案与解析</summary>

虚拟环境把每个项目的库隔离开：A 项目需要旧版 numpy、B 项目需要新版，也不会冲突。
</details>

<!-- exercise: int4 -->
### 练习 4：4 位量化

一个 70B 参数的模型量化成 4 位（每个参数 0.5 字节）后，参数大约占多少内存？

- 7 GB
- 35 GB
- 70 GB
- 140 GB

<details>
<summary>答案与解析</summary>

70 × 0.5 = 35 GB。用 BF16 则需要 140 GB——这就是量化能让大模型在普通硬件上运行的原因。
</details>

## 2.10 延伸阅读

- [uv 官方文档](https://docs.astral.sh/uv/)
- [PyTorch 安装页](https://pytorch.org/get-started/locally/)
- [Hugging Face 中文入门](https://huggingface.co/learn/llm-course/zh-CN/chapter1/1)
- 下一课：[AI 编程需要的 Python](../03-python-for-ai/zh-CN.md)
