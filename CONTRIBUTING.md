# 参与贡献 · Contributing

**中文**｜[English](#english)

## 写一课需要哪些文件？

```text
courses/lm/02-tokenization/
├── zh-CN.md          简体中文课文（必需，作为原文）
├── en.md             英文课文（必需）
├── ru.md             俄文课文（必需）
├── zh-TW.md          繁體中文：运行 tools/zh_tw.py 自动生成，不要手改
└── exercises.toml    练习的测试数据与参考答案（各语言共用）
```

文件夹名以两位数字开头（`03-numbers`），决定课程顺序。每个 `.md` 的第一个 `# ` 标题就是课名，例如 `# 3. 数字与计算`。

## 课文写作规范

1. **从零讲起**：第一次出现的概念都要解释；多用类比和图（用 ```text 代码块画字符图，各平台都能显示）。
2. **每段代码都要能运行**：可运行的代码块写成 ```python、```c 等；只是片段、故意写错的例子加 `norun`：```python norun。`tools/check.py --examples` 会运行所有可运行的代码块。其他标注：`input="3\n4"`（运行时的输入）、`requires=numpy`（需要第三方包，没装时检查会跳过，网页版会提示）、`timeout=120`（运行较久的例子，单位秒，默认 20）。
3. **常见错误**一节：贴出真实的报错信息，并解释它。
4. 提示框使用 GitHub 语法：`> [!TIP]`、`> [!NOTE]`、`> [!WARNING]`、`> [!IMPORTANT]`、`> [!CAUTION]`。
5. 公式用 ```formula 代码块，内容写成普通 Unicode 文字（如 `softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ`）。
6. 课与课之间用相对链接：`[下一课](../03-embeddings/zh-CN.md)`，学习程序会自动跳转。
7. 内容要现代：以当前稳定版本为准（Python 3.14、Java 25、C++23、Rust 2024……），老写法只在需要对比时提及。

## 练习怎么写？

在课文的「练习」一节，每道题以一个标记开头，后面跟三级标题和题目：

````markdown
<!-- exercise: sum-two -->
### 练习 2：两数之和

读入两行整数，输出它们的和。

<details>
<summary>提示</summary>

用 `int(input())` 读入整数。
</details>

<details>
<summary>答案与解析</summary>

（只在答对或查看答案后显示）
</details>
````

- `<summary>` 里含「答案 / 解析 / Answer / Explanation / Ответ / Разбор」的 `<details>` 是**解析**，其余 `<details>` 是**提示**（可以有多个，逐个显示）。
- **选择题**：题目最后的无序列表就是选项，各语言选项数量和顺序必须一致。
- **读代码写输出**：题目里放一个代码块。

`exercises.toml` 里写测试与答案：

```toml
[[exercise]]
id = "sum-two"          # 与 <!-- exercise: sum-two --> 对应
kind = "code"           # code（编程）| choice（选择）| output（读代码写输出）
starter = ""            # 初始代码，可写成 [exercise.starter] 按语言区分
solution = '''
a = int(input())
b = int(input())
print(a + b)
'''
tests = [
  { input = "1\n2\n", output = "3\n" },
  { input = "-5\n5\n", output = "0\n" },
]
# match = "exact"       # 默认逐行比较（忽略行尾空格）；也可以是 contains / unordered / number

[[exercise]]
id = "pick-one"
kind = "choice"
answer = 2              # 从 0 开始的选项序号

[[exercise]]
id = "predict"
kind = "output"
answer = '''
1
2
'''
```

如果期望输出包含文字（不同语言不同），把 `solution`、`tests` 写成按语言区分的表：

```toml
[exercise.tests]
zh-CN = [{ input = "", output = "你好\n" }]
en = [{ input = "", output = "Hello\n" }]
ru = [{ input = "", output = "Привет\n" }]
```

## 提交前检查

```bash
pip install opencc-python-reimplemented
python tools/zh_tw.py            # 生成繁體中文
python tools/check.py --examples # 检查结构，运行全部参考答案和例子
```

学习程序与 Our-tutorial 仓库共用同一份代码（`app/tutor/`，修改时请两边同步）：`content.py`（读取课程）、`markdown.py`（渲染）、`runners.py`（编译运行）、`errors.py`（报错解释）、`ui/`（界面）。界面文字在 `app/tutor/locales/*.json`；`zh-TW.json` 由 `tools/zh_tw.py` 生成。

---

<a name="english"></a>
## English

Each lesson is a folder `courses/<course>/<NN-name>/` with `zh-CN.md` (source), `en.md`, `ru.md`, an auto-generated `zh-TW.md` and a shared `exercises.toml`.

- Start every concept from zero; draw diagrams in ```text blocks.
- Every runnable code block must run (`tools/check.py --examples` runs them); mark fragments and deliberately broken code with `norun`. Other fence options: `input="3\n4"` (stdin), `requires=numpy` (third-party package; skipped when missing, the website shows a note), `timeout=120` (seconds, for slow demos; default 20).
- Use GitHub alerts (`> [!TIP]` …) and ```formula blocks with plain Unicode maths.
- Exercises: a `<!-- exercise: id -->` marker, a `###` title, the task text, `<details>` hints, and an optional `<details>` whose summary contains "Answer"/"Explanation" for the explanation. Choice options are the last bullet list. Tests and solutions go in `exercises.toml` (see the examples above); use per-language tables when expected output contains words.
- Before committing: `python tools/zh_tw.py && python tools/check.py --examples`.
