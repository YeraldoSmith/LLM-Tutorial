# 2. 分词：文本怎样变成 token

> 🎯 **本课目标**：明白模型为什么需要「分词」，以及按字符、按词、按子词切分各有什么利弊；理解 Unicode 与 **UTF-8 字节**；亲手实现 GPT 系列使用的 **BPE（字节对编码）** 分词器：训练、编码、解码；看到分词器的训练数据如何影响中文、俄文等非英语用户；认识特殊 token 和对话模板；理解大模型「数不清字母」等怪现象的原因。
>
> ⏱ **预计用时**：80 分钟　｜　📋 **前置知识**：本部分第 1 课

上一课我们把每个字符当成一个 token。真正的大模型并不这样做：它们用一个**分词器**（tokenizer）把文本切成**子词**（subword）片段，再把每个片段换成一个编号。分词器看起来只是个「预处理」步骤，但它深刻地影响着模型的能力、速度和使用成本——尤其是对不说英语的用户。

> [!NOTE]
> 本课的 BPE 实现参考了 Andrej Karpathy 的 [minbpe](https://github.com/karpathy/minbpe)（MIT 许可证）和视频 [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)。

## 2.1 为什么要分词？三种切法

神经网络只认数字，所以必须先把文本切成小块、给每块一个编号。怎么切是一个权衡：

| 切法 | 词表大小 | 序列长度 | 问题 |
|------|----------|----------|------|
| **按字符** | 小（英文几十个；加上汉字几万个） | 很长 | 序列太长，计算量大；单个字符几乎没有意义 |
| **按词** | 巨大（几十万以上） | 短 | 新词、拼错的词、人名无法表示（「未登录词」）；中文还要先分词 |
| **按子词**（BPE 等） | 适中（3 万～25 万） | 适中 | 常见词是一个 token，罕见词拆成几块——两全其美 |

现代大模型几乎都用**子词**分词。例如 `unbelievable` 可能被切成 `un` + `believ` + `able`，模型从没见过的新词也总能用已有的小块拼出来。

## 2.2 从字符到字节：Unicode 与 UTF-8

计算机里每个字符都有一个 **Unicode 编号**（码位），例如 `A` 是 65，`你` 是 20320。存储和传输时最常用 **UTF-8** 编码把码位变成 1～4 个**字节**（0～255 的数）：英文字母 1 个字节，俄文字母 2 个，汉字 3 个，表情符号 4 个。

```python
for ch in ["A", "é", "Ж", "你", "😀"]:
    b = ch.encode("utf-8")
    print(f"{ch!r}: 码位 {ord(ch):6d}  UTF-8 字节 {list(b)}（{len(b)} 个）")

text = "Hi 你好 Привет"
print(len(text), "个字符，", len(text.encode("utf-8")), "个字节")
print(bytes([228, 189, 160]).decode("utf-8"))     # 字节还原成文字
```

GPT-2 以来的大多数分词器都以**字节**为起点：任何文本（任何语言、表情、甚至乱码）都能表示成字节序列，所以**永远不会遇到「无法表示」的字符**。词表的前 256 个 token 就是这 256 种字节，后面的 token 都是由它们**合并**出来的。

## 2.3 BPE：字节对编码

**BPE**（Byte Pair Encoding，字节对编码）的训练过程极其简单：

1. 把训练文本变成字节序列（每个字节是一个 token，编号 0～255）；
2. 统计所有**相邻 token 对**出现的次数；
3. 把**出现最多的一对**合并成一个新 token（编号 256、257……），并记下这条「合并规则」；
4. 重复第 2～3 步，直到词表达到想要的大小。

**编码**新文本时，按学到的规则**按先后顺序**合并；**解码**时把每个 token 对应的字节拼起来再按 UTF-8 还原。

```python
EN = """A language model reads text as a sequence of tokens and learns to predict the next token.
The model never sees letters or words directly: it only sees numbers. Before training, we must
decide how to cut the text into pieces. If the pieces are single characters, the sequences become
very long. If the pieces are whole words, the vocabulary becomes huge and new words cannot be
represented. Byte pair encoding finds a middle way: it starts from bytes and repeatedly merges the
most frequent pair of neighbouring tokens into a new token. Frequent words become single tokens,
while rare words are split into smaller pieces. The same tokenizer is used for training the model
and for every request the model receives later, so the tokenizer is part of the model."""

def get_stats(ids):                       # 统计相邻 token 对出现的次数
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):             # 把 ids 里所有的 pair 替换成 new_id
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text, vocab_size):
    ids = list(text.encode("utf-8"))      # 从字节开始：0～255
    merges = {}                           # (a, b) → 新 token 的编号
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)  # 出现次数最多的一对
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))   # 先用最早学到的规则
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

merges, vocab = train(EN, 256 + 100)      # 学 100 条合并规则
print("最先学到的 10 个 token：", [vocab[i].decode("utf-8", "replace") for i in range(256, 266)])
sentence = "the model predicts the next token"
ids = encode(sentence, merges)
print("编码：", ids)
print("切分：", [vocab[i].decode("utf-8", "replace") for i in ids])
print(f"{len(sentence)} 个字符 → {len(ids)} 个 token；解码还原：{decode(ids, vocab) == sentence}")
```

可以看到，最先合并出来的是 `e `（词尾的 e 加空格）、` t`、`he ` 这样极其常见的片段，很快就出现了 ` the `、`model `、` token` 这样的整词。常见的词变成 1 个 token，不常见的（如 `predicts`）被拆成几块。

> [!NOTE]
> 真实的分词器在 BPE 之前还有一步**预分词**：先用正则表达式把文本按单词、数字、标点、空格切开，BPE 只在每一块内部合并。这样可以避免 `dog.`、`dog!`、`dog?` 被学成三个不同的 token。

## 2.4 分词器偏向谁？非英语用户的「token 税」

分词器是在**训练数据**上学出来的。如果训练数据主要是英文，那么英文会被压缩得很好，而其他语言只能拆成零碎的字节。下面分别用「只有英文」和「英文 + 中文 + 俄文」的数据训练分词器，比较同样意思的句子需要多少个 token：

```python
EN = """A language model reads text as a sequence of tokens and learns to predict the next token.
The model never sees letters or words directly: it only sees numbers. Before training, we must
decide how to cut the text into pieces. If the pieces are single characters, the sequences become
very long. If the pieces are whole words, the vocabulary becomes huge and new words cannot be
represented. Byte pair encoding finds a middle way: it starts from bytes and repeatedly merges the
most frequent pair of neighbouring tokens into a new token. Frequent words become single tokens,
while rare words are split into smaller pieces. The same tokenizer is used for training the model
and for every request the model receives later, so the tokenizer is part of the model."""
ZH = """语言模型把文本看成一串词元，学习预测下一个词元。模型看不到汉字本身，只能看到数字。
训练之前，我们必须决定怎样把文本切成小块。如果每个汉字都是一块，序列就会很长；
如果每个词都是一块，词表就会非常大，新词也无法表示。字节对编码从字节出发，
不断把最常见的相邻两块合并成新的一块，于是常见的词变成一个词元，少见的词被拆成更小的部分。
同一个分词器既用于训练模型，也用于模型以后收到的每一个请求，所以分词器也是模型的一部分。"""
RU = """Языковая модель читает текст как последовательность токенов и учится предсказывать
следующий токен. Модель не видит букв и слов, она видит только числа. Перед обучением нужно
решить, как разрезать текст на кусочки. Если кусочки — отдельные символы, последовательности
становятся очень длинными. Если кусочки — целые слова, словарь становится огромным, а новые
слова нельзя представить. Кодирование пар байтов находит середину: оно начинает с байтов и
снова и снова объединяет самую частую пару соседних токенов в новый токен."""

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text, vocab_size):
    ids, merges = list(text.encode("utf-8")), {}
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

for name, data in [("只用英文训练", EN), ("用英文 + 中文 + 俄文训练", EN + ZH + RU)]:
    merges, vocab = train(data, 256 + 300)
    print(name)
    for s in ["the model predicts the next token", "模型预测下一个词元", "модель предсказывает токен"]:
        ids = encode(s, merges)
        pieces = [vocab[i].decode("utf-8", "replace") for i in ids]
        print(f"   {len(s):2d} 个字符 → {len(ids):2d} 个 token  ", pieces[:6])
```

只用英文训练时，中文句子 9 个字被拆成 27 个 token（每个汉字 3 个字节，一个都没合并），俄文也几乎是一个字节一个 token；加入中文和俄文数据后，同样的句子只需要原来一半甚至更少的 token。`�` 表示这个 token 只是某个字符的**一部分字节**，单独显示不出来。

这在现实中很重要，因为：

- **按 token 计费**：API 按 token 数收费，同样的意思用 token 多的语言就更贵；
- **上下文更短**：模型一次能读的 token 数有限，token 多就意味着能放进去的文字少；
- **生成更慢**：模型一个 token 一个 token 地生成，token 越多越慢。

早期的 GPT-2 分词器对中文很不友好。如今的分词器（GPT-4o 的 `o200k_base`、Llama 3、Qwen、DeepSeek 等）都用了大量多语言数据训练，词表也更大，对中文、俄文的效率已经好得多。**Qwen、DeepSeek 等国产模型的分词器对中文尤其高效。**

| 模型 / 分词器 | 词表大小 |
|---------------|----------|
| GPT-2 | 50,257 |
| GPT-4（`cl100k_base`） | 约 10 万 |
| GPT-4o（`o200k_base`） | 约 20 万 |
| Llama 3 | 约 12.8 万 |
| Qwen 系列 | 约 15 万 |
| DeepSeek-V3 | 约 12.9 万 |

## 2.5 使用真实的分词器

真实项目里不用自己写 BPE，直接用现成的库。例如 OpenAI 的 `tiktoken`，或 Hugging Face 的 `transformers`（任何开源模型的分词器都能这样加载）：

```python norun
# pip install tiktoken
import tiktoken
enc = tiktoken.get_encoding("o200k_base")          # GPT-4o 使用的分词器
ids = enc.encode("大语言模型很有趣！")
print(len(ids), ids)
print([enc.decode([i]) for i in ids])               # 看看每个 token 是什么

# pip install transformers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")   # 第一次运行会从 Hugging Face 下载
print(tok.tokenize("大语言模型很有趣！"))
```

> [!TIP]
> 想直观地看一段文字怎么被切分，可以用在线工具，例如 OpenAI 的 [Tokenizer](https://platform.openai.com/tokenizer) 页面，或 Hugging Face 上的 [The Tokenizer Playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)。

## 2.6 特殊 token 与对话模板

除了普通文本片段，词表里还有一些**特殊 token**，它们不对应任何文字，而是起「标记」作用：

- `<|endoftext|>`：一篇文档结束（预训练时把不同文档隔开）；
- 对话模型的**角色标记**：表示「下面是系统提示 / 用户 / 助手说的话」。

你在聊天界面里看到的一问一答，送进模型时其实被拼成了一整段带特殊 token 的文本，这种格式叫**对话模板**（chat template）。例如 Qwen 等模型使用的格式大致是：

```text
<|im_start|>system
你是一个乐于助人的助手。<|im_end|>
<|im_start|>user
1 + 1 等于几？<|im_end|>
<|im_start|>assistant
1 + 1 等于 2。<|im_end|>
```

模型学会了：看到 `<|im_start|>assistant` 之后就该生成回答，生成到 `<|im_end|>` 就停下。不同模型的模板不一样，用错模板会让模型表现变差——Hugging Face 的 `tokenizer.apply_chat_template(...)` 会自动套用正确的模板。第五部分讲微调时还会用到。

## 2.7 分词带来的怪现象

很多让人困惑的大模型行为，根源都在分词：

| 现象 | 原因 |
|------|------|
| 数不清 `strawberry` 里有几个 `r` | 模型看到的是 `str` + `aw` + `berry` 这样的 token，而不是一个个字母 |
| 不擅长把单词倒着拼、按字母操作 | 同上，字母藏在 token 内部 |
| 大数字的算术容易出错 | 数字被切成不规则的块（如 `12345` → `123` + `45`），位数对不齐 |
| 句尾多一个空格，回答就变差 | `" token"`（带前导空格）和 `"token"` 是不同的 token，训练中极少出现孤立的空格 token |
| 中文、小语种更「贵」 | 分词器训练数据里这些语言少，压缩率低（2.4 节） |

> [!NOTE]
> 新一代的推理模型会把问题「拆开来想」（比如先把单词逐个字母写出来再数），所以这些问题已经缓解了很多，但原因仍然是分词。也有研究尝试完全去掉分词器，直接在字节上建模。

## 2.8 小结

- 模型只认数字：**分词器**把文本切成 token 并编号。子词分词在词表大小和序列长度之间取得平衡。
- 文本 → Unicode 码位 → **UTF-8 字节**（英文 1 字节、俄文 2 字节、汉字 3 字节）；以字节为起点，任何文本都能表示。
- **BPE**：从字节出发，反复合并最常见的相邻 token 对；编码时按学到的顺序合并，解码时拼回字节。
- 分词器的训练数据决定了各种语言的压缩率：token 越多，就越贵、能读的内容越少、生成越慢。现代多语言分词器已大大改善。
- **特殊 token** 和**对话模板**把多轮对话拼成一段文本；数字母、算术等怪现象常常源于分词。

## 2.9 练习

<!-- exercise: utf8-bytes -->
### 练习 1：字符与字节

输入一行文本，输出它的**字符数**和 **UTF-8 字节数**（空格隔开）。

```text
样例输入：Hi 你好
样例输出：5 9
```

<details>
<summary>提示</summary>

`len(s)` 是字符数，`len(s.encode("utf-8"))` 是字节数。
</details>

<!-- exercise: bpe-step -->
### 练习 2：BPE 的一步合并

输入一个不含空格的字符串，把它看成字符序列。找出**出现次数最多的相邻字符对**（统计时允许重叠，例如 `aaa` 里 `aa` 算 2 次；次数相同时取**最先出现**的那一对），在第一行输出这一对；然后**从左到右**把所有这一对合并成一个 token（不重叠），在第二行输出合并后的 token 序列，用空格隔开。

```text
样例输入：aaabdaaabac
样例输出：
aa
aa a b d aa a b a c
```

<details>
<summary>提示</summary>

用字典按出现顺序统计 `zip(seq, seq[1:])`；`max(counts, key=counts.get)` 在并列时会返回最先加入字典的那一个。合并可以照抄本课的 `merge` 函数，只是把新编号换成两个字符拼起来的字符串。
</details>

<!-- exercise: decode-bytes -->
### 练习 3：读代码，写输出

```python
tokens = [72, 105, 32, 228, 189, 160]
print(bytes(tokens).decode("utf-8"))
print(len(tokens))
```

<details>
<summary>答案与解析</summary>

`72, 105, 32` 是 `H`、`i` 和空格；`228, 189, 160` 这 3 个字节合起来是汉字「你」的 UTF-8 编码。所以输出 `Hi 你`，然后是 `6`（6 个字节，但只有 4 个字符）。
</details>

<!-- exercise: strawberry -->
### 练习 4：数字母

早期的大模型经常答错「strawberry 里有几个 r」。最根本的原因是？

- 模型没学过英语
- 模型看到的是 token（若干个字母组成的块），而不是一个个字母
- 模型的计算能力不够
- 训练数据里没有这个单词

<details>
<summary>答案与解析</summary>

单词被切成 `str`、`aw`、`berry` 这样的块，每个块对模型来说只是一个编号，块里有哪些字母并不直接可见。
</details>

<!-- exercise: vocab-size -->
### 练习 5：词表越大越好吗？

把分词器的词表从 3 万增加到 20 万，下面哪种说法**正确**？

- 同样的文本会变成更多 token
- 同样的文本会变成更少 token，但嵌入层和输出层的参数会变多
- 对模型没有任何影响
- 模型将无法处理没见过的词

<details>
<summary>答案与解析</summary>

词表大，能合并的片段多，序列就短（更快、能读更多内容）；代价是每个 token 都要一行嵌入向量、输出层也要为每个 token 打分，参数和计算量随之增加。字节级 BPE 无论词表多大都能表示任何文本。
</details>

## 2.10 延伸阅读

- Andrej Karpathy：[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)（视频）与 [minbpe](https://github.com/karpathy/minbpe)
- Sennrich 等：《Neural Machine Translation of Rare Words with Subword Units》（2016，把 BPE 引入 NLP 的论文）
- Hugging Face LLM 课程：[分词器](https://huggingface.co/learn/llm-course/zh-CN/chapter2/4)（有中文版）
- 下一课：[词向量：用一串数字表示意思](../03-embeddings/zh-CN.md)
