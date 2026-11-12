# 3. 词向量：用一串数字表示意思

> 🎯 **本课目标**：明白为什么不直接用 token 的编号或 one-hot 向量；理解**嵌入**（embedding）就是一张「查找表」；用**余弦相似度**衡量意思的远近，用向量做「国王 − 男人 + 女人 ≈ 女王」的类比；从「词的上下文」中自动学出词向量；在 PyTorch 里用 `nn.Embedding` 训练一个 Bengio 式的神经语言模型，看它自己学会把元音放在一起；认识位置编码和句向量。
>
> ⏱ **预计用时**：80 分钟　｜　📋 **前置知识**：本部分第 1、2 课；第一部分第 4 课（向量、点积）

分词之后，每个 token 都有一个编号，比如 `猫` 是 1234、`狗` 是 5678。但编号本身没有意义：1234 和 1235 并不比 1234 和 5678 更「像」。这一课我们给每个 token 一个**向量**，让意思相近的 token 在空间里也靠得近——这就是**词向量**（word vector），也叫**嵌入**（embedding），它是所有语言模型的第一层。

## 3.1 编号和 one-hot 的问题

上一课结尾，我们用 one-hot 向量表示字符：27 个字符就是 27 维，每个向量只有一个 1。这有两个大问题：

1. **太大**：词表有 15 万个 token，每个 one-hot 向量就有 15 万维，几乎全是 0；
2. **没有「相似」的概念**：任意两个不同的 one-hot 向量都互相垂直（点积为 0），`猫` 和 `狗` 的距离与 `猫` 和 `飞机` 的距离完全一样。

我们想要的是：每个 token 用一个**短得多的稠密向量**（比如 768 维，每一维都是一个小数）表示，并且意思相近的 token 向量也相近。

## 3.2 嵌入就是一张查找表

**嵌入层**其实就是一个矩阵 `E`，形状是「词表大小 × 向量维度」。第 i 行就是第 i 个 token 的向量。「把 token 变成向量」就是**按编号取出那一行**：

```text
              维度 0   维度 1   维度 2   维度 3
token 0  [    0.12,   -0.40,    0.88,    0.05 ]
token 1  [   -0.71,    0.33,    0.10,   -0.26 ]
token 2  [    0.45,    0.91,   -0.37,    0.64 ]     ← 输入编号 2，就取出这一行
…
```

数学上，「取第 i 行」等于「one-hot 向量乘以矩阵 E」——所以嵌入层就是一个线性层，只是用查表的方式算得更快。上一课神经网络版 bigram 中的 `W[a]` 正是这样一张表。嵌入矩阵一开始是随机的，和模型的其他参数一起**通过训练学出来**。

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
emb = nn.Embedding(num_embeddings=10, embedding_dim=4)   # 词表 10 个 token，每个 4 维
ids = torch.tensor([3, 1, 3])
print(emb(ids))                                          # 取出第 3、1、3 行（形状 3×4）
print(torch.equal(emb(ids)[0], emb.weight[3]))            # True：就是查表
onehot = nn.functional.one_hot(ids, 10).float()          # 3×10 的 one-hot 矩阵
print(torch.allclose(onehot @ emb.weight, emb(ids)))      # True：等价于 one-hot 乘矩阵
```

## 3.3 用向量表示意思：相似度与类比

假设（为了演示，我们**手工**设计）每个词的向量有 4 个维度，分别表示「皇室」「男性」「女性」「年轻」：

```python
import math

E = {                    # 皇室  男性  女性  年轻
    "king":     [1.0, 1.0, 0.0, 0.0],
    "queen":    [1.0, 0.0, 1.0, 0.0],
    "man":      [0.0, 1.0, 0.0, 0.0],
    "woman":    [0.0, 0.0, 1.0, 0.0],
    "prince":   [1.0, 1.0, 0.0, 1.0],
    "princess": [1.0, 0.0, 1.0, 1.0],
    "boy":      [0.0, 1.0, 0.0, 1.0],
    "girl":     [0.0, 0.0, 1.0, 1.0],
}

def cosine(a, b):         # 余弦相似度：方向越一致越接近 1
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

print("king 和 queen：", round(cosine(E["king"], E["queen"]), 3))
print("king 和 girl： ", round(cosine(E["king"], E["girl"]), 3))

def analogy(a, b, c):     # a 之于 b，相当于 c 之于 ?   → 找最接近 b − a + c 的词
    target = [vb - va + vc for va, vb, vc in zip(E[a], E[b], E[c])]
    return max((w for w in E if w not in (a, b, c)), key=lambda w: cosine(target, E[w]))

print("man : king = woman : ?   →", analogy("man", "king", "woman"))
print("man : boy  = woman : ?   →", analogy("man", "boy", "woman"))
print("king : prince = queen : ? →", analogy("king", "prince", "queen"))
```

`king − man + woman`：从「国王」里减去「男性」、加上「女性」，正好落在「女王」上。**向量的方向编码了意思上的关系**（性别、皇室、年龄……）。

**余弦相似度**（cosine similarity）是比较两个嵌入最常用的方法：两个向量的夹角越小，余弦值越接近 1；互相垂直是 0；方向相反是 −1。它只看方向、不看长度。

> [!NOTE]
> 真实模型的嵌入有几百到几千维，每一维**不是**人能直接看懂的「性别」「皇室」，意思分散在很多维度的组合里。但 2013 年 Google 的 **word2vec** 确实发现，自动学出的词向量里存在 `king − man + woman ≈ queen` 这样的规律，引起了轰动。

## 3.4 从上下文中学出词向量

真正的词向量当然不是手工设计的。语言学家 J. R. Firth 有一句名言：

> 「看一个词和谁做伴，就知道它的意思。」（You shall know a word by the company it keeps.）

这叫**分布假说**：出现在相似上下文中的词，意思也相似。`猫` 和 `狗` 都常出现在「喂」「养」「可爱」附近，所以它们的意思接近。

最直接的做法：统计每个词周围出现了哪些词（**共现矩阵**），每个词的那一行就是它的向量。再用 **PPMI**（正点互信息）降低「the」「a」这种到处都有的词的影响：

```python
import math

CORPUS = """the cat drinks milk . the dog drinks water . a cat chases a mouse . a dog chases a cat .
the cat sleeps on the sofa . the dog sleeps on the floor . my cat is cute . my dog is cute .
i eat an apple . i eat a banana . she eats an orange . he eats a banana . an apple is sweet .
a banana is sweet . an orange is sweet . i drink milk . she drinks water . he drinks juice .
paris is a city in france . berlin is a city in germany . rome is a city in italy .
i live in paris . she lives in berlin . he lives in rome . we visit paris . they visit rome ."""
words = CORPUS.split()
vocab = sorted(set(words))
idx = {w: i for i, w in enumerate(vocab)}

# 共现矩阵：counts[w][c] = 词 c 出现在词 w 前后 2 个位置以内的次数
window = 2
counts = [[0] * len(vocab) for _ in vocab]
for i, w in enumerate(words):
    for j in range(max(0, i - window), min(len(words), i + window + 1)):
        if j != i:
            counts[idx[w]][idx[words[j]]] += 1

# PPMI：比「随机碰到」更经常一起出现，才算有关系
total = sum(map(sum, counts))
row = [sum(r) for r in counts]
col = [sum(counts[i][j] for i in range(len(vocab))) for j in range(len(vocab))]
vec = [[max(0.0, math.log(counts[i][j] * total / (row[i] * col[j]))) if counts[i][j] else 0.0
        for j in range(len(vocab))] for i in range(len(vocab))]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)

def nearest(word, k=3):
    v = vec[idx[word]]
    skip = {".", "the", "a", "an", "is", "in"}
    scores = [(cosine(v, vec[idx[w]]), w) for w in vocab if w != word and w not in skip]
    return [(w, round(s, 2)) for s, w in sorted(scores, reverse=True)[:k]]

print("词表大小：", len(vocab), "，每个词的向量有", len(vocab), "维")
for w in ["cat", "apple", "paris", "mouse"]:
    print(f"{w:6s} 最相近的词：", nearest(w))
```

即使只有 30 句话，程序也自己发现了 `cat` 和 `dog`、`apple` 和 `banana`、`paris` 和 `rome` 意思相近——我们从没告诉它这些词是什么。

这种「数共现」的方法有两个缺点：向量维度等于词表大小（太大），而且很稀疏。**word2vec**（2013）、**GloVe**（2014）等方法改为**训练**一个短的稠密向量，让它能预测周围的词。而在现代大模型里，词向量不再单独训练，而是作为模型的第一层，**和整个模型一起**为「预测下一个 token」这个目标而学习。

## 3.5 动手：让神经语言模型自己学出嵌入

2003 年，Yoshua Bengio 等人提出了第一个成功的神经语言模型：

1. 把前面 n 个 token 分别查表得到嵌入向量；
2. 把这些向量**拼接**起来，送进一个 MLP；
3. 输出下一个 token 的概率（softmax）。

这比 bigram 强在：看的上下文更长（这里是 3 个字符），而且嵌入让相似的字符共享知识。我们把嵌入设成只有 **2 维**，这样就能直接看看它学到了什么：

```python requires=torch
import torch
from torch import nn

NAMES = """emma olivia ava isabella sophia mia charlotte amelia harper evelyn abigail emily
elizabeth sofia avery ella madison scarlett victoria aria grace chloe camila penelope riley
layla lillian nora zoey mila aubrey hannah lily addison eleanor natalie luna savannah leah
stella hazel ellie audrey violet claire bella aurora lucy anna samantha caroline sarah maya
liam noah william james oliver benjamin elijah lucas mason logan alexander ethan jacob
michael daniel henry jackson sebastian aiden matthew samuel david joseph carter owen wyatt
john jack luke dylan levi isaac gabriel julian mateo anthony lincoln joshua andrew caleb
ryan nathan thomas leo isaiah charles hudson hunter connor eli ezra aaron adrian nolan""".split()

chars = list(".abcdefghijklmnopqrstuvwxyz")
stoi = {c: i for i, c in enumerate(chars)}
block = 3                                   # 上下文长度：看前 3 个字符
X, Y = [], []
for name in NAMES:
    context = [0] * block                   # 开头用 "." 填充
    for ch in name + ".":
        X.append(context)
        Y.append(stoi[ch])
        context = context[1:] + [stoi[ch]]
X, Y = torch.tensor(X), torch.tensor(Y)
print("训练样本：", tuple(X.shape), " 例如", [chars[i] for i in X[7]], "→", chars[Y[7]])

torch.manual_seed(0)
class NameMLP(nn.Module):                   # Bengio 等人 2003 年神经语言模型的缩小版
    def __init__(self, vocab=27, d=2, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)   # 每个字符一个 d 维向量
        self.fc1 = nn.Linear(block * d, hidden)
        self.fc2 = nn.Linear(hidden, vocab)
    def forward(self, x):                   # x：(batch, 3) 个字符编号
        e = self.emb(x)                     # (batch, 3, d)
        h = torch.tanh(self.fc1(e.flatten(1)))   # 把 3 个向量拼接成 (batch, 3d)
        return self.fc2(h)                  # (batch, 27) 个 logits

model = NameMLP()
opt = torch.optim.AdamW(model.parameters(), lr=0.01)
for step in range(2001):
    ix = torch.randint(0, len(X), (64,))
    loss = nn.functional.cross_entropy(model(X[ix]), Y[ix])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        with torch.no_grad():
            full = nn.functional.cross_entropy(model(X), Y).item()
        print(f"step {step:4d}  loss {full:.3f}")

E = model.emb.weight.detach()
vowels = [stoi[c] for c in "aeiou"]
others = [stoi[c] for c in "bcdfghjklmnprstvwz"]
def avg_dist(A, B):
    d = [torch.dist(E[a], E[b]).item() for a in A for b in B if a != b]
    return sum(d) / len(d)
print(f"元音之间的平均距离 {avg_dist(vowels, vowels):.2f}，元音与辅音之间的平均距离 {avg_dist(vowels, others):.2f}")

torch.manual_seed(1)                        # 用训练好的模型生成名字
names = []
for _ in range(8):
    context, out = [0] * block, ""
    while len(out) < 20:
        probs = torch.softmax(model(torch.tensor([context])), dim=1)
        ix = torch.multinomial(probs, 1).item()
        if ix == 0:
            break
        out += chars[ix]
        context = context[1:] + [ix]
    names.append(out)
print("生成的名字：", names)
```

几点观察：

- 损失比 bigram（约 2.1）低得多，生成的名字也像样多了——因为模型能看到前 3 个字符。
- 我们从没告诉模型什么是元音，但学到的 2 维嵌入里，**元音彼此靠得更近**：因为它们在名字里起的作用相似（常常可以互相替换）。这就是嵌入学到「意思」的方式。
- 训练数据只有一百多个名字，所以模型也会「背出」一些训练集里原有的名字（第二部分第 6 课讲的过拟合）。真实模型用几十亿甚至几万亿个 token 训练。

## 3.6 位置也需要嵌入

上面的 MLP 把 3 个向量**按顺序拼接**，所以知道谁在前谁在后。但下一课要学的**注意力机制**会把所有 token「一视同仁」地放在一起处理，本身**不知道顺序**——而「狗咬人」和「人咬狗」意思完全不同。

所以 Transformer 还要给每个**位置**一个向量：

- **可学习的位置嵌入**（GPT-2）：再加一张「位置 × 维度」的表，第 0 个位置、第 1 个位置……各有一个向量，和 token 嵌入**相加**；
- **正弦位置编码**（原始 Transformer）：用不同频率的 sin/cos 函数生成，不需要学习；
- **RoPE 旋转位置编码**（Llama、Qwen、DeepSeek 等现代模型）：不加在输入上，而是在注意力里按位置「旋转」向量。下一课再详细讲。

```formula
输入向量 = token 嵌入[token 编号] + 位置嵌入[位置]
```

## 3.7 大模型里的嵌入

| 模型 | 词表大小 | 嵌入维度 | 嵌入层参数 |
|------|----------|----------|------------|
| GPT-2 small | 50,257 | 768 | 约 3,860 万（占全部 1.24 亿的约 31%） |
| Llama 3 8B | 128,256 | 4,096 | 约 5.25 亿 |
| Qwen3 系列（大模型） | 约 151,000 | 4,096～5,120 等 | 数亿 |

模型的最后一层要把向量变回「每个 token 的分数」（logits），这一层的权重形状和嵌入矩阵一样。很多模型让两者**共用同一个矩阵**（叫**权重绑定**，weight tying），既省参数，也让「输入时的意思」和「输出时的意思」保持一致。

## 3.8 句向量：给整段话一个向量

把整句话、整段文字变成一个向量，就能比较两段话的意思是否相近。这正是**语义搜索**和第七部分要讲的 **RAG**（检索增强生成）的基础：把文档和问题都变成向量，找出余弦相似度最高的文档。专门做这件事的模型叫**嵌入模型**，很多都支持多种语言，用中文提问也能找到英文资料：

```python norun
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")      # 支持 100 多种语言；也可以试试 Qwen/Qwen3-Embedding-0.6B
docs = ["猫喜欢喝牛奶", "The cat drinks milk", "巴黎是法国的首都", "Paris is the capital of France"]
vecs = model.encode(docs, normalize_embeddings=True)
query = model.encode(["小猫爱喝什么？"], normalize_embeddings=True)
print(query @ vecs.T)                           # 和两句关于猫的话相似度最高（不管中文还是英文）
```

## 3.9 小结

- token 编号和 one-hot 向量都不能表达「相似」；**嵌入**给每个 token 一个短的稠密向量，让意思相近的 token 向量也相近。
- 嵌入层就是一张**查找表**（词表大小 × 维度），等价于 one-hot 乘矩阵；它和模型一起训练。
- **余弦相似度**衡量方向的接近程度；向量之差可以编码关系（`king − man + woman ≈ queen`）。
- **分布假说**：上下文相似的词意思相似；从共现统计或「预测周围的词」中都能学出词向量。
- 神经语言模型（Bengio 2003）：嵌入 → 拼接 → MLP → softmax；嵌入会自动把作用相似的 token 放在一起。
- 注意力本身不知道顺序，需要**位置编码**（可学习、正弦、RoPE）；**句向量**是语义搜索和 RAG 的基础。

## 3.10 练习

<!-- exercise: nearest-word -->
### 练习 1：最相近的词

第一行是整数 `n`，接下来 `n` 行，每行是一个词和它的向量（若干个数，空格隔开），最后一行是要查询的词。输出与查询词**余弦相似度最高**的**另一个**词，以及相似度（保留 4 位小数），空格隔开。

```text
样例输入：
4
cat 0.9 0.8 0.1
dog 0.8 0.9 0.2
car 0.1 0.2 0.9
bus 0.2 0.1 0.8
cat
样例输出：dog 0.9899
```

<details>
<summary>提示</summary>

先把每行拆成 `parts = line.split()`，`parts[0]` 是词，`parts[1:]` 转成小数就是向量；用 `max(..., key=...)` 找相似度最高的词。
</details>

<!-- exercise: analogy -->
### 练习 2：词语类比

起始代码里给出了 3.3 节那张手工设计的词向量表。输入一行三个词 `a b c`，输出最接近 `b − a + c` 的词（不能是 `a`、`b`、`c` 本身）。

```text
样例输入：man king woman
样例输出：queen
```

<details>
<summary>提示</summary>

`target = [vb - va + vc for va, vb, vc in zip(E[a], E[b], E[c])]`，再在其他词里找余弦相似度最高的那个。
</details>

<!-- exercise: embedding-params -->
### 练习 3：嵌入层有多少参数？

输入一行三个整数 `V d tied`：词表大小、嵌入维度，以及输入嵌入和输出层是否共用一个矩阵（1 表示共用，0 表示不共用）。输出这两部分（不算偏置）一共有多少个参数。

```text
样例输入：50257 768 1
样例输出：38597376
```

<details>
<summary>提示</summary>

一个矩阵是 `V × d` 个参数；不共用时有两个。
</details>

<!-- exercise: why-not-onehot -->
### 练习 4：one-hot 的问题

用 one-hot 向量表示词，最主要的问题是？

- 计算机无法存储 0 和 1
- 任意两个不同的词都互相垂直、距离一样，无法表达「意思相近」，而且维度等于词表大小
- one-hot 向量不能输入神经网络
- one-hot 只能表示英文

<details>
<summary>答案与解析</summary>

不同的 one-hot 向量点积都是 0，彼此「一样远」；维度还和词表一样大（十几万）。嵌入用短的稠密向量解决了这两个问题。
</details>

<!-- exercise: why-position -->
### 练习 5：为什么需要位置编码？

- 为了让模型知道词表有多大
- 因为注意力机制本身不区分 token 的先后顺序，而「狗咬人」和「人咬狗」意思不同
- 为了减少参数量
- 为了让模型支持中文

<details>
<summary>答案与解析</summary>

注意力把所有 token 放在一起计算，如果不加位置信息，打乱顺序结果也不变。位置编码（可学习的、正弦的或 RoPE）把「第几个」的信息告诉模型。
</details>

## 3.11 延伸阅读

- Mikolov 等：《Efficient Estimation of Word Representations in Vector Space》（2013，word2vec）
- Bengio 等：《A Neural Probabilistic Language Model》（2003）
- Andrej Karpathy：[Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)（本课 3.5 节的灵感来源）
- Jay Alammar：[The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)（图解，网上有中文翻译）
- 下一课：[注意力机制：让每个词「看到」上下文](../04-attention/zh-CN.md)
