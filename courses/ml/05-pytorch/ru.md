# 5. Знакомство с PyTorch: тензоры, автодифференцирование и цикл обучения

> 🎯 **Цели урока**: познакомиться с основой PyTorch — **тензором**; научиться создавать тензоры, считать с ними, применять broadcasting и менять форму; увидеть, что автодифференцирование PyTorch работает точно как micrograd из прошлого урока; строить модели через `nn.Module`; написать стандартный цикл обучения из пяти шагов; обучать мини-батчами, использовать GPU, сохранять модели и узнавать самые частые ошибки.
>
> ⏱ **Время**: 90 минут　|　📋 **Что нужно знать**: уроки 1–4 этой части; установленный PyTorch (см. урок 2 части 1)

## 5.1 Зачем нужен PyTorch?

В прошлом уроке мы написали micrograd на чистом Python. По принципу он полноценный, но обрабатывает **по одному числу** — даже сеть из 17 параметров учится несколько секунд. У настоящих больших моделей миллиарды параметров, и им нужно:

1. работать сразу с **целыми блоками данных** (векторами, матрицами и многомерными массивами);
2. считать параллельно на **GPU** — в тысячи раз быстрее;
3. иметь готовые слои, функции потерь и оптимизаторы.

**PyTorch** — именно такой фреймворк. Его открыла компания Meta, а сейчас поддерживает PyTorch Foundation в составе Linux Foundation. Это самый популярный фреймворк глубокого обучения в науке и индустрии: код обучения и инференса Hugging Face Transformers, vLLM, Llama, DeepSeek, Qwen и большинства других построен на PyTorch. (Ещё один распространённый вариант — JAX от Google.)

> [!TIP]
> **Установка**: выполните `uv add torch` (или `pip install torch`) в папке проекта — подробности в уроке 2 части 1.
>
> Если учебная программа пишет, что не находит `torch`, значит, она использует не тот Python, куда вы установили torch. Откройте **Настройки** и укажите **путь к Python** из виртуального окружения проекта: `папка-проекта/.venv/bin/python` на macOS/Linux, `папка-проекта\.venv\Scripts\python.exe` в Windows.

## 5.2 Тензоры

**Тензор** — это «многомерный массив»: 0 измерений — одно число (скаляр), 1 измерение — вектор, 2 — матрица, а всё, где измерений больше, просто называют тензором. Тензоры почти такие же, как массивы NumPy, но умеют ещё две вещи: **работать на GPU** и **автоматически вычислять производные**.

```python requires=torch
import torch

print(torch.__version__)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(x)
print("shape:", x.shape, " dtype:", x.dtype, " device:", x.device)
print(torch.zeros(2, 3))                 # все нули
print(torch.ones(3))                     # все единицы
print(torch.arange(6).reshape(2, 3))     # 0..5, уложенные в 2 строки × 3 столбца
torch.manual_seed(0)                     # фиксируем зерно, чтобы результаты повторялись
print(torch.randn(2, 2))                 # случайные числа из стандартного нормального распределения
```

У каждого тензора три главных атрибута — **при ошибке смотрите на них в первую очередь**:

| Атрибут | Смысл | Пример |
|---------|-------|--------|
| `shape` | размер каждого измерения | `torch.Size([2, 3])`: 2 строки, 3 столбца |
| `dtype` | тип данных | `torch.float32` (по умолчанию для дробных), `torch.int64` (по умолчанию для целых), `torch.bfloat16` (часто в LLM) |
| `device` | где хранится | `cpu`, `cuda` (GPU NVIDIA), `mps` (Apple Silicon) |

## 5.3 Операции с тензорами

```python requires=torch
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
print(a + b)          # поэлементное сложение
print(a * b)          # поэлементное умножение (НЕ матричное!)
print(a @ b)          # матричное умножение
print(a.sum(), a.mean(), a.max())
print(a.sum(dim=0))   # сумма по измерению 0 (складываем строки) → суммы столбцов
print(a.sum(dim=1))   # сумма по измерению 1 → суммы строк
print(a.T)            # транспонирование
print(a[0], a[:, 1], a[1, 0].item())    # строка 0, столбец 1, один элемент
```

> [!WARNING]
> `*` умножает **поэлементно**, а `@` — это **матричное умножение**. Новички путают их чаще всего. А `.item()` превращает тензор из одного элемента в обычное число Python — это постоянно нужно при печати потерь.

Аргумент `dim` (измерение) важен: для тензора формы `(2, 3)` вызов `sum(dim=0)` «схлопывает» измерение 0, и результат имеет форму `(3,)`; `sum(dim=1)` даёт форму `(2,)`. Удобно запомнить: **`dim` — это измерение, которое исчезает.**

## 5.4 Broadcasting и изменение формы

**Broadcasting** (расширение, «вещание»): когда тензоры разной формы участвуют в поэлементной операции, PyTorch автоматически «растягивает» меньший до той же формы. Правило: **выравниваем формы по самому правому измерению**; каждое измерение должно совпадать, либо одно из них должно быть 1 (или отсутствовать).

```python requires=torch
import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])     # форма (2, 3)
bias = torch.tensor([10.0, 20.0, 30.0])  # форма (3,)
print(x + bias)                          # bias растягивается на каждую строку
col = torch.tensor([[100.0], [200.0]])   # форма (2, 1)
print(x + col)                           # col растягивается на каждый столбец
print(x.reshape(3, 2))                   # меняем форму (число элементов то же)
print(x.reshape(-1))                     # -1 означает «посчитай сам» → 6 элементов в линию
print(x.unsqueeze(0).shape)              # добавляем измерение спереди → (1, 2, 3)
```

```text
x          (2, 3)       x          (2, 3)
bias          (3,)  ✓   col        (2, 1)  ✓   (1 растягивается до 3)
результат  (2, 3)       результат  (2, 3)

(2, 3) и (2,) ✗: правые измерения 3 ≠ 2, и ни одно не равно 1 → ошибка
```

В нейросетях broadcasting повсюду: в линейном слое `x @ W.T + b` смещение `b` имеет форму `(out,)` и растягивается на каждый пример в батче.

## 5.5 Автодифференцирование: это micrograd

Повторим пример из прошлого урока `L = (a × b + c)²` на PyTorch:

```python requires=torch
import torch

a = torch.tensor(2.0, requires_grad=True)    # requires_grad=True: нужен градиент по нему
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(10.0, requires_grad=True)
L = (a * b + c) ** 2
L.backward()
print(L.item(), a.grad, b.grad, c.grad)      # 16.0 -24 16 8, как вручную и в micrograd
```

Точно так же! PyTorch тоже незаметно записывает вычислительный граф для каждой операции (у каждого результата есть `grad_fn`), а `backward()` передаёт градиенты назад в обратном топологическом порядке — ровно то, что вы сами написали в прошлом уроке.

## 5.6 Строим модели через nn.Module

В `torch.nn` есть готовые слои. Самый частый — **`nn.Linear(in_features, out_features)`**: внутри у него веса `weight` (форма `(out, in)`) и смещение `bias` (форма `(out,)`), а вычисляет он `x @ weight.T + bias`.

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
layer = nn.Linear(3, 2)          # вход размерности 3 → выход размерности 2
print(layer.weight.shape, layer.bias.shape)
x = torch.randn(4, 3)            # 4 примера (один батч), по 3 числа
print(layer(x).shape)            # (4, 2): по 2 выхода на пример

model = nn.Sequential(           # соединяем слои по порядку
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
print(model)
print("параметров:", sum(p.numel() for p in model.parameters()))   # 3×16+16 + 16×2+2 = 98
```

> [!NOTE]
> Первое измерение почти всегда — **батч** (сколько примеров обрабатывается вместе). Модели обрабатывают целый батч сразу, чтобы GPU работал параллельно. `nn.Linear(3, 2)` важно только, чтобы **последнее** измерение было 3; перед ним может быть сколько угодно измерений.

Более гибкий способ — **унаследовать `nn.Module`** и самому написать `forward` (прямой проход). Так написаны все большие модели, включая GPT и Llama:

```python requires=torch
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                    # сначала обязательно вызвать __init__ родителя
        self.fc1 = nn.Linear(d_in, d_hidden)  # слои-атрибуты автоматически регистрируются как параметры
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):                     # пишем только прямой проход; обратный сделает autograd
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

model = MLP(2, 8, 1)
print(model(torch.zeros(5, 2)).shape)         # вызов model(x) запускает forward
for name, p in model.named_parameters():
    print(name, tuple(p.shape))
```

## 5.7 Стандартный цикл обучения

Обучим XOR на PyTorch. Посмотрите на пять строк внутри цикла — **почти любой код обучения на PyTorch выглядит так**:

```python requires=torch
import torch
from torch import nn

torch.manual_seed(0)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
loss_fn = nn.BCEWithLogitsLoss()                  # потери бинарной классификации (сигмоида внутри)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for step in range(1, 501):
    logits = model(X)                             # ① прямой проход
    loss = loss_fn(logits, y)                     # ② считаем потери
    optimizer.zero_grad()                         # ③ обнуляем градиенты
    loss.backward()                               # ④ обратное распространение
    optimizer.step()                              # ⑤ обновляем параметры
    if step % 100 == 0:
        print(f"шаг {step}: потери {loss.item():.4f}")

with torch.no_grad():
    print(torch.sigmoid(model(X)).round().squeeze().tolist())   # [0.0, 1.0, 1.0, 0.0]
```

Сравним с прошлым уроком:

| micrograd | PyTorch |
|-----------|---------|
| сами пишем `(p - y) ** 2` | `nn.MSELoss()`, `nn.BCEWithLogitsLoss()`, `nn.CrossEntropyLoss()` и др. |
| `p.grad = 0.0` | `optimizer.zero_grad()` |
| `p.data -= 0.1 * p.grad` | `optimizer.step()` (SGD); чаще используют `torch.optim.AdamW` |

> [!TIP]
> В классификации **не** ставьте сигмоиду/softmax в конец модели сами: `BCEWithLogitsLoss` и `CrossEntropyLoss` принимают «сырые» оценки (**логиты**) и внутри обрабатывают их численно устойчиво. Применяйте `torch.sigmoid` / `torch.softmax` вручную, только когда нужны вероятности (например, при предсказании).

## 5.8 Полный пример: мини-батчи, GPU и оценка

Вот пример ближе к настоящему проекту: классификация точек на двух концентрических кольцах. Новые понятия:

- **Dataset / DataLoader**: делят данные на мини-батчи и перемешивают их каждый раунд. Градиентный спуск по мини-батчам называется **стохастическим градиентным спуском** (SGD): он требует меньше памяти, чем весь датасет сразу, и часто учится лучше.
- **эпоха** (epoch): один проход по всему обучающему набору.
- **device**: если есть GPU, используем его; и модели, и данным нужен `.to(device)`.
- **обучающая / тестовая выборка**: только проверка на данных, которых модель **никогда не видела**, показывает, научилась ли она на самом деле (а не запомнила).
- **`model.eval()` и `torch.no_grad()`**: при оценке включаем режим оценки и не записываем граф.

```python requires=torch
import math
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

# 1. Данные: два концентрических кольца. Внутреннее — класс 0, внешнее — класс 1
def make_rings(n):
    r = torch.cat([torch.rand(n) * 1.0, 1.5 + torch.rand(n) * 1.0])   # радиус
    theta = torch.rand(2 * n) * 2 * math.pi                          # угол
    X = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()            # метки классов должны быть целыми
    return X, y

X_train, y_train = make_rings(500)
X_test, y_test = make_rings(100)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# 2. Устройство: cuda для GPU NVIDIA, mps для Apple Silicon, иначе cpu
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print("устройство:", device)

# 3. Модель, функция потерь, оптимизатор
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()                     # многоклассовая перекрёстная энтропия (softmax внутри)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

def accuracy(X, y):
    model.eval()                                    # режим оценки
    with torch.no_grad():                           # градиенты не нужны: быстрее и экономнее
        pred = model(X.to(device)).argmax(dim=1)    # класс с наибольшей оценкой — это предсказание
    model.train()                                   # обратно в режим обучения
    return (pred == y.to(device)).float().mean().item()

# 4. Обучение: каждая эпоха проходит по всем данным мини-батчами
for epoch in range(1, 21):
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    if epoch % 5 == 0:
        print(f"эпоха {epoch:2d}  потери {total / len(X_train):.4f}  "
              f"точность на тесте {accuracy(X_test, y_test):.1%}")
```

Линейная модель никогда не разделит два концентрических кольца, а этот крошечный MLP быстро достигает почти 100% точности на **тестовой выборке**.

## 5.9 Сохранение и загрузка моделей

```python requires=torch
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
torch.save(model.state_dict(), "rings.pt")        # сохраняем только параметры (рекомендуется)

model2 = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
model2.load_state_dict(torch.load("rings.pt"))    # строим ту же архитектуру и загружаем параметры
print(torch.equal(model[0].weight, model2[0].weight))   # True
```

`state_dict()` — это словарь «имя параметра → тензор». Файлы моделей с Hugging Face (например, `model.safetensors`) по сути такие же словари, только в более безопасном и быстром формате.

## 5.10 Самые частые ошибки

| Сообщение об ошибке (фрагмент) | Причина | Что делать |
|--------------------------------|---------|------------|
| `mat1 and mat2 shapes cannot be multiplied (4x5 and 3x2)` | последнее измерение входа (5) ≠ `in_features` у `nn.Linear` (3) | `print(x.shape)`, исправить модель или данные |
| `The size of tensor a (3) must match the size of tensor b (2)` | формы двух тензоров нельзя согласовать | проверить оба `shape` |
| `Expected all tensors to be on the same device` | модель на GPU, данные на CPU (или наоборот) | `.to(device)` для обоих |
| `expected target dtype to be Long` | метки для `CrossEntropyLoss` не целые | `y = y.long()` |
| `mat1 and mat2 must have the same dtype` | один `float64` (часто из NumPy), другой `float32` | `.float()` |
| потери растут или становятся `nan` | слишком большая скорость обучения или забыт `zero_grad()` | уменьшить `lr`, проверить цикл обучения |

Учебная программа узнаёт эти ошибки и объясняет причину на вашем языке.

```python requires=torch
import torch
from torch import nn

layer = nn.Linear(3, 2)
x = torch.randn(4, 5)             # нарочно подаём вход размерности 5
try:
    layer(x)
except RuntimeError as e:
    print("ошибка:", e)
```

## 5.11 Итоги

- **Тензор** — основной тип данных PyTorch: многомерный массив, который может жить на GPU и поддерживает автодифференцирование. При ошибке сначала смотрите `shape`, `dtype` и `device`.
- `*` — поэлементное умножение, `@` — матричное; `dim` задаёт измерение, по которому идёт операция (оно исчезает); **broadcasting** выравнивает формы справа, каждое измерение должно совпадать или быть 1.
- `requires_grad=True` + `backward()` — это micrograd для тензоров.
- Модели строят из `nn.Linear`, `nn.Sequential` или наследника `nn.Module`; первое измерение обычно батч.
- Пять шагов обучения: **прямой проход → потери → `zero_grad()` → `backward()` → `step()`**; обучайте мини-батчами через `DataLoader` и оценивайте на тестовой выборке с `model.eval()` и `torch.no_grad()`.
- Сохранение — `torch.save(model.state_dict(), ...)`, загрузка — `load_state_dict`.

## 5.12 Задания

<!-- exercise: linear-shape -->
### Задание 1: Форма выхода

У `x` форма `(32, 10)`. Какой будет форма выхода после `nn.Linear(10, 5)`?

- `(10, 5)`
- `(32, 5)`
- `(32, 10)`
- `(5, 32)`

<details>
<summary>Ответ и разбор</summary>

`nn.Linear` меняет только **последнее** измерение: 10 → 5. Первое измерение, 32, — это размер батча, оно не меняется.
</details>

<!-- exercise: broadcast-shape -->
### Задание 2: Правило broadcasting

Реализуйте правило broadcasting сами, без PyTorch. Прочитайте две строки, каждая — форма (положительные целые числа через пробел). Если формы можно согласовать, выведите итоговую форму (через пробел); иначе выведите `error`.

```text
Пример ввода:
2 3
3
Пример вывода: 2 3
```

<details>
<summary>Подсказка</summary>

Сначала дополните более короткую форму единицами спереди, чтобы длины совпали; затем сравнивайте измерения по одному: если равны — берите это значение; если одно равно 1 — берите другое; иначе ошибка.

```python norun
a = list(map(int, input().split()))
b = list(map(int, input().split()))
n = max(len(a), len(b))
a = [1] * (n - len(a)) + a
b = [1] * (n - len(b)) + b
```
</details>

<!-- exercise: star-vs-at -->
### Задание 3: `*` или `@`?

Что выведет эта программа?

```python requires=torch
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 0], [0, 1]])
print((a * b).tolist())
print((a @ b).tolist())
```

<details>
<summary>Ответ и разбор</summary>

`b` — единичная матрица. Поэлементное `a * b` оставляет только диагональ: `[[1, 0], [0, 4]]`; матричное произведение `a @ b` равно самой `a`: `[[1, 2], [3, 4]]`.
</details>

<!-- exercise: torch-grad -->
### Задание 4: Производные на PyTorch

Прочитайте число `x`. С помощью автодифференцирования PyTorch вычислите `y = x³ + 2x` и производную `dy/dx`, выведите `y` и `dy/dx` (2 знака после точки, через пробел). (Нужен `torch`.)

```text
Пример ввода: 2
Пример вывода: 12.00 14.00
```

<details>
<summary>Подсказка</summary>

`x = torch.tensor(float(input()), requires_grad=True)`; вычислите `y`, вызовите `y.backward()`, и производная окажется в `x.grad`. Обычные числа получаются через `.item()`.
</details>

<!-- exercise: loop-order -->
### Задание 5: Порядок шагов обучения

Какой порядок шагов в цикле обучения правильный?

- `backward()` → `zero_grad()` → `step()` → прямой проход
- прямой проход → потери → `zero_grad()` → `backward()` → `step()`
- прямой проход → `step()` → `backward()` → `zero_grad()`
- `zero_grad()` → `step()` → прямой проход → `backward()`

<details>
<summary>Ответ и разбор</summary>

Сначала прямым проходом считаем потери, затем очищаем старые градиенты и обратным распространением получаем новые, и наконец обновляем ими параметры. `zero_grad()` можно поставить где угодно до `backward()` (некоторые ставят его в самое начало цикла).
</details>

<!-- exercise: no-grad -->
### Задание 6: Зачем при оценке `torch.no_grad()`?

- предсказания модели становятся точнее
- граф вычислений не записывается, это экономит память и время
- без него всегда будет ошибка
- все градиенты становятся равны 0

<details>
<summary>Ответ и разбор</summary>

При оценке обратное распространение не нужно, поэтому граф записывать незачем. `no_grad()` не меняет предсказания, но заметно экономит память и время — особенно важно при инференсе больших моделей.
</details>

## 5.13 Что почитать

- [Официальные уроки PyTorch](https://docs.pytorch.org/tutorials/): начните с «Learn the Basics»
- «Погружение в глубокое обучение» (*Dive into Deep Learning*): [d2l.ai](https://d2l.ai/) — бесплатная книга с полным кодом на PyTorch
- Андрей Карпатый: серия видео [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- Следующий урок: [Приёмы обучения: как помочь сети учиться лучше](../06-training-tricks/ru.md)
