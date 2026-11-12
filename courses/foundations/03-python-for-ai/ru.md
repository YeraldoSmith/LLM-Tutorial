# 3. Python, который нужен для ИИ

> 🎯 **Цели урока**: повторить приёмы Python, которые чаще всего встречаются в коде ИИ (включения, подсчёт словарём, сортировка, функции, классы); экспериментировать с `random` и `math`; читать и записывать текстовые файлы; впервые взглянуть на NumPy и понять, почему «векторизация» быстрая.
>
> ⏱ **Время**: 45 минут　|　📋 **Что нужно знать**: основы Python (переменные, условия, циклы, списки, словари, функции)

Этот урок не учит Python с нуля (для этого есть [курс Python в Our-tutorial](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/python)) — он собирает **приёмы, которые нужнее всего, чтобы читать и писать код ИИ**. Если что-то непонятно, вернитесь к соответствующему уроку Python.

## 3.1 Списковые включения: пакетная обработка в одну строку

Код ИИ постоянно делает «одно и то же с группой чисел»:

```python
scores = [2.0, -1.0, 0.5, 3.0]
doubled = [x * 2 for x in scores]
positive = [x for x in scores if x > 0]
relu = [max(0.0, x) for x in scores]        # ReLU: отрицательные становятся 0 (часто в нейросетях)
print(doubled)
print(positive)
print(relu)
```

Вложенные включения строят двумерные списки (матрицы):

```python
rows, cols = 2, 3
zeros = [[0.0 for _ in range(cols)] for _ in range(rows)]
table = [[r * cols + c for c in range(cols)] for r in range(rows)]
print(zeros)
print(table)
```

> [!WARNING]
> Не пишите `[[0.0] * cols] * rows`! Внешнее `* rows` копирует ссылку на **один и тот же** внутренний список, и изменение одной строки меняет все (урок 8 курса Python).

## 3.2 Словари: подсчёт и поиск

Токенизация, подсчёт слов и отображение «слово → номер» держатся на словарях.

```python
text = "to be or not to be"
counts = {}
for w in text.split():
    counts[w] = counts.get(w, 0) + 1
print(counts)

from collections import Counter
print(Counter(text.split()).most_common(2))   # счётчик из стандартной библиотеки удобнее
```

**Словарь** (vocabulary) — основа языковой модели: каждому различному слову (или символу) даём номер, превращая текст в числа:

```python
text = "hello world hello"
vocab = sorted(set(text.split()))                  # уникальные, отсортированные
stoi = {w: i for i, w in enumerate(vocab)}        # string to index
itos = {i: w for w, i in stoi.items()}            # index to string
ids = [stoi[w] for w in text.split()]
print(stoi)
print(ids)
print(" ".join(itos[i] for i in ids))             # обратно в текст
```

Имена `stoi` / `itos` встречаются во многих открытых проектах, например в nanoGPT Андрея Карпаты.

## 3.3 Сортировка с `key`

```python
words = {"the": 7, "cat": 3, "sat": 3, "on": 5}
by_count = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
print(by_count)
top2 = [w for w, c in by_count[:2]]
print(top2)
```

`key=lambda kv: kv[1]` означает «сортировать по второму элементу каждой пары (счётчику)». `lambda` — маленькая однострочная функция: `lambda аргументы: результат`.

Чтобы при равенстве сортировать по алфавиту, верните из `key` кортеж: `key=lambda kv: (-kv[1], kv[0])` — сначала по убыванию счётчика (знак минус), затем по слову.

## 3.4 Функции и классы: упаковываем модель

Каждый слой нейросети обычно оформляют **классом**: при создании он хранит параметры, при вызове вычисляет выход.

```python
class Linear:
    """Простейший «слой»: y = w * x + b"""
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, x):          # позволяет вызывать объект как функцию
        return self.w * x + self.b

layer = Linear(w=2.0, b=1.0)
print(layer(3.0))                   # 7.0
print([layer(x) for x in [0, 1, 2]])
```

`__call__` делает возможной запись `layer(3.0)` — именно так используются модели PyTorch: `output = model(input)`.

## 3.5 random и math: инструменты для экспериментов

```python
import math
import random

random.seed(42)                         # фиксируем зерно: одинаковые результаты при каждом запуске
print(random.random())                  # случайное число в [0, 1)
print(random.randint(1, 6))             # бросок кубика
print(random.choice(["камень", "ножницы", "бумага"]))
print(random.choices(["a", "b"], weights=[0.9, 0.1], k=5))   # выбор по вероятностям (так LM выбирает слова)
print(random.gauss(0, 1))               # нормальное распределение (для инициализации весов)

print(math.exp(1), math.log(math.e))    # экспонента и натуральный логарифм — каждый день для вероятностей
print(math.sqrt(2), math.pi)
```

> [!TIP]
> **Зерно** (seed) делает «случайное» повторяемым: одно и то же зерно даёт одну и ту же последовательность. Всегда задавайте его в экспериментах, иначе не понять, изменился ли результат из-за кода или из-за случайности.

## 3.6 Чтение и запись текстовых файлов

Первый шаг обучения языковой модели — прочитать текст:

```python
from pathlib import Path

Path("tiny.txt").write_text("строка один\nстрока два\nстрока три\n", encoding="utf-8")

text = Path("tiny.txt").read_text(encoding="utf-8")
print(len(text), "символов")
for i, line in enumerate(text.splitlines(), 1):
    print(i, line)
```

Всегда указывайте `encoding="utf-8"`, иначе в Windows текст может испортиться ([«Первые шаги», урок 6](https://github.com/YeraldoSmith/Our-tutorial/tree/HEAD/courses/start/06-numbers-and-text)).

## 3.7 Первый взгляд на NumPy: зачем «векторизовать»

Массовые вычисления на списках Python медленные: интерпретатор обрабатывает каждое число по отдельности. **NumPy** хранит данные в непрерывной памяти и с помощью скомпилированного кода на C **обрабатывает весь массив сразу** — часто в десятки и сотни раз быстрее. Практически все библиотеки ИИ (включая PyTorch) мыслят такими массивами.

```python requires=numpy
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])
print(a + b)            # поэлементно: [11. 22. 33.]
print(a * 2)            # каждый элемент умножить на 2
print(a @ b)            # скалярное произведение (герой следующего урока): 140.0
m = np.zeros((2, 3))    # матрица 2×3 из нулей
print(m.shape)
print(np.exp(a))        # e в степени каждого элемента
```

Для этого кода нужен NumPy (`pip install numpy` или `uv add numpy`). Задания частей 1–3 написаны на **чистом Python**, чтобы каждый шаг был виден и его можно было изменить; поняв идеи, вы увидите, что NumPy/PyTorch просто делают то же самое быстрее.

### Сравним скорость

```python requires=numpy
import time
import numpy as np

n = 1_000_000
xs = list(range(n))
t = time.perf_counter()
total = sum(x * x for x in xs)
t_list = time.perf_counter() - t

arr = np.arange(n, dtype=np.float64)
t = time.perf_counter()
total_np = float((arr * arr).sum())
t_np = time.perf_counter() - t
print(f"Список Python: {t_list * 1000:.1f} мс, NumPy: {t_np * 1000:.1f} мс")
```

## 3.8 Итоги

- Включения `[f(x) for x in xs if условие]` обрабатывают группы данных; двумерные списки — вложенными включениями, но не `[[0]*n]*m`.
- Словари считают и строят словари токенов (`stoi`, `itos`); `Counter.most_common` удобен.
- `sorted(..., key=lambda ...)` сортирует по любому правилу; кортеж в `key` даёт многоуровневую сортировку.
- Классы с `__call__` упаковывают «слои» и «модели».
- `random.seed` делает эксперименты воспроизводимыми; `random.choices` выбирает по вероятностям; `math.exp` и `math.log` — повседневные инструменты.
- Для файлов — `encoding="utf-8"`; NumPy сильно ускоряет вычисления за счёт векторизации.

## 3.9 Задания

<!-- exercise: top-words -->
### Задание 1: Три самых частых слова

Прочитайте строку английских слов в нижнем регистре через пробел. Выведите 3 самых частых слова с их количеством, по одному в строке, в виде `слово количество`. При равенстве — по алфавиту.

```text
Пример ввода: to be or not to be that is the question to be
Пример вывода:
be 3
to 3
is 1
```

<details>
<summary>Подсказка 1</summary>

Сначала посчитайте словарём (или `Counter`).
</details>

<details>
<summary>Подсказка 2</summary>

`sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]`: по убыванию количества, затем по слову.
</details>

<!-- exercise: vocab -->
### Задание 2: Словарь символов

Прочитайте строку текста и постройте словарь **символов**: пронумеруйте все различные символы в **отсортированном** порядке, начиная с 0, затем выведите номер каждого символа исходного текста через пробел.

```text
Пример ввода: hello
Пример вывода: 1 0 2 2 3
```

(Отсортированные символы — `e h l o`, номера 0, 1, 2, 3.)

<details>
<summary>Подсказка</summary>

`chars = sorted(set(text))`, `stoi = {ch: i for i, ch in enumerate(chars)}`.
</details>

<!-- exercise: grid-trap -->
### Задание 3: Ловушка с умножением списка

Чему равен `grid` после выполнения кода?

```python
grid = [[0] * 2] * 2
grid[0][0] = 9
print(grid)
```

<details>
<summary>Ответ и разбор</summary>

`[[9, 0], [9, 0]]`. `* 2` копирует ссылку на один и тот же внутренний список, так что обе строки — это один список. Правильно: `[[0] * 2 for _ in range(2)]`.
</details>

<!-- exercise: seed -->
### Задание 4: Зерно генератора

Зачем в экспериментах с ИИ задавать зерно (например, `random.seed(42)`)?

- Чтобы программа работала быстрее
- Чтобы результаты были воспроизводимы и можно было сравнивать эффект изменений
- Чтобы случайные числа были «случайнее»
- Чтобы модель была точнее

<details>
<summary>Ответ и разбор</summary>

Одно зерно — одна последовательность случайных чисел. При фиксированном зерне различия в результатах вызваны вашими изменениями, а не случайностью.
</details>

## 3.10 Что почитать

- [NumPy: основы для начинающих](https://numpy.org/doc/stable/user/absolute_beginners.html) (англ.)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/): понятная глава о NumPy, бесплатно онлайн.
- Следующий урок: [Векторы и матрицы](../04-vectors-matrices/ru.md)
