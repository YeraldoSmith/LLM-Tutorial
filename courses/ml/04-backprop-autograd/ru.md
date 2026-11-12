# 4. Обратное распространение и автодифференцирование: свой micrograd

> 🎯 **Цели урока**: понять, что такое «вычислительный граф»; вручную найти градиенты выражения по цепному правилу; самостоятельно написать движок автоматического дифференцирования примерно из 60 строк (по мотивам micrograd Карпатого); обучить с его помощью нейросеть решать XOR с нуля. Это и есть главный принцип PyTorch.
>
> ⏱ **Время**: 90 минут　|　📋 **Что нужно знать**: урок 5 части 1 (цепное правило) и урок 3 этой части

Веса XOR-сети из прошлого урока мы подобрали вручную. У настоящих сетей от тысяч до триллионов параметров, поэтому градиент каждого параметра нужно вычислять **автоматически**. Это называется **обратным распространением ошибки** (backpropagation), а программа, которая это делает, — **автоматическим дифференцированием** (autograd). Это один из самых важных уроков всего курса: поняв его, вы поймёте сердце любого фреймворка глубокого обучения.

> [!NOTE]
> Код урока основан на открытом проекте Андрея Карпатого [micrograd](https://github.com/karpathy/micrograd) (лицензия MIT). Очень советуем посмотреть вместе с уроком его видео [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) (на YouTube можно включить автоматический перевод субтитров).

## 4.1 Вычислительный граф

Любое вычисление можно нарисовать как **граф**: узлы — это значения, стрелки показывают, «из чего вычислено». Например, `L = (a × b + c)²` при `a = 2, b = −3, c = 10`:

```text
 a=2 ──┐
       ├─(×)── d=−6 ──┐
 b=−3 ─┘              ├─(+)── e=4 ──(²)── L=16
 c=10 ────────────────┘
```

**Прямой проход** (forward): слева направо вычисляем значение каждого узла.

**Обратный проход** (backward): справа налево вычисляем производную (градиент) L по каждому узлу.

## 4.2 Обратное распространение вручную

Начинаем с конца. На каждом шаге нужна только **локальная производная**, которую по цепному правилу умножаем на «градиент, пришедший сзади»:

```text
dL/dL = 1                                  (начало)
L = e²      → dL/de = 2e = 8
e = d + c   → dL/dd = dL/de × 1 = 8        (сложение: градиент копируется обоим входам)
              dL/dc = dL/de × 1 = 8
d = a × b   → dL/da = dL/dd × b = 8 × (−3) = −24    (умножение: умножаем на ДРУГОЙ вход)
              dL/db = dL/dd × a = 8 × 2 = 16
```

Два легко запоминающихся правила:

- **узел сложения** передаёт градиент каждому входу **без изменений**;
- **узел умножения** умножает градиент на **значение другого входа** и передаёт обратно.

Проверим численными производными:

```python
def L(a, b, c):
    return (a * b + c) ** 2

a, b, c, h = 2.0, -3.0, 10.0, 1e-6
print("dL/da ≈", round((L(a + h, b, c) - L(a - h, b, c)) / (2 * h), 3))
print("dL/db ≈", round((L(a, b + h, c) - L(a, b - h, c)) / (2 * h), 3))
print("dL/dc ≈", round((L(a, b, c + h) - L(a, b, c - h)) / (2 * h), 3))
```

## 4.3 Движок автодифференцирования: класс Value

Теперь пусть компьютер делает это сам. Идея:

1. каждое число оборачиваем в объект `Value`, который помнит **своё значение**, **свой градиент**, **из каких значений он получен** и **как передать им градиент**;
2. при прямом вычислении каждая операция создаёт новый `Value` и записывает, «что делать на обратном пути»;
3. `backward()` выполняет эти функции в **топологическом порядке** (от выхода ко входам, причём каждый узел обрабатывается после всех, что находятся «ниже по течению»).

```python
import math

class Value:
    def __init__(self, data, children=(), op=""):
        self.data = data            # само число
        self.grad = 0.0             # производная потерь по этому значению, сначала 0
        self._backward = lambda: None
        self._prev = set(children)  # из каких узлов получен
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad           # сложение: градиент проходит как есть
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad    # умножение: на другой вход
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):           # k — обычное число
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad   # производная tanh равна 1 − tanh²
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other

    def backward(self):
        order, seen = [], set()
        def build(v):                       # топологическая сортировка: сначала входы, потом сам узел
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                order.append(v)
        build(self)
        self.grad = 1.0                     # dL/dL = 1
        for v in reversed(order):           # от выхода ко входам
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) ** 2
L.backward()
print(L)
print(a.grad, b.grad, c.grad)     # -24.0 16.0 8.0: точно как вручную
```

### Почему градиенты «накапливаются» (`+=`)?

Если одно значение используется несколько раз, например `y = a + a`, то `a` влияет на `y` по двум путям, и полный градиент равен сумме по обоим путям (многомерное цепное правило). Поэтому каждый `_backward` использует `+=`, а не `=`. По той же причине в PyTorch перед каждым обратным проходом нужно **обнулять градиенты** (`optimizer.zero_grad()`) — иначе к ним добавятся градиенты прошлого шага.

```python norun
a = Value(3.0)
y = a + a
y.backward()
print(a.grad)     # 2.0
```

## 4.4 Строим нейросеть из Value

С автодифференцированием для построения сети достаточно описать прямой проход:

```python norun
import random

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, n_in, sizes):
        dims = [n_in] + sizes
        self.layers = [Layer(dims[i], dims[i + 1]) for i in range(len(sizes))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

## 4.5 Обучение с нуля: сеть учит XOR

Соединим код из 4.3 и 4.4 и добавим цикл обучения (четыре шага из урока 1 этой части). На этот раз ни одного градиента вручную:

```python
import math
import random

class Value:
    def __init__(self, data, children=(), op=""):
        self.data, self.grad = data, 0.0
        self._backward, self._prev = (lambda: None), set(children)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad; other.grad += out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def backward(self):
        order, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                order.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()
    def parameters(self):
        return self.w + [self.b]

class MLP:
    def __init__(self, n_in, sizes):
        dims = [n_in] + sizes
        self.layers = [[Neuron(dims[i]) for _ in range(dims[i + 1])] for i in range(len(sizes))]
    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0]
    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]

random.seed(42)
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]                  # на выходе tanh, поэтому метки -1 / 1
model = MLP(2, [4, 1])
print("параметров:", len(model.parameters()))

for step in range(1, 201):
    preds = [model(x) for x in xs]                              # ① прямой проход
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))         # ② потери (сумма квадратов ошибок)
    for p in model.parameters():                                # ③ обнуляем градиенты и делаем обратный проход
        p.grad = 0.0
    loss.backward()
    for p in model.parameters():                                # ④ градиентный спуск
        p.data -= 0.1 * p.grad
    if step % 50 == 0:
        print(f"шаг {step}: потери {loss.data:.4f}")

for x, y in zip(xs, ys):
    print(x, "цель", y, "прогноз", round(model(x).data, 3))
```

Сеть **сама выучила** XOR. Вы только что с нуля реализовали ядро полноценного фреймворка глубокого обучения: **автодифференцирование + нейросеть + градиентный спуск**.

## 4.6 Соответствие с PyTorch

| Наш micrograd | PyTorch |
|---------------|---------|
| `Value` (одно число) | `torch.Tensor` (многомерный массив, в котором могут быть миллиарды чисел) |
| `v.grad` | `t.grad` |
| `loss.backward()` | `loss.backward()` |
| обнуляем каждый `p.grad` вручную | `optimizer.zero_grad()` |
| `p.data -= lr * p.grad` | `optimizer.step()` |
| для каждой операции пишем `_backward` | в PyTorch обратные функции уже написаны для каждой операции (умножение матриц, softmax…) и эффективно выполняются на GPU |

Разница только в том, что PyTorch работает с **целыми тензорами**, а не с отдельными числами, поэтому он в миллионы раз быстрее; принцип тот же самый. Следующий урок — про PyTorch.

## 4.7 Итоги

- Вычисление можно нарисовать как **вычислительный граф**; прямой проход считает значения, **обратное распространение** считает градиенты от выхода назад.
- Каждому узлу нужна только **локальная производная**: сложение передаёт градиент как есть, умножение умножает его на другой вход; цепное правило перемножает их.
- Автодифференцирование: каждая операция записывает, «как идти назад», а `backward()` выполняет это в **обратном топологическом порядке**.
- Если значение используется несколько раз, его градиент **накапливается**, поэтому перед каждым обратным проходом градиенты нужно **обнулять**.
- micrograd ≈ главная идея PyTorch; PyTorch расширяет её на тензоры и запускает на GPU.

## 4.8 Задания

<!-- exercise: manual-backprop -->
### Задание 1: Градиенты вручную

Для `f = (a × b + c)²` прочитайте строку с тремя числами `a b c` и выведите `df/da`, `df/db` и `df/dc` (2 знака после точки, через пробел). Запишите формулы напрямую по цепному правилу из раздела 4.2 — без численных производных.

```text
Пример ввода: 2 -3 10
Пример вывода: -24.00 16.00 8.00
```

<details>
<summary>Подсказка</summary>

`e = a*b + c`, `df/de = 2e`; `df/da = 2e·b`, `df/db = 2e·a`, `df/dc = 2e`.
</details>

<!-- exercise: neuron-grad -->
### Задание 2: Градиент одного нейрона

Нейрон вычисляет `y = tanh(w·x + b)`. Прочитайте строку `w x b` и выведите `y`, `dy/dw` и `dy/db` (4 знака после точки, через пробел).

```text
Пример ввода: 1 1 0
Пример вывода: 0.7616 0.4200 0.4200
```

<details>
<summary>Подсказка</summary>

Пусть `z = w·x + b`. Локальная производная tanh равна `1 − y²`; производная `z` по `w` равна `x`, по `b` — 1. Значит, `dy/dw = (1 − y²)·x`, `dy/db = 1 − y²`.
</details>

<!-- exercise: add-relu -->
### Задание 3: Добавьте ReLU в Value

В начальном коде есть упрощённый класс `Value` (сложение, умножение, обратное распространение), а ввод и вывод уже написаны. Добавьте на месте `# TODO` метод `relu()`: прямой проход — `max(0, x)`; на обратном пути градиент проходит без изменений, если вход был больше 0, иначе он равен 0. Программа читает `a b c`, вычисляет `y = relu(a * b + c)` и выводит значение `y` и градиенты `a`, `b`, `c`.

```text
Пример ввода: 2 3 -1
Пример вывода: 5.00 3.00 2.00 1.00
```

<details>
<summary>Подсказка</summary>

```python norun
def relu(self):
    out = Value(max(0.0, self.data), (self,))
    def _backward():
        self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out
```
</details>

<!-- exercise: accumulate -->
### Задание 4: Накопление градиентов

С помощью `Value` из этого урока вычислите `y = a * a + a` при `a = 3` и вызовите `y.backward()`. Чему равен `a.grad`?

- 3
- 6
- 7
- 10

<details>
<summary>Ответ и разбор</summary>

`dy/da = 2a + 1 = 7`. `a` встречается в трёх местах, и градиенты по трём путям (3 + 3 + 1) складываются.
</details>

<!-- exercise: zero-grad -->
### Задание 5: Зачем обнулять градиенты?

Что произойдёт, если в цикле обучения забыть обнулять градиенты перед каждым `backward()`?

- ничего
- к градиентам добавятся градиенты всех прошлых шагов, и направление, и величина обновлений будут неверными
- программа упадёт с ошибкой
- обучение ускорится

<details>
<summary>Ответ и разбор</summary>

Поскольку `_backward` использует `+=`, без обнуления градиенты растут и растут. В PyTorch это забытый `optimizer.zero_grad()` — очень частая ошибка.
</details>

## 4.9 Что почитать

- Андрей Карпатый: [micrograd](https://github.com/karpathy/micrograd) и видео [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) — источник вдохновения для этого урока.
- 3Blue1Brown: [Математика обратного распространения](https://www.youtube.com/watch?v=tIeHLnjs5U8) (можно включить автоматический перевод субтитров)
- Следующий урок: [Знакомство с PyTorch](../05-pytorch/ru.md)
