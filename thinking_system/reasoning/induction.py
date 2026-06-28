"""Few-shot индукция правил/программ: учиться из 2-3 примеров ПОИСКОМ, без корпуса.

Дано несколько пар вход→выход. Система ИЩЕТ в маленьком языке операций кратчайшую
программу (композицию примитивов), согласованную со ВСЕМИ примерами (бритва Оккама),
и применяет её к новым входам. Правило выводится из 2-3 примеров и обобщается на
бесконечно много случаев — статистический учитель из малых данных так не может.

Это рассуждение, а не подгонка: поиск по пространству гипотез + проверка
непротиворечивости. Сила ограничена ЯЗЫКОМ (приоры: какие примитивы есть) и
ГЛУБИНОЙ поиска (экспонента) — поэтому решённые программы ДОБАВЛЯЮТСЯ в библиотеку
как новые примитивы (абстракция): сложное начинает решаться меньшим поиском, и
система рассуждает всё лучше с опытом (ср. DreamCoder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Primitive:
    """Именованная операция значение→значение (с ценой для бритвы Оккама)."""

    name: str
    fn: Callable[[Any], Any]
    cost: float = 1.0


class Program:
    """Композиция примитивов, применяемая слева направо: x → f1 → f2 → …"""

    def __init__(self, steps: list[Primitive]) -> None:
        self.steps = steps

    def __call__(self, x: Any) -> Any:
        for p in self.steps:
            x = p.fn(x)
        return x

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def cost(self) -> float:
        return sum(p.cost for p in self.steps)

    def __str__(self) -> str:
        return " ▸ ".join(p.name for p in self.steps) if self.steps else "id"


def _eq(a: Any, b: Any) -> bool:
    try:
        return type(a) == type(b) and a == b
    except Exception:  # noqa: BLE001
        return False


def induce(examples: list[tuple[Any, Any]], primitives: list[Primitive], *, max_depth: int = 3) -> Program | None:
    """Найти КРАТЧАЙШУЮ программу из примитивов, согласованную со всеми примерами.

    Поиск в ширину по длине композиции (короткие гипотезы первыми = Оккам). Примитив,
    бросивший исключение на каком-то входе (несовместимый тип), отбрасывается.
    Возвращает Program или None, если в данном языке/глубине решения нет.

    ВНИМАНИЕ: «решено» здесь = согласовано с ДАННЫМИ примерами. Это не гарантирует
    обобщение — на одном примере (или когда выход совпал с входом, как flip_v на
    одной строке) подойти может и неверное/тождественное правило. Чтобы отличить
    настоящее правило от совпадения, проверяйте найденную программу на ОТЛОЖЕННОЙ
    паре (как делает run_benchmark.py).
    """
    inputs = [i for i, _ in examples]
    outputs = [o for _, o in examples]

    def consistent(vals: list[Any]) -> bool:
        return all(_eq(v, o) for v, o in zip(vals, outputs))

    if consistent(inputs):
        return Program([])                                  # тождество уже подходит

    frontier: list[tuple[list[Primitive], list[Any]]] = [([], list(inputs))]
    for _ in range(max_depth):
        nxt: list[tuple[list[Primitive], list[Any]]] = []
        for steps, vals in frontier:
            for p in primitives:
                try:
                    nv = [p.fn(v) for v in vals]
                except Exception:  # noqa: BLE001 — несовместимый тип = недопустимый шаг
                    continue
                ns = steps + [p]
                if consistent(nv):
                    return Program(ns)
                nxt.append((ns, nv))
        frontier = nxt
    return None


class Library:
    """Растущий набор примитивов: решённые программы становятся абстракциями.

    После решения задачи её программу можно добавить как ОДИН примитив — тогда
    более глубокие композиции достижимы при той же глубине поиска (рассуждение
    улучшается с опытом).
    """

    def __init__(self, primitives: list[Primitive]) -> None:
        self.prims = list(primitives)
        self.abstractions: list[str] = []

    def induce(self, examples: list[tuple[Any, Any]], *, max_depth: int = 3) -> Program | None:
        return induce(examples, self.prims, max_depth=max_depth)

    def add_abstraction(self, name: str, program: Program) -> None:
        """Добавить выученную программу как новый примитив (макрос за 1 шаг поиска).

        Цена макроса = сумма цен его шагов (а не 1.0): для поиска он стоит ОДИН шаг
        глубины, но MDL/Оккам-учёт (`Program.cost`) остаётся честным — макрос не
        «дешевле» эквивалентной развёрнутой программы.
        """
        self.prims.append(Primitive(name, program.__call__, cost=program.cost))
        self.abstractions.append(name)


# ── базовый язык: типобезопасные примитивы (несовместимый тип → TypeError → отсев) ──

def _int(x: Any) -> int:
    if not isinstance(x, int):
        raise TypeError
    return x


def _lst(x: Any) -> list:
    if not isinstance(x, list):
        raise TypeError
    return x


def default_primitives() -> list[Primitive]:
    """Базовый язык над целыми и списками целых."""
    P: list[Primitive] = []
    for k in (1, 2, 3):                                      # целые
        P.append(Primitive(f"+{k}", lambda x, k=k: _int(x) + k))
        P.append(Primitive(f"*{k}", lambda x, k=k: _int(x) * k))
    P.append(Primitive("-1", lambda x: _int(x) - 1))
    P.append(Primitive("neg", lambda x: -_int(x)))
    P.append(Primitive("square", lambda x: _int(x) * _int(x)))
    P.append(Primitive("reverse", lambda x: _lst(x)[::-1]))  # список → список
    P.append(Primitive("sort", lambda x: sorted(_lst(x))))
    P.append(Primitive("sort_desc", lambda x: sorted(_lst(x), reverse=True)))
    P.append(Primitive("each+1", lambda x: [e + 1 for e in _lst(x)]))
    P.append(Primitive("each*2", lambda x: [e * 2 for e in _lst(x)]))
    P.append(Primitive("evens", lambda x: [e for e in _lst(x) if e % 2 == 0]))
    P.append(Primitive("tail", lambda x: _lst(x)[1:]))
    P.append(Primitive("sum", lambda x: sum(_lst(x))))       # список → целое
    P.append(Primitive("len", lambda x: len(_lst(x))))
    P.append(Primitive("max", lambda x: max(_lst(x))))
    P.append(Primitive("head", lambda x: _lst(x)[0]))
    P.append(Primitive("last", lambda x: _lst(x)[-1]))
    return P


def default_library() -> Library:
    return Library(default_primitives())
