"""Параметрические примитивы: ПЕРЕМЕННАЯ-аргумент связывается из данных задачи.

Первый вид «переменных» в языке: семейство операций с аргументом-цветом
(keep[c] / drop[c] / paint[c]) инстанцируется по ПАЛИТРЕ конкретной задачи —
цветам, реально встречающимся в её train-парах. Поиск получает не «операцию
вообще», а операции с уже связанной переменной; связывание идёт из данных,
а не из рук автора.

ВЫЧИСЛЯЕМЫЕ аргументы — следующий уровень: keep[top] / drop[rare] / paint[top]
связывают цвет НЕ перебором, а функцией от самого входа (top — самый частый
ненулевой цвет, rare — самый редкий; ничьи — меньший цвет, как у top_color).
Одна и та же программа работает на парах с РАЗНЫМИ палитрами — перечисляемый
paint[c] такого обобщения дать не может в принципе.
"""

from __future__ import annotations

import re
from collections import Counter

from thinking_system.reasoning.grids import Grid
from thinking_system.reasoning.induction import Primitive

_NAME_RE = re.compile(r"^(keep|drop|paint)\[(top|rare|\d)\]$")


def _keep(g: Grid, c: int) -> Grid:
    """Оставить только клетки цвета c (остальное — пусто)."""
    return tuple(tuple(v if v == c else 0 for v in r) for r in g)


def _drop(g: Grid, c: int) -> Grid:
    """Стереть клетки цвета c."""
    return tuple(tuple(0 if v == c else v for v in r) for r in g)


def _paint(g: Grid, c: int) -> Grid:
    """Перекрасить все занятые клетки в цвет c."""
    return tuple(tuple(c if v != 0 else 0 for v in r) for r in g)


_FAMILIES = {"keep": _keep, "drop": _drop, "paint": _paint}


def _color_counts(g: Grid) -> Counter:
    counts = Counter(v for r in g for v in r if v != 0)
    if not counts:
        raise ValueError("пустая сетка: цвет не вычислить")
    return counts


def _top(g: Grid) -> int:
    return min(_color_counts(g), key=lambda c: (-_color_counts(g)[c], c))


def _rare(g: Grid) -> int:
    return min(_color_counts(g), key=lambda c: (_color_counts(g)[c], c))


_COMPUTED = {"top": _top, "rare": _rare}


def make(family: str, color: int) -> Primitive:
    fn = _FAMILIES[family]
    return Primitive(f"{family}[{color}]", lambda g, _fn=fn, _c=color: _fn(g, _c))


def make_computed(family: str, which: str) -> Primitive:
    """Аргумент-цвет ВЫЧИСЛЯЕТСЯ из входа при каждом применении (top/rare)."""
    fn, arg = _FAMILIES[family], _COMPUTED[which]
    return Primitive(f"{family}[{which}]",
                     lambda g, _fn=fn, _a=arg: _fn(g, _a(g)))


def by_name(name: str) -> Primitive | None:
    """«keep[3]» / «paint[top]» → примитив (восстановление из имён состояния)."""
    m = _NAME_RE.match(name)
    if not m:
        return None
    family, arg = m.group(1), m.group(2)
    return make_computed(family, arg) if arg in _COMPUTED else make(family, int(arg))


def task_palette(pairs: list[tuple[Grid, Grid]]) -> list[int]:
    """Ненулевые цвета, встречающиеся в train-парах задачи (отсортированы)."""
    colors: set[int] = set()
    for gin, gout in pairs:
        for g in (gin, gout):
            colors.update(v for r in g for v in r if v != 0)
    return sorted(colors)


def instantiate(pairs: list[tuple[Grid, Grid]]) -> list[Primitive]:
    """Семейства × (палитра задачи ∪ вычисляемые top/rare) → связанные переменные."""
    return ([make(fam, c) for fam in _FAMILIES for c in task_palette(pairs)]
            + [make_computed(fam, w) for fam in _FAMILIES for w in _COMPUTED])
