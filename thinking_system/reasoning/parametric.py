"""Параметрические примитивы: ПЕРЕМЕННАЯ-аргумент связывается из данных задачи.

Первый вид «переменных» в языке: семейство операций с аргументом-цветом
(keep[c] / drop[c] / paint[c]) инстанцируется по ПАЛИТРЕ конкретной задачи —
цветам, реально встречающимся в её train-парах. Поиск получает не «операцию
вообще», а операции с уже связанной переменной; связывание идёт из данных,
а не из рук автора. Второй вид — дырки в шаблонах (templates.py): там переменной
становится целый шаг программы, и разные привязки цвета анти-унифицируются
в один шаблон.
"""

from __future__ import annotations

import re

from thinking_system.reasoning.grids import Grid
from thinking_system.reasoning.induction import Primitive

_NAME_RE = re.compile(r"^(keep|drop|paint)\[(\d)\]$")


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


def make(family: str, color: int) -> Primitive:
    fn = _FAMILIES[family]
    return Primitive(f"{family}[{color}]", lambda g, _fn=fn, _c=color: _fn(g, _c))


def by_name(name: str) -> Primitive | None:
    """«keep[3]» → примитив (для восстановления решений/абстракций из имён)."""
    m = _NAME_RE.match(name)
    return make(m.group(1), int(m.group(2))) if m else None


def task_palette(pairs: list[tuple[Grid, Grid]]) -> list[int]:
    """Ненулевые цвета, встречающиеся в train-парах задачи (отсортированы)."""
    colors: set[int] = set()
    for gin, gout in pairs:
        for g in (gin, gout):
            colors.update(v for r in g for v in r if v != 0)
    return sorted(colors)


def instantiate(pairs: list[tuple[Grid, Grid]]) -> list[Primitive]:
    """Все семейства × палитра задачи → примитивы со связанной переменной."""
    return [make(fam, c) for fam in _FAMILIES for c in task_palette(pairs)]
