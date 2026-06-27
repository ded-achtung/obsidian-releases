"""Общий seed примитивов над СЕТКАМИ — чтобы тот же ростовой механизм брал ARC-домен.

Сетка — кортеж кортежей чисел (цвета). Примитивы здесь ОБЩИЕ (группа симметрий
сетки: отражения, поворот, транспонирование), а не «решатель задачи X». На этом
seed та же самонаращивающаяся библиотека (library_learning) растит грид-абстракции
из решённых задач — путь к ARC через рост языка, а не ручной DSL под бенчмарк.
"""

from __future__ import annotations

from thinking_system.reasoning.induction import Primitive

Grid = tuple


def flip_h(g: Grid) -> Grid:
    return tuple(r[::-1] for r in g)


def flip_v(g: Grid) -> Grid:
    return g[::-1]


def transpose(g: Grid) -> Grid:
    return tuple(zip(*g))


def rot90(g: Grid) -> Grid:
    return tuple(zip(*g[::-1]))


def rot180(g: Grid) -> Grid:
    return rot90(rot90(g))


def recolor1(g: Grid) -> Grid:
    """Перекрасить: каждый цвет +1 (общая операция над цветами, не из группы симметрий)."""
    return tuple(tuple(v + 1 for v in r) for r in g)


def to_grid(rows) -> Grid:
    """Список списков → сетка (кортеж кортежей) для сравнения/хеширования."""
    return tuple(tuple(r) for r in rows)


def grid_primitives() -> list[Primitive]:
    """Общий набор грид-операций (симметрии); rot180 НЕ дан — растится абстракцией."""
    return [
        Primitive("id", lambda g: g),
        Primitive("flip_h", flip_h),
        Primitive("flip_v", flip_v),
        Primitive("transpose", transpose),
        Primitive("rot90", rot90),
        Primitive("color+1", recolor1),
    ]
