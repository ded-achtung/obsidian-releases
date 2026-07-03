"""Объект как ПЕРЕМЕННАЯ: each[f] применяет f к каждому связному объекту, pick[k] выбирает.

Третий вид переменных в языке (после цвета и шага-дырки): квантификация по
ОБЪЕКТАМ сетки. each[f] раскладывает сетку на связные объекты (та же 4-связность,
что у keep_largest), применяет f к вырезке каждого объекта и собирает сетку
обратно — «отрази КАЖДЫЙ объект» вместо «отрази сетку». pick[k] оставляет k-й
по размеру объект — обобщение keep_largest ранговой переменной.

В each-семейство входят только операции, для которых пообъектное применение
ОТЛИЧАЕТСЯ от глобального (симметрии, gravity); f, меняющая размер вырезки,
честно недопустима (исключение → поиск отбрасывает шаг).
"""

from __future__ import annotations

import re

from thinking_system.reasoning.grids import Grid, flip_h, flip_v, rot90, to_grid, transpose
from thinking_system.reasoning.induction import Primitive
from thinking_system.reasoning.perception import _components, _dims, gravity

_EACH_RE = re.compile(r"^each\[(.+)\]$")
_PICK_RE = re.compile(r"^pick\[(\d)\]$")

# операции, пообъектное применение которых не совпадает с глобальным
_INNER = {"flip_h": flip_h, "flip_v": flip_v, "transpose": transpose,
          "rot90": rot90, "gravity": gravity}
_PICK_RANKS = (2, 3)                     # ранг 1 — это существующий keep_largest


def _crop(g: Grid, cells: list) -> tuple[Grid, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    sub = [[0] * (max(cs) - c0 + 1) for _ in range(max(rs) - r0 + 1)]
    for r, c in cells:
        sub[r - r0][c - c0] = g[r][c]
    return to_grid(sub), r0, c0


def each_apply(g: Grid, f) -> Grid:
    """f к вырезке каждого связного объекта; сборка на прежних местах."""
    comps = _components(g)
    if not comps:
        raise ValueError("нет объектов")
    rows, cols = _dims(g)
    out = [[0] * cols for _ in range(rows)]
    for cells in comps:
        sub, r0, c0 = _crop(g, cells)
        res = f(sub)
        if _dims(res) != _dims(sub):
            raise ValueError("объект изменил размер")       # честно недопустимый шаг
        for i, row in enumerate(res):
            for j, v in enumerate(row):
                if v != 0:
                    out[r0 + i][c0 + j] = v
    return to_grid(out)


def pick_apply(g: Grid, k: int) -> Grid:
    """Оставить k-й по размеру связный объект (1 = крупнейший), остальное обнулить."""
    comps = _components(g)
    if len(comps) < k:
        raise ValueError("объектов меньше k")
    keep = set(sorted(comps, key=len, reverse=True)[k - 1])
    rows, cols = _dims(g)
    return to_grid([[g[r][c] if (r, c) in keep else 0 for c in range(cols)]
                    for r in range(rows)])


def make_each(inner_name: str) -> Primitive:
    fn = _INNER[inner_name]
    return Primitive(f"each[{inner_name}]", lambda g, _f=fn: each_apply(g, _f))


def make_pick(k: int) -> Primitive:
    return Primitive(f"pick[{k}]", lambda g, _k=k: pick_apply(g, _k))


def instantiate() -> list[Primitive]:
    """Все объектные операции (не зависят от палитры — переменная здесь сам объект)."""
    return [make_each(n) for n in _INNER] + [make_pick(k) for k in _PICK_RANKS]


def by_name(name: str) -> Primitive | None:
    """«each[flip_h]» / «pick[2]» → примитив (восстановление решений из имён)."""
    m = _EACH_RE.match(name)
    if m and m.group(1) in _INNER:
        return make_each(m.group(1))
    m = _PICK_RE.match(name)
    return make_pick(int(m.group(1))) if m else None
