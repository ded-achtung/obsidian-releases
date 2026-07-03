"""Объект как ПЕРЕМЕННАЯ: each[f] — к каждому объекту, pick[k] — выбор, big/small/one[f] — по условию.

Третий вид переменных в языке (после цвета и шага-дырки): квантификация по
ОБЪЕКТАМ сетки. each[f] раскладывает сетку на связные объекты (та же 4-связность,
что у keep_largest), применяет f к вырезке каждого объекта и собирает сетку
обратно — «отрази КАЖДЫЙ объект» вместо «отрази сетку». pick[k] оставляет k-й
по размеру объект — обобщение keep_largest ранговой переменной.

Предикатные семейства — УСЛОВНАЯ переменная «какому объекту»: big[f] применяет
f только к самому большому объекту, small[f] — к самому маленькому, one[f] — ко
всем объектам размера 1; остальные объекты не трогаются. Действия f — перекраска
paint[c] по палитре задачи и стирание paint[0]: «перекрась самый большой в 3» =
big[paint[3]], «сотри одиночки» = one[paint[0]].

В each-семейство входят только операции, для которых пообъектное применение
ОТЛИЧАЕТСЯ от глобального (симметрии, gravity); f, меняющая размер вырезки,
честно недопустима (исключение → поиск отбрасывает шаг).
"""

from __future__ import annotations

import re

from thinking_system.reasoning import parametric
from thinking_system.reasoning.grids import Grid, flip_h, flip_v, rot90, to_grid, transpose
from thinking_system.reasoning.induction import Primitive
from thinking_system.reasoning.perception import _components, _dims, gravity

_EACH_RE = re.compile(r"^each\[(.+)\]$")
_PICK_RE = re.compile(r"^pick\[(\d)\]$")
_WHERE_RE = re.compile(r"^(big|small|one)\[(.+)\]$")

# операции, пообъектное применение которых не совпадает с глобальным
_INNER = {"flip_h": flip_h, "flip_v": flip_v, "transpose": transpose,
          "rot90": rot90, "gravity": gravity}
_PICK_RANKS = (2, 3)                     # ранг 1 — это существующий keep_largest

# предикат → какие объекты выбрать из списка компонент
_PREDS = {
    "big": lambda comps: [max(comps, key=len)],
    "small": lambda comps: [min(comps, key=len)],
    "one": lambda comps: [c for c in comps if len(c) == 1],
}


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


def where_apply(g: Grid, pred, f) -> Grid:
    """f к вырезке каждого объекта, выбранного предикатом; прочие — без изменений."""
    comps = _components(g)
    if not comps:
        raise ValueError("нет объектов")
    out = [list(r) for r in g]
    for cells in pred(comps):
        sub, r0, c0 = _crop(g, cells)
        res = f(sub)
        if _dims(res) != _dims(sub):
            raise ValueError("объект изменил размер")
        for r, c in cells:                                   # стереть выбранный объект…
            out[r][c] = 0
        for i, row in enumerate(res):                        # …и вписать результат f
            for j, v in enumerate(row):
                if v != 0:
                    out[r0 + i][c0 + j] = v
    return to_grid(out)


def make_each(inner_name: str) -> Primitive:
    fn = _INNER[inner_name]
    return Primitive(f"each[{inner_name}]", lambda g, _f=fn: each_apply(g, _f))


def make_pick(k: int) -> Primitive:
    return Primitive(f"pick[{k}]", lambda g, _k=k: pick_apply(g, _k))


def make_where(pred_name: str, inner: Primitive) -> Primitive:
    pred = _PREDS[pred_name]
    return Primitive(f"{pred_name}[{inner.name}]",
                     lambda g, _p=pred, _f=inner.fn: where_apply(g, _p, _f))


def instantiate() -> list[Primitive]:
    """Объектные операции, не зависящие от палитры (each/pick)."""
    return [make_each(n) for n in _INNER] + [make_pick(k) for k in _PICK_RANKS]


def instantiate_predicates(pairs: list) -> list[Primitive]:
    """Условная переменная «какому объекту»: big/small/one × paint[палитра ∪ {0}]."""
    actions = [parametric.make("paint", c) for c in parametric.task_palette(pairs) + [0]]
    return [make_where(p, a) for p in _PREDS for a in actions]


def _inner_by_name(name: str) -> Primitive | None:
    if name in _INNER:
        return Primitive(name, _INNER[name])
    return parametric.by_name(name)


def by_name(name: str) -> Primitive | None:
    """«each[flip_h]» / «pick[2]» / «big[paint[3]]» → примитив (из имён состояния)."""
    m = _EACH_RE.match(name)
    if m and m.group(1) in _INNER:
        return make_each(m.group(1))
    m = _PICK_RE.match(name)
    if m:
        return make_pick(int(m.group(1)))
    m = _WHERE_RE.match(name)
    if m:
        inner = _inner_by_name(m.group(2))
        if inner is not None:
            return make_where(m.group(1), inner)
    return None
