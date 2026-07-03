"""Объект как ПЕРЕМЕННАЯ: each[f], pick[k], big/small/one[f] — в двух связностях.

Третий вид переменных в языке (после цвета и шага-дырки): квантификация по
ОБЪЕКТАМ сетки. each[f] раскладывает сетку на связные объекты, применяет f к
вырезке каждого и собирает обратно — «отрази КАЖДЫЙ объект» вместо «отрази
сетку». pick[k] оставляет k-й по размеру объект. Предикатные семейства —
условная переменная «какому объекту»: big[f] — самому большому, small[f] —
самому маленькому, one[f] — всем объектам размера 1.

СВЯЗНОСТЬ — тоже переменная. Обычные имена (each/big/…) считают объектом
разноцветную связную область (как keep_largest); имена с суффиксом «c»
(eachc/bigc/onec/…) — ОДНОЦВЕТНУЮ: клетки соединяются, только если цвет
совпадает. Например onec[paint[0]] стирает одиночные клетки чужого цвета
внутри фигуры — цветослепая связность их вообще не видит как объекты.
Поиск сам связывает нужный режим, как и остальные переменные.

В each-семейство входят только операции, для которых пообъектное применение
ОТЛИЧАЕТСЯ от глобального (симметрии, gravity); f, меняющая размер вырезки,
честно недопустима (исключение → поиск отбрасывает шаг).
"""

from __future__ import annotations

import re

from thinking_system.reasoning import parametric
from thinking_system.reasoning.grids import Grid, flip_h, flip_v, rot90, to_grid, transpose
from thinking_system.reasoning.induction import Primitive
from thinking_system.reasoning.perception import _N4, _components, _dims, gravity

_EACH_RE = re.compile(r"^each(c?)\[(.+)\]$")
_PICK_RE = re.compile(r"^pick(c?)\[(\d)\]$")
_WHERE_RE = re.compile(r"^(big|small|one)(c?)\[(.+)\]$")

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


def _components_same_color(g: Grid) -> list:
    """Связные ОДНОЦВЕТНЫЕ области: сосед присоединяется, только если цвет тот же."""
    rows, cols = _dims(g)
    seen = [[False] * cols for _ in range(rows)]
    comps = []
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != 0 and not seen[r][c]:
                color, stack, cells = g[r][c], [(r, c)], []
                seen[r][c] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in _N4:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < rows and 0 <= nx < cols and not seen[ny][nx]
                                and g[ny][nx] == color):
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _comps(g: Grid, same_color: bool) -> list:
    comps = _components_same_color(g) if same_color else _components(g)
    if not comps:
        raise ValueError("нет объектов")
    return comps


def _crop(g: Grid, cells: list) -> tuple[Grid, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    sub = [[0] * (max(cs) - c0 + 1) for _ in range(max(rs) - r0 + 1)]
    for r, c in cells:
        sub[r - r0][c - c0] = g[r][c]
    return to_grid(sub), r0, c0


def each_apply(g: Grid, f, *, same_color: bool = False) -> Grid:
    """f к вырезке каждого связного объекта; сборка на прежних местах."""
    rows, cols = _dims(g)
    out = [[0] * cols for _ in range(rows)]
    for cells in _comps(g, same_color):
        sub, r0, c0 = _crop(g, cells)
        res = f(sub)
        if _dims(res) != _dims(sub):
            raise ValueError("объект изменил размер")       # честно недопустимый шаг
        for i, row in enumerate(res):
            for j, v in enumerate(row):
                if v != 0:
                    out[r0 + i][c0 + j] = v
    return to_grid(out)


def pick_apply(g: Grid, k: int, *, same_color: bool = False) -> Grid:
    """Оставить k-й по размеру связный объект (1 = крупнейший), остальное обнулить."""
    comps = _comps(g, same_color)
    if len(comps) < k:
        raise ValueError("объектов меньше k")
    keep = set(sorted(comps, key=len, reverse=True)[k - 1])
    rows, cols = _dims(g)
    return to_grid([[g[r][c] if (r, c) in keep else 0 for c in range(cols)]
                    for r in range(rows)])


def where_apply(g: Grid, pred, f, *, same_color: bool = False) -> Grid:
    """f к вырезке каждого объекта, выбранного предикатом; прочие — без изменений."""
    out = [list(r) for r in g]
    for cells in pred(_comps(g, same_color)):
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


def _sfx(same_color: bool) -> str:
    return "c" if same_color else ""


def make_each(inner_name: str, *, same_color: bool = False) -> Primitive:
    fn = _INNER[inner_name]
    return Primitive(f"each{_sfx(same_color)}[{inner_name}]",
                     lambda g, _f=fn, _s=same_color: each_apply(g, _f, same_color=_s))


def make_pick(k: int, *, same_color: bool = False) -> Primitive:
    return Primitive(f"pick{_sfx(same_color)}[{k}]",
                     lambda g, _k=k, _s=same_color: pick_apply(g, _k, same_color=_s))


def make_where(pred_name: str, inner: Primitive, *, same_color: bool = False) -> Primitive:
    pred = _PREDS[pred_name]
    return Primitive(f"{pred_name}{_sfx(same_color)}[{inner.name}]",
                     lambda g, _p=pred, _f=inner.fn, _s=same_color:
                     where_apply(g, _p, _f, same_color=_s))


def instantiate() -> list[Primitive]:
    """each/pick в обеих связностях (палитра не нужна — переменная здесь сам объект)."""
    return ([make_each(n, same_color=s) for s in (False, True) for n in _INNER]
            + [make_pick(k, same_color=s) for s in (False, True) for k in _PICK_RANKS])


def instantiate_predicates(pairs: list) -> list[Primitive]:
    """big/small/one × paint[палитра ∪ {0}] × обе связности."""
    actions = [parametric.make("paint", c) for c in parametric.task_palette(pairs) + [0]]
    return [make_where(p, a, same_color=s)
            for s in (False, True) for p in _PREDS for a in actions]


def _inner_by_name(name: str) -> Primitive | None:
    if name in _INNER:
        return Primitive(name, _INNER[name])
    return parametric.by_name(name)


def by_name(name: str) -> Primitive | None:
    """«each[flip_h]» / «pickc[2]» / «bigc[paint[3]]» → примитив (из имён состояния)."""
    m = _EACH_RE.match(name)
    if m and m.group(2) in _INNER:
        return make_each(m.group(2), same_color=bool(m.group(1)))
    m = _PICK_RE.match(name)
    if m:
        return make_pick(int(m.group(2)), same_color=bool(m.group(1)))
    m = _WHERE_RE.match(name)
    if m:
        inner = _inner_by_name(m.group(3))
        if inner is not None:
            return make_where(m.group(1), inner, same_color=bool(m.group(2)))
    return None
