"""Слой ОТНОШЕНИЙ: бинарные операции над половинами сетки (наложение, маска, разность).

Все прежние слои унарны и не сопоставляют ЧАСТИ сетки друг с другом, поэтому
второй по частоте мотив ARC — «вход состоит из двух половин, выход — их
поклеточная комбинация» — был невыразим ни на какой глубине. Здесь общие
операции наложения (алгебра, не решатели):

  • and_h / and_v   — клетка занята в ОБЕИХ половинах (цвет — из первой);
  • or_h  / or_v    — занята хотя бы в одной (приоритет первой);
  • xor_h / xor_v   — занята ровно в ОДНОЙ (её цвет);
  • diff_h / diff_v — занята в первой и пуста во второй (маска-вычитание).

Половины: по горизонтали лево|право, по вертикали верх/низ; при нечётном
размере средний ряд/столбец (обычно линия-разделитель) отбрасывается — это
общая геометрическая конвенция, а не знание о конкретной задаче. Итоговый
цвет обычно задаётся композицией с paint[c]: «xor_h ▸ paint[3]».
"""

from __future__ import annotations

from thinking_system.reasoning.grids import Grid, to_grid
from thinking_system.reasoning.induction import Primitive

_RULES = {
    "and": lambda a, b: a if a and b else 0,
    "or": lambda a, b: a if a else b,
    "xor": lambda a, b: (a or b) if bool(a) != bool(b) else 0,
    "diff": lambda a, b: a if a and not b else 0,
}


def _halves_h(g: Grid) -> tuple[Grid, Grid]:
    cols = len(g[0]) if g else 0
    if cols < 2:
        raise ValueError("не из чего взять половины")
    half = cols // 2
    return (tuple(r[:half] for r in g),
            tuple(r[cols - half:] for r in g))               # нечётная ширина → средний столбец прочь


def _halves_v(g: Grid) -> tuple[Grid, Grid]:
    rows = len(g)
    if rows < 2:
        raise ValueError("не из чего взять половины")
    half = rows // 2
    return g[:half], g[rows - half:]


def _combine(a: Grid, b: Grid, rule) -> Grid:
    return to_grid([[rule(x, y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)])


def make(rule_name: str, axis: str) -> Primitive:
    rule = _RULES[rule_name]
    halves = _halves_h if axis == "h" else _halves_v
    def fn(g, _r=rule, _h=halves):
        a, b = _h(g)
        return _combine(a, b, _r)
    return Primitive(f"{rule_name}_{axis}", fn)


def pairwise_primitives() -> list[Primitive]:
    """Общие операции наложения половин (шестой слой seed)."""
    return [make(r, ax) for r in _RULES for ax in ("h", "v")]
