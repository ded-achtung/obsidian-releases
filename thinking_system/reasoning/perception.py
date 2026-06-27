"""Перцептивные примитивы: операции, ЛОМАЮЩИЕ инварианты геометрического seed.

Геометрические примитивы (симметрии + равномерная перекраска) замкнуты: каждая
клетка едет по фиксированной геометрии и читает ровно одну входную клетку. Поэтому
композицией из них не выразить операции, которые СМОТРЯТ НА СОДЕРЖИМОЕ/СОСЕДЕЙ и
АГРЕГИРУЮТ. Здесь — именно такие общие кирпичи:

  • gravity        — уронить непустые клетки вниз (сдвиг ЗАВИСИТ от данных);
  • keep_largest   — оставить крупнейший СВЯЗНЫЙ объект (анализ соседей);
  • denoise        — убрать одиночные клетки без соседей (соседство);
  • bounding_box   — обрезать до содержимого (меняет число клеток);
  • count_nonzero  — число непустых клеток как 1×1 (агрегация многих в одно);
  • fill_holes     — залить замкнутый фон цветом рамки (транзитивная достижимость).

Та же индукция/рост библиотеки находит и КОМПОНУЕТ их с геометрией — система
получает выход за пределы замыкания, не кодируя решатели под конкретные задачи.
"""

from __future__ import annotations

from thinking_system.reasoning.induction import Primitive
from thinking_system.reasoning.grids import to_grid

_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dims(g):
    return len(g), (len(g[0]) if g else 0)


def gravity(g):
    """Уронить непустые клетки вниз в каждом столбце (сдвиг зависит от содержимого)."""
    rows, cols = _dims(g)
    out = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        vals = [g[r][c] for r in range(rows) if g[r][c] != 0]
        for k, v in enumerate(vals):
            out[rows - len(vals) + k][c] = v
    return to_grid(out)


def _components(g):
    rows, cols = _dims(g)
    seen = [[False] * cols for _ in range(rows)]
    comps = []
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != 0 and not seen[r][c]:
                stack = [(r, c)]
                seen[r][c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in _N4:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and g[ny][nx] != 0 and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def keep_largest(g):
    """Оставить крупнейший связный объект (непустой), остальное обнулить."""
    comps = _components(g)
    if not comps:
        return g
    keep = set(max(comps, key=len))
    rows, cols = _dims(g)
    return to_grid([[g[r][c] if (r, c) in keep else 0 for c in range(cols)] for r in range(rows)])


def denoise(g):
    """Убрать непустые клетки без непустых 4-соседей (одиночный шум)."""
    rows, cols = _dims(g)

    def has_nb(r, c):
        return any(0 <= r + dy < rows and 0 <= c + dx < cols and g[r + dy][c + dx] != 0 for dy, dx in _N4)

    return to_grid([[g[r][c] if (g[r][c] != 0 and has_nb(r, c)) else 0 for c in range(cols)] for r in range(rows)])


def bounding_box(g):
    """Обрезать сетку до прямоугольника, охватывающего непустые клетки."""
    rows, cols = _dims(g)
    rs = [r for r in range(rows) if any(g[r])]
    cs = [c for c in range(cols) if any(g[r][c] for r in range(rows))]
    if not rs or not cs:
        return g
    return to_grid([[g[r][c] for c in range(min(cs), max(cs) + 1)] for r in range(min(rs), max(rs) + 1)])


def count_nonzero(g):
    """Число непустых клеток как сетка 1×1 (агрегация многих в одно)."""
    return ((sum(1 for r in g for v in r if v != 0),),)


def fill_holes(g):
    """Залить замкнутый фон (нули, не достижимые от рамки) цветом-меткой (max-цвет)."""
    rows, cols = _dims(g)
    border_bg = [[False] * cols for _ in range(rows)]
    stack = []
    for r in range(rows):
        for c in (0, cols - 1):
            if g[r][c] == 0 and not border_bg[r][c]:
                border_bg[r][c] = True; stack.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if g[r][c] == 0 and not border_bg[r][c]:
                border_bg[r][c] = True; stack.append((r, c))
    while stack:                                              # фон, достижимый от рамки
        y, x = stack.pop()
        for dy, dx in _N4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and g[ny][nx] == 0 and not border_bg[ny][nx]:
                border_bg[ny][nx] = True; stack.append((ny, nx))
    fill = max((v for row in g for v in row), default=0)
    return to_grid([[fill if (g[r][c] == 0 and not border_bg[r][c]) else g[r][c] for c in range(cols)] for r in range(rows)])


def perception_primitives() -> list[Primitive]:
    """Общие перцептивные операции, ломающие инварианты геометрического seed."""
    return [
        Primitive("gravity", gravity),
        Primitive("keep_largest", keep_largest),
        Primitive("denoise", denoise),
        Primitive("bbox", bounding_box),
        Primitive("count", count_nonzero),
        Primitive("fill_holes", fill_holes),
    ]
