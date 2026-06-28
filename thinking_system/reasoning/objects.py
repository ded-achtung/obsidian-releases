"""Третий слой общих примитивов: направленная гравитация, восстановление симметрии,
калейдоскоп, обрезка рамки.

Продолжаем расширять словарь восприятия (а не подгонять решатели под задачи). Все
операции общие и параметро-свободные; каждая ломает инвариант геометрии:

  • gravity_up/left/right — гравитация в 3 стороны (down уже есть в perception);
                            сдвиг ЗАВИСИТ от содержимого, не фиксированная перестановка;
  • restore_symmetry     — починить узор под ОККЛЮДЕРОМ: цвет-заслонку система
                            находит сама (тот, чьё стирание делает узор симметричным),
                            и заполняет его клетки по зеркалам — большое семейство ARC;
  • mirror_quad          — калейдоскоп 2×2 (узор и три его отражения), размер ×2;
  • trim_border          — снять сплошную внешнюю рамку (одно кольцо одного цвета).

Та же индукция находит и компонует их с прежними слоями.
"""

from __future__ import annotations

from thinking_system.reasoning.induction import Primitive
from thinking_system.reasoning.grids import to_grid, flip_h, flip_v
from thinking_system.reasoning.perception import _dims


def gravity_up(g):
    """Уронить непустые клетки вверх в каждом столбце."""
    rows, cols = _dims(g)
    out = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        vals = [g[r][c] for r in range(rows) if g[r][c] != 0]
        for k, v in enumerate(vals):
            out[k][c] = v
    return to_grid(out)


def gravity_left(g):
    """Сдвинуть непустые клетки влево в каждой строке."""
    rows, cols = _dims(g)
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        vals = [v for v in g[r] if v != 0]
        for k, v in enumerate(vals):
            out[r][k] = v
    return to_grid(out)


def gravity_right(g):
    """Сдвинуть непустые клетки вправо в каждой строке."""
    rows, cols = _dims(g)
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        vals = [v for v in g[r] if v != 0]
        for k, v in enumerate(vals):
            out[r][cols - len(vals) + k] = v
    return to_grid(out)


def _syms(rows, cols):
    syms = [lambda r, c: (r, cols - 1 - c), lambda r, c: (rows - 1 - r, c),
            lambda r, c: (rows - 1 - r, cols - 1 - c)]
    if rows == cols:
        syms.append(lambda r, c: (c, r))                       # транспонирование — для квадрата
    return syms


def _restore_for_hole(g, hole):
    """Заполнить клетки цвета `hole` по симметриям, которые УВАЖАЕТ видимая часть.

    Окклюдер — цвет, чьё стирание оставляет симметричный узор: берём только те
    симметрии (h/v/180/диаг), под которыми все ВИДИМЫЕ пары клеток совпадают, и по ним
    достраиваем заслонённые клетки. Если ни одна симметрия не уважается — цвет не
    заслонка. Возвращает (сетка, сколько_заполнено) или (None, 0).
    """
    rows, cols = _dims(g)
    good = []
    for s in _syms(rows, cols):
        ok = all(g[r][c] == hole or g[s(r, c)[0]][s(r, c)[1]] == hole
                 or g[r][c] == g[s(r, c)[0]][s(r, c)[1]]
                 for r in range(rows) for c in range(cols))
        if ok:
            good.append(s)
    if not good:
        return None, 0
    out = [list(row) for row in g]
    filled = 0
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != hole:
                continue
            cands = [g[mr][mc] for mr, mc in (s(r, c) for s in good) if g[mr][mc] != hole]
            if cands:
                out[r][c] = cands[0]; filled += 1
    return to_grid(out), filled


def restore_symmetry(g):
    """Починить симметрию под заслонкой: цвет-окклюдер находится автоматически.

    Для каждого цвета-гипотезы заслонки заполняем его клетки по симметриям, которые
    уважает видимая часть, и берём цвет, дающий больше всего заполнений. Так система
    сама решает, что закрывает узор, и восстанавливает его — без подсказки извне.

    Если узор УЖЕ симметричен по горизонтали и вертикали, восстанавливать нечего:
    возвращаем без изменений (иначе «диагональная» симметрия квадрата ложно приняла бы
    обычный цвет-узор за заслонку и переписала корректные клетки).
    """
    if flip_h(g) == g and flip_v(g) == g:
        return g
    best, best_filled = None, 0
    for hole in sorted({v for row in g for v in row}):
        out, filled = _restore_for_hole(g, hole)
        if out is not None and filled > best_filled:
            best, best_filled = out, filled
    return best if best is not None else g


def mirror_quad(g):
    """Калейдоскоп: узор и три его отражения в сетке 2×2 (размер удваивается)."""
    top = tuple(row + row[::-1] for row in g)                  # g | flip_h(g)
    return top + top[::-1]                                     # стек с flip_v


def trim_border(g):
    """Снять сплошную внешнюю рамку одного цвета (одно кольцо)."""
    rows, cols = _dims(g)
    if rows < 3 or cols < 3:
        return g
    ring = set()
    for c in range(cols):
        ring.add(g[0][c]); ring.add(g[rows - 1][c])
    for r in range(rows):
        ring.add(g[r][0]); ring.add(g[r][cols - 1])
    if len(ring) == 1:
        return to_grid([[g[r][c] for c in range(1, cols - 1)] for r in range(1, rows - 1)])
    return g


def object_primitives() -> list[Primitive]:
    """Третий слой общих примитивов (гравитация по сторонам, симметрия, рамка)."""
    return [
        Primitive("grav_up", gravity_up),
        Primitive("grav_left", gravity_left),
        Primitive("grav_right", gravity_right),
        Primitive("restore_sym", restore_symmetry),
        Primitive("mirror_quad", mirror_quad),
        Primitive("trim_border", trim_border),
    ]
