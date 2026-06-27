"""Структурно-объектные примитивы: второй слой выхода за барьер замыкания.

`perception.py` дал первые кирпичи, ломающие инварианты геометрии (гравитация,
связные объекты, счёт, заливка). Здесь — следующий слой ОБЩИХ операций, каждая
из которых тоже не выразима композицией симметрий, но покрывает свои семейства
задач ARC:

  • complete_symmetry — достроить симметричный узор по зеркальным двойникам
                        (одна выходная клетка читает несколько входных);
  • replicate_by_self — фрактал: разложить копии входа туда, где вход непуст
                        (размер выхода = rows²×cols², зависит от содержимого);
  • outline          — оставить только границу объектов (анализ соседей);
  • count_objects    — число связных объектов как 1×1 (агрегация);
  • most_common_color— самый частый ненулевой цвет как 1×1 (агрегация по гистограмме).

Это не решатели под конкретные задачи: каждая операция общая, а ТА ЖЕ индукция
находит и компонует их с геометрией и перцепцией. Путь дальше — больше таких
общих примитивов, а не подгонка под бенчмарк.
"""

from __future__ import annotations

from collections import Counter

from thinking_system.reasoning.induction import Primitive
from thinking_system.reasoning.grids import to_grid
from thinking_system.reasoning.perception import _components, _dims, _N4   # переиспользуем разбор сцены


def complete_symmetry(g):
    """Достроить узор: пустую клетку заполнить значением её зеркальных двойников.

    Для каждого нуля смотрим h-, v- и 180°-зеркала. Если непустые двойники есть и
    СОГЛАСОВАНЫ (один цвет) — заполняем. Никогда не стираем: чинит повреждённую
    симметрию, не ломая остального. Одна выходная клетка читает несколько входных —
    это и есть нарушение инварианта «клетка ← ровно одна клетка».
    """
    rows, cols = _dims(g)
    out = [list(row) for row in g]
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != 0:
                continue
            cands = []
            for mr, mc in ((r, cols - 1 - c), (rows - 1 - r, c), (rows - 1 - r, cols - 1 - c)):
                v = g[mr][mc]
                if v != 0:
                    cands.append(v)
            if cands and all(v == cands[0] for v in cands):
                out[r][c] = cands[0]
    return to_grid(out)


def replicate_by_self(g):
    """Фрактал: выход rows²×cols², в блок (R,C) кладём копию g, если g[R][C]≠0.

    Размещение копий ЗАВИСИТ от содержимого входа, а размер выхода растёт — ни то,
    ни другое недостижимо фиксированной перестановкой клеток. Классическое семейство
    ARC «нарисуй вход там, где вход непуст».
    """
    rows, cols = _dims(g)
    out = [[0] * (cols * cols) for _ in range(rows * rows)]
    for big_r in range(rows):
        for big_c in range(cols):
            if g[big_r][big_c] != 0:
                for r in range(rows):
                    for c in range(cols):
                        out[big_r * rows + r][big_c * cols + c] = g[r][c]
    return to_grid(out)


def outline(g):
    """Оставить только граничные клетки объектов (есть пустой/краевой 4-сосед)."""
    rows, cols = _dims(g)

    def on_border(r, c):
        for dy, dx in _N4:
            ny, nx = r + dy, c + dx
            if not (0 <= ny < rows and 0 <= nx < cols) or g[ny][nx] == 0:
                return True
        return False

    return to_grid([[g[r][c] if (g[r][c] != 0 and on_border(r, c)) else 0
                     for c in range(cols)] for r in range(rows)])


def count_objects(g):
    """Число связных непустых объектов как сетка 1×1 (агрегация сцены в число)."""
    return ((len(_components(g)),),)


def most_common_color(g):
    """Самый частый ненулевой цвет как 1×1 (при равенстве — меньший цвет)."""
    cnt = Counter(v for row in g for v in row if v != 0)
    if not cnt:
        return ((0,),)
    best = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return ((best,),)


def structural_primitives() -> list[Primitive]:
    """Структурно-объектные операции — второй слой выхода за барьер замыкания."""
    return [
        Primitive("complete_sym", complete_symmetry),
        Primitive("fractal", replicate_by_self),
        Primitive("outline", outline),
        Primitive("count_obj", count_objects),
        Primitive("top_color", most_common_color),
    ]
