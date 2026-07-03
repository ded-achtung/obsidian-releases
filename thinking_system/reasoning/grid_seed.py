"""Полный перцептивный seed над сетками — единая точка сборки всех слоёв.

Словарь восприятия рос слоями (каждый ломает свой инвариант геометрии):

  • grid_primitives        — геометрия (симметрии + перекраска);
  • perception_primitives  — гравитация, связные объекты, счёт, заливка;
  • structural_primitives  — симметрия-достройка, фрактал, контур, агрегации;
  • object_primitives      — гравитация по сторонам, симметрия под заслонкой, рамка.

`full_grid_seed()` собирает их вместе — это вход для library_learning, которая
сама абстрагирует частые КОМБИНАЦИИ (например «bbox ▸ fractal», встреченную на
реальном ARC) в новые именованные операции, сокращая глубину поиска.
"""

from __future__ import annotations

from thinking_system.reasoning.grids import grid_primitives
from thinking_system.reasoning.induction import Primitive
from thinking_system.reasoning.perception import perception_primitives
from thinking_system.reasoning.structural import structural_primitives
from thinking_system.reasoning.objects import object_primitives


def full_grid_seed() -> list:
    """Все слои перцептивных примитивов над сетками в одном наборе."""
    return (grid_primitives() + perception_primitives()
            + structural_primitives() + object_primitives())


def guard(p: Primitive, *, max_cells: int = 10_000) -> Primitive:
    """Примитив с ограничителем: гигантская промежуточная сетка = недопустимый шаг.

    Выходы ARC ≤ 30×30 = 900 клеток; взрыв вроде fractal∘fractal на большой сетке —
    тупик, который иначе съедает время/память поиска.
    """
    def fn(g, _f=p.fn):
        out = _f(g)
        if isinstance(out, tuple) and out and isinstance(out[0], tuple) \
                and len(out) * len(out[0]) > max_cells:
            raise ValueError("сетка слишком велика")
        return out
    return Primitive(p.name, fn, p.cost)


def guarded_grid_seed(*, max_cells: int = 10_000) -> list:
    """Полный seed с ограничителем размера промежуточных сеток (для реального ARC)."""
    return [guard(p, max_cells=max_cells) for p in full_grid_seed()]
