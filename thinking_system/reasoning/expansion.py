"""Слой РАСШИРЕНИЯ: операции, закономерно меняющие размер сетки (алгебра, не решатели).

Диагноз замеров (AUDIT, дополнения 6–7): охват упёрся одновременно в глубину
поиска И в выразительность языка — существующий seed почти целиком сохраняет
размер сетки, поэтому самые частые мотивы ARC «клетка → блок n×n», «повтори
сетку», «сетка + её зеркало» невыразимы ни на какой глубине. Этот слой добавляет
ОБЩИЕ операции алгебры сеток (как и прежние слои — параметро-свободные и не
знающие ни о каких конкретных задачах):

  • upscale2 / upscale3 — каждая клетка → блок 2×2 / 3×3;
  • tile_h / tile_v     — сетка, повторённая дважды по горизонтали / вертикали;
  • mirror_h / mirror_v — сетка рядом со своим зеркалом (симметризация).

Взрывной рост в композициях (upscale3∘upscale3∘…) отсекает общий guard размера.
"""

from __future__ import annotations

from thinking_system.reasoning.grids import Grid
from thinking_system.reasoning.induction import Primitive


def _upscale(g: Grid, k: int) -> Grid:
    return tuple(tuple(v for v in row for _ in range(k)) for row in g for _ in range(k))


def upscale2(g: Grid) -> Grid:
    """Каждая клетка → блок 2×2 (размер ×2)."""
    return _upscale(g, 2)


def upscale3(g: Grid) -> Grid:
    """Каждая клетка → блок 3×3 (размер ×3)."""
    return _upscale(g, 3)


def tile_h(g: Grid) -> Grid:
    """Сетка, повторённая дважды по горизонтали."""
    return tuple(row + row for row in g)


def tile_v(g: Grid) -> Grid:
    """Сетка, повторённая дважды по вертикали."""
    return g + g


def mirror_h(g: Grid) -> Grid:
    """Сетка и её горизонтальное зеркало рядом (симметризация вширь)."""
    return tuple(row + row[::-1] for row in g)


def mirror_v(g: Grid) -> Grid:
    """Сетка и её вертикальное зеркало друг под другом (симметризация ввысь)."""
    return g + g[::-1]


def expansion_primitives() -> list[Primitive]:
    """Общие размер-меняющие операции (пятый слой seed)."""
    return [
        Primitive("upscale2", upscale2),
        Primitive("upscale3", upscale3),
        Primitive("tile_h", tile_h),
        Primitive("tile_v", tile_v),
        Primitive("mirror_h", mirror_h),
        Primitive("mirror_v", mirror_v),
    ]
