"""Тесты перцептивных примитивов: корректность, барьер замыкания, выход за него."""

from __future__ import annotations

from thinking_system.reasoning.grids import grid_primitives, to_grid, flip_h
from thinking_system.reasoning.perception import (perception_primitives, gravity, keep_largest,
                                                  denoise, bounding_box, count_nonzero, fill_holes)
from thinking_system.reasoning.induction import induce

GEOM = grid_primitives()
FULL = GEOM + perception_primitives()


def test_primitives_correct() -> None:
    assert gravity(to_grid([[5, 0], [0, 0], [0, 7]])) == ((0, 0), (0, 0), (5, 7))
    assert keep_largest(to_grid([[1, 1, 0, 0], [1, 0, 0, 2], [0, 0, 0, 0]])) == ((1, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0))
    assert count_nonzero(to_grid([[1, 0, 1], [0, 1, 0]])) == ((3,),)
    assert denoise(to_grid([[1, 0, 0], [0, 0, 5]])) == ((0, 0, 0), (0, 0, 0))           # обе клетки одиночны
    assert bounding_box(to_grid([[0, 0, 0], [0, 7, 0], [0, 0, 0]])) == ((7,),)
    assert fill_holes(to_grid([[3, 3, 3], [3, 0, 3], [3, 3, 3]])) == ((3, 3, 3), (3, 3, 3), (3, 3, 3))


def test_geometric_seed_cannot_express_perceptual() -> None:
    g = to_grid([[5, 0], [0, 0], [0, 7]])
    assert induce([(g, gravity(g))], GEOM, max_depth=3) is None            # барьер замыкания
    c = to_grid([[1, 0, 1], [0, 1, 0]])
    assert induce([(c, count_nonzero(c))], GEOM, max_depth=3) is None


def test_enriched_seed_solves_and_composes() -> None:
    g1 = to_grid([[5, 0], [0, 0], [0, 7]]); g2 = to_grid([[0, 1, 0], [2, 0, 3], [0, 0, 0]])
    assert str(induce([(g, gravity(g)) for g in (g1, g2)], FULL, max_depth=3)) == "gravity"
    k1 = to_grid([[1, 1, 0, 0], [1, 0, 0, 2], [0, 0, 0, 0]]); k2 = to_grid([[0, 3, 3, 3], [0, 0, 0, 0], [5, 0, 0, 0]])
    assert str(induce([(g, keep_largest(g)) for g in (k1, k2)], FULL, max_depth=3)) == "keep_largest"
    comp = induce([(g, flip_h(gravity(g))) for g in (g1, g2)], FULL, max_depth=3)       # перцепция ∘ геометрия
    assert comp is not None and comp(g1) == flip_h(gravity(g1))
