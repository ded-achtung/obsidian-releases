"""Тесты структурно-объектных примитивов: корректность, барьер, выход за него."""

from __future__ import annotations

from thinking_system.reasoning.grids import grid_primitives, to_grid, flip_h
from thinking_system.reasoning.perception import perception_primitives
from thinking_system.reasoning.structural import (structural_primitives, complete_symmetry,
                                                  replicate_by_self, outline, count_objects,
                                                  most_common_color)
from thinking_system.reasoning.induction import induce

GEOM = grid_primitives()
FULL = GEOM + perception_primitives() + structural_primitives()


def test_primitives_correct() -> None:
    # достроить симметрию: выбита одна клетка (0,0) — восстановлена по зеркалам
    holed = to_grid([[0, 2, 2, 1], [2, 3, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]])
    whole = to_grid([[1, 2, 2, 1], [2, 3, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]])
    assert complete_symmetry(holed) == whole

    # фрактал: 2×2 → 4×4, копия по диагонали (где вход непуст)
    assert replicate_by_self(to_grid([[1, 0], [0, 1]])) == ((1, 0, 0, 0), (0, 1, 0, 0),
                                                            (0, 0, 1, 0), (0, 0, 0, 1))
    # контур: сплошной квадрат → бублик (центр без пустого соседа уходит)
    assert outline(to_grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])) == ((1, 1, 1), (1, 0, 1), (1, 1, 1))
    # счёт объектов и самый частый цвет — агрегации в 1×1
    assert count_objects(to_grid([[1, 1, 0], [0, 0, 2], [0, 0, 2]])) == ((2,),)
    assert most_common_color(to_grid([[1, 1, 2], [1, 0, 2]])) == ((1,),)


def test_geometric_seed_cannot_express_structural() -> None:
    g = to_grid([[1, 0], [0, 1]])
    assert induce([(g, replicate_by_self(g))], GEOM, max_depth=3) is None      # барьер замыкания
    h = to_grid([[0, 2, 2, 1], [2, 3, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]])
    assert induce([(h, complete_symmetry(h))], GEOM, max_depth=3) is None


def test_enriched_seed_solves_and_composes() -> None:
    f1 = to_grid([[1, 0], [0, 1]]); f2 = to_grid([[0, 2], [2, 2]])
    assert str(induce([(g, replicate_by_self(g)) for g in (f1, f2)], FULL, max_depth=3)) == "fractal"
    o1 = to_grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]]); o2 = to_grid([[5, 5], [5, 5]])
    assert str(induce([(g, outline(g)) for g in (o1, o2)], FULL, max_depth=3)) == "outline"
    # композиция структурного с геометрией: отражение ∘ фрактал
    comp = induce([(g, flip_h(replicate_by_self(g))) for g in (f1, f2)], FULL, max_depth=3)
    assert comp is not None and comp(f1) == flip_h(replicate_by_self(f1))
