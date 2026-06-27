"""Тесты третьего слоя примитивов: гравитация по сторонам, симметрия, рамка, барьер."""

from __future__ import annotations

from thinking_system.reasoning.grids import grid_primitives, to_grid, flip_h
from thinking_system.reasoning.perception import perception_primitives
from thinking_system.reasoning.structural import structural_primitives
from thinking_system.reasoning.objects import (object_primitives, gravity_up, gravity_left,
                                               gravity_right, restore_symmetry, mirror_quad,
                                               trim_border)
from thinking_system.reasoning.induction import induce

GEOM = grid_primitives()
FULL = GEOM + perception_primitives() + structural_primitives() + object_primitives()


def test_primitives_correct() -> None:
    assert gravity_up(to_grid([[0, 0], [5, 0], [0, 7]])) == ((5, 7), (0, 0), (0, 0))
    assert gravity_left(to_grid([[0, 5, 0, 7], [0, 0, 0, 0]])) == ((5, 7, 0, 0), (0, 0, 0, 0))
    assert gravity_right(to_grid([[0, 5, 0, 7], [0, 0, 0, 0]])) == ((0, 0, 5, 7), (0, 0, 0, 0))
    assert mirror_quad(to_grid([[1, 2], [3, 4]])) == ((1, 2, 2, 1), (3, 4, 4, 3),
                                                      (3, 4, 4, 3), (1, 2, 2, 1))
    assert trim_border(to_grid([[5, 5, 5, 5], [5, 1, 2, 5], [5, 3, 4, 5], [5, 5, 5, 5]])) == ((1, 2), (3, 4))
    assert trim_border(to_grid([[1, 2], [3, 4]])) == ((1, 2), (3, 4))     # нет рамки — без изменений


def test_restore_symmetry_finds_occluder() -> None:
    # симметричный узор, заслонённый цветом 8 в левом-верхнем углу; цвет-заслонку находим сами
    occ = to_grid([[8, 8, 2, 1], [8, 8, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]])
    src = to_grid([[1, 2, 2, 1], [2, 3, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]])
    assert restore_symmetry(occ) == src
    # частый цвет-узор (2) НЕ должен быть принят за заслонку — иначе вышло бы не src


def test_geometric_seed_cannot_express() -> None:
    occ = to_grid([[8, 8, 2, 1], [8, 8, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]])
    assert induce([(occ, restore_symmetry(occ))], GEOM, max_depth=3) is None      # барьер
    q = to_grid([[1, 2], [3, 4]])
    assert induce([(q, mirror_quad(q))], GEOM, max_depth=3) is None                # меняет размер


def test_enriched_seed_solves_and_composes() -> None:
    o1 = to_grid([[8, 8, 2, 1], [8, 8, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]])
    o2 = to_grid([[1, 9, 9, 1], [4, 9, 9, 4], [4, 5, 5, 4], [1, 6, 6, 1]])         # заслонка 9
    assert str(induce([(g, restore_symmetry(g)) for g in (o1, o2)], FULL, max_depth=3)) == "restore_sym"
    q1 = to_grid([[1, 2], [3, 4]]); q2 = to_grid([[5, 0], [0, 6]])
    assert str(induce([(g, mirror_quad(g)) for g in (q1, q2)], FULL, max_depth=3)) == "mirror_quad"
    # композиция с геометрией: калейдоскоп ∘ отражение
    comp = induce([(g, flip_h(mirror_quad(g))) for g in (q1, q2)], FULL, max_depth=3)
    assert comp is not None and comp(q1) == flip_h(mirror_quad(q1))
