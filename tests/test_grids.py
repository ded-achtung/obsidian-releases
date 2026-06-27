"""Тесты грид-домена: корректность примитивов и рост библиотеки над сетками."""

from __future__ import annotations

from thinking_system.reasoning.grids import grid_primitives, rot90, flip_h, flip_v, transpose, recolor1, to_grid
from thinking_system.reasoning.library_learning import LibraryLearner

G = to_grid([[1, 2], [3, 4]])


def test_grid_primitives_correct() -> None:
    assert flip_h(G) == ((2, 1), (4, 3))
    assert flip_v(G) == ((3, 4), (1, 2))
    assert transpose(G) == ((1, 3), (2, 4))
    assert rot90(G) == ((3, 1), (4, 2))
    assert recolor1(G) == ((2, 3), (4, 5))


def _tasks():
    g1 = to_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]]); g2 = to_grid([[1, 0], [0, 1], [2, 2]]); g3 = to_grid([[3, 1], [4, 1], [5, 9]])
    c2 = lambda g: recolor1(recolor1(g))
    return [
        [(g, c2(g)) for g in [g1, g2]],
        [(g, c2(g)) for g in [g3, g1]],
        [(g, transpose(flip_h(g))) for g in [g1, g3]],
        [(g, c2(rot90(g))) for g in [g1, g2, g3]],        # глубина 3
    ]


def test_library_growth_over_grids() -> None:
    L = LibraryLearner(seed=grid_primitives())
    hist = L.learn(_tasks(), rounds=3, max_depth=2, abstractions_per_round=1)
    assert hist[0]["solved"] == 3                          # глубина-3 грид-задача пока недостижима
    assert hist[-1]["solved"] == 4                         # после роста грид-языка — решена
    assert any("color+1" in a for a in L.lib.abstractions)  # выучила грид-абстракцию сама
