"""Тесты синтеза примитивов из данных: выход за рамки фиксированного словаря."""

from __future__ import annotations

from thinking_system.reasoning.grids import to_grid, flip_h
from thinking_system.reasoning.grid_seed import full_grid_seed
from thinking_system.reasoning.induction import induce
from thinking_system.reasoning.synthesis import (
    synthesize, synth_colormap, synth_upscale, synth_tile, synth_downscale,
)


def _cm(g, m):
    return to_grid([[m.get(v, v) for v in row] for row in g])


def test_colormap_synthesized_and_generalizes() -> None:
    m = {3: 7, 5: 2, 0: 0}
    train = [(to_grid([[3, 5], [0, 3]]), _cm([[3, 5], [0, 3]], m)),
             (to_grid([[5, 0], [3, 5]]), _cm([[5, 0], [3, 5]], m))]
    p = synth_colormap(train)
    assert p is not None and p.name == "colormap*"
    held = to_grid([[3, 3], [5, 0]])
    assert p.fn(held) == _cm([[3, 3], [5, 0]], m)        # обобщает на новый вход


def test_colormap_rejects_inconsistent() -> None:
    # один и тот же цвет 3 отображается по-разному → не функция цвета → не colormap
    train = [(to_grid([[3]]), to_grid([[7]])), (to_grid([[3]]), to_grid([[2]]))]
    assert synth_colormap(train) is None


def test_upscale_tile_downscale_synthesized() -> None:
    g = to_grid([[1, 2], [3, 4]])
    up = synth_upscale([(g, to_grid([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]))])
    assert up is not None and up.name == "upscale*2x2"
    tile = synth_tile([(g, to_grid([[1, 2, 1, 2], [3, 4, 3, 4]]))])
    assert tile is not None and tile.name == "tile*1x2"
    big = to_grid([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    down = synth_downscale([(big, g)])
    assert down is not None and down.name == "downscale*2x2"


def test_synthesis_breaks_fixed_vocabulary() -> None:
    # рекраска, которой НЕТ в seed (не color+1): фиксированный словарь не решает, синтез — решает
    m = {1: 4, 2: 8, 0: 0}
    train = [(to_grid([[1, 2], [0, 1]]), _cm([[1, 2], [0, 1]], m)),
             (to_grid([[2, 0], [1, 2]]), _cm([[2, 0], [1, 2]], m))]
    seed = full_grid_seed()
    assert induce(train, seed, max_depth=2) is None                  # вне фиксированного словаря
    grown = seed + synthesize(train)                                 # язык вырос из данных
    prog = induce(train, grown, max_depth=2)
    assert prog is not None and any("*" in s.name for s in prog.steps)


def test_synthesized_colormap_is_positionwise() -> None:
    # colormap* — чистая поцельная замена: коммутирует с пространственными операциями.
    # Поэтому он НЕ синтезируется из задач со сдвигом/поворотом (там позиции расходятся),
    # но если синтезирован — применим к любой сетке и совместим с seed-операциями.
    m = {1: 9, 2: 5, 0: 0}
    train = [(to_grid([[1, 2], [0, 1]]), _cm([[1, 2], [0, 1]], m))]
    p = synth_colormap(train)
    assert p is not None
    g = to_grid([[1, 2, 0], [2, 0, 1]])
    assert p.fn(flip_h(g)) == flip_h(p.fn(g))                        # коммутирует с flip_h
    # из задачи colormap∘flip_h позиционный colormap не выводится (расхождение позиций)
    spatial = [(g, flip_h(_cm(g, m))) for g in (to_grid([[1, 2], [0, 1]]),)]
    assert synth_colormap(spatial) is None


def test_synthesize_empty_when_no_pattern() -> None:
    # разные формы, не кратные, без цветовой функции → синтезировать нечего
    train = [(to_grid([[1, 2, 3]]), to_grid([[1], [2]]))]
    assert synthesize(train) == []
