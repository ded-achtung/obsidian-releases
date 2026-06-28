"""Тест синтезаторов «изобрести примитив из данных»: строят функцию из пар и проверяют."""

from __future__ import annotations

from thinking_system.reasoning.invent import (
    synth_colormap, synth_upscale, synth_downscale, synth_mosaic, invent,
)


def test_colormap_inferred_from_data() -> None:
    train = [(((1, 2), (2, 1)), ((3, 4), (4, 3)))]      # 1→3, 2→4
    fn = synth_colormap(train)
    assert fn is not None
    assert fn(((1, 1), (2, 2))) == ((3, 3), (4, 4))      # обобщает на новый вход


def test_upscale_factor_inferred() -> None:
    train = [(((1, 2),), ((1, 1, 2, 2), (1, 1, 2, 2)))]  # масштаб ×(2,2) репликацией клетки
    fn = synth_upscale(train)
    assert fn is not None
    assert fn(((3,),)) == ((3, 3), (3, 3))


def test_downscale_majority_inferred() -> None:
    train = [(((1, 1, 2, 2), (1, 1, 2, 2)), ((1, 2),))]  # блок 2×2 → мажоритарный цвет
    fn = synth_downscale(train)
    assert fn is not None
    assert fn(((5, 5, 6, 6), (5, 5, 6, 6))) == ((5, 6),)


def test_mosaic_mirror_inferred() -> None:
    train = [(((1, 2),), ((1, 2, 2, 1),))]               # 1×2: вход | его зеркало
    fn = synth_mosaic(train)
    assert fn is not None
    assert fn(((3, 4),)) == ((3, 4, 4, 3),)


def test_invent_dispatches_and_is_consistent() -> None:
    train = [(((1, 2), (2, 1)), ((7, 2), (2, 7)))]       # 1→7
    name, fn = invent(train)
    assert fn is not None and name == "colormap"
    assert all(fn(i) == o for i, o in train)             # согласована со всеми парами


def test_invent_returns_none_when_no_theory() -> None:
    """Если ни один синтезатор не объясняет данные — честно None (а не выдумка)."""
    train = [(((1, 2),), ((9, 9, 9),))]                  # форма не кратна, не перекраска
    name, fn = invent(train)
    assert fn is None and name is None
