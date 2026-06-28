"""Тест композиционного синтеза: система САМА строит схему из атомов + хвост из данных."""

from __future__ import annotations

from thinking_system.reasoning.synthesize import synthesize


def test_composes_transform_then_inferred_recolor() -> None:
    """Схема «flip_h ▸ colormap» должна РОДИТЬСЯ поиском (её никто не писал как целое)."""
    def task(g):
        flipped = tuple(r[::-1] for r in g)
        return tuple(tuple(v + 4 for v in r) for r in flipped)   # flip_h затем +4 ко всем цветам
    train = [(((1, 2, 3), (4, 5, 6)), task(((1, 2, 3), (4, 5, 6)))),
             (((2, 1, 0), (3, 3, 1)), task(((2, 1, 0), (3, 3, 1))))]
    label, fn = synthesize(train, max_prefix=2)
    assert fn is not None
    assert "flip_h" in label                                  # префикс найден поиском
    # обобщает на новый вход (цвета из числа уже виденных — colormap не выдумывает новых)
    g = ((6, 5), (0, 1))
    assert fn(g) == task(g)


def test_pure_deterministic_composition() -> None:
    """Чисто детерминированная композиция тоже находится (без хвоста)."""
    def task(g):                                              # flip_v затем flip_h = rot180
        return tuple(r[::-1] for r in g[::-1])
    train = [(((1, 2), (3, 4)), task(((1, 2), (3, 4)))),
             (((5, 6), (7, 8)), task(((5, 6), (7, 8))))]
    label, fn = synthesize(train, max_prefix=2)
    assert fn is not None
    assert all(fn(i) == o for i, o in train)


def test_returns_none_when_no_composition_explains() -> None:
    train = [(((1, 2),), ((9, 8, 7, 6, 5),))]                 # ничем из атомов+хвоста не объяснимо
    label, fn = synthesize(train, max_prefix=2)
    assert fn is None and label is None
