"""Тест композиционного синтеза: система САМА строит схему из атомов + хвост из данных."""

from __future__ import annotations

from thinking_system.reasoning.synthesize import synthesize


def test_composes_crop_then_inferred_recolor() -> None:
    """Схема «crop ▸ colormap» должна РОДИТЬСЯ поиском (меняет форму → не локальное правило)."""
    def task(g):                                              # обрезать до объекта, затем +4
        cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != 0]
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        return tuple(tuple(g[r][c] + 4 for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))
    g1 = ((0, 0, 0, 0), (0, 1, 2, 0), (0, 3, 4, 0), (0, 0, 0, 0))
    g2 = ((0, 0, 0), (0, 2, 1), (0, 4, 3))
    train = [(g1, task(g1)), (g2, task(g2))]
    label, fn = synthesize(train, max_prefix=2)
    assert fn is not None
    assert "▸" in label and "crop" in label                   # реально скомпонована (≥2)
    g3 = ((0, 0, 0, 0), (0, 4, 3, 0), (0, 2, 1, 0), (0, 0, 0, 0))
    assert fn(g3) == task(g3)                                  # обобщает (цвета виденные)


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
