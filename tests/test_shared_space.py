"""Тест единого пространства: двунаправленный кросс-модальный поиск язык↔мир."""

from __future__ import annotations

import numpy as np

from thinking_system.language.grounding import CharNgram, generate_commands
from thinking_system.language.shared_space import SharedSpace
from run_shared_space import lang_to_cell_acc, cell_to_lang_acc, split


def _build(trained: bool):
    size = 7
    cells = [(s // size, s % size) for s in range(size * size)]
    train, test = split()
    feat = CharNgram([t for t, _ in train])
    sp = SharedSpace(dim=24, seed=0, trained=trained)
    sp.fit_language(train, feat).fit_world(cells, size)
    return sp, test, cells


def test_shared_space_grounds_language_to_world() -> None:
    sp, test, cells = _build(trained=True)
    base, _, _ = _build(trained=False)
    aligned = lang_to_cell_acc(sp, test, cells)
    control = lang_to_cell_acc(base, test, cells)
    assert aligned >= 0.9
    assert aligned - control >= 0.4          # выравнивание решает (контроль ~случайность)


def test_same_space_describes_world_in_language() -> None:
    """Двунаправленность: ТО ЖЕ пространство описывает клетку словами (классификатор не может)."""
    sp, test, cells = _build(trained=True)
    base, _, _ = _build(trained=False)
    aligned = cell_to_lang_acc(sp, test, cells)
    control = cell_to_lang_acc(base, test, cells)
    assert aligned >= 0.6
    assert aligned - control >= 0.3


def test_embeddings_share_one_space() -> None:
    """Язык и мир дают векторы ОДНОЙ размерности (одна валюта)."""
    sp, test, cells = _build(trained=True)
    zt = sp.embed_text(test[0][0])
    zc = sp.embed_cell(cells[0])
    assert zt.shape == zc.shape == (sp.dim,)
