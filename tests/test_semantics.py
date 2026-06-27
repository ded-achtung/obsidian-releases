"""Тест дистрибутивной семантики: синонимы без общих букв грунтятся по смыслу."""

from __future__ import annotations

import numpy as np

from thinking_system.language.distributional import DistributionalEmbeddings
from run_semantics import CONCEPTS, build_corpus


def _fit():
    emb = DistributionalEmbeddings(dim=16).fit(build_corpus())
    cents = {cid: emb.phrase(cfg["known"]) for cid, cfg in CONCEPTS.items()}
    cm = np.array([cents[c] for c in sorted(cents)])
    return emb, cm


def test_synonyms_grounded_by_meaning_not_form() -> None:
    """ТЕСТ-синонимы (нет в обучении, не делят подслов) попадают в свой концепт."""
    emb, cm = _fit()
    ok = tot = 0
    for cid, cfg in CONCEPTS.items():
        for w in cfg["test"]:
            q = emb.phrase([w])
            assert q is not None                       # синоним есть в корпусе → имеет вектор
            pred = int(np.argmin(((cm - q) ** 2).sum(1)))
            ok += int(pred == cid); tot += 1
    assert ok / tot >= 0.85                            # смысл из контекста работает


def test_known_and_test_synonyms_cluster() -> None:
    """Синоним ближе к центроиду своего концепта, чем к чужим."""
    emb, cm = _fit()
    for cid, cfg in CONCEPTS.items():
        for w in cfg["test"]:
            q = emb.phrase([w])
            d = ((cm - q) ** 2).sum(1)
            assert int(np.argmin(d)) == cid

def test_out_of_corpus_word_has_no_vector() -> None:
    """Честность: слово, которого нет в корпусе, вектора не получает (смысл — из употребления)."""
    emb, _ = _fit()
    assert emb.vec("zzqqx_not_a_word") is None
