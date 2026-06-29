"""Тест метапознания: калибровка + отказ «не знаю» на OOD."""

from __future__ import annotations

import numpy as np

from thinking_system.language.grounding import CharNgram, GoalClassifier
from thinking_system.agent.metacog import SelectiveClassifier
from run_metacog import OOD, split


def _build():
    train, test = split()
    feat = CharNgram([t for t, _ in train])
    clf = GoalClassifier(feat); clf.fit(train, epochs=300)
    sel = SelectiveClassifier(clf)
    sel.calibrate(test)
    return sel, test


def test_confidence_separates_valid_from_ood() -> None:
    sel, test = _build()
    vconf = np.mean([sel.probs(t).max() for t, _ in test])
    oconf = np.mean([sel.probs(t).max() for t in OOD])
    assert vconf - oconf >= 0.2          # валидные увереннее OOD


def test_abstains_on_ood_but_answers_valid() -> None:
    sel, test = _build()
    vconf = np.array([sel.probs(t).max() for t, _ in test])
    sel.set_threshold(float(np.quantile(vconf, 0.05)))
    # OOD: отказ почти всегда
    ood_abstain = np.mean([sel.predict(t)[0] is None for t in OOD])
    assert ood_abstain >= 0.8
    # валидные: высокое покрытие и точность среди отвеченных
    ans = [(sel.predict(t), c) for t, c in test]
    cov = np.mean([p is not None for (p, _), _ in ans])
    acc = np.mean([p == c for (p, _), c in ans if p is not None])
    assert cov >= 0.85 and acc >= 0.95


def test_calibration_does_not_worsen_ece() -> None:
    sel, test = _build()
    ece_cal = sel.ece(test)
    ece_raw = SelectiveClassifier(sel.clf).ece(test)   # T=1
    assert ece_cal <= ece_raw + 1e-6
