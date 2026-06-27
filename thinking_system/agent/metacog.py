"""Метапознание: калиброванная уверенность и отказ «я не знаю».

Разрыв «система не знает, чего не знает»: классификатор всегда выдаёт класс, даже на
мусоре. Здесь — селективное предсказание: (1) температурная калибровка (уверенность
совпадает с реальной точностью), (2) порог отказа — на неуверенных/OOD входах система
говорит «не знаю» вместо уверенной ошибки. Это шаг к метапознанию: знать границы знания.

Чистый numpy поверх обученного GoalClassifier.
"""

from __future__ import annotations

import numpy as np


class SelectiveClassifier:
    """Калиброванная уверенность + отказ от ответа на OOD/неуверенных входах."""

    def __init__(self, clf) -> None:
        self.clf = clf                       # обученный GoalClassifier
        self.T = 1.0                         # температура (калибровка)
        self.tau = 0.0                       # порог уверенности для ответа

    def _logits(self, text: str) -> np.ndarray:
        return self.clf.bow.vec(text) @ self.clf.W + self.clf.b

    def probs(self, text: str, *, T: float | None = None) -> np.ndarray:
        z = self._logits(text) / (T if T is not None else self.T)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def calibrate(self, val: list[tuple[str, int]], *, grid=None) -> float:
        """Подобрать температуру T, минимизируя NLL на валидации (температурное шкалирование)."""
        grid = grid if grid is not None else np.linspace(0.3, 6.0, 40)
        best_T, best_nll = 1.0, 1e18
        for T in grid:
            nll = np.mean([-np.log(self.probs(t, T=T)[c] + 1e-12) for t, c in val])
            if nll < best_nll:
                best_nll, best_T = nll, T
        self.T = float(best_T)
        return self.T

    def set_threshold(self, tau: float) -> None:
        self.tau = tau

    def predict(self, text: str):
        """(класс, уверенность) если уверены, иначе (None, уверенность) = «не знаю»."""
        p = self.probs(text)
        c, conf = int(p.argmax()), float(p.max())
        return (c if conf >= self.tau else None), conf

    def ece(self, data: list[tuple[str, int]], *, n_bins: int = 10) -> float:
        """Expected Calibration Error: |уверенность − фактическая точность| по бинам."""
        confs, correct = [], []
        for t, c in data:
            p = self.probs(t)
            confs.append(float(p.max())); correct.append(int(p.argmax() == c))
        confs, correct = np.array(confs), np.array(correct)
        e = 0.0
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            m = (confs > lo) & (confs <= hi)
            if m.any():
                e += m.mean() * abs(confs[m].mean() - correct[m].mean())
        return float(e)
