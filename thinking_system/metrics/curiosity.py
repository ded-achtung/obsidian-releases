"""Метрики любопытства: куда агент вкладывал внимание и что выучил.

Главный вопрос шага 2: распределяет ли агент внимание разумно — вкладывается в
выучиваемые каналы и покидает шумный/освоенные? Отсюда метрики:

  • visit_fractions   — доля посещений по каналам (как делил внимание);
  • final_error       — финальная ошибка предсказания по каналам (что выучил);
  • wasted_on_noise   — доля внимания, потраченная на нередуцируемый шум.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CuriosityTracker:
    """Накопление по-шаговых данных активного цикла.

    Args:
        n_actions: число каналов/действий.
    """

    n_actions: int
    actions: list[int] = field(default_factory=list)
    errors: list[float | None] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    _epistemic_log: list[np.ndarray] = field(default_factory=list)

    def record(self, *, t: int, action: int, error: float | None, epistemic: np.ndarray) -> None:
        self.steps.append(t)
        self.actions.append(action)
        self.errors.append(error)
        self._epistemic_log.append(np.asarray(epistemic, dtype=np.float64).copy())

    def visit_counts(self) -> np.ndarray:
        counts = np.zeros(self.n_actions, dtype=int)
        for a in self.actions:
            counts[a] += 1
        return counts

    def visit_fractions(self) -> np.ndarray:
        counts = self.visit_counts()
        total = counts.sum()
        return counts / total if total else counts.astype(float)

    def final_error(self, *, last_k: int = 30) -> np.ndarray:
        """Средняя ошибка предсказания по каждому каналу на последних last_k его визитах.

        Это IN-SAMPLE ошибка (те же предсказания, что служили сигналом обучения), а не
        ошибка на отложенной выборке — мера «насколько освоен канал», не строгого
        обобщения. Для held-out оценки см. ContinualEvaluator.
        """
        out = np.full(self.n_actions, np.nan)
        for k in range(self.n_actions):
            errs = [e for a, e in zip(self.actions, self.errors) if a == k and e is not None]
            if errs:
                out[k] = float(np.mean(errs[-last_k:]))
        return out

    def wasted_on_noise(self, learnable_mask: np.ndarray) -> float:
        """Доля визитов, потраченных на невыучиваемые (шумные) каналы."""
        fr = self.visit_fractions()
        noise = ~np.asarray(learnable_mask, dtype=bool)
        return float(fr[noise].sum())

    def windowed_visit_fraction(self, channel: int, *, n_windows: int = 60) -> np.ndarray:
        """Доля внимания к каналу по n_windows последовательным окнам времени."""
        acts = np.asarray(self.actions, dtype=int)
        if acts.size == 0:
            return np.zeros(n_windows)
        return np.array([float((w == channel).mean()) if w.size else 0.0 for w in np.array_split(acts, n_windows)])

    def phase_visit_fraction(self, channel: int, *, lo: float, hi: float) -> float:
        """Доля внимания к каналу на отрезке [lo, hi] от прогона (доли, 0..1)."""
        acts = np.asarray(self.actions, dtype=int)
        if acts.size == 0:
            return 0.0
        a, b = int(lo * acts.size), int(hi * acts.size)
        seg = acts[a:b]
        return float((seg == channel).mean()) if seg.size else 0.0
