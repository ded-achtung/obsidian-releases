"""Ансамбль предикторов — даёт оценку ЭПИСТЕМИЧЕСКОЙ неопределённости (для шага 2).

Несколько MLP-предикторов с разной инициализацией, обучаемых на bootstrap-выборках
из одного replay-батча. Два сигнала:

  • predict(context)      — среднее по членам (точечный прогноз);
  • disagreement(context) — дисперсия прогнозов между членами = эпистемическая
                            (редуцируемая) неопределённость.

Почему это правильно различает структуру и шум:
  • выучиваемый, ещё не выученный канал → члены расходятся → высокое disagreement
    → высокая ценность; по мере обучения сходятся → ценность падает → внимание
    уходит дальше (естественный куррикулум);
  • чистый шум → MSE-оптимум для всех членов = предсказывать среднее → они
    СОГЛАШАЮТСЯ (низкое disagreement) → низкая ценность. Это и решает проблему
    «шумного телевизора», на которой залипает наивное любопытство «иди на бОльшую
    ошибку» (ср. Pathak et al., 2019, Self-Supervised Exploration via Disagreement).
"""

from __future__ import annotations

import numpy as np

from thinking_system.predictors.base import Predictor
from thinking_system.predictors.mlp import MLPPredictor


class EnsemblePredictor(Predictor):
    """Ансамбль MLP-предикторов с оценкой рассогласования.

    Args:
        context_dim, latent_dim, hidden_dim, lr: как у MLPPredictor (на каждого члена).
        n_members: число членов ансамбля.
        seed: базовое зерно (члены получают разные производные зёрна).
    """

    def __init__(
        self,
        context_dim: int,
        latent_dim: int,
        *,
        n_members: int = 4,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        if n_members < 2:
            raise ValueError("ensemble needs >= 2 members for disagreement")
        self.members = [
            MLPPredictor(context_dim, latent_dim, hidden_dim=hidden_dim, lr=lr, seed=seed + 7 * i + 1)
            for i in range(n_members)
        ]
        self._rng = np.random.default_rng(seed)

    def _stack(self, context: np.ndarray) -> np.ndarray:
        return np.stack([m.predict(context) for m in self.members])  # (M, latent)

    def predict(self, context: np.ndarray) -> np.ndarray:
        return self._stack(context).mean(axis=0)

    def disagreement(self, context: np.ndarray) -> float:
        """Дисперсия прогнозов между членами, усреднённая по латентным размерностям."""
        return float(self._stack(context).var(axis=0).mean())

    def update(self, contexts: np.ndarray, targets: np.ndarray) -> float:
        X = np.atleast_2d(np.asarray(contexts, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(targets, dtype=np.float64))
        n = X.shape[0]
        losses = []
        for m in self.members:
            idx = self._rng.integers(0, n, size=n)  # bootstrap — поддерживает разнообразие членов
            losses.append(m.update(X[idx], Y[idx]))
        return float(np.mean(losses))
