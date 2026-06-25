"""Интерфейс энкодера.

Энкодер превращает сырое наблюдение (вектор любой природы — домен-агностично)
в латентное представление, в котором живёт предсказание и понимание.

MVP-реализация (RandomProjectionEncoder) — фиксированная, необучаемая: она
сознательно проста, чтобы цикл предсказания был в фокусе и не возникало
коллапса представлений. ШОВ ДЛЯ РАЗВИТИЯ: заменить на обучаемый JEPA-энкодер
(joint-embedding predictive architecture) с EMA-таргет-энкодером и stop-grad —
тогда сам энкодер начнёт учить инвариантные/абстрактные признаки. Интерфейс
ниже специально это допускает (encode + опциональный update_target).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    """Абстрактный энкодер obs → латент."""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Размерность латентного пространства."""

    @abstractmethod
    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Закодировать одно наблюдение (1D) в латент формы (latent_dim,)."""

    def update_target(self) -> None:
        """Хук EMA-обновления таргет-энкодера (для будущего JEPA). No-op в MVP."""
        return None
