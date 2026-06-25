"""Интерфейс предиктора (модели мира).

Предиктор предсказывает следующий латент из латентного контекста. Ошибка его
предсказания — это и сигнал обучения, и главная метрика «понимания»: чем точнее
предсказание во времени, тем лучше система смоделировала причинно-временную
структуру потока.

MVP — MLPPredictor (numpy). ШОВ ДЛЯ РАЗВИТИЯ: предсказание в латентном
пространстве замороженного/EMA-энкодера = ядро JEPA; позже сюда же встаёт
вероятностная модель мира для планирования через active inference (pymdp /
RxInfer), где предиктор становится генеративной моделью POMDP.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Predictor(ABC):
    """Абстрактный предиктор контекст → следующий латент."""

    @abstractmethod
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Предсказать следующий латент (1D) из контекста (1D)."""

    @abstractmethod
    def update(self, contexts: np.ndarray, targets: np.ndarray) -> float:
        """Обучающий шаг по батчу (contexts, targets). Вернуть значение лосса."""
