"""Интерфейс потока наблюдений.

Поток — это домен-агностичный источник: он лишь выдаёт векторы фиксированной
размерности (obs_dim). Любой реальный домен (текст, аудио, сенсоры робота,
события лога) подключается реализацией этого интерфейса + подходящим энкодером —
остальной цикл не меняется.

MVP — SyntheticStream (управляемая временная структура, чтобы честно проверить,
что система учится предсказывать). Поток конечен или бесконечен; PredictiveLoop
сам берёт нужное число шагов.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np


class Stream(ABC):
    """Абстрактный поток наблюдений-векторов."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Размерность одного наблюдения."""

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """Итератор по наблюдениям (каждое — 1D-массив формы (obs_dim,))."""
