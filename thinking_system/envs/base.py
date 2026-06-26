"""Интерфейс управляемой среды.

Среда предлагает агенту НЕСКОЛЬКО действий (например, каналов/источников); агент
выбирает одно и получает наблюдение. Именно наличие выбора делает осмысленным
активный вывод: «какое действие принесёт больше всего информации о мире».
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Environment(ABC):
    """Абстрактная среда: action → observation."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Размерность наблюдения."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Число доступных действий."""

    @abstractmethod
    def reset(self) -> None:
        """Сбросить внутреннее состояние среды."""

    @abstractmethod
    def step(self, action: int) -> np.ndarray:
        """Выполнить действие, вернуть наблюдение формы (obs_dim,)."""

    def action_labels(self) -> list[str]:
        """Человекочитаемые метки действий (по умолчанию — индексы)."""
        return [str(i) for i in range(self.n_actions)]
