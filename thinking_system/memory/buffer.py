"""Кольцевой буфер эпизодов с равномерным replay (MVP).

Фиксированная ёмкость: при переполнении новые эпизоды затирают старые (FIFO по
позиции записи). sample() возвращает равномерно случайный батч — простейшая
форма experience replay.
"""

from __future__ import annotations

import numpy as np

from thinking_system.core.experience import Experience
from thinking_system.memory.base import EpisodicMemory


class EpisodicBuffer(EpisodicMemory):
    """Кольцевой буфер опыта.

    Args:
        capacity: максимальное число хранимых эпизодов.
        seed: зерно ГСЧ для сэмплирования replay-батчей.
    """

    def __init__(self, capacity: int = 10_000, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: list[Experience] = []
        self._pos = 0
        self._rng = np.random.default_rng(seed)

    def add(self, experience: Experience) -> None:
        if len(self._data) < self.capacity:
            self._data.append(experience)
        else:
            self._data[self._pos] = experience
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._data:
            raise ValueError("cannot sample from empty memory")
        n = min(batch_size, len(self._data))
        idx = self._rng.integers(0, len(self._data), size=n)
        contexts = np.stack([self._data[i].context for i in idx])
        targets = np.stack([self._data[i].target for i in idx])
        return contexts, targets

    def __len__(self) -> int:
        return len(self._data)
