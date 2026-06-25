"""Интерфейс эпизодической памяти.

Память хранит следы опыта (Experience) и отдаёт батчи для replay — это
brain-inspired механизм: переигрывание прошлого опыта (как консолидация во сне)
не даёт предиктору забывать ранее выученное при обучении на новом.

MVP — EpisodicBuffer (равномерный replay из кольцевого буфера). ШОВ ДЛЯ
РАЗВИТИЯ по дорожной карте:
  • приоритетный replay (по новизне/ошибке/важности — борьба с замусориванием);
  • двойственная память (CLS): эпизодика → консолидация в семантическую память;
  • generative replay (синтез прошлых эпизодов без хранения сырых данных).
Всё это — новые реализации этого же интерфейса.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from thinking_system.core.experience import Experience


class EpisodicMemory(ABC):
    """Абстрактная эпизодическая память с поддержкой replay."""

    @abstractmethod
    def add(self, experience: Experience) -> None:
        """Добавить эпизод в память."""

    @abstractmethod
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть replay-батч (contexts, targets) формы (B, ctx) и (B, latent)."""

    @abstractmethod
    def __len__(self) -> int:
        """Текущее число эпизодов в памяти."""
