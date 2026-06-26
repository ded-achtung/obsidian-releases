"""Символьный словарь: текст ↔ последовательность id символов."""

from __future__ import annotations

import numpy as np


class CharVocab:
    """Отображение символ ↔ id, построенное по тексту.

    Args:
        text: корпус, по которому строится словарь (берутся все встретившиеся символы).
    """

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self._stoi = {c: i for i, c in enumerate(self.chars)}
        self._itos = {i: c for c, i in self._stoi.items()}

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        """Текст → массив id (символы вне словаря пропускаются)."""
        return np.array([self._stoi[c] for c in text if c in self._stoi], dtype=np.int64)

    def decode(self, ids: np.ndarray | list[int]) -> str:
        """Массив id → текст."""
        return "".join(self._itos[int(i)] for i in ids)
