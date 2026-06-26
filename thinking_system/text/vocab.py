"""Словари: текст ↔ последовательность id.

Два уровня:
  • ByteVocab  — УНИВЕРСАЛЬНЫЙ (UTF-8 байты, фиксированный словарь 256). Одинаково
                 кодирует любой язык и любой код без языковой настройки. Рекомендуется.
  • CharVocab  — символы Unicode; многоязычен, но словарь зависит от корпуса.
Оба дают одинаковый интерфейс (size / encode / decode), поэтому предиктор и
датасет работают с любым из них без изменений.
"""

from __future__ import annotations

import numpy as np


class ByteVocab:
    """Универсальный байтовый словарь (UTF-8): любой язык и код → байты 0..255.

    Словарь фиксирован (256) и не зависит от корпуса — новый язык/символ никогда
    не требует менять словарь. Это и есть «учить все языки и код одинаково».
    """

    size = 256

    def encode(self, text: str) -> np.ndarray:
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)

    def decode(self, ids: np.ndarray | list[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


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
