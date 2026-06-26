"""Byte-level BPE — субсловная токенизация поверх БАЙТОВ (универсально для всех языков+кода).

Сливает самые частые пары токенов в новые токены (как GPT-2 BPE, без regex-предтокенизации).
Старт со словаря из 256 байтов → учим слияния до нужного размера. Меньше токенов на
текст = длиннее эффективный контекст и лучше обобщение, чем у символьной модели.
"""

from __future__ import annotations

from collections import Counter


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BytePairTokenizer:
    """Учит BPE-слияния над UTF-8 байтами; encode/decode текста ↔ токены."""

    def __init__(self) -> None:
        self.merge_list: list[tuple[tuple[int, int], int]] = []
        self.vocab: dict[int, bytes] = {}
        self.vocab_size = 256

    def train(self, text: str, vocab_size: int) -> "BytePairTokenizer":
        ids = list(text.encode("utf-8"))
        self.merge_list = []
        nid = 256
        while nid < vocab_size:
            pairs = Counter(zip(ids, ids[1:]))
            if not pairs:
                break
            top = max(pairs, key=pairs.get)
            if pairs[top] < 2:
                break
            ids = _merge(ids, top, nid)
            self.merge_list.append((top, nid))
            nid += 1
        self.vocab_size = nid
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), i in self.merge_list:
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        return self

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merge_list:  # применяем слияния в порядке обучения
            ids = _merge(ids, pair, new_id)
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")
