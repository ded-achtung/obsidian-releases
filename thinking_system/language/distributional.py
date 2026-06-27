"""Дистрибутивная семантика: смысл слова из его контекстов (PPMI + SVD).

Разрыв «подслова берут морфологию, но не синонимы без общих букв»: «hub»/«core» близки
к «center» не по форме, а по УПОТРЕБЛЕНИЮ. Здесь учим эмбеддинги из со-встречаемости в
корпусе (PPMI-матрица → усечённое SVD). Тогда синоним, которого не было в размеченных
командах, получает смысл из своих контекстов и грунтится по близости к знакомым словам.

Чистый numpy, без предобучения: со-встречаемость → PPMI → SVD.
"""

from __future__ import annotations

import numpy as np


class DistributionalEmbeddings:
    """Эмбеддинги слов из со-встречаемости в корпусе (PPMI + усечённое SVD)."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.stoi: dict[str, int] = {}
        self.E: np.ndarray | None = None

    def fit(self, corpus: list[list[str]]) -> "DistributionalEmbeddings":
        vocab = sorted({w for s in corpus for w in s})
        self.stoi = {w: i for i, w in enumerate(vocab)}
        n = len(vocab)
        C = np.zeros((n, n))
        for sent in corpus:                                  # со-встречаемость в пределах предложения
            ids = [self.stoi[w] for w in sent]
            for a in range(len(ids)):
                for b in range(len(ids)):
                    if a != b:
                        C[ids[a], ids[b]] += 1.0
        total = C.sum()
        if total == 0:
            self.E = np.zeros((n, self.dim))
            return self
        Pij = C / total
        Pi = C.sum(1, keepdims=True) / total
        Pj = C.sum(0, keepdims=True) / total
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((Pij + 1e-12) / (Pi * Pj + 1e-12))
        ppmi = np.maximum(pmi, 0.0)                          # положительная PMI
        U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
        d = min(self.dim, U.shape[1])
        self.E = U[:, :d] * np.sqrt(S[:d])                   # эмбеддинг = U·√S
        return self

    def vec(self, word: str) -> np.ndarray | None:
        i = self.stoi.get(word)
        return None if i is None else self.E[i]

    def phrase(self, words: list[str]) -> np.ndarray | None:
        vs = [self.vec(w) for w in words]
        vs = [v for v in vs if v is not None]
        if not vs:
            return None
        v = np.mean(vs, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
