"""Единое пространство репрезентаций: язык и мир в ОДНОМ векторном пространстве.

Разрыв «нет общей валюты»: модули говорят на разных представлениях, интеграция — клей.
Здесь язык (команда) и мир (клетка) отображаются в ОДНО пространство, привязанное к
общим якорям регионов. Тогда ОДНА операция (близость) служит в обе стороны:

    язык → клетка   (грунтинг команды в локацию)
    клетка → язык   (описание локации словами)

Это нельзя сделать классификатором (он односторонний, язык→класс). Обучение — гребневая
регрессия в замкнутой форме: каждую модальность отображаем к общим якорям региона.
Честно: «общая валюта» здесь — выровненное пространство над фиксированными якорями, шаг к
единому «языку мысли», а не полный универсальный субстрат для всех модулей.
"""

from __future__ import annotations

import numpy as np

from thinking_system.language.grounding import N_CLASSES, goal_cell


def cell_descriptor(cell: tuple[int, int], size: int) -> np.ndarray:
    """Нелинейные пространственные признаки клетки (чтобы центр отделялся от углов)."""
    r, c = cell[0] / (size - 1), cell[1] / (size - 1)
    dr, dc = r - 0.5, c - 0.5
    return np.array([1.0, r, c, dr * dr, dc * dc, abs(dr) + abs(dc), r * c])


def region_of_cell(cell: tuple[int, int], size: int) -> int:
    """Ближайший регион-якорь (4 угла + центр) к клетке — по сеточному расстоянию."""
    anchors = [goal_cell(k, size) for k in range(N_CLASSES)]
    return int(np.argmin([abs(cell[0] - a[0]) + abs(cell[1] - a[1]) for a in anchors]))


def _ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Гребневая регрессия в замкнутой форме: W = (XᵀX+λI)⁻¹ XᵀY."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


class SharedSpace:
    """Выровненное пространство: язык и мир → общие якоря регионов."""

    def __init__(self, dim: int = 24, *, seed: int = 0, trained: bool = True) -> None:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((N_CLASSES, dim))
        self.anchors = a / np.linalg.norm(a, axis=1, keepdims=True)   # общие якоря регионов
        self.dim = dim
        self.trained = trained
        self.W_L = self.W_W = None
        self._rng = rng

    def fit_language(self, commands, featurizer) -> "SharedSpace":
        X = np.array([featurizer.vec(t) for t, _ in commands])
        Y = np.array([self.anchors[c] for _, c in commands])
        self.W_L = _ridge(X, Y) if self.trained else self._rng.standard_normal((X.shape[1], self.dim))
        self._feat = featurizer
        return self

    def fit_world(self, cells, size) -> "SharedSpace":
        X = np.array([cell_descriptor(c, size) for c in cells])
        Y = np.array([self.anchors[region_of_cell(c, size)] for c in cells])
        self.W_W = _ridge(X, Y) if self.trained else self._rng.standard_normal((X.shape[1], self.dim))
        self.size = size
        return self

    def embed_text(self, text: str) -> np.ndarray:
        return self._feat.vec(text) @ self.W_L

    def embed_cell(self, cell: tuple[int, int]) -> np.ndarray:
        return cell_descriptor(cell, self.size) @ self.W_W
