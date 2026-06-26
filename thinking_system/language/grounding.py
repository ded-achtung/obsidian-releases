"""Грунтинг языка в цель: текстовая команда → клетка-цель.

Небольшой обучаемый классификатор сопоставляет команду на естественном языке
одной из пространственных целей (углы + центр). Обучается на одних формулировках,
проверяется на ДРУГИХ — если обобщает на невиданные фразы, значит выучил смысл
слов («top»→верх, «left»→лево), а не запомнил предложения. Затем действующий
агент идёт к понятой цели.
"""

from __future__ import annotations

import re

import numpy as np

# класс → (имя, формулировки)
PLACES: dict[int, tuple[str, list[str]]] = {
    0: ("top-left", ["top left", "upper left", "top-left corner", "north west", "northwest", "upper-left corner"]),
    1: ("top-right", ["top right", "upper right", "top-right corner", "north east", "northeast", "upper-right corner"]),
    2: ("bottom-left", ["bottom left", "lower left", "bottom-left corner", "south west", "southwest", "lower-left corner"]),
    3: ("bottom-right", ["bottom right", "lower right", "bottom-right corner", "south east", "southeast", "lower-right corner"]),
    4: ("center", ["center", "middle", "centre", "central cell", "the middle", "the center"]),
}
VERBS = ["go to", "navigate to", "head to", "move to", "reach", "get to", "walk to", "proceed to"]

N_CLASSES = len(PLACES)


def goal_cell(cls: int, size: int) -> tuple[int, int]:
    """Клетка-цель для класса в сетке size×size."""
    return {0: (0, 0), 1: (0, size - 1), 2: (size - 1, 0), 3: (size - 1, size - 1), 4: (size // 2, size // 2)}[cls]


def generate_commands() -> list[tuple[str, int]]:
    """Все команды (глагол × формулировка × детерминант) с метками классов."""
    data: list[tuple[str, int]] = []
    for cls, (_, phrases) in PLACES.items():
        for v in VERBS:
            for p in phrases:
                data.append((f"{v} {p}", cls))
                data.append((f"{v} the {p}", cls))
    return data


def tokenize(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z]+", text.lower()) if w]


class BagOfWords:
    """Мешок слов: текст → бинарный вектор по словарю обучающих команд."""

    def __init__(self, texts: list[str]) -> None:
        vocab = sorted({w for t in texts for w in tokenize(t)})
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.size = len(vocab)

    def vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.size)
        for w in tokenize(text):
            if w in self.stoi:
                v[self.stoi[w]] = 1.0
        return v


class GoalClassifier:
    """Линейный softmax-классификатор команда → класс цели (обучение кросс-энтропией)."""

    def __init__(self, bow: BagOfWords, n_classes: int = N_CLASSES, *, lr: float = 0.5) -> None:
        self.bow = bow
        self.C = n_classes
        self.W = np.zeros((bow.size, n_classes))
        self.b = np.zeros(n_classes)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, data: list[tuple[str, int]], *, epochs: int = 400) -> None:
        X = np.array([self.bow.vec(t) for t, _ in data])
        y = np.array([c for _, c in data])
        Y = np.eye(self.C)[y]
        for _ in range(epochs):
            P = self._softmax(X @ self.W + self.b)
            g = (P - Y) / len(X)
            self.W -= 0.5 * (X.T @ g)
            self.b -= 0.5 * g.sum(axis=0)

    def predict(self, text: str) -> tuple[int, float]:
        p = self._softmax(self.bow.vec(text)[None] @ self.W + self.b)[0]
        return int(p.argmax()), float(p.max())
