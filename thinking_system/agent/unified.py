"""UnifiedAgent — действие в зашумлённом частично-наблюдаемом мире по языковой команде.

Объединяет все три расширения в одном агенте:
  • ВОСПРИЯТИЕ (п.1) — обученный классификатор распознаёт локальный обзор сквозь шум;
  • ВЕРА (п.2) — байесовский фильтр над клетками снимает неоднозначность обзоров;
  • ЯЗЫК (п.3) — цель задаётся текстом, грунтится в клетку.
Действие выбирается активным выводом: уточнить, где я (эпистемика) → дойти до
заданной словами цели (прагматика).
"""

from __future__ import annotations

import numpy as np

from thinking_system.agent.belief import BeliefAgent
from thinking_system.language.grounding import goal_cell


class SoftmaxClassifier:
    """Линейный softmax-классификатор (для распознавания обзора из шумного вектора)."""

    def __init__(self, dim: int, n_classes: int, *, lr: float = 0.3) -> None:
        self.W = np.zeros((dim, n_classes))
        self.b = np.zeros(n_classes)
        self.lr = lr
        self.C = n_classes

    def proba(self, X: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(X) @ self.W + self.b
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 300) -> None:
        Y = np.eye(self.C)[y]
        for _ in range(epochs):
            P = self.proba(X)
            g = (P - Y) / len(X)
            self.W -= self.lr * (X.T @ g)
            self.b -= self.lr * g.sum(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.proba(X).argmax(axis=1)


class UnifiedAgent(BeliefAgent):
    """Belief-агент с ОБУЧЕННЫМ восприятием и целью, задаваемой ЯЗЫКОМ.

    Args:
        grid: карта мира.
        perception: обученный классификатор «шумный вектор → класс обзора».
        pattern_classes: отображение обзор(tuple) → класс восприятия.
        localized_thresh, seed: как у BeliefAgent.
    """

    def __init__(self, grid, perception: SoftmaxClassifier, pattern_classes: dict, *, localized_thresh: float = 0.6, seed: int = 0) -> None:
        super().__init__(grid, localized_thresh=localized_thresh, seed=seed)
        self.perception = perception
        self.cell_class = np.array([pattern_classes[self.obs[i]] for i in range(self.F)])

    def observe_noisy(self, obs_vec: np.ndarray) -> None:
        """Обновить веру по ЗАШУМЛЁННОМУ наблюдению через обученное восприятие (мягкое правдоподобие)."""
        proba = self.perception.proba(obs_vec)[0]      # P(класс обзора | шумное наблюдение)
        like = proba[self.cell_class]                  # → правдоподобие каждой клетки
        self.b = self.b * like
        tot = self.b.sum()
        self.b = self.b / tot if tot > 0 else np.ones(self.F) / self.F

    def set_goal_from_text(self, text: str, goal_clf, size: int) -> tuple[int, tuple[int, int]]:
        """Понять команду и задать цель."""
        cls, _ = goal_clf.predict(text)
        cell = goal_cell(cls, size)
        self.set_goal(self.g.sid(cell))
        return cls, cell
