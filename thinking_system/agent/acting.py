"""ActingAgent — агент, который ДЕЙСТВУЕТ в мире и учится достигать цели.

Не зная карту, агент выбирает действия по ожидаемой свободной энергии — балансу:

  • ПРАГМАТИЧЕСКАЯ ценность — приблизиться к предпочитаемому (целевому) состоянию,
    используя ВЫУЧЕННУЮ модель мира (кратчайший путь по известным переходам);
  • ЭПИСТЕМИЧЕСКАЯ ценность — уменьшить неопределённость, пробуя наименее изученные
    действия (исследование/любопытство).

Пока путь к цели не выучен — правит исследование; как только карта связала старт
с целью — правит достижение цели. Модель мира (куда ведут действия) — это выучен-
ная когнитивная карта; она копится между эпизодами, поэтому агент доходит до цели
всё быстрее. Это переносит принципы системы (предсказание + активный вывод +
память) с ЧТЕНИЯ на ДЕЙСТВИЕ, меняющее состояние мира.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class ActingAgent:
    """Действующий агент с выучиваемой моделью мира и активным выводом.

    Args:
        n_actions: число действий.
        goal_state: индекс целевого (предпочитаемого) состояния.
        seed: зерно ГСЧ.
    """

    def __init__(self, n_actions: int, goal_state: int, *, seed: int = 0) -> None:
        self.nA = n_actions
        self.goal = goal_state
        self.model: dict[tuple[int, int], int] = {}      # (s,a) → s'  — выученная карта мира
        self.counts: dict[tuple[int, int], int] = defaultdict(int)  # сколько раз пробовали (s,a)
        self.rng = np.random.default_rng(seed)

    def dist_to_goal(self) -> dict[int, int]:
        """Расстояние до цели по ИЗВЕСТНЫМ переходам (BFS назад от цели)."""
        radj: dict[int, list[int]] = defaultdict(list)
        for (s, a), sp in self.model.items():
            radj[sp].append(s)
        dist = {self.goal: 0}
        q = deque([self.goal])
        while q:
            u = q.popleft()
            for v in radj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def act(self, s: int, *, epsilon: float = 0.05) -> int:
        """Выбрать действие: к цели (прагматика) либо исследовать (эпистемика)."""
        dist = self.dist_to_goal()
        best_a, best_d = None, float("inf")
        for a in range(self.nA):
            sp = self.model.get((s, a))
            if sp is not None and sp in dist and dist[sp] < best_d:
                best_d, best_a = dist[sp], a
        if best_a is not None and self.rng.random() > epsilon:
            return best_a  # ПРАГМАТИКА: шаг к цели по выученной карте
        counts = np.array([self.counts[(s, a)] for a in range(self.nA)], dtype=float)
        cand = np.where(counts == counts.min())[0]
        return int(self.rng.choice(cand))  # ЭПИСТЕМИКА: наименее изученное действие

    def learn(self, s: int, a: int, sp: int) -> None:
        """Обновить модель мира наблюдённым переходом."""
        self.model[(s, a)] = sp
        self.counts[(s, a)] += 1

    def greedy_path(self, start: int, *, max_len: int = 200) -> list[int]:
        """Путь старт→цель по выученной карте (что агент знает о мире)."""
        dist = self.dist_to_goal()
        path = [start]
        s = start
        for _ in range(max_len):
            if s == self.goal:
                break
            best_a, best_d, best_sp = None, float("inf"), None
            for a in range(self.nA):
                sp = self.model.get((s, a))
                if sp is not None and sp in dist and dist[sp] < best_d:
                    best_d, best_a, best_sp = dist[sp], a, sp
            if best_sp is None:
                break
            path.append(best_sp)
            s = best_sp
        return path
