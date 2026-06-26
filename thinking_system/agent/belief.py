"""BeliefAgent — действует под частичной наблюдаемостью, ведя веру о своём состоянии.

Поскольку из одного наблюдения позицию не определить, агент держит распределение
(belief) над всеми клетками и обновляет его байесовским фильтром:
  • observe(o): оставить клетки, чьё наблюдение совпало с o;
  • predict(a): сдвинуть веру по модели перехода действия.
Действие выбирается по активному выводу:
  • ЭПИСТЕМИКА — пока не локализовался, выбирать действие с наибольшим ОЖИДАЕМЫМ
    приростом информации (снижением энтропии веры) — двигаться, чтобы понять, где ты;
  • ПРАГМАТИКА — локализовавшись, идти к цели по известной карте (BFS).

Агент знает карту мира (как устроены стены), но НЕ знает свою позицию — классическая
локализация робота, решаемая интеграцией наблюдений во времени.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.world.partial import local_pattern


class BeliefAgent:
    """Агент с байесовской верой о позиции и активным выбором действий.

    Args:
        grid: карта мира (стены/цель известны; позиция — нет).
        localized_thresh: энтропия веры (бит), ниже которой считаем «локализовался».
        seed: зерно ГСЧ.
    """

    def __init__(self, grid: GridWorld, *, localized_thresh: float = 0.6, seed: int = 0) -> None:
        self.g = grid
        self.free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
        self.idx = {s: i for i, s in enumerate(self.free)}
        self.F = len(self.free)
        self.obs = [local_pattern(grid, s) for s in self.free]
        self.uniq = list(set(self.obs))
        self.thresh = localized_thresh
        self.rng = np.random.default_rng(seed)

        self.move = np.zeros((self.F, 4), dtype=int)  # модель перехода (по карте)
        for i, s in enumerate(self.free):
            r, c = divmod(s, grid.size)
            for a, (dr, dc) in enumerate(GridWorld.MOVES):
                nr, nc = r + dr, c + dc
                free = 0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls
                self.move[i, a] = self.idx[grid.sid((nr, nc))] if free else i

        self.b = np.ones(self.F) / self.F
        self.set_goal(grid.goal_state)

    def set_goal(self, goal_state: int) -> None:
        """Задать цель (например, по языковой команде) — пересчитать расстояния."""
        radj: list[list[int]] = [[] for _ in range(self.F)]
        for i in range(self.F):
            for a in range(4):
                j = self.move[i, a]
                if j != i:
                    radj[j].append(i)
        dist = np.full(self.F, np.inf)
        gi = self.idx[goal_state]
        dist[gi] = 0
        q = deque([gi])
        while q:
            u = q.popleft()
            for v in radj[u]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self.dist = dist

    # --- байесовский фильтр ---
    def observe(self, o) -> None:
        mask = np.fromiter((1.0 if self.obs[i] == o else 0.0 for i in range(self.F)), dtype=np.float64)
        self.b = self.b * mask
        tot = self.b.sum()
        self.b = self.b / tot if tot > 0 else np.ones(self.F) / self.F

    def predict(self, a: int) -> None:
        nb = np.zeros(self.F)
        np.add.at(nb, self.move[:, a], self.b)
        self.b = nb

    @staticmethod
    def _H(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p))) if p.size else 0.0

    def entropy(self) -> float:
        return self._H(self.b)

    def support(self) -> int:
        return int(np.sum(self.b > 1e-9))

    def map_state(self) -> int:
        return self.free[int(np.argmax(self.b))]

    def _info_gain(self, a: int) -> float:
        nb = np.zeros(self.F)
        np.add.at(nb, self.move[:, a], self.b)
        h_prior = self._H(nb)
        h_post = 0.0
        for o in self.uniq:
            mask = np.fromiter((1.0 if self.obs[i] == o else 0.0 for i in range(self.F)), dtype=np.float64)
            mass = float(np.sum(nb * mask))
            if mass > 0:
                h_post += mass * self._H(nb * mask / mass)
        return h_prior - h_post  # ожидаемый прирост информации

    def act(self) -> int:
        if self.entropy() <= self.thresh:  # ПРАГМАТИКА: локализован → к цели
            i = int(np.argmax(self.b))
            return int(min(range(4), key=lambda a: self.dist[self.move[i, a]]))
        gains = [self._info_gain(a) for a in range(4)]  # ЭПИСТЕМИКА: уточнить, где я
        best = max(gains)
        cand = [a for a in range(4) if gains[a] >= best - 1e-9]
        return int(self.rng.choice(cand))
