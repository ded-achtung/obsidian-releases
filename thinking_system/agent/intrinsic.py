"""IntrinsicAgent — САМ ставит себе цели (без внешней награды или языка).

Внутренняя мотивация: агент выбирает цели на ФРОНТИРЕ известного — клетки, у
которых ещё есть неиспробованные действия (где больше всего нового узнать). Это
форма learning-progress / novelty-мотивации (ср. IMGEP / SAGG-RIAC, Oudeyer):
агент сам расширяет свою компетенцию, без указаний извне.

Возникает АВТОКУРРИКУЛУМ: сначала близкие цели, потом всё дальше — карта мира
осваивается изнутри быстрее, чем при случайных самоцелях.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.gridworld import GridWorld


class IntrinsicAgent:
    """Агент с самостоятельным формированием целей и расширением компетенции.

    Args:
        grid: мир.
        intrinsic: True — цели на фронтире (внутренняя мотивация); False — случайные.
        seed: зерно.
    """

    def __init__(self, grid: GridWorld, *, intrinsic: bool = True, seed: int = 0) -> None:
        self.g = grid
        self.free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
        self.start = grid.sid(grid.start)
        self.map: dict[tuple[int, int], int] = {}
        self.tried: set[tuple[int, int]] = set()
        self.intrinsic = intrinsic
        self.rng = np.random.default_rng(seed)

    def _move(self, s: int, a: int) -> int:
        r, c = divmod(s, self.g.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.g.size and 0 <= nc < self.g.size and (nr, nc) not in self.g.walls:
            return self.g.sid((nr, nc))
        return s

    def reachable(self) -> set[int]:
        """Клетки, достижимые из старта по ВЫУЧЕННОЙ карте (компетенция)."""
        seen = {self.start}
        q = deque([self.start])
        while q:
            u = q.popleft()
            for a in range(4):
                v = self.map.get((u, a))
                if v is not None and v != u and v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    def _dist_to(self, goal: int) -> dict[int, int]:
        radj: dict[int, list[int]] = {}
        for (s, a), sp in self.map.items():
            if sp != s:
                radj.setdefault(sp, []).append(s)
        dist = {goal: 0}
        q = deque([goal])
        while q:
            u = q.popleft()
            for v in radj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _true_dist(self, goal: int) -> int:
        """Истинная длина пути старт→goal (для метрики сложности/куррикулума)."""
        seen = {self.start}
        q = deque([(self.start, 0)])
        while q:
            u, d = q.popleft()
            if u == goal:
                return d
            for a in range(4):
                v = self._move(u, a)
                if v != u and v not in seen:
                    seen.add(v)
                    q.append((v, d + 1))
        return -1

    def select_goal(self) -> int:
        """САМ выбрать цель: интринсик — на фронтире известного; иначе — случайно."""
        if not self.intrinsic:
            return int(self.rng.choice(self.free))
        R = self.reachable()
        frontier = [s for s in R if any((s, a) not in self.tried for a in range(4))]
        return int(self.rng.choice(frontier if frontier else list(R)))

    def episode(self, *, max_steps: int = 100) -> dict:
        """Один эпизод: поставить себе цель-фронтир, открывать новое на месте и идти к ней."""
        goal = self.select_goal()
        goal_dist = self._true_dist(goal)
        s = self.start
        for _ in range(max_steps):
            untried = [a for a in range(4) if (s, a) not in self.tried]
            if untried:                                          # сначала пробуем новое ЗДЕСЬ (расширяем карту)
                a = int(self.rng.choice(untried))
            else:                                                # локально всё известно → идём к цели-фронтиру
                dist = self._dist_to(goal)
                a = min(range(4), key=lambda a: dist.get(self._move(s, a), 10 ** 9)) if s in dist else int(self.rng.integers(4))
            sp = self._move(s, a)
            self.map[(s, a)] = sp
            self.tried.add((s, a))
            s = sp
        return {"goal": goal, "goal_dist": goal_dist, "reachable": len(self.reachable()), "coverage": len(self.tried)}
