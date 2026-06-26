"""AutotelicAgent — любопытство движет ВСЕМ поведением, в шумном мире.

Без всякой внешней цели или награды агент сам ставит себе задачи (цели на фронтире
знаний) И достигает их, воспринимая свою позицию обученной JEPA-моделью мира сквозь
шум. Слияние внутренней мотивации (п. «сам ставит цели») с действием в перцептивном
мире (п. JEPA-восприятие): один автономный контур, где любопытство — единственный
двигатель.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.gridworld import GridWorld


class AutotelicAgent:
    """Автономный агент: сам ставит цели (любопытство) и достигает их через JEPA-восприятие.

    Args:
        grid: мир.
        jepa: обученная LatentWorldModel (восприятие obs → латент).
        prototypes: средний латент каждой свободной клетки (F, latent_dim).
        free: свободные клетки (в порядке prototypes).
        intrinsic: True — цели на фронтире (любопытство); False — случайные самоцели.
        seed: зерно.
    """

    def __init__(self, grid: GridWorld, jepa, prototypes: np.ndarray, free: list[int], *, intrinsic: bool = True, seed: int = 0) -> None:
        self.g = grid
        self.jepa = jepa
        self.protos = prototypes
        self.free = free
        self.start = grid.sid(grid.start)
        self.map: dict[tuple[int, int], int] = {}
        self.tried: set[tuple[int, int]] = set()
        self.intrinsic = intrinsic
        self.rng = np.random.default_rng(seed)

    def perceive(self, obs: np.ndarray) -> int:
        """JEPA-восприятие позиции сквозь шум: латент → ближайший прототип клетки."""
        z = self.jepa.encode(obs)[0]
        return self.free[int(np.argmin(((self.protos - z) ** 2).sum(axis=1)))]

    def _move(self, s: int, a: int) -> int:
        r, c = divmod(s, self.g.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.g.size and 0 <= nc < self.g.size and (nr, nc) not in self.g.walls:
            return self.g.sid((nr, nc))
        return s

    def reachable(self) -> set[int]:
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

    def select_goal(self) -> int:
        """САМ выбрать цель: на фронтире известного (любопытство) либо случайно."""
        if not self.intrinsic:
            return int(self.rng.choice(self.free))
        R = self.reachable()
        frontier = [s for s in R if any((s, a) not in self.tried for a in range(4))]
        return int(self.rng.choice(frontier if frontier else list(R)))

    def episode(self, world, *, max_steps: int = 120) -> dict:
        """Автономный эпизод в ШУМНОМ мире: цель сам, навигация через JEPA-восприятие."""
        goal = self.select_goal()
        obs = world.reset(self.start)
        s_est = self.perceive(obs)
        pn = pc = 0
        for _ in range(max_steps):
            pn += 1
            pc += int(s_est == world.true)
            untried = [a for a in range(4) if (s_est, a) not in self.tried]
            if untried:
                a = int(self.rng.choice(untried))
            else:
                dist = self._dist_to(goal)
                a = min(range(4), key=lambda a: dist.get(self._move(s_est, a), 10 ** 9)) if s_est in dist else int(self.rng.integers(4))
            obs, _ = world.step(a)
            s2 = self.perceive(obs)
            self.map[(s_est, a)] = s2
            self.tried.add((s_est, a))
            s_est = s2
        return {"reachable": len(self.reachable()), "perception_acc": pc / pn}
