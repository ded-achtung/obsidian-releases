"""MegaAgent — слияние JEPA-модели мира + иерархической памяти + языка.

Один действующий агент, в котором вместе работают три блока:
  • ВОСПРИЯТИЕ — обученная JEPA-модель мира денойзит наблюдение: латент → ближайший
    прототип клетки (агент понимает, где он, несмотря на шум);
  • ПАМЯТЬ — иерархическая (карта + ориентиры + навыки) даёт маршрут к цели;
  • ЯЗЫК — цель задаётся командой и грунтится в клетку.
Замкнутый контур: каждый шаг агент воспринимает позицию через JEPA и выбирает
действие к языковой цели по выученной карте.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.language.grounding import goal_cell


class MegaAgent:
    """Действующий агент: JEPA-восприятие + иерархическая память + языковая цель.

    Args:
        grid: карта мира.
        jepa: обученная LatentWorldModel (восприятие obs → латент).
        prototypes: средний латент каждой свободной клетки (F, latent_dim).
        free: список свободных клеток (в порядке prototypes).
        hier_mem: HierarchicalMemory с выученной картой/ориентирами/навыками.
        goal_clf: классификатор языковой команды → цель.
    """

    def __init__(self, grid: GridWorld, jepa, prototypes: np.ndarray, free: list[int], hier_mem, goal_clf) -> None:
        self.g = grid
        self.jepa = jepa
        self.protos = prototypes
        self.free = free
        self.mem = hier_mem
        self.goal_clf = goal_clf

    def perceive(self, obs: np.ndarray) -> int:
        """JEPA-восприятие: денойз наблюдения → ближайший прототип клетки."""
        z = self.jepa.encode(obs)[0]
        i = int(np.argmin(((self.protos - z) ** 2).sum(axis=1)))
        return self.free[i]

    def _move(self, s: int, a: int) -> int:
        return self.g.move_sid(s, a)

    def _dist_to(self, goal: int) -> dict[int, int]:
        """Расстояния до цели по ВЫУЧЕННОЙ карте (mem.map)."""
        radj: dict[int, list[int]] = {}
        for (s, a), sp in self.mem.map.items():
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

    def navigate(self, command: str, world, start: int, size: int, *, max_steps: int = 120):
        """Понять команду → план по иерархии → дойти, воспринимая позицию через JEPA."""
        cls, _ = self.goal_clf.predict(command)
        cell = goal_cell(cls, size)
        self.g.goal = cell                     # мир сообщает «дошёл» по языковой цели
        goal = self.g.sid(cell)
        dist = self._dist_to(goal)
        plan = self.mem.plan(start, goal)      # высокоуровневый маршрут (ориентиры)

        obs = world.reset(start)
        correct = total = 0
        for t in range(1, max_steps + 1):
            s_est = self.perceive(obs)         # JEPA-восприятие
            total += 1
            correct += int(s_est == world.true)
            a = min(range(4), key=lambda a: dist.get(self._move(s_est, a), 10 ** 9))
            obs, done = world.step(a)
            if done:
                return {"reached": True, "steps": t, "perception_acc": correct / total, "landmarks": plan["landmarks"] if plan else [], "goal": cell, "cls": cls}
        return {"reached": False, "steps": max_steps, "perception_acc": correct / total, "landmarks": plan["landmarks"] if plan else [], "goal": cell, "cls": cls}
