"""Навыки как ПОЛИТИКИ (опции), а не фиксированные маршруты — иерархический RL.

Навык-маршрут (последовательность действий) ломается, если агента сбить с пути.
Навык-ОПЦИЯ — это политика π_L(s)→действие к ориентиру L (жадная по ценности =
расстоянию до L по выученной карте). Она ведёт к цели из ЛЮБОЙ клетки, поэтому
устойчива к возмущениям: сбило — политика просто перенаводит. Высокий уровень
компонует опции по ориентирам; финальный участок — политика к самой цели.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.gridworld import GridWorld


class OptionPolicies:
    """Опции (политики к ориентирам) поверх иерархической памяти."""

    def __init__(self, grid: GridWorld, hier_mem) -> None:
        self.g = grid
        self.mem = hier_mem
        self.free = grid.free_sids()
        self._cache: dict[int, dict[int, int]] = {}
        for L in hier_mem.landmarks:
            self._cache[L] = self._dist(L)

    def _move(self, s: int, a: int) -> int:
        return self.g.move_sid(s, a)

    def _dist(self, target: int) -> dict[int, int]:
        """Ценность опции = расстояние до target по ВЫУЧЕННОЙ карте (BFS назад)."""
        radj: dict[int, list[int]] = {}
        for (s, a), sp in self.mem.map.items():
            if sp != s:
                radj.setdefault(sp, []).append(s)
        dist = {target: 0}
        q = deque([target])
        while q:
            u = q.popleft()
            for v in radj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _value(self, target: int) -> dict[int, int]:
        if target not in self._cache:
            self._cache[target] = self._dist(target)
        return self._cache[target]

    def option_action(self, target: int, s: int) -> int:
        """Политика опции: действие к target из состояния s (жадно по ценности)."""
        d = self._value(target)
        return min(range(4), key=lambda a: d.get(self._move(s, a), 10 ** 9))

    def navigate_options(self, start: int, goal: int, *, perturb: float = 0.0, max_steps: int = 120, seed: int = 0) -> bool:
        """Дойти, компонуя опции по ориентирам; политика перенаводит при сбоях."""
        rng = np.random.default_rng(seed)
        plan = self.mem.plan(start, goal)
        targets = (plan["landmarks"] if plan else []) + [goal]
        s, ti = start, 0
        for _ in range(max_steps):
            if s == goal:
                return True
            s = self._move(s, self.option_action(targets[ti], s))
            if rng.random() < perturb:                      # сбой: телепорт в случайную клетку
                s = int(rng.choice(self.free))
            if s == targets[ti] and ti < len(targets) - 1:
                ti += 1
        return s == goal

    def navigate_route(self, start: int, goal: int, *, perturb: float = 0.0, seed: int = 0) -> bool:
        """Baseline: слепо выполнить ФИКСИРОВАННЫЙ маршрут (ломается при сбоях)."""
        rng = np.random.default_rng(seed)
        plan = self.mem.plan(start, goal)
        if plan is None:
            return False
        s = start
        for a in plan["actions"]:
            s = self._move(s, a)
            if rng.random() < perturb:
                s = int(rng.choice(self.free))
            if s == goal:
                return True
        return s == goal
