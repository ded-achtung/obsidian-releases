"""Зашумлённый частично наблюдаемый мир: локальный обзор закодирован в шумный вектор.

Объединяет п.1 (непрерывные ЗАШУМЛЁННЫЕ наблюдения) и п.2 (частичная наблюдаемость):
наблюдение — это проекция 4-битного локального обзора клетки в непрерывный вектор
плюс гауссов шум. Агент должен и распознать обзор сквозь шум (обученное восприятие),
и снять неоднозначность многих одинаковых обзоров (вера).
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.world.partial import local_pattern


class NoisyPartialWorld:
    """Наблюдение клетки = проекция её локального обзора (4 бита) + шум."""

    def __init__(self, grid: GridWorld, *, dim: int = 16, noise: float = 0.5, seed: int = 0) -> None:
        self.g = grid
        self.dim = dim
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.proj = np.random.default_rng(seed + 1).standard_normal((4, dim))  # фикс. проекция бит→вектор
        self.true = grid.sid(grid.start)

    def _obs(self, state: int) -> np.ndarray:
        bits = np.asarray(local_pattern(self.g, state), dtype=np.float64) * 2 - 1  # ±1
        return bits @ self.proj + self.noise * self.rng.standard_normal(self.dim)

    def observe_at(self, state: int) -> np.ndarray:
        return self._obs(state)

    def reset(self, true_state: int) -> np.ndarray:
        self.true = true_state
        return self._obs(self.true)

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        r, c = divmod(self.true, self.g.size)
        dr, dc = GridWorld.MOVES[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.g.size and 0 <= nc < self.g.size and (nr, nc) not in self.g.walls:
            self.true = self.g.sid((nr, nc))
        return self._obs(self.true), self.true == self.g.goal_state


def unified_maze() -> GridWorld:
    """7×7 со «столбами»-препятствиями; углы и центр свободны (цели для языка)."""
    walls = {(1, 1), (1, 3), (1, 5), (3, 1), (3, 5), (5, 1), (5, 3), (5, 5)}
    return GridWorld(7, walls, start=(0, 0), goal=(6, 6))
