"""Зашумлённые непрерывные наблюдения клеток + мир, выдающий их.

Используется обучаемой моделью мира (JEPA): наблюдение = скрытый признак клетки
плюс шум, поэтому восприятие нужно учить (а не читать чистый id клетки).
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld


class CellFeatures:
    """Наблюдение клетки = её скрытый признак полной амплитуды + гауссов шум."""

    def __init__(self, n_states: int, dim: int, *, noise: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.feat = rng.standard_normal((n_states, dim))
        self.noise = noise
        self.dim = dim
        self.rng = rng

    def observe(self, state: int) -> np.ndarray:
        return self.feat[state] + self.noise * self.rng.standard_normal(self.dim)


class FeatureWorld:
    """Сетка, выдающая ЗАШУМЛЁННЫЕ наблюдения (через CellFeatures)."""

    def __init__(self, grid: GridWorld, features: CellFeatures) -> None:
        self.g = grid
        self.f = features
        self.true = grid.sid(grid.start)

    def reset(self, true_state: int) -> np.ndarray:
        self.true = true_state
        return self.f.observe(self.true)

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        r, c = divmod(self.true, self.g.size)
        dr, dc = GridWorld.MOVES[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.g.size and 0 <= nc < self.g.size and (nr, nc) not in self.g.walls:
            self.true = self.g.sid((nr, nc))
        return self.f.observe(self.true), self.true == self.g.goal_state
