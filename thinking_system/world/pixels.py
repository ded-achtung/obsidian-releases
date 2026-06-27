"""Сырое пиксельное восприятие: тип клетки → фиксированный глиф + шум.

В отличие от CellFeatures (заданный руками вектор-признак на клетку — оракул), здесь
наблюдение — это СЫРЫЕ пиксели: каждый ТИП клетки (локальный паттерн стен) рендерится
в фиксированный бинарный глиф, а наблюдение = глиф + гауссов шум (каждый раз разный).
Клетки одного типа выглядят одинаково (как в реальном восприятии) — представление
приходится выучивать из самих наблюдений, а не получать готовым.
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.partial import local_pattern


class GlyphWorld:
    """Рендер клеток в зашумлённые пиксельные глифы по типу (локальному паттерну стен)."""

    def __init__(self, grid, *, patch: int = 5, noise: float = 0.7, seed: int = 0) -> None:
        self.g = grid
        self.noise = noise
        self.dim = patch * patch
        size = grid.size
        self.free = [s for s in range(grid.n_states) if (s // size, s % size) not in grid.walls]
        patterns = sorted({local_pattern(grid, s) for s in self.free})
        self.type_of = {s: patterns.index(local_pattern(grid, s)) for s in self.free}
        self.n_types = len(patterns)
        rng = np.random.default_rng(seed)
        # фиксированный бинарный глиф на каждый тип (сырое «изображение»)
        self.glyph = rng.integers(0, 2, size=(self.n_types, self.dim)).astype(np.float64)
        self._rng = np.random.default_rng(seed + 1)

    def clean(self, cell: int) -> np.ndarray:
        return self.glyph[self.type_of[cell]]

    def observe(self, cell: int) -> np.ndarray:
        """Сырое наблюдение клетки: глиф её типа + гауссов шум."""
        return self.clean(cell) + self.noise * self._rng.standard_normal(self.dim)

    def sample_pairs(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """n пар (видA, видB) одной случайной клетки + её тип (тип НЕ используется при обучении)."""
        cells = [self.free[i] for i in self._rng.integers(0, len(self.free), n)]
        A = np.array([self.observe(c) for c in cells])
        B = np.array([self.observe(c) for c in cells])
        y = np.array([self.type_of[c] for c in cells])
        return A, B, y
