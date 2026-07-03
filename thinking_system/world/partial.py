"""Частично наблюдаемая сетка: агент видит только локальную картину, не позицию.

Наблюдение клетки — 4-битный код «стена ли у соседа» (↑ ↓ ← →). Множество клеток
дают ОДИН и тот же код (перцептивный алиасинг), поэтому из одного наблюдения нельзя
понять, где ты. Нужно интегрировать наблюдения и действия во времени.
"""

from __future__ import annotations

from thinking_system.world.gridworld import GridWorld


def local_pattern(grid: GridWorld, state: int) -> tuple[int, int, int, int]:
    """4-битный локальный обзор: 1 = у соседа стена/край, 0 = свободно (↑ ↓ ← →)."""
    r, c = divmod(state, grid.size)
    return tuple(0 if grid.is_free((r + dr, c + dc)) else 1
                 for dr, dc in GridWorld.MOVES)  # type: ignore[return-value]


class PartialGridWorld:
    """Обёртка над GridWorld: истинная позиция скрыта, наблюдается только local_pattern."""

    def __init__(self, grid: GridWorld) -> None:
        self.g = grid
        self.true = grid.sid(grid.start)

    def reset(self, true_state: int) -> tuple[int, int, int, int]:
        self.true = true_state
        return local_pattern(self.g, self.true)

    def step(self, action: int) -> tuple[tuple[int, int, int, int], bool]:
        self.true = self.g.move_sid(self.true, action)
        return local_pattern(self.g, self.true), self.true == self.g.goal_state
