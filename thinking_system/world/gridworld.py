"""Мир-сетка с препятствиями: агент перемещается, действие меняет его положение.

Полностью наблюдаемый дискретный мир. Агент НЕ знает карту заранее — он должен
сам исследовать и выучить, куда ведут действия (где стены), и научиться доходить
до цели. Это полигон для действующего агента (модель мира + активный вывод).
"""

from __future__ import annotations

from collections import deque


class GridWorld:
    """Сетка size×size со стенами, стартом и целью. Действия: ↑ ↓ ← →.

    Args:
        size: сторона сетки.
        walls: множество клеток-стен (r, c).
        start: стартовая клетка.
        goal: целевая клетка.
    """

    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ARROWS = ["↑", "↓", "←", "→"]

    def __init__(self, size: int, walls: set[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int]) -> None:
        self.size = size
        self.walls = set(walls)
        self.start = start
        self.goal = goal
        self.pos = start

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def sid(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    @property
    def goal_state(self) -> int:
        return self.sid(self.goal)

    def reset(self) -> int:
        self.pos = self.start
        return self.sid(self.pos)

    def step(self, action: int) -> tuple[int, bool]:
        dr, dc = self.MOVES[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.walls:
            self.pos = (nr, nc)  # шаг; иначе упёрся в стену/границу — стоит на месте
        return self.sid(self.pos), self.pos == self.goal

    def optimal_steps(self) -> int:
        """Истинная длина кратчайшего пути старт→цель (BFS по свободным клеткам)."""
        q = deque([(self.start, 0)])
        seen = {self.start}
        while q:
            (r, c), d = q.popleft()
            if (r, c) == self.goal:
                return d
            for dr, dc in self.MOVES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.walls and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append(((nr, nc), d + 1))
        return -1  # недостижимо


def default_maze() -> GridWorld:
    """Серпантинный лабиринт 7×7: три стены-перегородки с разнесёнными проходами."""
    size = 7
    walls: set[tuple[int, int]] = set()
    for c in range(0, 5):      # стена row1, проход справа (col5,6)
        walls.add((1, c))
    for c in range(2, 7):      # стена row3, проход слева (col0,1)
        walls.add((3, c))
    for c in range(0, 5):      # стена row5, проход справа (col5,6)
        walls.add((5, c))
    return GridWorld(size, walls, start=(0, 0), goal=(6, 0))
