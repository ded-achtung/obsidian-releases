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

    # ── общая геометрия: ЕДИНСТВЕННОЕ место с логикой стен/границ/шага ──────────
    # Все среды и агенты (полные, частичные, шумные, стохастические, со сдвигами)
    # ходят через эти хелперы, а не через собственные копии проверки границ.

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def is_free(self, cell: tuple[int, int]) -> bool:
        """Клетка внутри границ и не стена."""
        return self.in_bounds(cell) and cell not in self.walls

    def move_from(self, cell: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
        """Клетка + вектор смещения; упёрся в стену/границу — остался на месте."""
        nxt = (cell[0] + delta[0], cell[1] + delta[1])
        return nxt if self.is_free(nxt) else cell

    def move_sid(self, s: int, action: int) -> int:
        """То же в id состояний: шаг по MOVES[action]."""
        return self.sid(self.move_from(divmod(s, self.size), self.MOVES[action]))

    def free_cells(self) -> list[tuple[int, int]]:
        """Свободные клетки в row-major порядке (канонический порядок для выборок)."""
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) not in self.walls]

    def free_sids(self) -> list[int]:
        return [self.sid(c) for c in self.free_cells()]

    def reset(self) -> int:
        self.pos = self.start
        return self.sid(self.pos)

    def step(self, action: int) -> tuple[int, bool]:
        self.pos = self.move_from(self.pos, self.MOVES[action])
        return self.sid(self.pos), self.pos == self.goal

    def optimal_steps(self) -> int:
        """Истинная длина кратчайшего пути старт→цель (BFS по свободным клеткам)."""
        q = deque([(self.start, 0)])
        seen = {self.start}
        while q:
            cell, d = q.popleft()
            if cell == self.goal:
                return d
            for dr, dc in self.MOVES:
                nxt = (cell[0] + dr, cell[1] + dc)
                if self.is_free(nxt) and nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
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
