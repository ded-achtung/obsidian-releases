"""Распределение лабиринтов: случайные стены при гарантии связности (для переноса).

Один и тот же навык, обученный на одних лабиринтах, проверяется на ДРУГИХ из того
же распределения — так измеряется перенос (а не заучивание одной карты). Связность
свободных клеток гарантирована: любой старт может дойти до подцели.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.gridworld import GridWorld


def _all_connected(size: int, walls: set[tuple[int, int]], start: tuple[int, int]) -> bool:
    free = {(r, c) for r in range(size) for c in range(size) if (r, c) not in walls}
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in GridWorld.MOVES:
            nr, nc = r + dr, c + dc
            if (nr, nc) in free and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == free


def random_maze(seed: int, *, size: int = 7, n_walls: int = 8,
                start: tuple[int, int] = (0, 0), goal: tuple[int, int] = (6, 6),
                keep: tuple[tuple[int, int], ...] = ((3, 3),)) -> GridWorld:
    """Случайный лабиринт size×size с n_walls стенами; свободные клетки связны.

    keep, start, goal остаются свободными. Стена добавляется только если не рвёт
    связность — поэтому навык может дойти до подцели из любой клетки.
    """
    rng = np.random.default_rng(seed)
    keepset = set(keep) | {start, goal}
    cand = [(r, c) for r in range(size) for c in range(size) if (r, c) not in keepset]
    order = rng.permutation(len(cand))
    walls: set[tuple[int, int]] = set()
    for i in order:
        if len(walls) >= n_walls:
            break
        walls.add(cand[i])
        if not _all_connected(size, walls, start):
            walls.discard(cand[i])
    return GridWorld(size, walls, start=start, goal=goal)
