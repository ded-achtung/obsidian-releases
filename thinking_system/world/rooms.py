"""Мир-комнаты: 4 комнаты, соединённые дверными проёмами (для иерархической памяти).

Между комнатами — стены с одиночными проходами. Любой путь между комнатами идёт
через проёмы — естественные «ориентиры», которые всплывут при консолидации опыта.
"""

from __future__ import annotations

from thinking_system.world.gridworld import GridWorld


def rooms_world() -> GridWorld:
    """7×7: горизонтальная (row 3) и вертикальная (col 3) стены с 4 проёмами."""
    size = 7
    walls = set()
    for c in range(size):
        if c not in (1, 5):           # горизонтальная стена, проёмы (3,1) и (3,5)
            walls.add((3, c))
    for r in range(size):
        if r not in (1, 5):           # вертикальная стена, проёмы (1,3) и (5,3)
            walls.add((r, 3))
    return GridWorld(size, walls, start=(0, 0), goal=(6, 6))
