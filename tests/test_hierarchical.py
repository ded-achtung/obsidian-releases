"""Тесты иерархической памяти: консолидация в ориентиры/навыки и планирование."""

from __future__ import annotations

import numpy as np

from thinking_system.memory.hierarchical import HierarchicalMemory
from thinking_system.world.rooms import rooms_world


def _mem():
    grid = rooms_world()
    mem = HierarchicalMemory(grid)
    mem.explore(15000, seed=0)
    mem.consolidate(n_landmarks=4)
    return grid, mem


def test_consolidation_compresses_and_builds_skills() -> None:
    grid, mem = _mem()
    free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    assert len(mem.landmarks) == 4
    assert len(mem.landmarks) < len(free)   # семантическая компрессия
    assert len(mem.skills) > 0              # есть навыки-маршруты


def test_hierarchical_plan_reaches_goal() -> None:
    grid, mem = _mem()
    free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    rng = np.random.default_rng(1)
    reached = 0
    trials = 30
    for _ in range(trials):
        start, goal = (int(x) for x in rng.choice(free, 2, replace=False))
        plan = mem.plan(start, goal)
        assert plan is not None
        s = start
        for a in plan["actions"]:
            s = mem._move(s, a)
        reached += s == goal
        assert plan["n_high"] >= 1
    assert reached == trials  # план приводит в цель во всех случаях


def test_skills_are_reused_across_goals() -> None:
    grid, mem = _mem()
    free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    rng = np.random.default_rng(2)
    used = set()
    for _ in range(40):
        start, goal = (int(x) for x in rng.choice(free, 2, replace=False))
        plan = mem.plan(start, goal)
        if plan:
            for li, lj in zip(plan["landmarks"], plan["landmarks"][1:]):
                used.add((li, lj))
    assert len(used) <= len(mem.skills)  # небольшой набор навыков переиспользуется
