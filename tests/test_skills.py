"""Тесты навыков-политик (опций): достижение цели и устойчивость к сбоям."""

from __future__ import annotations

import numpy as np

from thinking_system.memory.hierarchical import HierarchicalMemory
from thinking_system.memory.skill_policies import OptionPolicies
from thinking_system.world.rooms import rooms_world


def _opt():
    grid = rooms_world()
    free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    mem = HierarchicalMemory(grid)
    mem.explore(15000, seed=0)
    mem.consolidate(n_landmarks=4)
    return grid, free, OptionPolicies(grid, mem)


def test_options_reach_goal_without_perturbation() -> None:
    grid, free, opt = _opt()
    rng = np.random.default_rng(0)
    succ = 0
    for _ in range(20):
        start, goal = (int(x) for x in rng.choice(free, 2, replace=False))
        succ += opt.navigate_options(start, goal, perturb=0.0, seed=0)
    assert succ == 20


def test_options_more_robust_than_routes_under_perturbation() -> None:
    grid, free, opt = _opt()
    rng = np.random.default_rng(1)
    route = options = 0
    for _ in range(40):
        start, goal = (int(x) for x in rng.choice(free, 2, replace=False))
        seed = int(rng.integers(10 ** 6))
        route += opt.navigate_route(start, goal, perturb=0.15, seed=seed)
        options += opt.navigate_options(start, goal, perturb=0.15, seed=seed)
    assert options > route  # политика устойчивее фиксированного маршрута
