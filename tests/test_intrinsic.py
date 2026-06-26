"""Тесты внутренней мотивации: освоение мира и автокуррикулум."""

from __future__ import annotations

import numpy as np

from thinking_system.agent.intrinsic import IntrinsicAgent
from thinking_system.world.gridworld import default_maze


def _n_free(grid):
    return len([s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls])


def test_intrinsic_masters_world_at_least_as_fast() -> None:
    grid = default_maze()
    ai = IntrinsicAgent(grid, intrinsic=True, seed=0)
    ar = IntrinsicAgent(grid, intrinsic=False, seed=0)
    for _ in range(150):
        ai.episode()
        ar.episode()
    assert len(ai.reachable()) == _n_free(grid)            # интринсик осваивает всё
    assert len(ai.reachable()) >= len(ar.reachable())      # не хуже случайных самоцелей


def test_emergent_curriculum_goals_get_farther() -> None:
    grid = default_maze()
    ag = IntrinsicAgent(grid, intrinsic=True, seed=0)
    dists = [ag.episode()["goal_dist"] for _ in range(60)]
    early = np.mean([d for d in dists[:5] if d > 0])
    later = np.mean([d for d in dists[20:50] if d > 0])
    assert later > early + 2  # цели быстро уходят дальше (автокуррикулум)
