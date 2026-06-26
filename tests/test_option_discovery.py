"""Тесты внутренней мотивации над опциями: открытие подцелей, навыки, покрытие."""

from __future__ import annotations

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.agent.option_discovery import OptionDiscoverer


TRUE_DOORS = [(1, 3), (5, 3), (3, 1), (3, 5)]


def _discovered():
    grid = rooms_world()
    disc = OptionDiscoverer(grid, seed=0)
    disc.explore(6000)
    subgoals = disc.discover(k=4)
    return grid, disc, subgoals


def test_discovers_bottleneck_doorways() -> None:
    grid, _, subgoals = _discovered()
    true = {grid.sid(c) for c in TRUE_DOORS}
    assert set(subgoals) == true                       # сам открыл ровно проёмы (их ему не давали)


def test_options_to_discovered_learn_from_experience() -> None:
    _, disc, subgoals = _discovered()
    curves = disc.learn_options(episodes_each=1200)
    for sg in subgoals:
        c = curves[sg]
        assert np.mean(c[:100]) > np.mean(c[-100:])    # каждый навык учится из опыта
    prog = disc.learning_progress(curves)
    assert all(p > 0 for p in prog.values())           # learning progress положителен (компетенция растёт)


def test_discovered_subgoals_cover_world_better_than_random() -> None:
    grid, disc, subgoals = _discovered()
    disc.learn_options(episodes_each=1200)
    cov = disc.coverage(subgoals)
    rng = np.random.default_rng(1)
    rand = float(np.mean([disc.coverage(list(rng.choice(disc.free, size=4, replace=False))) for _ in range(30)]))
    assert cov >= 0.95 and cov > rand                  # открытые горлышки — лучшие путевые точки

    start = grid.sid((0, 0))                            # навыки доводят вглубь каждой комнаты
    deep = [(1, 1), (1, 5), (5, 1), (5, 5)]
    assert all(disc.navigate(start, grid.sid(c)) for c in deep)
