"""Тесты стохастического мира: индукция падает, вероятностная модель + планирование."""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.stochastic import StochasticGridEnv, ProbabilisticModel
from thinking_system.reasoning.grounded import induce_dynamics


def _obs(env, grid, steps=7000, seed=0):
    rng = np.random.default_rng(seed)
    s = grid.start
    obs = []
    for _ in range(steps):
        a = int(rng.integers(4)); sp = env.transition(s, a); obs.append((s, a, sp)); s = sp
    return obs


def test_deterministic_induction_fails_in_noise() -> None:
    grid = rooms_world()
    env = StochasticGridEnv(grid, slip=0.25, seed=0)
    rules = induce_dynamics(_obs(env, grid))
    assert len(rules) < 4                                   # шум ломает детерминированное правило


def test_probabilistic_model_estimates_distribution() -> None:
    grid = rooms_world()
    env = StochasticGridEnv(grid, slip=0.25, seed=0)
    pm = ProbabilisticModel(grid).fit(_obs(env, grid))
    dist = pm.transitions((0, 0), 3)                        # действие → (вправо)
    assert dist.get((0, 1), 0) > 0.6                        # намеренный исход доминирует (~1−slip)
    assert abs(sum(dist.values()) - 1.0) < 1e-9            # это распределение


def test_value_iteration_policy_robust_to_noise() -> None:
    grid = rooms_world(); goal = (6, 6)
    env = StochasticGridEnv(grid, slip=0.25, seed=0)
    pm = ProbabilisticModel(grid).fit(_obs(env, grid))
    pm.value_iteration(goal)
    free = [(s // grid.size, s % grid.size) for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    starts = [c for c in free if c != goal]
    rng = np.random.default_rng(1)
    vi = det = 0
    N = 120
    for _ in range(N):
        st = starts[int(rng.integers(len(starts)))]
        vi += pm.reach(StochasticGridEnv(grid, slip=0.25, seed=int(rng.integers(10 ** 6))), st, goal)[0]
        e = StochasticGridEnv(grid, slip=0.25, seed=int(rng.integers(10 ** 6)))
        s = st
        for a in _bfs(grid, st, goal):
            s = e.transition(s, a)
        det += s == goal
    assert vi / N >= 0.9                                    # реактивная политика устойчива к шуму
    assert vi > det                                        # и бьёт фиксированный план


def _bfs(grid, start, goal):
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            acts = []
            while prev[u] is not None:
                p, a = prev[u]; acts.append(a); u = p
            return acts[::-1]
        for a in range(4):
            r, c = u; dr, dc = GridWorld.MOVES[a]; v = (r + dr, c + dc)
            if 0 <= v[0] < grid.size and 0 <= v[1] < grid.size and v not in grid.walls and v not in prev:
                prev[v] = (u, a); q.append(v)
    return []
