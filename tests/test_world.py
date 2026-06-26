"""Тесты действующего агента: среда-мир и обучение достижению цели."""

from __future__ import annotations

import numpy as np

from thinking_system.agent.acting import ActingAgent
from thinking_system.world.gridworld import GridWorld, default_maze


def test_gridworld_mechanics() -> None:
    env = default_maze()
    assert env.optimal_steps() > 0           # лабиринт решаем
    s0 = env.reset()
    assert s0 == 0                            # старт (0,0)
    s_right, _ = env.step(3)                  # вправо → (0,1)
    assert s_right == 1
    env.reset()
    s_wall, _ = env.step(1)                   # вниз в стену (1,0) → стоит на месте
    assert s_wall == 0


def test_step_into_goal_is_done() -> None:
    env = GridWorld(2, walls=set(), start=(0, 0), goal=(0, 1))
    env.reset()
    sid, done = env.step(3)                   # вправо в цель
    assert sid == env.goal_state and done


def test_acting_agent_learns_to_reach_goal() -> None:
    env = default_maze()
    agent = ActingAgent(env.n_actions, env.goal_state, seed=0)
    steps = []
    for ep in range(35):
        eps = max(0.05, 0.5 * (0.88 ** ep))
        s = env.reset()
        for t in range(1, 1501):
            a = agent.act(s, epsilon=eps)
            sp, done = env.step(a)
            agent.learn(s, a, sp)
            s = sp
            if done:
                break
        steps.append(t)

    assert steps[-1] < steps[0]                                   # доходит быстрее, чем вначале
    assert min(steps[-5:]) <= env.optimal_steps() * 1.5           # выходит на ~оптимум
    path = agent.greedy_path(env.reset())
    assert path[-1] == env.goal_state                             # выученный путь ведёт к цели
