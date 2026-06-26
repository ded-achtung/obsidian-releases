"""Тесты частичной наблюдаемости: алиасинг, локализация, достижение цели."""

from __future__ import annotations

from collections import Counter

from thinking_system.agent.belief import BeliefAgent
from thinking_system.world.gridworld import default_maze
from thinking_system.world.partial import PartialGridWorld, local_pattern


def _free(grid):
    return [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls and s != grid.goal_state]


def test_observation_aliasing_exists() -> None:
    grid = default_maze()
    pats = [local_pattern(grid, s) for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    assert len(pats[0]) == 4
    assert max(Counter(pats).values()) > 1  # разные клетки выглядят одинаково


def test_belief_agent_localizes_and_reaches_goal() -> None:
    grid = default_maze()
    po = PartialGridWorld(grid)
    agent = BeliefAgent(grid, seed=0)
    start = _free(grid)[0]
    agent.observe(po.reset(start))
    reached = False
    for _ in range(300):
        a = agent.act()
        o, done = po.step(a)
        agent.predict(a)
        agent.observe(o)
        if done:
            reached = True
            break
    assert reached
    assert agent.support() == 1            # локализовался до одной клетки
    assert agent.map_state() == po.true    # и вера совпала с истиной


def test_belief_reaches_goal_from_many_starts() -> None:
    grid = default_maze()
    starts = _free(grid)[:10]
    success = 0
    for start in starts:
        po = PartialGridWorld(grid)
        agent = BeliefAgent(grid, seed=0)
        agent.observe(po.reset(start))
        for _ in range(300):
            a = agent.act()
            o, done = po.step(a)
            agent.predict(a)
            agent.observe(o)
            if done:
                success += 1
                break
    assert success == len(starts)  # доходит из всех стартов
