"""Тест объединённого агента: восприятие сквозь шум + вера + язык в одном контуре."""

from __future__ import annotations

import numpy as np

from thinking_system.agent.unified import SoftmaxClassifier, UnifiedAgent
from thinking_system.language.grounding import BagOfWords, GoalClassifier, generate_commands
from thinking_system.world.noisy_partial import NoisyPartialWorld, unified_maze
from thinking_system.world.partial import local_pattern


def _setup():
    grid = unified_maze()
    size = grid.size
    free = [s for s in range(grid.n_states) if (s // size, s % size) not in grid.walls]
    patterns = sorted(set(local_pattern(grid, s) for s in free))
    pc = {p: i for i, p in enumerate(patterns)}
    world = NoisyPartialWorld(grid, dim=16, noise=0.5, seed=0)
    X, y = [], []
    for s in free:
        c = pc[local_pattern(grid, s)]
        for _ in range(30):
            X.append(world.observe_at(s)); y.append(c)
    perc = SoftmaxClassifier(16, len(patterns))
    perc.fit(np.array(X), np.array(y), epochs=200)
    return grid, size, free, pc, world, perc


def test_perception_recognizes_view_through_noise() -> None:
    grid, size, free, pc, world, perc = _setup()
    acc = np.mean([perc.predict(world.observe_at(s)[None])[0] == pc[local_pattern(grid, s)] for s in free for _ in range(10)])
    assert acc > 0.9


def test_unified_agent_understands_and_reaches() -> None:
    grid, size, free, pc, world, perc = _setup()
    gc = GoalClassifier(BagOfWords([t for t, _ in generate_commands()]))
    gc.fit(generate_commands(), epochs=200)
    agent = UnifiedAgent(grid, perc, pc, seed=0)

    cls, cell = agent.set_goal_from_text("go to the top right corner", gc, size)
    grid.goal = cell
    assert cls == 1  # понял команду

    start = next(s for s in free if s != grid.sid(cell))
    agent.b = np.ones(agent.F) / agent.F
    agent.observe_noisy(world.reset(start))
    reached = False
    for _ in range(200):
        a = agent.act()
        obs, done = world.step(a)
        agent.predict(a)
        agent.observe_noisy(obs)
        if done:
            reached = True
            break
    assert reached  # дошёл до заданной языком цели под шумом и частичной наблюдаемостью
