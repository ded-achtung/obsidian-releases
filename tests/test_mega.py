"""Тест мега-слияния: JEPA-восприятие + иерархическая память + язык в одном агенте."""

from __future__ import annotations

import numpy as np

from thinking_system.agent.mega import MegaAgent
from thinking_system.language.grounding import BagOfWords, GoalClassifier, generate_commands, goal_cell
from thinking_system.memory.hierarchical import HierarchicalMemory
from thinking_system.world.features import CellFeatures, FeatureWorld
from thinking_system.world.gridworld import GridWorld
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.world.rooms import rooms_world


def _collect(grid, feats, n, seed):
    rng = np.random.default_rng(seed)
    s = grid.sid(grid.start)
    O, A, O2 = [], [], []
    for _ in range(n):
        a = int(rng.integers(4))
        r, c = divmod(s, grid.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        sp = grid.sid((nr, nc)) if (0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls) else s
        O.append(feats.observe(s)); A.append(a); O2.append(feats.observe(sp)); s = sp
    return np.array(O), np.array(A), np.array(O2)


def test_mega_agent_understands_perceives_navigates() -> None:
    grid = rooms_world()
    size = grid.size
    free = [s for s in range(grid.n_states) if (s // size, s % size) not in grid.walls]
    feats = CellFeatures(grid.n_states, 24, noise=0.3, seed=0)

    O, A, O2 = _collect(grid, feats, 8000, 1)
    jepa = LatentWorldModel(24, 4, latent_dim=16, hidden=64, lr=5e-4, var_coef=0.02, seed=0)
    rng = np.random.default_rng(2)
    for _ in range(8000):
        b = rng.integers(0, len(O), 128)
        jepa.update(O[b], A[b], O2[b])
    protos = np.array([jepa.encode(np.array([feats.observe(s) for _ in range(30)])).mean(0) for s in free])

    mem = HierarchicalMemory(grid)
    mem.explore(12000, seed=0)
    mem.consolidate(n_landmarks=4)
    gc = GoalClassifier(BagOfWords([t for t, _ in generate_commands()]))
    gc.fit(generate_commands(), epochs=200)

    agent = MegaAgent(grid, jepa, protos, free, mem, gc)
    goal_sid = grid.sid(goal_cell(1, size))
    start = next(s for s in free if s != goal_sid)
    res = agent.navigate("go to the top right corner", FeatureWorld(grid, feats), start, size)

    assert res["cls"] == 1                 # язык понят
    assert res["perception_acc"] > 0.8     # JEPA-восприятие работает сквозь шум
    assert res["reached"]                  # дошёл до языковой цели по иерархии
