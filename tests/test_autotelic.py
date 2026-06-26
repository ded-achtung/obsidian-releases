"""Тест автотелического агента: автономно осваивает шумный мир без внешних целей."""

from __future__ import annotations

import numpy as np

from thinking_system.agent.autotelic import AutotelicAgent
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


def test_autotelic_masters_noisy_world_without_external_goals() -> None:
    grid = rooms_world()
    free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    feats = CellFeatures(grid.n_states, 24, noise=0.3, seed=0)
    O, A, O2 = _collect(grid, feats, 8000, 1)
    jepa = LatentWorldModel(24, 4, latent_dim=16, hidden=64, lr=5e-4, var_coef=0.02, seed=0)
    rng = np.random.default_rng(2)
    for _ in range(8000):
        b = rng.integers(0, len(O), 128)
        jepa.update(O[b], A[b], O2[b])
    protos = np.array([jepa.encode(np.array([feats.observe(s) for _ in range(30)])).mean(0) for s in free])

    ag = AutotelicAgent(grid, jepa, protos, free, intrinsic=True, seed=0)
    accs = []
    for _ in range(120):
        accs.append(ag.episode(FeatureWorld(grid, feats))["perception_acc"])
    assert np.mean(accs) > 0.8                 # восприятие сквозь шум работает
    assert len(ag.reachable()) == len(free)    # автономно освоил весь мир
