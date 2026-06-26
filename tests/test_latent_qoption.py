"""Тесты Q-learning над восприятием (функциональная аппроксимация вместо таблицы)."""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.world.features import CellFeatures
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.agent.latent_qoption import TileEncoder, LatentQOption, jepa_encoder


def _open_grid():
    return GridWorld(7, set(), start=(0, 0), goal=(6, 6))


def test_tile_fa_fewer_params_and_generalizes() -> None:
    grid = _open_grid()
    sub = grid.sid((3, 3))
    enc = TileEncoder(grid.size, n_tiles=4, seed=0)
    assert enc.dim < grid.n_states                              # параметров меньше, чем клеток (не таблица)

    free = [s for s in range(grid.n_states) if s != sub]
    perm = np.random.default_rng(0).permutation(free)
    train_starts, held = list(perm[:int(0.6 * len(perm))]), list(perm[int(0.6 * len(perm)):])

    opt = LatentQOption(grid, sub, enc, enc.dim, train_starts=train_starts, seed=0)
    curve = opt.train(3000)
    assert np.mean(curve[:100]) > np.mean(curve[-100:])         # учится из опыта
    out = np.mean([opt.reach(s, seed=i) for i, s in enumerate(held)])
    assert out >= 0.8                                           # доходит из ОТЛОЖЕННЫХ стартов = обобщает


def test_tile_fa_robust_to_perception_noise() -> None:
    grid = _open_grid()
    sub = grid.sid((3, 3))
    enc = TileEncoder(grid.size, n_tiles=4, noise=0.4, seed=1)  # координаты приходят с шумом
    opt = LatentQOption(grid, sub, enc, enc.dim, seed=0)
    opt.train(3000)
    acc = np.mean([opt.reach(s, seed=i) for i, s in enumerate(range(grid.n_states)) if s != sub])
    assert acc >= 0.8                                           # работает из зашумлённого восприятия


def test_q_over_jepa_latents_learns() -> None:
    grid = _open_grid()
    sub = grid.sid((3, 3))
    feats = CellFeatures(grid.n_states, dim=24, noise=0.15, seed=0)

    def move(s, a):
        r, c = divmod(s, grid.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        return grid.sid((nr, nc)) if 0 <= nr < grid.size and 0 <= nc < grid.size else s

    jepa = LatentWorldModel(obs_dim=24, n_actions=4, latent_dim=16, lr=5e-4, var_coef=0.02, seed=0)
    rng = np.random.default_rng(0)
    O, A, NO, s = [], [], [], 0
    for _ in range(5000):
        a = int(rng.integers(4)); sp = move(s, a)
        O.append(feats.observe(s)); A.append(a); NO.append(feats.observe(sp)); s = sp
    O, A, NO = np.array(O), np.array(A), np.array(NO)
    for _ in range(40):
        idx = rng.permutation(len(O))
        for k in range(0, len(O), 128):
            b = idx[k:k + 128]; jepa.update(O[b], A[b], NO[b])

    enc = jepa_encoder(jepa, feats.observe, normalize=True)     # φ = нормированный латент JEPA
    opt = LatentQOption(grid, sub, enc, jepa.Ld + 1, alpha=0.1, seed=0)
    curve = opt.train(4000)
    assert np.mean(curve[:100]) > np.mean(curve[-100:])         # учится из опыта над латентом
    reach = np.mean([opt.reach(s, seed=i) for i, s in enumerate(range(grid.n_states)) if s != sub])
    assert reach >= 0.7                                         # навык работает прямо над выученным восприятием
