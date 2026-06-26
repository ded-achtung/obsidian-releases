"""Тесты переноса навыка между мирами: связность распределения, zero-shot, few-shot."""

from __future__ import annotations

import numpy as np

from thinking_system.world.maze_dist import random_maze, _all_connected
from thinking_system.agent.latent_qoption import TileEncoder, LatentQOption
from thinking_system.agent.transfer import transfer_option, train_across_worlds, reach_rate

SUB = (3, 3)


def test_random_maze_is_connected() -> None:
    for sd in range(12):
        g = random_maze(sd)
        assert _all_connected(7, g.walls, (0, 0))          # все свободные клетки связны → подцель достижима


def test_zero_shot_transfer_beats_untrained() -> None:
    enc = TileEncoder(7, n_tiles=4, seed=0)
    A = random_maze(1000)
    optA = LatentQOption(A, A.sid(SUB), enc, enc.dim, seed=0)
    optA.train(5000)
    seeds = range(60, 70)
    trans = np.mean([reach_rate(transfer_option(random_maze(sd), SUB, enc, enc.dim, optA.W), seed=sd) for sd in seeds])
    unt = np.mean([reach_rate(LatentQOption(random_maze(sd), random_maze(sd).sid(SUB), enc, enc.dim, seed=0), seed=sd) for sd in seeds])
    assert trans >= 0.4                                    # перенесённый навык реально доходит на свежих мирах
    assert trans > unt + 0.2                               # и сильно лучше необученного (перенос, не случайность)


def test_few_shot_jumpstart() -> None:
    enc = TileEncoder(7, n_tiles=4, seed=0)
    W = train_across_worlds([random_maze(s) for s in range(8)], SUB, enc, enc.dim, episodes=5000, seed=0)
    B = random_maze(77)
    warm = transfer_option(B, SUB, enc, enc.dim, W, seed=5).train(120)
    scratch = LatentQOption(B, B.sid(SUB), enc, enc.dim, seed=5).train(120)
    assert np.mean(warm[:20]) < np.mean(scratch[:20])      # тёплый старт компетентен с первых эпизодов
