"""Тест заземления: представление из сырых пикселей (без меток) бьёт сырые пиксели."""

from __future__ import annotations

import numpy as np

from thinking_system.encoders.learned import DenoisingAutoencoder
from thinking_system.world.gridworld import default_maze
from thinking_system.world.pixels import GlyphWorld
from run_grounding import centroid_accuracy


def _train_encoder():
    grid = default_maze()
    world = GlyphWorld(grid, patch=5, noise=0.8, seed=0)
    A, B, _ = world.sample_pairs(5000)
    enc = DenoisingAutoencoder(world.dim, latent_dim=12, hidden=64, lr=5e-3, seed=0)
    enc.fit(A, B, epochs=30, batch=128)
    return world, enc


def test_noise2noise_reduces_noise_in_latent() -> None:
    """noise2noise-латент инвариантен к шуму: два вида клетки кодируются близко."""
    world, enc = _train_encoder()
    same, diff = [], []
    cells = world.free
    rng = np.random.default_rng(3)
    for _ in range(200):
        c = cells[int(rng.integers(len(cells)))]
        za, zb = enc.encode(world.observe(c))[0], enc.encode(world.observe(c))[0]
        same.append(np.linalg.norm(za - zb))
        d = cells[int(rng.integers(len(cells)))]
        if world.type_of[d] != world.type_of[c]:
            diff.append(np.linalg.norm(za - enc.encode(world.observe(d))[0]))
    assert np.mean(same) < np.mean(diff)          # внутри-клетки << меж-типами


def test_learned_repr_beats_raw_under_label_scarcity() -> None:
    """Ключевой тест: при k=1 метке/тип выученный латент заметно лучше сырых пикселей."""
    world, enc = _train_encoder()
    raw = lambda X: np.atleast_2d(X)
    learned = lambda X: enc.encode(X)
    a_raw = centroid_accuracy(raw, world, k=1, n_test=800, seed=1)
    a_lat = centroid_accuracy(learned, world, k=1, n_test=800, seed=1)
    assert a_lat - a_raw >= 0.08                  # представление выучено без меток и полезнее


def test_encoder_trained_without_labels() -> None:
    """Честность: энкодер обучается на парах наблюдений, метки типов не подаются."""
    import inspect
    sig = inspect.signature(DenoisingAutoencoder.fit)
    assert "y" not in sig.parameters and "labels" not in sig.parameters
