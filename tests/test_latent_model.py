"""Тест JEPA-модели мира: учится, не схлопывается, предсказывает последствия."""

from __future__ import annotations

import numpy as np

from thinking_system.world.latent_model import LatentWorldModel


def test_latent_model_learns_and_predicts_consequences() -> None:
    rng = np.random.default_rng(0)
    n_cells, dim = 6, 16
    feat = rng.standard_normal((n_cells, dim))

    def obs(s):  # зашумлённое наблюдение клетки
        return feat[s] + 0.3 * rng.standard_normal(dim)

    def nxt(s, a):  # кольцо: 0 → вперёд, 1 → назад
        return (s + 1) % n_cells if a == 0 else (s - 1) % n_cells

    O, A, O2, s = [], [], [], 0
    for _ in range(4000):
        a = int(rng.integers(2))
        O.append(obs(s)); A.append(a); s2 = nxt(s, a); O2.append(obs(s2)); s = s2
    O, A, O2 = np.array(O), np.array(A), np.array(O2)

    model = LatentWorldModel(dim, 2, latent_dim=12, hidden=48, lr=5e-4, var_coef=0.02, seed=0)
    first = None
    loss = std = 0.0
    for _ in range(6000):
        b = rng.integers(0, len(O), 64)
        loss, std = model.update(O[b], A[b], O2[b])
        if first is None:
            first = loss

    assert loss < first * 0.5     # учится динамике
    assert std > 0.05             # латент не схлопнулся

    cell_lat = np.array([model.encode(np.array([obs(s) for _ in range(20)])).mean(0) for s in range(n_cells)])
    correct = 0
    for _ in range(200):
        s = int(rng.integers(n_cells))
        a = int(rng.integers(2))
        pred = model.predict_next(obs(s), a)[0]
        nearest = int(np.argmin(np.sum((cell_lat - pred) ** 2, axis=1)))
        correct += nearest == nxt(s, a)
    assert correct / 200 > 0.8    # верно предсказывает клетку-последствие
