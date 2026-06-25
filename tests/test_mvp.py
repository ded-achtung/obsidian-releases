"""Smoke-тесты MVP: компоненты и факт обучения (ошибка падает, обходит baseline)."""

from __future__ import annotations

import numpy as np

from thinking_system import (
    EpisodicBuffer,
    Experience,
    MLPPredictor,
    PredictiveLoop,
    RandomProjectionEncoder,
    SyntheticStream,
)


def test_encoder_shape_and_determinism() -> None:
    enc = RandomProjectionEncoder(obs_dim=16, latent_dim=8, seed=1)
    obs = np.ones(16)
    z1 = enc.encode(obs)
    z2 = enc.encode(obs)
    assert z1.shape == (8,)
    assert np.allclose(z1, z2)  # детерминизм при фиксированном seed


def test_predictor_learns_constant_mapping() -> None:
    pred = MLPPredictor(context_dim=8, latent_dim=4, hidden_dim=32, lr=1e-2, seed=0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 8))
    Y = np.tanh(X[:, :4])  # выучиваемая зависимость
    first = pred.update(X, Y)
    for _ in range(300):
        last = pred.update(X, Y)
    assert last < first  # лосс падает


def test_episodic_buffer_capacity_and_sample() -> None:
    buf = EpisodicBuffer(capacity=10, seed=0)
    for t in range(25):
        buf.add(Experience(context=np.zeros(4), target=np.ones(2), t=t))
    assert len(buf) == 10  # кольцевой: не растёт сверх capacity
    ctx, tgt = buf.sample(5)
    assert ctx.shape == (5, 4)
    assert tgt.shape == (5, 2)


def test_loop_reduces_prediction_error() -> None:
    stream = SyntheticStream(obs_dim=24, noise=0.02, seed=0)
    enc = RandomProjectionEncoder(24, 12, seed=0)
    pred = MLPPredictor(context_dim=2 * 12, latent_dim=12, hidden_dim=64, lr=1e-3, seed=0)
    mem = EpisodicBuffer(capacity=5000, seed=0)
    loop = PredictiveLoop(enc, pred, mem, context_len=2)

    tracker = loop.run(stream, n_steps=4000)
    s = tracker.summary()

    assert s["final_error"] < s["initial_error"]   # система учится
    assert s["beats_baseline"] == 1.0              # обходит persistence
