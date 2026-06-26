"""Тесты torch-бэкенда (пропускаются, если torch не установлен)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from thinking_system.predictors.base import Predictor
from thinking_system.predictors.torch_backend import TorchMLPPredictor, TorchSymbolicPredictor
from thinking_system.text.dataset import make_pairs


def test_torch_mlp_is_predictor_and_learns() -> None:
    pred = TorchMLPPredictor(8, 4, hidden_dim=32, lr=1e-2, seed=0)
    assert isinstance(pred, Predictor)  # тот же интерфейс
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 8))
    Y = np.tanh(X[:, :4])
    first = pred.update(X, Y)
    last = first
    for _ in range(300):
        last = pred.update(X, Y)
    assert last < first
    assert pred.predict(X[0]).shape == (4,)


def test_torch_symbolic_learns() -> None:
    ids = np.array([0, 1, 2, 3] * 200)
    ctx, tgt = make_pairs(ids, 3)
    pred = TorchSymbolicPredictor(4, 3, emb_dim=8, hidden_dim=32, lr=5e-3, seed=0)
    first = pred.nll_bits(ctx, tgt)
    rng = np.random.default_rng(0)
    for _ in range(500):
        b = rng.integers(0, len(ctx), 64)
        pred.update(ctx[b], tgt[b])
    assert pred.nll_bits(ctx, tgt) < first * 0.5
    assert pred.logits(ctx[0]).shape == (4,)
