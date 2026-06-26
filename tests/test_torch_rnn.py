"""Тест char-LSTM генератора (пропускается без torch)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from thinking_system.predictors.torch_rnn import CharRNN


def test_charrnn_learns_and_generates() -> None:
    ids = np.array([0, 1, 2, 3] * 300)  # выучиваемый циклический паттерн
    rnn = CharRNN(4, emb=16, hidden=32, layers=1, lr=3e-3, seed=0)
    first = rnn.bpc(ids)
    rnn.fit(ids, seq_len=16, batch=16, steps=600)
    assert rnn.bpc(ids) < first              # учится (bits/char падает)

    gen = rnn.generate([0, 1, 2], 20, temp=0.5)
    assert len(gen) == 20
    assert all(0 <= g < 4 for g in gen)
