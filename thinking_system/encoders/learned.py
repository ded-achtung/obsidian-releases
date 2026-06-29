"""Обучаемый энкодер восприятия: представление из СЫРЫХ пикселей без меток.

Заземление символов: вместо заданного руками вектора-признака клетки агент видит
сырой зашумлённый глиф и сам учит представление. Самонаблюдение (noise2noise): две
независимые зашумлённые версии ОДНОГО объекта — обучаемся предсказывать одну по другой
(MSE). Сеть не может воспроизвести шум, поэтому в латенте остаётся СИГНАЛ (форма глифа),
а не шум — представление выучено без единой метки.

Чистый numpy: энкодер (in→hidden→latent, ReLU) + декодер (latent→hidden→in, ReLU).
"""

from __future__ import annotations

import numpy as np


def _relu(x):
    return np.maximum(0.0, x)


class DenoisingAutoencoder:
    """noise2noise автоэнкодер: латент кодирует сигнал, инвариантный к шуму наблюдения."""

    def __init__(self, in_dim: int, latent_dim: int = 12, hidden: int = 64, *, lr: float = 5e-3, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
        self.W1, self.b1 = s(in_dim, hidden), np.zeros(hidden)          # enc
        self.W2, self.b2 = s(hidden, latent_dim), np.zeros(latent_dim)
        self.W3, self.b3 = s(latent_dim, hidden), np.zeros(hidden)      # dec
        self.W4, self.b4 = s(hidden, in_dim), np.zeros(in_dim)
        self.lr = lr

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return _relu(X @ self.W1 + self.b1) @ self.W2 + self.b2

    def _forward(self, X):
        h1 = _relu(X @ self.W1 + self.b1)
        z = h1 @ self.W2 + self.b2
        h2 = _relu(z @ self.W3 + self.b3)
        out = h2 @ self.W4 + self.b4
        return h1, z, h2, out

    def fit(self, A: np.ndarray, B: np.ndarray, *, epochs: int = 40, batch: int = 128) -> "DenoisingAutoencoder":
        """Обучить предсказывать вид B по виду A того же объекта (оба зашумлены)."""
        n = len(A)
        rng = np.random.default_rng(0)
        for _ in range(epochs):
            for i in range(0, n, batch):
                idx = rng.integers(0, n, min(batch, n))
                a, b = A[idx], B[idx]
                h1, z, h2, out = self._forward(a)
                m = len(a)
                d_out = 2.0 * (out - b) / m                              # MSE grad
                dW4 = h2.T @ d_out; db4 = d_out.sum(0)
                d_h2 = d_out @ self.W4.T; d_h2[h2 <= 0] = 0.0
                dW3 = z.T @ d_h2; db3 = d_h2.sum(0)
                d_z = d_h2 @ self.W3.T
                dW2 = h1.T @ d_z; db2 = d_z.sum(0)
                d_h1 = d_z @ self.W2.T; d_h1[h1 <= 0] = 0.0
                dW1 = a.T @ d_h1; db1 = d_h1.sum(0)
                for p, g in ((self.W1, dW1), (self.b1, db1), (self.W2, dW2), (self.b2, db2),
                             (self.W3, dW3), (self.b3, db3), (self.W4, dW4), (self.b4, db4)):
                    p -= self.lr * g
        return self
