"""MLP-предиктор в латентном пространстве (numpy, обучаемый).

Архитектура: context -> Linear -> tanh -> Linear -> предсказанный латент.
Лосс: MSE между предсказанным и реальным следующим латентом.
Оптимизатор: Adam (реализован на numpy, чтобы MVP не зависел от torch).

Это «обучаемый нейрокомпонент» из брифа. Backprop выписан вручную и компактно;
когда понадобится GPU/масштаб — этот класс заменяется на torch-реализацию с тем
же интерфейсом Predictor, остальной цикл не меняется.
"""

from __future__ import annotations

import numpy as np

from thinking_system.predictors.base import Predictor


class _Adam:
    """Минимальный Adam для списка параметров-массивов."""

    def __init__(self, shapes: list[tuple[int, ...]], lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        self.t += 1
        out: list[np.ndarray] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            out.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return out


class MLPPredictor(Predictor):
    """Одно-слойный MLP-предиктор латент-контекст → следующий латент.

    Args:
        context_dim: размерность входного контекста (context_len * latent_dim).
        latent_dim: размерность цели (латента).
        hidden_dim: ширина скрытого слоя.
        lr: learning rate Adam.
        seed: зерно инициализации весов.
    """

    def __init__(
        self,
        context_dim: int,
        latent_dim: int,
        *,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        # He-инициализация для tanh-сети — разумный старт.
        self.W1 = rng.standard_normal((context_dim, hidden_dim)) * np.sqrt(2.0 / context_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, latent_dim)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(latent_dim)
        self._opt = _Adam([self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape], lr=lr)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        h = np.tanh(X @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out, (X, h)

    def predict(self, context: np.ndarray) -> np.ndarray:
        X = np.asarray(context, dtype=np.float64).reshape(1, -1)
        out, _ = self._forward(X)
        return out[0]

    def update(self, contexts: np.ndarray, targets: np.ndarray) -> float:
        X = np.atleast_2d(np.asarray(contexts, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(targets, dtype=np.float64))
        n = X.shape[0]

        out, (X_, h) = self._forward(X)
        diff = out - Y
        loss = float(np.mean(np.sum(diff * diff, axis=1)))

        # Backprop (MSE = mean_batch sum_dim diff²).
        d_out = (2.0 / n) * diff
        dW2 = h.T @ d_out
        db2 = d_out.sum(axis=0)
        d_h = d_out @ self.W2.T
        d_z1 = d_h * (1.0 - h * h)          # d/dx tanh = 1 - tanh²
        dW1 = X_.T @ d_z1
        db1 = d_z1.sum(axis=0)

        self.W1, self.b1, self.W2, self.b2 = self._opt.step(
            [self.W1, self.b1, self.W2, self.b2], [dW1, db1, dW2, db2]
        )
        return loss
