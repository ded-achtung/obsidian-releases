"""JEPA-style обучаемая модель мира: кодировать наблюдение → латент,
предсказать СЛЕДУЮЩИЙ латент по действию (в латентном пространстве).

В отличие от табличной модели (точные переходы по дискретным id), эта модель
работает с НЕПРЕРЫВНЫМИ ЗАШУМЛЁННЫМИ наблюдениями: она учит инвариантное к шуму
представление и предсказывает последствия действий в латенте. Это инженерный
аналог JEPA (joint-embedding predictive architecture):

    pred = g( f(obs), action )            ≈   f_target(next_obs)

  • f       — онлайн-энкодер (обучается),
  • f_target — EMA-копия f (stop-grad) — мишень предсказания (против коллапса),
  • g       — предиктор латента по действию.

Дополнительно — мягкая регуляризация дисперсии латента (как в VICReg) страхует от
схлопывания представления в точку.
"""

from __future__ import annotations

import numpy as np

from thinking_system.predictors.mlp import _Adam


class _MLP:
    """Двухслойный MLP (tanh-скрытый), с опциональным tanh на выходе и градиентом по входу."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, *, lr: float, out_act: str = "none", seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out)
        self.out_act = out_act
        self.opt = _Adam([self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape], lr=lr)

    def forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        if self.out_act == "tanh":
            out = np.tanh(out)
        return out, (X, h, out)

    def backward(self, dout, cache):
        X, h, out = cache
        if self.out_act == "tanh":
            dout = dout * (1.0 - out * out)
        dW2 = h.T @ dout
        db2 = dout.sum(axis=0)
        dh = dout @ self.W2.T
        dz1 = dh * (1.0 - h * h)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        dX = dz1 @ self.W1.T
        return dX, [dW1, db1, dW2, db2]

    def step(self, grads):
        self.W1, self.b1, self.W2, self.b2 = self.opt.step([self.W1, self.b1, self.W2, self.b2], grads)

    def ema_from(self, other: "_MLP", tau: float) -> None:
        self.W1 = tau * self.W1 + (1 - tau) * other.W1
        self.b1 = tau * self.b1 + (1 - tau) * other.b1
        self.W2 = tau * self.W2 + (1 - tau) * other.W2
        self.b2 = tau * self.b2 + (1 - tau) * other.b2


class LatentWorldModel:
    """Обучаемая модель мира: f(obs)→латент, g(латент, действие)→следующий латент.

    Args:
        obs_dim: размерность наблюдения.
        n_actions: число действий.
        latent_dim: размерность латента.
        hidden: ширина скрытых слоёв.
        lr: learning rate.
        tau: коэффициент EMA для таргет-энкодера (ближе к 1 = медленнее).
        var_coef: сила анти-коллапс регуляризации дисперсии латента.
        seed: зерно.
    """

    def __init__(self, obs_dim, n_actions, *, latent_dim=16, hidden=64, lr=1e-3, tau=0.99, var_coef=0.5, seed=0):
        self.f = _MLP(obs_dim, hidden, latent_dim, lr=lr, out_act="tanh", seed=seed)
        self.ft = _MLP(obs_dim, hidden, latent_dim, lr=lr, out_act="tanh", seed=seed)
        self.ft.ema_from(self.f, 0.0)  # инициализировать таргет копией онлайн-энкодера
        self.g = _MLP(latent_dim + n_actions, hidden, latent_dim, lr=lr, out_act="tanh", seed=seed + 1)
        self.nA = n_actions
        self.Ld = latent_dim
        self.tau = tau
        self.var_coef = var_coef

    def _onehot(self, a):
        a = np.atleast_1d(np.asarray(a, dtype=int))
        oh = np.zeros((a.size, self.nA))
        oh[np.arange(a.size), a] = 1.0
        return oh

    def encode(self, obs):
        return self.f.forward(np.atleast_2d(np.asarray(obs, float)))[0]

    def predict_next(self, obs, a):
        z = self.encode(obs)
        gx = np.concatenate([z, self._onehot(a)], axis=1)
        return self.g.forward(gx)[0]

    def update(self, obs, a, next_obs):
        """Шаг обучения по батчу переходов (obs, action, next_obs). Вернуть (loss, latent_std)."""
        obs = np.atleast_2d(np.asarray(obs, float))
        next_obs = np.atleast_2d(np.asarray(next_obs, float))
        n = obs.shape[0]

        z, fc = self.f.forward(obs)
        gx = np.concatenate([z, self._onehot(a)], axis=1)
        pred, gc = self.g.forward(gx)
        target = self.ft.forward(next_obs)[0]  # stop-grad (EMA-таргет)

        diff = pred - target
        loss = float(np.mean(np.sum(diff * diff, axis=1)))
        dpred = 2.0 * diff / n

        dgx, gg = self.g.backward(dpred, gc)
        dz = dgx[:, : self.Ld]

        # анти-коллапс: поощряем дисперсию латента (минимизируем -var)
        dz = dz + self.var_coef * (-2.0) * (z - z.mean(axis=0)) / (n * self.Ld)

        _, fg = self.f.backward(dz, fc)
        self.g.step(gg)
        self.f.step(fg)
        self.ft.ema_from(self.f, self.tau)

        return loss, float(z.std())
