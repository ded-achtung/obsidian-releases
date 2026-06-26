"""Символьный предиктор следующего токена (для текстовых данных).

Предиктивный движок, адаптированный под ДИСКРЕТНЫЕ символы (predictive coding для
текста): обучаемый эмбеддинг + MLP + softmax над словарём, обучение кросс-энтропией.
Предсказывает распределение следующего символа из контекста (окна прошлых символов).

Качество измеряется в БИТАХ НА СИМВОЛ (bits-per-character): сколько неопределённости
о следующем символе осталось. Падение BPC на ОТЛОЖЕННОЙ части = система реально
выучила структуру текста (а не запомнила обучающую выборку). Эпизодическая память и
replay — те же, что в шагах 1–3.
"""

from __future__ import annotations

import numpy as np

from thinking_system.predictors.mlp import _Adam


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SymbolicPredictor:
    """Нейронный предиктор следующего символа (контекст символов → распределение).

    Args:
        vocab_size: размер словаря символов.
        context_len: сколько прошлых символов в контексте.
        emb_dim: размерность эмбеддинга символа.
        hidden_dim: ширина скрытого слоя.
        lr: learning rate Adam.
        seed: зерно инициализации.
    """

    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        *,
        emb_dim: int = 24,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.emb_dim = emb_dim
        self.weight_decay = weight_decay
        in_dim = context_len * emb_dim
        self.E = rng.standard_normal((vocab_size, emb_dim)) * 0.1
        self.W1 = rng.standard_normal((in_dim, hidden_dim)) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, vocab_size)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(vocab_size)
        self._opt = _Adam([self.E.shape, self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape], lr=lr)

    def _forward(self, ctx: np.ndarray):
        ctx = np.atleast_2d(np.asarray(ctx, dtype=np.int64))
        emb = self.E[ctx]                       # (B, L, emb)
        x = emb.reshape(ctx.shape[0], -1)       # (B, L*emb)
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2          # (B, vocab)
        return logits, (ctx, x, h)

    def logits(self, ctx: np.ndarray) -> np.ndarray:
        """Логиты следующего символа для одного контекста (1D массив id)."""
        return self._forward(ctx)[0][0]

    def nll_bits(self, ctx: np.ndarray, targets: np.ndarray) -> float:
        """Средняя кросс-энтропия в БИТАХ на символ (без обучения) — для оценки."""
        logits, _ = self._forward(ctx)
        p = _softmax_rows(logits)
        idx = np.asarray(targets, dtype=np.int64).reshape(-1)
        nll_nats = float(-np.mean(np.log(p[np.arange(len(idx)), idx] + 1e-12)))
        return nll_nats / np.log(2.0)

    def update(self, ctx: np.ndarray, targets: np.ndarray) -> float:
        """Шаг обучения по батчу (контексты, целевые символы). Вернуть лосс в битах."""
        logits, (ctx2, x, h) = self._forward(ctx)
        idx = np.asarray(targets, dtype=np.int64).reshape(-1)
        n = ctx2.shape[0]
        p = _softmax_rows(logits)
        loss_nats = float(-np.mean(np.log(p[np.arange(n), idx] + 1e-12)))

        # Backprop кросс-энтропии через softmax.
        dlogits = p
        dlogits[np.arange(n), idx] -= 1.0
        dlogits /= n
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dz1 = dh * (1.0 - h * h)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        dx = (dz1 @ self.W1.T).reshape(n, self.context_len, self.emb_dim)
        dE = np.zeros_like(self.E)
        np.add.at(dE, ctx2, dx)  # аккумулируем градиент эмбеддингов

        if self.weight_decay:  # L2-регуляризация весов (не биасов) — против переобучения
            dE += self.weight_decay * self.E
            dW1 += self.weight_decay * self.W1
            dW2 += self.weight_decay * self.W2

        self.E, self.W1, self.b1, self.W2, self.b2 = self._opt.step(
            [self.E, self.W1, self.b1, self.W2, self.b2], [dE, dW1, db1, dW2, db2]
        )
        return loss_nats / np.log(2.0)
