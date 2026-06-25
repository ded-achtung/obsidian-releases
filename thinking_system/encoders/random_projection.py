"""Фиксированный random-projection энкодер (MVP).

obs (obs_dim) --W,b--> tanh --> (опц.) L2-нормировка --> латент (latent_dim).

Случайная проекция — легитимный домен-агностичный энкодер (random features):
сохраняет структуру входа, ничего не обучает, не подвержен коллапсу. Это даёт
устойчивую «подложку», на которой предиктор может честно показать рост
понимания. Когда дойдём до обучаемого восприятия — этот класс меняется на
JEPA-энкодер без изменения остального цикла (тот же интерфейс Encoder).
"""

from __future__ import annotations

import numpy as np

from thinking_system.encoders.base import Encoder


class RandomProjectionEncoder(Encoder):
    """Необучаемый энкодер на фиксированной случайной проекции.

    Args:
        obs_dim: размерность наблюдения.
        latent_dim: размерность латента.
        seed: зерно ГСЧ (детерминизм проекции).
        nonlinearity: "tanh" или "none".
        normalize: L2-нормировать латент (стабилизирует масштаб ошибки).
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        *,
        seed: int = 0,
        nonlinearity: str = "tanh",
        normalize: bool = True,
    ) -> None:
        if nonlinearity not in ("tanh", "none"):
            raise ValueError("nonlinearity must be 'tanh' or 'none'")
        rng = np.random.default_rng(seed)
        self._obs_dim = obs_dim
        self._latent_dim = latent_dim
        self._W = rng.standard_normal((obs_dim, latent_dim)) / np.sqrt(obs_dim)
        self._b = rng.standard_normal(latent_dim) * 0.1
        self._nonlinearity = nonlinearity
        self._normalize = normalize

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def encode(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs.shape[0] != self._obs_dim:
            raise ValueError(f"expected obs_dim={self._obs_dim}, got {obs.shape[0]}")
        z = obs @ self._W + self._b
        if self._nonlinearity == "tanh":
            z = np.tanh(z)
        if self._normalize:
            z = z / (np.linalg.norm(z) + 1e-8)
        return z
