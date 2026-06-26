"""Синтетический поток с управляемой временной структурой (MVP).

Скрытое состояние — набор вращающихся осцилляторов (разные частоты/фазы):

    latent(t) = [cos(f_k·t + φ_k), sin(f_k·t + φ_k)]_k        (предсказуемо)
    obs(t)    = latent(t) · P  +  noise·ε                     (P — фикс. проекция)

Это даёт гладкий, квази-периодический, многомерный сигнал: в нём ЕСТЬ временная
структура, которую предиктор может выучить (ошибка падает), но есть и шумовой
пол (ошибка не падает в ноль) — реалистичная проверка «роста понимания».
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from thinking_system.streams.base import Stream


class SyntheticStream(Stream):
    """Поток из вращающихся осцилляторов, спроецированных в obs-пространство.

    Args:
        obs_dim: размерность наблюдения.
        n_modes: число осцилляторов (скрытая размерность = 2 * n_modes).
        dt: шаг времени.
        noise: ст. отклонение аддитивного шума наблюдения.
        seed: зерно ГСЧ (частоты, фазы, проекция, шум).
    """

    def __init__(
        self,
        obs_dim: int = 32,
        *,
        n_modes: int = 3,
        dt: float = 0.1,
        noise: float = 0.02,
        freq_lo: float = 0.5,
        freq_hi: float = 2.0,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._obs_dim = obs_dim
        self._n_modes = n_modes
        self._dt = dt
        self._noise = noise
        self._freqs = rng.uniform(freq_lo, freq_hi, size=n_modes)
        self._phases = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
        self._proj = rng.standard_normal((2 * n_modes, obs_dim)) / np.sqrt(2 * n_modes)
        self._rng = rng

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def __iter__(self) -> Iterator[np.ndarray]:
        t = 0.0
        while True:
            angles = self._freqs * t + self._phases
            latent = np.concatenate([np.cos(angles), np.sin(angles)])
            obs = latent @ self._proj + self._noise * self._rng.standard_normal(self._obs_dim)
            yield obs
            t += self._dt
