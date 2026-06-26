"""Среда из нескольких каналов разной «обучаемости» (MVP для шага 2).

Каждое действие = выбрать канал и получить его текущее наблюдение. Канал
продвигает свои внутренние «часы» только когда его наблюдают (агент читает каждый
канал в своём темпе). Каналы намеренно разной природы:

    easy    — 1 осциллятор, низкий шум        → быстро выучивается
    medium  — 2 осциллятора                    → выучивается среднеплавно
    hard    — 4 осциллятора, выше частоты      → выучивается долго
    noise   — чистый гауссов шум               → НЕ выучивается (нет структуры)

Это создаёт ландшафт для проверки любопытства: правильная активная стратегия
должна вкладывать внимание в выучиваемые каналы и со временем покидать как
«шумный телевизор» (noise — прироста информации нет), так и уже освоенные каналы.
"""

from __future__ import annotations

import numpy as np

from thinking_system.envs.base import Environment


class _OscillatorChannel:
    """Канал из вращающихся осцилляторов (выучиваемая временная структура)."""

    def __init__(self, obs_dim: int, n_modes: int, freq_lo: float, freq_hi: float, dt: float, noise: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.freqs = rng.uniform(freq_lo, freq_hi, size=n_modes)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
        self.proj = rng.standard_normal((2 * n_modes, obs_dim)) / np.sqrt(2 * n_modes)
        self.dt = dt
        self.noise = noise
        self._rng = rng
        self.t = 0.0

    def reset(self) -> None:
        self.t = 0.0

    def observe(self) -> np.ndarray:
        angles = self.freqs * self.t + self.phases
        latent = np.concatenate([np.cos(angles), np.sin(angles)])
        obs = latent @ self.proj + self.noise * self._rng.standard_normal(self.obs_dim)
        self.t += self.dt
        return obs


class _NoiseChannel:
    """Канал чистого шума (нередуцируемая неопределённость — «шумный телевизор»)."""

    def __init__(self, obs_dim: int, scale: float, seed: int) -> None:
        self.obs_dim = obs_dim
        self.scale = scale
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def observe(self) -> np.ndarray:
        return self.scale * self._rng.standard_normal(self.obs_dim)


class MultiChannelEnv(Environment):
    """Среда из набора каналов; действие = индекс канала.

    Используйте конструктор по умолчанию: ``MultiChannelEnv.default(obs_dim, seed)``.
    """

    def __init__(self, channels: list[object], labels: list[str]) -> None:
        if not channels:
            raise ValueError("need at least one channel")
        self._channels = channels
        self._labels = labels
        self._obs_dim = channels[0].obs_dim  # type: ignore[attr-defined]

    @classmethod
    def default(cls, obs_dim: int = 32, *, seed: int = 0) -> "MultiChannelEnv":
        """Стандартный набор: easy / medium / hard / noise."""
        channels = [
            _OscillatorChannel(obs_dim, n_modes=1, freq_lo=0.2, freq_hi=0.4, dt=0.1, noise=0.02, seed=seed + 1),
            _OscillatorChannel(obs_dim, n_modes=3, freq_lo=0.6, freq_hi=1.4, dt=0.1, noise=0.02, seed=seed + 2),
            _OscillatorChannel(obs_dim, n_modes=8, freq_lo=1.5, freq_hi=4.0, dt=0.1, noise=0.03, seed=seed + 3),
            _NoiseChannel(obs_dim, scale=0.5, seed=seed + 4),
        ]
        labels = ["easy", "medium", "hard", "noise"]
        return cls(channels, labels)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def n_actions(self) -> int:
        return len(self._channels)

    def reset(self) -> None:
        for ch in self._channels:
            ch.reset()  # type: ignore[attr-defined]

    def step(self, action: int) -> np.ndarray:
        if not 0 <= action < len(self._channels):
            raise IndexError(f"action {action} out of range [0,{len(self._channels)})")
        return self._channels[action].observe()  # type: ignore[attr-defined]

    def action_labels(self) -> list[str]:
        return list(self._labels)

    def learnable_mask(self) -> np.ndarray:
        """Булева маска: какие каналы в принципе выучиваемы (не шум)."""
        return np.array([not isinstance(ch, _NoiseChannel) for ch in self._channels])
