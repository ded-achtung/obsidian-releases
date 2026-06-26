"""Опции, обученные Q-learning с ФУНКЦИОНАЛЬНОЙ АППРОКСИМАЦИЕЙ над восприятием.

Табличная опция (qoption.py) держит Q[s,a] и требует точный id клетки. Здесь
ценность — линейная функция Q(φ(o),a) над признаками восприятия φ(o), обучаемая
semi-gradient TD из опыта. Признаков МЕНЬШЕ, чем клеток, поэтому ценность не
запоминается поклеточно, а ОБОБЩАЕТСЯ между похожими наблюдениями:

  • агент доходит даже из стартов, не виденных при обучении;
  • работает прямо из ЗАШУМЛЁННОГО восприятия (а не из чистого индекса клетки).

Признаки φ задаются энкодером:
  • TileEncoder — гладкие RBF-тайлы над координатами (перекрытие → обобщение);
  • латент обученной JEPA-модели (model.encode(obs)) — «Q над латентами JEPA».
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from thinking_system.world.gridworld import GridWorld


class TileEncoder:
    """RBF-тайлы над координатами клетки: гладкое обобщающее восприятие (+ шум).

    Признаков (n_tiles² + bias) обычно меньше, чем клеток, а перекрытие тайлов
    переносит ценность на соседей — основа обобщения функциональной аппроксимации.

    Args:
        size: сторона сетки.
        n_tiles: число центров по стороне (всего n_tiles² RBF + bias).
        noise: σ гауссова шума восприятия координат (зашумлённое восприятие).
        seed: зерно.
    """

    def __init__(self, size: int, *, n_tiles: int = 4, noise: float = 0.0, seed: int = 0) -> None:
        cs = np.linspace(0.0, size - 1, n_tiles)
        self.centers = np.array([(r, c) for r in cs for c in cs], dtype=float)
        self.width = max(1e-6, (size - 1) / max(1, n_tiles - 1))
        self.size = size
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.dim = len(self.centers) + 1  # + bias

    def __call__(self, state: int) -> np.ndarray:
        r, c = divmod(state, self.size)
        p = np.array([r, c], float) + self.noise * self.rng.standard_normal(2)
        d2 = ((self.centers - p) ** 2).sum(axis=1)
        phi = np.exp(-d2 / (2.0 * self.width ** 2))
        return np.concatenate([phi, [1.0]])


def jepa_encoder(model, observe: Callable[[int], np.ndarray], *, normalize: bool = True) -> Callable[[int], np.ndarray]:
    """φ из латента обученной JEPA-модели: encode(observe(s)) ⊕ bias.

    L2-нормировка ограничивает амплитуду латента — линейный TD над ним устойчив
    (иначе «смертельная триада» FA+бутстрэп легко расходится). Размер признака —
    model.Ld + 1.

    Args:
        model: LatentWorldModel (есть .encode(obs)→(1, Ld)).
        observe: s → наблюдение клетки (например, CellFeatures.observe).
        normalize: L2-нормировать признак (рекомендуется).
    """

    def f(state: int) -> np.ndarray:
        z = np.asarray(model.encode(observe(state))[0], dtype=float)
        z = np.concatenate([z, [1.0]])
        return z / (np.linalg.norm(z) + 1e-8) if normalize else z

    return f


class LatentQOption:
    """Навык-опция к подцели: линейная Q(φ(o),a), semi-gradient TD из опыта.

    Args:
        grid: мир.
        subgoal: целевая клетка опции.
        encode: φ — отображение состояния в вектор признаков восприятия.
        n_features: размер вектора признаков φ.
        alpha: скорость обучения; gamma: дисконт; step_penalty: штраф за шаг.
        train_starts: из каких клеток стартуют обучающие эпизоды (для проверки
            обобщения держим часть клеток отложенными). По умолчанию — все свободные.
        seed: зерно.
    """

    def __init__(self, grid: GridWorld, subgoal: int, encode: Callable[[int], np.ndarray], n_features: int, *,
                 alpha: float = 0.05, gamma: float = 0.95, step_penalty: float = 0.01,
                 train_starts: list[int] | None = None, seed: int = 0) -> None:
        self.g = grid
        self.subgoal = subgoal
        self.encode = encode
        self.W = np.zeros((4, n_features))
        self.alpha = alpha
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
        self.train_starts = train_starts if train_starts is not None else [s for s in self.free if s != subgoal]
        self.rng = np.random.default_rng(seed)

    def _move(self, s: int, a: int) -> int:
        r, c = divmod(s, self.g.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.g.size and 0 <= nc < self.g.size and (nr, nc) not in self.g.walls:
            return self.g.sid((nr, nc))
        return s

    def q(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x

    def train(self, episodes: int, *, eps: float = 0.2, max_steps: int = 100) -> list[int]:
        """ε-жадные действия, награда за подцель, semi-gradient TD-обновление весов.

        Возвращает число шагов до подцели по эпизодам (падает по мере обучения).
        """
        steps_hist: list[int] = []
        for _ in range(episodes):
            s = int(self.rng.choice(self.train_starts))
            x = self.encode(s)
            used = max_steps
            for t in range(1, max_steps + 1):
                a = int(self.rng.integers(4)) if self.rng.random() < eps else int(np.argmax(self.q(x)))
                sp = self._move(s, a)
                done = sp == self.subgoal
                r = 1.0 if done else -self.step_penalty
                xp = self.encode(sp)
                target = r + (0.0 if done else self.gamma * float(np.max(self.q(xp))))
                self.W[a] += self.alpha * (target - self.q(x)[a]) * x  # semi-gradient TD
                s, x = sp, xp
                if done:
                    used = t
                    break
            steps_hist.append(used)
        return steps_hist

    def reach(self, start: int, *, max_steps: int = 80, perturb: float = 0.0, seed: int = 0) -> bool:
        """Исполнить выученную политику (через восприятие φ) до подцели."""
        rng = np.random.default_rng(seed)
        s = start
        for _ in range(max_steps):
            if s == self.subgoal:
                return True
            s = self._move(s, int(np.argmax(self.q(self.encode(s)))))
            if rng.random() < perturb:
                s = int(rng.choice(self.free))
        return s == self.subgoal
