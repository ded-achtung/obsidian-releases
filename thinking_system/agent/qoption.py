"""Опции, обученные ИЗ ОПЫТА (Q-learning) — иерархический RL без готовой карты.

В отличие от опции-из-карты (ценность = BFS по известным переходам), здесь навык
учится методом проб и ошибок: агент действует, получает награду за достижение
подцели и обновляет Q(s,a) по временной разности. Карта переходов не нужна —
политика возникает из опыта. Обученные опции компонуются на верхнем уровне.
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld


class QOption:
    """Навык-опция к подцели, обучаемый табличным Q-learning из опыта.

    Args:
        grid: мир.
        subgoal: целевая клетка опции (подцель).
        alpha: скорость обучения; gamma: дисконт; step_penalty: штраф за шаг.
        seed: зерно.
    """

    def __init__(self, grid: GridWorld, subgoal: int, *, alpha: float = 0.5, gamma: float = 0.95, step_penalty: float = 0.01, seed: int = 0) -> None:
        self.g = grid
        self.subgoal = subgoal
        self.alpha = alpha
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.Q = np.zeros((grid.n_states, 4))
        self.free = grid.free_sids()
        self.rng = np.random.default_rng(seed)

    def _move(self, s: int, a: int) -> int:
        return self.g.move_sid(s, a)

    def policy(self, s: int) -> int:
        return int(np.argmax(self.Q[s]))

    def train(self, episodes: int, *, eps: float = 0.2, max_steps: int = 100) -> list[int]:
        """Обучение из опыта: ε-жадные действия, награда за подцель, TD-обновление Q.

        Возвращает число шагов до подцели по эпизодам (падает по мере обучения).
        """
        steps_hist: list[int] = []
        for _ in range(episodes):
            s = int(self.rng.choice(self.free))
            used = max_steps
            for t in range(1, max_steps + 1):
                a = int(self.rng.integers(4)) if self.rng.random() < eps else self.policy(s)
                sp = self._move(s, a)
                done = sp == self.subgoal
                r = 1.0 if done else -self.step_penalty
                target = r + (0.0 if done else self.gamma * float(np.max(self.Q[sp])))
                self.Q[s, a] += self.alpha * (target - self.Q[s, a])
                s = sp
                if done:
                    used = t
                    break
            steps_hist.append(used)
        return steps_hist

    def reach(self, start: int, *, max_steps: int = 80, perturb: float = 0.0, seed: int = 0) -> bool:
        """Исполнить выученную политику до подцели (устойчиво к сбоям — это политика)."""
        rng = np.random.default_rng(seed)
        s = start
        for _ in range(max_steps):
            if s == self.subgoal:
                return True
            s = self._move(s, self.policy(s))
            if rng.random() < perturb:
                s = int(rng.choice(self.free))
        return s == self.subgoal


class QOptionLibrary:
    """Набор Q-обученных опций (по подцели на ориентир) + их композиция."""

    def __init__(self, grid: GridWorld, subgoals: list[int], *, seed: int = 0) -> None:
        self.g = grid
        self.subgoals = subgoals
        self.options = {L: QOption(grid, L, seed=seed + i) for i, L in enumerate(subgoals)}

    def train(self, episodes_each: int = 1500) -> dict[int, list[int]]:
        return {L: opt.train(episodes_each) for L, opt in self.options.items()}

    def compose(self, start: int, landmark_seq: list[int], *, perturb: float = 0.0, max_steps_each: int = 80, seed: int = 0) -> bool:
        """Иерархически пройти по ориентирам, исполняя обученные опции по очереди."""
        rng = np.random.default_rng(seed)
        s = start
        for L in landmark_seq:
            opt = self.options[L]
            for _ in range(max_steps_each):
                if s == L:
                    break
                s = opt._move(s, opt.policy(s))
                if rng.random() < perturb:
                    s = int(rng.choice(opt.free))
            if s != L:
                return False
        return True
