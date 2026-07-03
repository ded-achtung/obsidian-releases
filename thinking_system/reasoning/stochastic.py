"""Стохастические миры: вероятностная модель + планирование под неопределённостью.

Когда мир шумный (действие иногда «соскальзывает»), детерминированное правило не
вывести: одна и та же пара клетка×действие даёт РАЗНЫЕ исходы. Тогда агент
оценивает РАСПРЕДЕЛЕНИЕ переходов P(s'|s,a) из наблюдений и планирует под
неопределённостью (итерация по ценности). Получается РЕАКТИВНАЯ политика, устойчивая
к сбоям, — а не хрупкий фиксированный план.

Это стохастическое обобщение индукции правил и смычка с причинной моделью: шум —
это «экзогенная случайность в механизме» (ср. SCM), которую агент оценивает и
учитывает при выборе действий.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from thinking_system.world.gridworld import GridWorld


class StochasticGridEnv:
    """Скользкий мир: с вероятностью slip действие подменяется случайным (соскальзывание)."""

    def __init__(self, grid: GridWorld, *, slip: float = 0.25, seed: int = 0) -> None:
        self.g = grid
        self.slip = slip
        self.rng = np.random.default_rng(seed)
        self.free = grid.free_cells()

    def transition(self, s: tuple[int, int], a: int) -> tuple[int, int]:
        if self.rng.random() < self.slip:
            a = int(self.rng.integers(4))                    # соскользнул на случайное направление
        return self.g.move_from(s, GridWorld.MOVES[a])


class ProbabilisticModel:
    """Оценка P(s'|s,a) из наблюдений + планирование итерацией по ценности."""

    def __init__(self, grid: GridWorld) -> None:
        self.g = grid
        self.free = grid.free_cells()
        self.counts: dict[tuple[tuple[int, int], int], Counter] = defaultdict(Counter)
        self.V: dict[tuple[int, int], float] = {}
        self.pi: dict[tuple[int, int], int] = {}

    def fit(self, observations) -> "ProbabilisticModel":
        for s, a, sp in observations:
            self.counts[(s, a)][sp] += 1
        return self

    def transitions(self, s: tuple[int, int], a: int) -> dict[tuple[int, int], float]:
        """Оценённое распределение исходов; невиданная пара — считаем, что остаёмся на месте."""
        c = self.counts.get((s, a))
        if not c:
            return {s: 1.0}
        tot = sum(c.values())
        return {sp: n / tot for sp, n in c.items()}

    def _q(self, s, a, goal, gamma):
        return sum(p * ((1.0 if sp == goal else 0.0) + gamma * self.V[sp]) for sp, p in self.transitions(s, a).items())

    def value_iteration(self, goal: tuple[int, int], *, gamma: float = 0.95, iters: int = 300) -> dict:
        """Планирование под неопределённостью: V(s)=max_a E[reward+γV(s')]; политика — argmax."""
        self.V = {c: 0.0 for c in self.free}
        for _ in range(iters):
            newV = {}
            for s in self.free:
                newV[s] = 0.0 if s == goal else max(self._q(s, a, goal, gamma) for a in range(4))
            self.V = newV
        self.pi = {s: max(range(4), key=lambda a: self._q(s, a, goal, gamma)) for s in self.free if s != goal}
        return self.V

    def policy(self, s: tuple[int, int]) -> int:
        return self.pi.get(s, 0)

    def reach(self, env: StochasticGridEnv, start: tuple[int, int], goal: tuple[int, int], *, max_steps: int = 200) -> tuple[bool, int]:
        """Исполнить РЕАКТИВНУЮ политику в шумной среде (выбор действия по текущей клетке)."""
        s = start
        for t in range(max_steps):
            if s == goal:
                return True, t
            s = env.transition(s, self.policy(s))
        return s == goal, max_steps
