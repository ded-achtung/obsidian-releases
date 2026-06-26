"""Рассуждение, ЗАЗЕМЛЁННОЕ в мире: индукция правила прямо из наблюдений агента.

Замыкает рассуждение с остальной системой. Агент делает НЕСКОЛЬКО шагов в мире и
наблюдает тройки (клетка, действие, следующая клетка). Тем же движком индукции
(reasoning/induction) он выводит ПРАВИЛО динамики на каждое действие — из 1-2
примеров, — а не запоминает таблицу переходов. Выученной моделью он предсказывает
и ПЛАНИРУЕТ во всём мире.

Это и есть «учиться из малого»: ~2 наблюдения на действие задают правило для всех
клеток (тогда как табличной модели нужно увидеть каждую пару клетка×действие).
Стены — исключения: правило даёт смещение, а упор в стену/границу оставляет на
месте (агент знает препятствия из восприятия).
"""

from __future__ import annotations

from collections import defaultdict, deque

from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.induction import Primitive, Program, induce


def _pair(s):
    if not (isinstance(s, tuple) and len(s) == 2):
        raise TypeError
    return s


def pair_primitives() -> list[Primitive]:
    """Язык смещений над клеткой (r, c): из них индуцируется эффект каждого действия."""
    return [
        Primitive("row-1", lambda s: (_pair(s)[0] - 1, s[1])),
        Primitive("row+1", lambda s: (_pair(s)[0] + 1, s[1])),
        Primitive("col-1", lambda s: (_pair(s)[0], _pair(s)[1] - 1)),
        Primitive("col+1", lambda s: (_pair(s)[0], _pair(s)[1] + 1)),
    ]


def observe(grid: GridWorld, steps: int, *, seed: int = 0) -> list[tuple[tuple[int, int], int, tuple[int, int]]]:
    """Агент гуляет по миру и записывает наблюдения (клетка, действие, след. клетка)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    grid.reset()
    obs = []
    for _ in range(steps):
        a = int(rng.integers(4))
        s = grid.pos
        grid.step(a)
        obs.append((s, a, grid.pos))
    return obs


def induce_dynamics(observations, *, max_depth: int = 2) -> dict[int, Program]:
    """Из наблюдений вывести правило смещения на каждое действие (где агент сдвинулся)."""
    by_a: dict[int, list] = defaultdict(list)
    for s, a, sp in observations:
        if sp != s:                                          # учимся по ходам, где реально сдвинулись
            by_a[a].append((s, sp))
    prims = pair_primitives()
    rules: dict[int, Program] = {}
    for a, ex in by_a.items():
        prog = induce(ex, prims, max_depth=max_depth)
        if prog is not None:
            rules[a] = prog
    return rules


class WorldRule:
    """Выученная индукцией модель мира: предсказывает переходы и планирует пути."""

    def __init__(self, grid: GridWorld, rules: dict[int, Program]) -> None:
        self.g = grid
        self.rules = rules

    def predict(self, s: tuple[int, int], a: int) -> tuple[int, int]:
        """Применить выученное смещение; упор в стену/границу — остаться на месте."""
        prog = self.rules.get(a)
        if prog is None:
            return s
        r, c = prog(s)
        if 0 <= r < self.g.size and 0 <= c < self.g.size and (r, c) not in self.g.walls:
            return (r, c)
        return s

    def plan(self, start: tuple[int, int], goal: tuple[int, int], *, max_steps: int = 200) -> list[int] | None:
        """BFS по ВЫУЧЕННОЙ динамике: последовательность действий start→goal (или None)."""
        prev: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            if u == goal:
                acts = []
                while prev[u] is not None:
                    p, a = prev[u]
                    acts.append(a)
                    u = p
                return acts[::-1]
            for a in self.rules:
                v = self.predict(u, a)
                if v != u and v not in prev:
                    prev[v] = (u, a)
                    q.append(v)
        return None
