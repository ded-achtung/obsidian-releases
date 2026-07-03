"""OptionDiscoverer — агент САМ открывает подцели для навыков (без подсказок).

Внутренняя мотивация над опциями: никто не даёт агенту список проёмов. Блуждая,
он строит карту и находит БУТЫЛОЧНЫЕ ГОРЛЫШКИ — узкие клетки (ровно 2 соседа) с
высокой посреднической центральностью. Это дверные проёмы: через них идут почти
все межкомнатные пути (классическое обнаружение подцелей-горлышек, McGovern&Barto).

Найденные горлышки становятся подцелями НАВЫКОВ: к каждой агент учит Q-опцию из
опыта. Кривая обучения даёт learning progress — автокуррикулум (во что вкладывать
усилия). Открытые из опыта навыки покрывают мир (доходят куда угодно), и делают
это лучше, чем навыки к случайным подцелям.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.agent.qoption import QOption


class OptionDiscoverer:
    """Открывает подцели-горлышки из опыта и учит к ним Q-навыки.

    Args:
        grid: мир.
        seed: зерно блуждания.
    """

    def __init__(self, grid: GridWorld, *, seed: int = 0) -> None:
        self.g = grid
        self.free = grid.free_sids()
        self.start = grid.sid(grid.start)
        self.adj: dict[int, set[int]] = {s: set() for s in self.free}
        self.subgoals: list[int] = []
        self.options: dict[int, QOption] = {}
        self.rng = np.random.default_rng(seed)

    def _move(self, s: int, a: int) -> int:
        return self.g.move_sid(s, a)

    def explore(self, steps: int) -> None:
        """Случайно блуждать, накапливая неориентированную карту соседств."""
        s = self.start
        for _ in range(steps):
            sp = self._move(s, int(self.rng.integers(4)))
            if sp != s:
                self.adj[s].add(sp)
                self.adj[sp].add(s)
            s = sp

    def _betweenness(self) -> Counter:
        """Через сколько кратчайших путей проходит клетка (по выученной карте)."""
        bc: Counter = Counter()
        for src in self.free:
            prev: dict[int, int | None] = {src: None}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if v not in prev:
                        prev[v] = u
                        q.append(v)
            for dst in self.free:
                if dst == src or dst not in prev:
                    continue
                u = prev[dst]
                while u is not None and u != src:
                    bc[u] += 1
                    u = prev[u]
        return bc

    def discover(self, k: int = 4) -> list[int]:
        """Подцели = узкие клетки (ровно 2 соседа) с наибольшей центральностью."""
        bc = self._betweenness()
        narrow = [s for s in self.free if len(self.adj[s]) == 2]
        self.subgoals = sorted(narrow, key=lambda s: -bc[s])[:k]
        return self.subgoals

    def learn_options(self, subgoals: list[int] | None = None, *, episodes_each: int = 1500, seed: int = 0) -> dict[int, list[int]]:
        """К каждой подцели выучить Q-опцию из опыта; вернуть кривые обучения."""
        subgoals = subgoals if subgoals is not None else self.subgoals
        self.options = {}
        curves: dict[int, list[int]] = {}
        for i, sg in enumerate(subgoals):
            opt = QOption(self.g, sg, seed=seed + i)
            curves[sg] = opt.train(episodes_each)
            self.options[sg] = opt
        return curves

    @staticmethod
    def learning_progress(curves: dict[int, list[int]], *, w: int = 100) -> dict[int, float]:
        """Прирост компетенции = насколько упало число шагов до подцели (автокуррикулум)."""
        return {sg: float(np.mean(c[:w]) - np.mean(c[-w:])) for sg, c in curves.items()}

    def _map_dist(self, src: int) -> dict[int, int]:
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def coverage(self, subgoals: list[int] | None = None, *, local: int = 4) -> float:
        """Доля клеток, достижимых через ближайшую подцель + короткий локальный шаг.

        Хорошие подцели-горлышки — переиспользуемые путевые точки: к любой клетке
        ведёт «доехать опцией до ближайшего горлышка, потом локально дойти».
        """
        sg = list(subgoals if subgoals is not None else self.subgoals)
        dist = {s: self._map_dist(s) for s in sg}
        ok = tot = 0
        for g in self.free:
            if g in sg:
                continue
            tot += 1
            v = min(sg, key=lambda s: dist[s].get(g, 10 ** 9))
            if dist[v].get(g, 10 ** 9) <= local:
                ok += 1
        return ok / tot

    def navigate(self, start: int, goal: int, *, local: int = 8, max_steps: int = 120) -> bool:
        """Дойти до ПРОИЗВОЛЬНОЙ цели: опция к ближайшему горлышку + локальный доход."""
        if goal in self.options:
            return self.options[goal].reach(start)
        sgd = {s: self._map_dist(s) for s in self.subgoals}
        v = min(self.subgoals, key=lambda s: sgd[s].get(goal, 10 ** 9))  # путевая точка-горлышко
        s, opt = start, self.options[v]
        for _ in range(max_steps):
            if s == v:
                break
            s = self._move(s, opt.policy(s))
        gd = self._map_dist(goal)
        for _ in range(local):                                          # короткий локальный доход
            if s == goal:
                return True
            a = min(range(4), key=lambda a: gd.get(self._move(s, a), 10 ** 9))
            s = self._move(s, a)
        return s == goal
