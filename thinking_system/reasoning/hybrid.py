"""Частичный успех рассуждения: правило для регулярной части + заплатки на исключения.

Реальный мир редко целиком регулярен. Вместо «или вывели правило, или учим всё с
нуля» агент выводит правило для БОЛЬШИНСТВА переходов, а немногие нерегулярные
клетки запоминает как ИСКЛЮЧЕНИЯ. Стоимость пропорциональна доле нерегулярного, а
не размеру мира.

  induce_majority — программа, объясняющая как можно больше примеров (остальные —
                    исключения), а не обязательно все (как induce).
  HybridWorldModel — модель = правило-на-действие ⊕ таблица исключений (s,a)→s'.
                     predict: исключение важнее правила; правило обобщает регулярное.
"""

from __future__ import annotations

from collections import defaultdict, deque

from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.induction import Primitive, Program, _eq
from thinking_system.reasoning.grounded import pair_primitives


def induce_majority(examples, primitives: list[Primitive], *, max_depth: int = 2):
    """Найти программу, объясняющую МАКСИМУМ примеров; вернуть (программа, индексы исключений)."""
    outs = [o for _, o in examples]
    ins = [i for i, _ in examples]

    def misfit(vals):
        return [i for i, (v, o) in enumerate(zip(vals, outs)) if not _eq(v, o)]

    best_steps: list[Primitive] = []
    best_miss = misfit(ins)                                  # глубина 0: тождество
    frontier = [([], list(ins))]
    for _ in range(max_depth):
        nxt = []
        for steps, vals in frontier:
            for p in primitives:
                try:
                    nv = [p.fn(v) for v in vals]
                except (TypeError, ValueError, IndexError, KeyError):  # примитив неприменим к значению → отсев
                    continue
                ns = steps + [p]
                miss = misfit(nv)
                if len(miss) < len(best_miss):
                    best_steps, best_miss = ns, miss
                nxt.append((ns, nv))
        frontier = nxt
    return Program(best_steps), best_miss


class HybridWorldModel:
    """Модель мира = правило-на-действие (для регулярного) ⊕ исключения (для остального)."""

    def __init__(self, grid: GridWorld) -> None:
        self.g = grid
        self.rules: dict[int, Program] = {}
        self.exceptions: dict[tuple[tuple[int, int], int], tuple[int, int]] = {}

    def fit(self, observations) -> "HybridWorldModel":
        """Из наблюдений: на каждое действие — правило большинства + заплатки исключений."""
        by_a: dict[int, list] = defaultdict(list)
        for s, a, sp in observations:
            if sp != s:                                      # правило учим по ходам (упор в стену — отдельно)
                by_a[a].append((s, sp))
        prims = pair_primitives()
        for a, ex in by_a.items():
            self.rules[a], _ = induce_majority(ex, prims, max_depth=2)
        for s, a, sp in observations:                        # вторая проходка: любая ошибка правила → исключение
            if self._rule_predict(s, a) != sp:               # ловит и неверный ход, и неверный «упор»
                self.exceptions[(s, a)] = sp
        return self

    def _rule_predict(self, s: tuple[int, int], a: int) -> tuple[int, int]:
        prog = self.rules.get(a)
        if prog is None:
            return s
        r, c = prog(s)
        if 0 <= r < self.g.size and 0 <= c < self.g.size and (r, c) not in self.g.walls:
            return (r, c)
        return s

    def predict(self, s: tuple[int, int], a: int) -> tuple[int, int]:
        if (s, a) in self.exceptions:                        # исключение важнее правила
            return self.exceptions[(s, a)]
        return self._rule_predict(s, a)

    def plan(self, start: tuple[int, int], goal: tuple[int, int]) -> list[int] | None:
        """BFS по гибридной модели (правило + исключения)."""
        prev = {start: None}
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

    def reach(self, env, start: tuple[int, int], goal: tuple[int, int], *, max_steps: int = 400) -> tuple[bool, int]:
        """Замкнутый цикл: спланировать → шаг → при СЮРПРИЗЕ дописать исключение → переспланировать.

        Так агент доходит даже при неполной модели, доучивая нерегулярное по ходу.
        """
        s = start
        for t in range(max_steps):
            if s == goal:
                return True, t
            path = self.plan(s, goal)
            if not path:
                return False, t
            a = path[0]
            sp = env.transition(s, a)
            if sp != self.predict(s, a):                     # сюрприз → выучить исключение и переспланировать
                self.exceptions[(s, a)] = sp
            s = sp
        return s == goal, max_steps
