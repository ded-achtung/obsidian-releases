"""Причинно-модельное рассуждение: вмешательства do() и контрфактика (лестница Пёрла).

Структурная причинная модель (SCM): переменные в топологическом порядке, у каждой —
родители и механизм, корни заданы распределением. На малой модели выводы ТОЧНЫ
(перечисление корней), и видно главное, чего не может статистика:

  • наблюдение P(Y|X)        — что СВЯЗАНО (корреляция);
  • вмешательство P(Y|do(X)) — что будет, если ЗАДАТЬ X (разрывает входящие связи);
  • контрфактика             — что было бы С ЭТИМ случаем, будь X иным (абдукция шума
                               → действие → предсказание, 3 шага Пёрла).

Корреляция ≠ причинность: общий предок делает X и Y связанными, но do(X) на Y не
влияет. Эта модель различает «увидеть X» и «сделать X», отвечает на «что если» и
планирует — вмешиваясь в настоящую причину.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable


class SCM:
    """Структурная причинная модель: корни (распределения) + механизмы (функции родителей)."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.parents: dict[str, list[str]] = {}
        self.mech: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.roots: dict[str, dict[Any, float]] = {}

    def root(self, name: str, dist: dict[Any, float]) -> "SCM":
        """Экзогенная переменная-корень с распределением {значение: вероятность}."""
        self.order.append(name)
        self.parents[name] = []
        self.roots[name] = dist
        return self

    def eq(self, name: str, parents: list[str], fn: Callable[[dict[str, Any]], Any]) -> "SCM":
        """Структурное уравнение: name = fn(значения родителей)."""
        self.order.append(name)
        self.parents[name] = parents
        self.mech[name] = fn
        return self

    def simulate(self, root_vals: dict[str, Any], *, do: dict[str, Any] | None = None) -> dict[str, Any]:
        """Один прогон модели: корни заданы, остальное вычисляется; do переопределяет."""
        do = do or {}
        v: dict[str, Any] = {}
        for name in self.order:
            if name in do:
                v[name] = do[name]
            elif name in self.roots:
                v[name] = root_vals[name]
            else:
                v[name] = self.mech[name]({p: v[p] for p in self.parents[name]})
        return v

    def _worlds(self, *, do: dict[str, Any] | None = None):
        """Перечислить все миры с весами (точный вывод по корням, не входящим в do)."""
        do = do or {}
        free = [r for r in self.roots if r not in do]
        supports = [list(self.roots[r].items()) for r in free]
        for combo in itertools.product(*supports) if free else [()]:
            rv = {r: val for r, (val, _) in zip(free, combo)}
            rv.update({r: do[r] for r in self.roots if r in do})
            w = 1.0
            for _, p in combo:
                w *= p
            yield self.simulate(rv, do=do), w

    def prob(self, var: str, val: Any, *, given: dict[str, Any] | None = None, do: dict[str, Any] | None = None) -> float:
        """P(var=val | наблюдение given, вмешательство do). Точно, перечислением миров."""
        given = given or {}
        num = den = 0.0
        for world, w in self._worlds(do=do):
            if all(world[k] == v for k, v in given.items()):
                den += w
                if world[var] == val:
                    num += w
        return num / den if den else 0.0

    def affects(self, x: str, y: str, *, lo: Any = 0, hi: Any = 1, val: Any = 1) -> float:
        """Причинный эффект x на y: P(y=val|do(x=hi)) − P(y=val|do(x=lo))."""
        return self.prob(y, val, do={x: hi}) - self.prob(y, val, do={x: lo})

    def correlation(self, x: str, y: str, *, lo: Any = 0, hi: Any = 1, val: Any = 1) -> float:
        """Наблюдательная связь x и y: P(y=val|x=hi) − P(y=val|x=lo) (видеть, не делать)."""
        return self.prob(y, val, given={x: hi}) - self.prob(y, val, given={x: lo})

    def counterfactual(self, evidence: dict[str, Any], do: dict[str, Any], query: str) -> dict[Any, float]:
        """Контрфактика: при ЭТОМ наблюдении — что было бы с query, будь do (3 шага Пёрла)."""
        consistent = []                                     # абдукция: миры, совместимые с наблюдением
        for rv in self._root_assignments():
            if all(self.simulate(rv)[k] == v for k, v in evidence.items()):
                consistent.append(rv)
        out: dict[Any, float] = {}
        wsum = 0.0
        for rv in consistent:                               # действие + предсказание под do
            w = 1.0
            for r, val in rv.items():
                w *= self.roots[r][val]
            res = self.simulate(rv, do=do)[query]
            out[res] = out.get(res, 0.0) + w
            wsum += w
        return {k: v / wsum for k, v in out.items()} if wsum else {}

    def _root_assignments(self):
        names = list(self.roots)
        for combo in itertools.product(*[list(self.roots[r]) for r in names]):
            yield dict(zip(names, combo))

    def plan(self, goal_var: str, goal_val: Any, candidates: list[tuple[str, Any]]) -> tuple[tuple[str, Any], float]:
        """Найти вмешательство (переменная, значение), максимизирующее P(goal_var=goal_val)."""
        scored = [((x, v), self.prob(goal_var, goal_val, do={x: v})) for x, v in candidates]
        return max(scored, key=lambda s: s[1])
