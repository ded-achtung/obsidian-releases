"""Выученный приор поиска: искать программы по вероятности, а не вслепую.

При росте библиотеки наивный перебор замедляется (больше примитивов = шире ветвление).
Решение (как в DreamCoder): выучить ПРИОР над примитивами — какие чаще полезны — и
вести поиск в порядке убывания вероятности (best-first по описательной длине, MDL).
Тогда решение находится после ГОРАЗДО меньшего числа проверенных программ, и поиск
ускоряется по мере роста библиотеки, а не замедляется.

  usage_prior      — частоты использования примитивов в решениях (с сглаживанием);
  best_first_induce — поиск с приоритетом по стоимости −log P(примитив); считает,
                      сколько программ проверено до решения.
"""

from __future__ import annotations

import heapq
import itertools
import math

from thinking_system.reasoning.induction import Program, Primitive, _eq


def usage_prior(solutions: list[Program], primitives: list[Primitive]) -> dict[str, float]:
    """Веса примитивов из частоты их использования в решениях (add-one сглаживание)."""
    from collections import Counter

    c: Counter = Counter()
    for prog in solutions:
        for step in prog.steps:
            c[step.name] += 1
    return {p.name: c.get(p.name, 0) + 1.0 for p in primitives}


def best_first_induce(examples: list[tuple], primitives: list[Primitive], weights: dict[str, float] | None = None, *,
                      max_depth: int = 3, budget: int = 200000) -> tuple[Program | None, int]:
    """Поиск программы в порядке вероятности; вернуть (программа, число проверенных)."""
    inputs = [i for i, _ in examples]
    outputs = [o for _, o in examples]

    def consistent(vals):
        return all(_eq(v, o) for v, o in zip(vals, outputs))

    if consistent(inputs):
        return Program([]), 0

    w = weights or {}
    total = sum(w.get(p.name, 1.0) for p in primitives) or 1.0
    cost = {p.name: -math.log(w.get(p.name, 1.0) / total + 1e-12) for p in primitives}

    counter = itertools.count()
    pq = [(0.0, next(counter), [], list(inputs))]          # (стоимость, тай-брейк, шаги, значения)
    generated = 0
    while pq:
        c, _, steps, vals = heapq.heappop(pq)
        if len(steps) >= max_depth:
            continue
        for p in primitives:
            try:
                nv = [p.fn(v) for v in vals]
            except (TypeError, ValueError, IndexError, KeyError):  # примитив неприменим к значению → отсев
                continue
            generated += 1
            if generated > budget:
                return None, generated
            ns = steps + [p]
            if consistent(nv):
                return Program(ns), generated
            heapq.heappush(pq, (c + cost[p.name], next(counter), ns, nv))
    return None, generated


def search_cost(tasks: list[list[tuple]], primitives: list[Primitive], weights: dict[str, float] | None = None, *,
                max_depth: int = 3) -> tuple[int, int]:
    """Суммарно по задачам: (число решённых, всего проверено программ)."""
    solved = total = 0
    for ex in tasks:
        prog, n = best_first_induce(ex, primitives, weights, max_depth=max_depth)
        total += n
        solved += int(prog is not None)
    return solved, total
