"""Умный поиск глубины 3+: биграммный приор из опыта + направляющая эвристика.

Диагноз предыдущих шагов (AUDIT, дополнение 6): на глубине ≤2 узкое место — не
язык, а ГЛУБИНА поиска: перебор глубины 3 при ~90 примитивах — сотни тысяч
программ на задачу. Здесь поиск делается умнее, а не шире:

  • биграммный приор: из накопленных решений учится, какой шаг вероятен ПОСЛЕ
    какого (униграммный приор search_prior этого не видит) — поиск сначала
    пробует последовательности, похожие на прежние решения;
  • направляющая эвристика: узлы, чьи промежуточные сетки БЛИЖЕ к целевой
    (совпадает форма, меньше несовпадающих клеток), разворачиваются раньше.

Эвристика только ПЕРЕУПОРЯДОЧИВАЕТ обход (без отсечения) — любая программа
в пределах бюджета остаётся достижимой; жадности вида «дошли до цели» нет,
принимается только точное совпадение на ВСЕХ парах.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter

from thinking_system.reasoning.induction import Primitive, Program, _eq

_START = "^"                             # маркер начала программы для биграмм


def bigram_prior(solutions: list[list[str]]) -> dict:
    """Счётчики пар (предыдущий шаг → следующий) из решений; ключ (_START, x) — первый шаг."""
    counts: Counter = Counter()
    for names in solutions:
        prev = _START
        for n in names:
            counts[(prev, n)] += 1
            prev = n
    return dict(counts)


def _mismatch(vals, outputs) -> float:
    """Несходство с целями ∈ [0, 1]: 1 — другая форма, иначе доля несовпавших клеток."""
    total = 0.0
    for v, o in zip(vals, outputs):
        if not (isinstance(v, tuple) and v and isinstance(v[0], tuple)):
            total += 1.0
        elif (len(v), len(v[0])) != (len(o), len(o[0])):
            total += 1.0
        else:
            cells = len(o) * len(o[0])
            bad = sum(1 for rv, ro in zip(v, o) for a, b in zip(rv, ro) if a != b)
            total += bad / cells if cells else 0.0
    return total / max(len(outputs), 1)


def guided_induce(examples: list[tuple], primitives: list[Primitive],
                  bigram: dict | None = None, *, max_depth: int = 3,
                  budget: int = 30000, guide_weight: float = 3.0) -> tuple[Program | None, int]:
    """Best-first по (стоимость пути по биграммам + λ·несходство); (программа, проверено)."""
    inputs = [i for i, _ in examples]
    outputs = [o for _, o in examples]

    def consistent(vals) -> bool:
        return all(_eq(v, o) for v, o in zip(vals, outputs))

    if consistent(inputs):
        return Program([]), 0

    bigram = bigram or {}
    uni: Counter = Counter()
    for (_, n), k in bigram.items():
        uni[n] += k
    norm = math.log(1.0 + 5.0 * max(uni.values(), default=1))

    def step_cost(prev: str, name: str) -> float:
        w = 1.0 + 5.0 * bigram.get((prev, name), 0) + uni.get(name, 0)
        return norm - math.log(w)

    counter = itertools.count()
    pq = [(guide_weight * _mismatch(inputs, outputs),
           next(counter), 0.0, _START, [], list(inputs))]   # (приоритет, тай-брейк, путь, prev, шаги, значения)
    checked = 0
    while pq:
        _, _, pcost, prev, steps, vals = heapq.heappop(pq)
        expandable = len(steps) + 1 < max_depth
        for p in primitives:
            try:
                nv = [p.fn(v) for v in vals]
            except Exception:  # noqa: BLE001 — недопустимый шаг
                continue
            checked += 1
            if checked > budget:
                return None, checked
            if consistent(nv):
                return Program(steps + [p]), checked
            if expandable:                                   # предельную глубину не заталкиваем:
                ncost = pcost + step_cost(prev, p.name)      # её узлы только проверяются
                heapq.heappush(pq, (ncost + guide_weight * _mismatch(nv, outputs),
                                    next(counter), ncost, p.name, steps + [p], nv))
    return None, checked
