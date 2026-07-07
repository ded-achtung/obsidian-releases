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


def make_step_cost(bigram: dict | None, primitives: list[Primitive], smooth: float):
    """Стоимость шага −log P(name|prev), P = (1−smooth)·P_биграмм + smooth/V.

    Свойство (урок дополнения 11 AUDIT): при smooth > 0 стоимость ЛЮБОГО шага
    ограничена −log(smooth/V), какие бы счётчики ни накопились в приоре —
    перераспределение вероятностей не может заморить путь голодом."""
    bigram = bigram or {}
    uni: Counter = Counter()
    for (_, n), k in bigram.items():
        uni[n] += k
    n_prims = len(primitives) or 1
    base_w = {p.name: 1.0 + uni.get(p.name, 0) for p in primitives}
    sum_base = sum(base_w.values())
    prev_sums: dict[str, float] = {}                         # Σw по prev (кэш нормировки)

    def step_cost(prev: str, name: str) -> float:
        s = prev_sums.get(prev)
        if s is None:
            s = sum_base + 5.0 * sum(k for (pv, _), k in bigram.items() if pv == prev)
            prev_sums[prev] = s
        w = base_w.get(name, 1.0) + 5.0 * bigram.get((prev, name), 0)
        return -math.log((1.0 - smooth) * (w / s) + smooth / n_prims)

    return step_cost


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
                  budget: int = 30000, guide_weight: float = 3.0,
                  smooth: float = 0.1) -> tuple[Program | None, int]:
    """Best-first по (стоимость пути по биграммам + λ·несходство); (программа, проверено).

    Приор СГЛАЖИВАЕТСЯ смесью с равномерным: P = (1−smooth)·P_биграмм + smooth/V.
    Урок дополнения 11 (AUDIT): без сглаживания перераспределение вероятностей
    при росте опыта вытесняет прежние находки за границу бюджета; смесь
    ограничивает стоимость любого шага величиной −log(smooth/V) — ни один
    путь не голодает, какие бы биграммы ни накопились. Замер чувствительности
    (дополнение 12): smooth=0.3 уже РАЗМЫВАЕТ приор настолько, что находки
    приора теряются; 0.1 сохраняет и находки приора, и границу стоимости.
    """
    inputs = [i for i, _ in examples]
    outputs = [o for _, o in examples]

    def consistent(vals) -> bool:
        return all(_eq(v, o) for v, o in zip(vals, outputs))

    if consistent(inputs):
        return Program([]), 0

    step_cost = make_step_cost(bigram, primitives, smooth)

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
