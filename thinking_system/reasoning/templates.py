"""Анти-унификация: разные решения с общей структурой → ШАБЛОН с переменной-дыркой.

Второй вид «переменных» в языке. Из пары решений одинаковой длины, совпадающих
во всех позициях кроме одной («keep[3] ▸ bbox» и «keep[2] ▸ bbox»), выводится
шаблон «? ▸ bbox»: конкретные привязки обобщены в переменную. Шаблонный поиск
перебирает только заполнение дырки (|P| программ вместо |P|^len) — так структуры
глубины 3+ становятся достижимыми по цене одного шага. Чем богаче опыт решений,
тем больше шаблонов — способность растёт из жизни агента, не из рук автора.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from thinking_system.reasoning.induction import Primitive, Program, _eq

HOLE = None  # переменная-дырка в шаблоне: сюда подставляются примитивы

Template = tuple  # кортеж из имён шагов и HOLE, напр. ("flip_h", None, "bbox")


def anti_unify(solutions: list[list[str]], *, max_templates: int = 20) -> list[Template]:
    """Шаблоны из пар решений одной длины, различающихся РОВНО в одной позиции.

    Возвращает шаблоны, отсортированные по поддержке (скольким парам решений
    шаблон обязан) и длине; длина ≥ 2, ровно одна дырка.
    """
    support: Counter = Counter()
    for a, b in combinations([tuple(s) for s in solutions], 2):
        if len(a) != len(b) or len(a) < 2:
            continue
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(diff) == 1:
            support[tuple(HOLE if i == diff[0] else x for i, x in enumerate(a))] += 1
    ranked = sorted(support, key=lambda t: (-support[t], -len(t)))
    return ranked[:max_templates]


def template_search(pairs: list, templates: list[Template],
                    primitives: list[Primitive],
                    resolve=None) -> tuple[Program | None, int]:
    """Заполнить дырку каждого шаблона каждым примитивом; проверить на всех парах.

    resolve(имя) → Primitive для фиксированных шагов шаблона (по умолчанию —
    поиск по имени среди primitives). Возвращает (программа | None, проверено).
    """
    by_name = {p.name: p for p in primitives}
    resolve = resolve or by_name.get
    inputs = [i for i, _ in pairs]
    outputs = [o for _, o in pairs]
    checked = 0
    for tpl in templates:
        fixed = [resolve(n) if n is not HOLE else HOLE for n in tpl]
        if any(s is None and n is not HOLE for s, n in zip(fixed, tpl)):
            continue                                        # фиксированный шаг вне языка
        for cand in primitives:
            steps = [cand if s is HOLE else s for s in fixed]
            checked += 1
            try:
                vals = list(inputs)
                for p in steps:
                    vals = [p.fn(v) for v in vals]
            except Exception:  # noqa: BLE001 — несовместимый шаг
                continue
            if all(_eq(v, o) for v, o in zip(vals, outputs)):
                return Program(steps), checked
    return None, checked
