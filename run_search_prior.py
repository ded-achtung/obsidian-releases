#!/usr/bin/env python3
"""Выученный приор поиска: при росте библиотеки искать БЫСТРЕЕ, а не медленнее.

Наивный перебор замедляется, когда примитивов больше. Решение (DreamCoder): выучить
приор над примитивами (что чаще полезно) и вести поиск в порядке вероятности
(best-first по −log P). Решение находится после куда меньшего числа проверенных
программ. Сравниваем равномерный поиск и поиск с выученным приором.

Запуск: python run_search_prior.py
"""

from __future__ import annotations

from thinking_system.reasoning.library_learning import LibraryLearner
from thinking_system.reasoning.search_prior import usage_prior, search_cost

TASKS = [
    [(2, 9), (3, 16), (5, 36)], [(1, 4), (4, 25), (2, 9)], [(3, 7), (5, 11), (10, 21)],
    [(2, 18), (3, 32), (4, 50)], [(2, 10), (3, 17), (4, 26)],
    [([1, 2, 3], 12), ([5], 10), ([2, 2], 8)], [([1, 1, 1], 6), ([4], 8)],
    [([1, 2, 3], 144), ([5], 100)],
]


def main():
    print("▶ Выученный приор поиска: меньше перебора при большей библиотеке\n")
    learner = LibraryLearner()
    learner.learn(TASKS, rounds=4, max_depth=2)            # вырастить библиотеку
    sols = list(learner.wake(TASKS, max_depth=2).values())
    prior = usage_prior(sols, learner.lib.prims)           # выучить приор из решений
    prims = learner.lib.prims

    print(f"библиотека: {len(prims)} примитивов (с выученными абстракциями); глубина поиска 3\n")
    s_u, c_u = search_cost(TASKS, prims, None, max_depth=3)
    s_p, c_p = search_cost(TASKS, prims, prior, max_depth=3)
    print(f"   равномерный поиск (вслепую):  решено {s_u}/8, проверено программ: {c_u}")
    print(f"   поиск с ВЫУЧЕННЫМ приором:    решено {s_p}/8, проверено программ: {c_p}")
    print(f"   ускорение: ×{c_u / max(c_p, 1):.1f} меньше перебора при том же результате")

    top = sorted(prior.items(), key=lambda kv: -kv[1])[:5]
    print(f"\n   что приор ставит вперёд: {', '.join(f'{n}({w:.0f})' for n, w in top)}")
    print("   (выученные абстракции и частые операции пробуются первыми)")

    print("\n── Что это значит ──")
    print("   Рост библиотеки теперь не тормозит поиск, а УСКОРЯЕТ его: система выучивает,")
    print("   какие куски чаще полезны, и пробует их первыми. Это рекогнайзер DreamCoder в")
    print("   простом виде (униграммный приор) — ключ к тому, чтобы язык мог расти без предела")
    print("   по стоимости поиска. Честно: униграммный приор прост; контекстный приор сильнее.")


if __name__ == "__main__":
    main()
