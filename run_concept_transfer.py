#!/usr/bin/env python3
"""Перенос: понятия, выученные на семействе A, помогают РЕШАТЬ другое семейство B.

Разрыв «нет переноса» — честный измеримый шаг. Система открывает понятия (MDL) на
обучающем семействе A и применяет их к ДРУГОМУ, held-out семейству B (новые задачи,
переиспользующие те же понятия в новых композициях). Метрика: сколько B решается при
ФИКСИРОВАННОМ бюджете глубины поиска.

Ключевой честный контроль: понятия, выученные на НЕРЕЛЕВАНТНОМ семействе C, B НЕ
помогают. Значит выигрыш — именно от переноса нужной структуры, а не просто от того,
что примитивов стало больше.

Запуск: python run_concept_transfer.py
"""

from __future__ import annotations

from thinking_system.reasoning.induction import Library, default_primitives
from thinking_system.reasoning.mdl_abstraction import MDLLearner
from thinking_system.reasoning.search_prior import search_cost

# A — обучающее семейство (учит понятия +1∘square и each*2∘sum)
FAMILY_A = [
    [(2, 9), (3, 16), (5, 36)], [(2, 18), (3, 32), (4, 50)],
    [(1, 5), (2, 10), (3, 17)], [(2, 11), (3, 18), (4, 27)],
    [([1, 2, 3], 12), ([5], 10), ([2, 2], 8)], [([1, 2, 3], 144), ([5], 100), ([2, 2], 64)],
    [([1, 2, 3], 13), ([5], 11), ([2, 2], 9)], [([1, 2, 3], 14), ([5], 12), ([2, 2], 10)],
]

# C — НЕРЕЛЕВАНТНОЕ семейство: учит понятия над переупорядочиванием списков
# (reverse∘each+1 и tail∘reverse), каждое встречается 4 раза — MDL их откроет, но для B
# они бесполезны. Композиции подобраны так, чтобы не схлопываться в один примитив.
def rev_inc(x): return [e + 1 for e in x[::-1]]                 # reverse ▸ each+1
def tail_rev(x): return x[1:][::-1]                             # tail ▸ reverse
FAMILY_C = [
    [(a, rev_inc(a)) for a in ([1, 2, 3], [5, 1], [0, 4, 2])],                       # reverse ▸ each+1
    [(a, rev_inc(a)[1:]) for a in ([1, 2, 3], [5, 1, 0], [0, 4, 2])],                # reverse ▸ each+1 ▸ tail
    [(a, [e + 1 for e in rev_inc(a)]) for a in ([1, 2, 3], [5, 1], [0, 4, 2])],      # reverse ▸ each+1 ▸ each+1
    [(a, rev_inc(a)[0]) for a in ([1, 2, 3], [5, 1], [0, 4, 2])],                    # reverse ▸ each+1 ▸ head
    [(a, tail_rev(a)) for a in ([1, 2, 3], [5, 1, 0], [0, 4, 2, 7])],                # tail ▸ reverse
    [(a, [e + 1 for e in tail_rev(a)]) for a in ([1, 2, 3], [5, 1, 0], [0, 4, 2])],  # tail ▸ reverse ▸ each+1
    [(a, tail_rev(a)[0]) for a in ([1, 2, 3], [5, 1, 0], [0, 4, 2])],                # tail ▸ reverse ▸ head
    [(a, tail_rev(a)[1:]) for a in ([1, 2, 3, 8], [5, 1, 0], [0, 4, 2, 7])],         # tail ▸ reverse ▸ tail
]

# B — held-out ЦЕЛЕВОЕ семейство: новые задачи, нужны те же понятия, что в A,
# но композиции глубже (3 базовых шага) — при бюджете глубины 2 базовый язык их НЕ берёт.
FAMILY_B = [
    ("3·(x+1)²", [(2, 27), (3, 48), (4, 75)]),                  # +1 ▸ square ▸ *3
    ("((x+1)²)²", [(1, 16), (2, 81), (3, 256)]),                # +1 ▸ square ▸ square
    ("(2·sum)²", [([1, 2], 36), ([3], 36), ([2, 2], 64)]),      # each*2 ▸ sum ▸ square
    ("3·(2·sum)", [([1, 2], 18), ([4], 24), ([2, 2], 24)]),     # each*2 ▸ sum ▸ *3
]


def learn_concepts(family):
    base = default_primitives()
    lib = Library(base)
    corpus = []
    for ex in family:
        prog = lib.induce(ex, max_depth=3)
        if prog is not None:
            corpus.append([s.name for s in prog.steps])
    learner = MDLLearner(base)
    learner.compress(corpus, max_abstractions=6)
    return learner


def solved_at_depth(prims, depth):
    tasks = [ex for _, ex in FAMILY_B]
    solved, checked = search_cost(tasks, prims, None, max_depth=depth)
    return solved, checked


def main():
    base = default_primitives()
    print("▶ Перенос понятий: A → B (с контролем на нерелевантном C)\n")

    a = learn_concepts(FAMILY_A)
    c = learn_concepts(FAMILY_C)
    print(f"   понятия из A (релевантные):    {list(a.abstractions)}")
    print(f"   понятия из C (нерелевантные):  {list(c.abstractions)}")

    DEPTH = 2
    print(f"\n   Решаемость held-out семейства B при бюджете ГЛУБИНЫ {DEPTH} (всего {len(FAMILY_B)} задач):")
    s_base, c_base = solved_at_depth(base, DEPTH)
    s_a, c_a = solved_at_depth(a.to_library().prims, DEPTH)
    s_c, c_c = solved_at_depth(c.to_library().prims, DEPTH)
    print(f"   {'без переноса (базовый язык)':<34}{s_base}/{len(FAMILY_B)}   проверено программ: {c_base}")
    print(f"   {'+ понятия из A (перенос)':<34}{s_a}/{len(FAMILY_B)}   проверено программ: {c_a}")
    print(f"   {'+ понятия из C (контроль)':<34}{s_c}/{len(FAMILY_B)}   проверено программ: {c_c}")

    print("\n   по задачам (глубина 2):")
    grown_a = a.to_library()
    seed_lib = Library(base)
    for label, ex in FAMILY_B:
        before = seed_lib.induce(ex, max_depth=2)
        after = grown_a.induce(ex, max_depth=2)
        print(f"   {label:<14} базовый: {str(before):<8}  →  с понятиями A: «{after}»")

    print("\n── Что это значит ──")
    print("   Понятия, выученные на A, позволяют решать ДРУГОЕ семейство B в пределах тесного")
    print("   бюджета глубины, которого раньше не хватало, — это перенос. Контроль подтверждает")
    print("   честность: нерелевантные понятия из C ту же B НЕ берут (выигрыш — от структуры,")
    print("   а не от лишних примитивов). Честно: перенос работает, когда B переиспользует")
    print("   ровно те абстракции; чужая структура не помогает (и не должна).")


if __name__ == "__main__":
    main()
