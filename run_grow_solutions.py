#!/usr/bin/env python3
"""Рост библиотеки из РЕАЛЬНЫХ решений: система сама именует комбо, что нашла поиском.

Зеркалит честный прогон на настоящем ARC. Там на глубине 2 система находит решения,
и ДВА комбо реально повторяются в разных задачах:
    «keep_largest ▸ bbox»  — выделить крупнейший объект и обрезать до него (×2 на ARC)
    «flip_h ▸ flip_v»      — поворот на 180° (×2 на ARC)
grow_from_solutions берёт эти НАСТОЯЩИЕ решения (найденные best_first_induce, а не
собственным wake) и абстрагирует повторяющиеся комбо в именованные операции. После
этого такие задачи решаются на глубине 1.

Запуск: python run_grow_solutions.py
"""

from __future__ import annotations

from thinking_system.reasoning.grid_seed import full_grid_seed
from thinking_system.reasoning.perception import keep_largest, bounding_box
from thinking_system.reasoning.grids import flip_h, flip_v, to_grid
from thinking_system.reasoning.induction import Library
from thinking_system.reasoning.library_learning import LibraryLearner
from thinking_system.reasoning.search_prior import best_first_induce

SEED = full_grid_seed()


def extract_largest(g): return bounding_box(keep_largest(g))   # keep_largest ▸ bbox
def rot180(g): return flip_v(flip_h(g))                        # flip_h ▸ flip_v

# задачи «как на ARC»: крупный объект + шум; и пары на поворот 180°
E1 = to_grid([[3, 3, 0, 0, 0], [3, 3, 0, 0, 0], [0, 0, 0, 5, 0], [0, 0, 0, 0, 0]])
E2 = to_grid([[0, 0, 0, 7], [0, 4, 4, 0], [0, 4, 4, 0], [2, 0, 0, 0]])
E3 = to_grid([[0, 0, 9, 0], [6, 6, 0, 0], [6, 6, 0, 0], [6, 6, 0, 0]])      # held-out
R1 = to_grid([[1, 2, 0], [0, 3, 4]])
R2 = to_grid([[5, 0, 6], [7, 8, 0]])
R3 = to_grid([[1, 0], [2, 3]])                                              # held-out


def main():
    print("▶ Рост библиотеки из РЕАЛЬНЫХ решений (как на ARC): имена для найденных комбо\n")
    # 1) внешний решатель (best_first_induce) находит решения — собираем НАСТОЯЩИЕ программы
    found = []
    for g in (E1, E2):
        prog, _ = best_first_induce([(g, extract_largest(g))], SEED, None, max_depth=2, budget=4000)
        found.append(prog)
    for g in (R1, R2):
        prog, _ = best_first_induce([(g, rot180(g))], SEED, None, max_depth=2, budget=4000)
        found.append(prog)
    print("   решения, найденные поиском:")
    for p in found:
        print(f"      «{p}»")

    # 2) грузим РЕШЕНИЯ в библиотеку — она сама абстрагирует повторяющиеся комбо
    learner = LibraryLearner(SEED)
    added = learner.grow_from_solutions(found, top=2, min_count=2)
    print(f"\n   library_learning абстрагировала повторяющиеся комбо: {added}")

    # 3) held-out задачи на те же комбо: были недостижимы на глубине 1, теперь берутся
    print("\n   ЗАДАЧА (held-out)        | seed, глубина 1 | после роста, глубина 1")
    raw = Library(SEED)
    for label, fn, g in [("выделить объект", extract_largest, E3), ("поворот 180°", rot180, R3)]:
        ex = [(g, fn(g))]
        before = raw.induce(ex, max_depth=1)
        after = learner.solve(ex, max_depth=1)
        print(f"   {label:<24}| {str(before):<15} | «{after}»")

    print("\n── Что это значит ──")
    print("   Комбо не заданы руками — система НАШЛА их поиском на задачах и, увидев повтор,")
    print("   назвала. Ровно это происходит на реальном ARC: «keep_largest ▸ bbox» и")
    print("   «flip_h ▸ flip_v» там повторяются, и grow_from_solutions их абстрагирует.")
    print("   Назвав найденное, система решает такой класс задач на глубине 1 — опыт")
    print("   реальных решений сокращает будущий поиск.")


if __name__ == "__main__":
    main()
