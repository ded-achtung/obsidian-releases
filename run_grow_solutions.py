#!/usr/bin/env python3
"""Рост библиотеки из найденных решений: система сама именует комбо, что нашла поиском.

ЧЕСТНО О СТАТУСЕ: это СИНТЕТИЧЕСКАЯ иллюстрация механизма, НЕ прогон на реальном ARC.
Датасета ARC в репозитории нет — поэтому конкретные числа «на настоящем ARC» (сколько
задач решено, какие комбо и сколько раз повторились) здесь НЕ воспроизводимы и не
заявляются. Демо показывает только сам механизм grow_from_solutions на рукотворных
сетках: best_first_induce находит решения, повторяющиеся комбо абстрагируются в
именованные операции:
    «keep_largest ▸ bbox»  — выделить крупнейший объект и обрезать до него
    «flip_h ▸ flip_v»      — поворот на 180°

И ВАЖНО (честная планка): рост даёт ЭФФЕКТИВНОСТЬ (эти комбо теперь берутся на глубине
1 вместо 2), а НЕ новый охват — сырой seed и так решает их на глубине 2. Сравнение
«до = None на глубине 1» само по себе тривиально: 2-шаговую программу на глубине 1
найти нельзя. Поэтому ниже показаны обе планки: глубина 1 и глубина 2 сырого seed.

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
    print("▶ Рост библиотеки из найденных поиском решений (синтетическая иллюстрация механизма)\n")
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

    # 3) held-out задачи на те же комбо — с ЧЕСТНЫМ сравнением планок (глубина 1 и 2 сырого seed)
    print("\n   ЗАДАЧА (held-out)        | seed гл.1 | seed гл.2 | после роста гл.1")
    raw = Library(SEED)
    for label, fn, g in [("выделить объект", extract_largest, E3), ("поворот 180°", rot180, R3)]:
        ex = [(g, fn(g))]
        raw1 = raw.induce(ex, max_depth=1)
        raw2 = raw.induce(ex, max_depth=2)
        after = learner.solve(ex, max_depth=1)
        assert after is not None and after(g) == fn(g)        # held-out корректность, не только «нашлась программа»
        print(f"   {label:<24}| {str(raw1):<9} | {str(raw2):<9} | «{after}»")

    print("\n── Что это значит (честно) ──")
    print("   Комбо не заданы руками — best_first_induce НАШЁЛ их поиском, и при повторе")
    print("   grow_from_solutions их назвал. Но это выигрыш в ЭФФЕКТИВНОСТИ: семейство теперь")
    print("   берётся на глубине 1, тогда как сырой seed решает его на глубине 2 (колонка выше).")
    print("   Охват НЕ вырос. И это лишь иллюстрация механизма — не результат на реальном ARC,")
    print("   которого в репозитории нет.")


if __name__ == "__main__":
    main()
