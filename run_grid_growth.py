#!/usr/bin/env python3
"""Тот же ростовой механизм — над СЕТКАМИ (домен ARC), а не под один бенчмарк.

Общий seed грид-операций (симметрии сетки + перекраска) — не «решатель задачи X».
На нём та же самонаращивающаяся библиотека растит ГРИД-абстракции из решённых задач:
при фиксированной глубине поиска сложные грид-задачи становятся достижимы после
того, как система сама поднимет переиспользуемый кусок в язык.

Запуск: python run_grid_growth.py
"""

from __future__ import annotations

from thinking_system.reasoning.grids import grid_primitives, rot90, recolor1, transpose, flip_h, to_grid
from thinking_system.reasoning.library_learning import LibraryLearner

g1 = to_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
g2 = to_grid([[1, 0], [0, 1], [2, 2]])
g3 = to_grid([[3, 1], [4, 1], [5, 9]])


def task(f, grids):
    return [(g, f(g)) for g in grids]


def c2(g):
    return recolor1(recolor1(g))                # перекрасить дважды (+2)


def main():
    print("▶ Рост библиотеки над СЕТКАМИ (домен ARC), тот же механизм\n")
    tasks = [
        ("перекрасить ×2 (+2)",          task(c2, [g1, g2])),
        ("перекрасить ×2 (+2)",          task(c2, [g3, g1])),
        ("транспонировать после отражения", task(lambda g: transpose(flip_h(g)), [g1, g3])),
        ("повернуть, потом перекрасить ×2  [глубина 3]", task(lambda g: c2(rot90(g)), [g1, g2, g3])),
    ]
    learner = LibraryLearner(seed=grid_primitives())
    print(f"общий seed грид-операций: {[p.name for p in learner.lib.prims]}")
    print("глубина поиска ФИКСИРОВАНА = 2; поток из 4 грид-задач (одна требует 3 шага)\n")

    for h in learner.learn([ex for _, ex in tasks], rounds=3, max_depth=2, abstractions_per_round=1):
        ab = ", ".join(h["abstractions"]) or "—"
        print(f"   раунд {h['round']}: решено {h['solved']}/4 | библиотека {h['library']} | выучено: {ab}")

    print(f"\n   выученная ГРИД-абстракция: {learner.lib.abstractions}")
    print("\n── Что это значит ──")
    print("   Самонаращивание языка работает и над сетками — тем же кодом, что над числами и")
    print("   списками: общий seed симметрий + перекраски, а грид-абстракции система растит сама.")
    print("   Это путь к ARC через РОСТ языка, а не ручной DSL под бенчмарк. На настоящем ARC")
    print("   тот же приём строил бы примитивы сеток из решённых задач (замер на корпусе ARC")
    print("   здесь не проводится — датасета ARC в репозитории нет).")


if __name__ == "__main__":
    main()
