#!/usr/bin/env python3
"""Рост библиотеки над ПОЛНЫМ перцептивным seed: система сама именует частые комбо.

Раньше мы добавили слои примитивов руками. Теперь — следующий уровень: пусть
library_learning сама находит частые КОМБИНАЦИИ примитивов в решённых задачах и
абстрагирует их в новые именованные операции. Ключевой факт: комбо, которые она
выделяет здесь («bbox ▸ fractal», «trim_border ▸ mirror_quad»), — те же, что
реально встретились на ARC (задачу 8f2ea7aa система взяла именно как «bbox ▸ fractal»).
Назвав комбо, система решает такие задачи на МЕНЬШЕЙ глубине (depth 2 → depth 1).

Запуск: python run_grid_library.py
"""

from __future__ import annotations

from thinking_system.reasoning.grid_seed import full_grid_seed
from thinking_system.reasoning.perception import bounding_box
from thinking_system.reasoning.structural import replicate_by_self
from thinking_system.reasoning.objects import trim_border, mirror_quad
from thinking_system.reasoning.induction import Library
from thinking_system.reasoning.library_learning import LibraryLearner


def pad0(core):
    """Окружить ядро рамкой нулей (чтобы bbox был нетривиален)."""
    cols = len(core[0])
    z = tuple([0] * (cols + 2))
    return (z,) + tuple(tuple([0] + list(r) + [0]) for r in core) + (z,)


def frame(core, c):
    """Окружить ядро сплошной рамкой цвета c (чтобы trim_border был нужен)."""
    cols = len(core[0])
    top = tuple([c] * (cols + 2))
    return (top,) + tuple(tuple([c] + list(r) + [c]) for r in core) + (top,)


def combo1(g): return replicate_by_self(bounding_box(g))      # bbox ▸ fractal
def combo2(g): return mirror_quad(trim_border(g))             # trim_border ▸ mirror_quad

CORES = [((1, 2), (3, 4)), ((5, 0), (6, 7)), ((2, 3), (3, 2)), ((8, 1), (1, 8))]


def main():
    print("▶ Рост библиотеки над полным перцептивным seed: имена для частых комбо\n")
    seed = full_grid_seed()
    # задачи: по две на каждое комбо (чтобы оно встретилось ≥2 раз → абстрагируется)
    tasks = []
    for core in CORES[:2]:
        g = pad0(core); tasks.append([(g, combo1(g))])
    for core in CORES[:2]:
        g = frame(core, 9); tasks.append([(g, combo2(g))])

    learner = LibraryLearner(seed)
    print(f"   стартовый seed: {len(seed)} примитивов (геометрия+перцепция+структура+объекты)")
    hist = learner.learn(tasks, rounds=2, max_depth=2, abstractions_per_round=1)
    for h in hist:
        print(f"   раунд {h['round']}: решено {h['solved']}/4, библиотека {h['library']}, "
              f"абстракции={h['abstractions']}")
    print(f"   итог абстракций: {list(learner.lib.abstractions)}")

    # held-out задачи: те же комбо на НОВЫХ ядрах — теперь решаются на глубине 1
    print("\n   ЗАДАЧА (held-out)        | seed, глубина 1 | после роста, глубина 1")
    raw = Library(seed)
    for label, fn, core in [("bbox ▸ fractal", combo1, CORES[2]),
                            ("trim ▸ mirror_quad", combo2, CORES[3])]:
        g = pad0(core) if fn is combo1 else frame(core, 9)
        ex = [(g, fn(g))]
        before = raw.induce(ex, max_depth=1)
        after = learner.solve(ex, max_depth=1)
        print(f"   {label:<24}| {str(before):<15} | «{after}»")

    print("\n── Что это значит ──")
    print("   Система не получила эти комбо в руки — она ВЫДЕЛИЛА их из решённых задач и")
    print("   назвала. Те же комбо встречаются на реальном ARC (8f2ea7aa = «bbox ▸ fractal»).")
    print("   Назвав их, она берёт целое семейство задач на глубине 1 вместо 2 — поиск")
    print("   сокращается по мере роста опыта. Это и есть путь к общности: не больше ручных")
    print("   примитивов, а самонаращиваемый словарь поверх них.")


if __name__ == "__main__":
    main()
