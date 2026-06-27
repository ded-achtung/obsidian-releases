"""Тесты самонаращивающейся библиотеки: wake/sleep, рост, разблокировка сложных задач."""

from __future__ import annotations

from thinking_system.reasoning.library_learning import LibraryLearner

TASKS = [
    [(2, 9), (3, 16), (5, 36)],          # (x+1)^2
    [(1, 4), (4, 25), (2, 9)],           # (x+1)^2
    [(3, 7), (5, 11), (10, 21)],         # 2x+1
    [(2, 18), (3, 32), (4, 50)],         # ((x+1)^2)*2   глубина 3
    [(2, 10), (3, 17), (4, 26)],         # (x+1)^2+1      глубина 3
    [([1, 2, 3], 12), ([5], 10), ([2, 2], 8)],   # sum(2*xs)
    [([1, 1, 1], 6), ([4], 8)],                  # sum(2*xs)
    [([1, 2, 3], 144), ([5], 100)],              # sum(2*xs)^2    глубина 3
]


def test_wake_solves_shallow_tasks_only_at_first() -> None:
    L = LibraryLearner()
    sols = L.wake(TASKS, max_depth=2)
    assert len(sols) == 5                                   # глубина-3 задачи пока недостижимы
    assert 3 not in sols and 4 not in sols and 7 not in sols


def test_sleep_abstracts_recurring_subprogram() -> None:
    L = LibraryLearner()
    added = L.sleep(L.wake(TASKS, max_depth=2), top=1)
    assert added and "∘" in added[0]                       # подняла переиспользуемый кусок в примитив
    assert added[0] in {p.name for p in L.lib.prims}


def test_library_growth_unlocks_deeper_tasks() -> None:
    L = LibraryLearner()
    hist = L.learn(TASKS, rounds=4, max_depth=2, abstractions_per_round=1)
    assert hist[0]["solved"] == 5                           # старт: только короткие
    assert hist[-1]["solved"] == 8                          # после самонаращивания — все, в т.ч. глубина 3
    assert len(L.lib.abstractions) >= 2                     # выучила ≥2 абстракции сама
    # та же глубина поиска, но язык вырос → решается больше
    assert hist[-1]["solved"] > hist[0]["solved"]


def test_grows_across_two_domains() -> None:
    L = LibraryLearner()
    L.learn(TASKS, rounds=4, max_depth=2)
    abstr = set(L.lib.abstractions)
    assert any("square" in a for a in abstr)               # абстракция из домена чисел
    assert any("sum" in a for a in abstr)                  # и из домена списков — не привязана к одному
