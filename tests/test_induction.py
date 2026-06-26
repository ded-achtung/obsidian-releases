"""Тесты few-shot индукции правил: обобщение, Оккам, абстракция, границы языка."""

from __future__ import annotations

from thinking_system.reasoning.induction import default_library, Program


def test_induces_rules_and_generalizes_to_unseen() -> None:
    cases = [
        ([(3, 9), (4, 16)],                         [(7, 49), (10, 100), (11, 121)]),         # x²
        ([(3, 7), (5, 11)],                         [(10, 21), (0, 1), (100, 201)]),          # x·2+1
        ([(2, 9), (3, 16)],                         [(5, 36), (9, 100)]),                     # (x+1)²
        ([([1, 2, 3], [3, 2, 1]), ([4, 5], [5, 4])], [([7, 8, 9], [9, 8, 7]), ([1], [1])]),   # reverse
        ([([1, 2, 3], 12), ([5], 10)],             [([10], 20), ([1, 1, 1, 1], 8)]),         # sum of doubled
    ]
    for ex, test in cases:
        prog = default_library().induce(ex, max_depth=3)
        assert prog is not None
        assert all(prog(i) == o for i, o in test)              # выведенное правило обобщает на новые входы


def test_prefers_shortest_program_occam() -> None:
    assert default_library().induce([(5, 5), (9, 9)]).length == 0   # тождество, если вход=выход
    mx = default_library().induce([([3, 1, 2], 3), ([5, 9, 2], 9)], max_depth=3)
    assert mx is not None and mx.length == 1                  # «max» (1 шаг), а не «sort ▸ last» (2 шага)


def test_abstraction_enables_harder_with_same_search_depth() -> None:
    lib = default_library()
    hard = [(2, 18), (3, 32)]                                 # ((x+1)²)·2 — глубина 3 из базовых
    assert lib.induce(hard, max_depth=2) is None              # при глубине 2 недостижимо
    sub = lib.induce([(2, 9), (3, 16)], max_depth=2)          # выучить (x+1)²
    assert isinstance(sub, Program)
    lib.add_abstraction("inc_sq", sub)
    sol = lib.induce(hard, max_depth=2)                       # теперь достижимо за ту же глубину
    assert sol is not None and sol(10) == ((10 + 1) ** 2) * 2  # и обобщает на новый вход


def test_unsolvable_without_right_primitive() -> None:
    assert default_library().induce([(2, 8), (3, 27)], max_depth=3) is None  # нет куба в языке — честно None
