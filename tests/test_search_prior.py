"""Тесты выученного приора поиска: корректность, веса, экономия перебора."""

from __future__ import annotations

from thinking_system.reasoning.induction import default_primitives
from thinking_system.reasoning.library_learning import LibraryLearner
from thinking_system.reasoning.search_prior import usage_prior, best_first_induce, search_cost

TASKS = [
    [(2, 9), (3, 16), (5, 36)], [(1, 4), (4, 25), (2, 9)], [(3, 7), (5, 11), (10, 21)],
    [(2, 18), (3, 32), (4, 50)], [(2, 10), (3, 17), (4, 26)],
    [([1, 2, 3], 12), ([5], 10), ([2, 2], 8)], [([1, 1, 1], 6), ([4], 8)],
    [([1, 2, 3], 144), ([5], 100)],
]


def test_best_first_finds_correct_program() -> None:
    prog, n = best_first_induce([(2, 9), (3, 16)], default_primitives(), None, max_depth=2)
    assert prog is not None and prog(5) == 36 and n >= 1      # (x+1)^2, и счётчик перебора растёт


def test_usage_prior_weights_used_primitives_higher() -> None:
    L = LibraryLearner()
    L.learn(TASKS, rounds=4, max_depth=2)
    prior = usage_prior(list(L.wake(TASKS, max_depth=2).values()), L.lib.prims)
    assert prior["+1∘square"] > prior["neg"]                  # использованная абстракция важнее неиспользуемого


def test_prior_reduces_search_cost() -> None:
    L = LibraryLearner()
    L.learn(TASKS, rounds=4, max_depth=2)
    prior = usage_prior(list(L.wake(TASKS, max_depth=2).values()), L.lib.prims)
    s_u, c_u = search_cost(TASKS, L.lib.prims, None, max_depth=3)
    s_p, c_p = search_cost(TASKS, L.lib.prims, prior, max_depth=3)
    assert s_u == s_p == 8                                    # обе версии решают все
    assert c_p < c_u                                          # приор экономит перебор


def test_abstraction_extends_reach_of_guided_search() -> None:
    """Язык среднего уровня: абстракция делает глубину 4 достижимой на глубине 3.

    У абстракции НЕТ биграммных счётчиков в приоре — она достижима благодаря
    сглаживанию (урок дополнения 12 AUDIT): опыт не запирает язык в старых именах.
    """
    from thinking_system.reasoning.deep_search import guided_induce
    from thinking_system.reasoning.induction import Primitive, Program

    exs = [(-1, 27), (0, 36), (1, 45), (2, 54)]              # 9·(x+4): глубина 4 в базовых именах
    seed = default_primitives()
    bigram = {("^", "+1"): 5, ("+1", "square"): 5}           # счётчики только у базовых имён
    none_prog, _ = guided_induce(exs, seed, bigram, max_depth=3, budget=30000)
    assert none_prog is None                                  # честно: глубины 3 не хватает
    plus3 = next(p for p in seed if p.name == "+3")
    mul3 = next(p for p in seed if p.name == "*3")
    ab = Primitive("+3∘*3", Program([plus3, mul3]).__call__, cost=1.0)
    prog, _ = guided_induce(exs, seed + [ab], bigram, max_depth=3, budget=30000)
    assert prog is not None and prog(10) == 126               # (10+4)·9, найдено через «+3∘*3»
