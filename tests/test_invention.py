"""Тесты ИЗОБРЕТЕНИЯ операций из наблюдений: новое из регулярности, не из воздуха."""

from __future__ import annotations

from thinking_system.reasoning.invention import invent_primitive
from thinking_system.system import ThinkingSystem


def test_invents_affine_from_observations() -> None:
    p = invent_primitive([(2, 11), (4, 17), (5, 20)])      # 3x+5
    assert p is not None and p.fn(10) == 35 and p.fn(0) == 5


def test_invents_quadratic_not_composable_from_seed() -> None:
    p = invent_primitive([(2, 6), (3, 12), (4, 20)])       # x²+x
    assert p is not None and p.fn(5) == 30 and p.fn(6) == 42


def test_invents_elementwise_over_lists() -> None:
    p = invent_primitive([([1, 2], [5, 8]), ([0, 4], [2, 14])])  # each 3e+2
    assert p is not None and p.fn([10]) == [32]


def test_two_points_underdetermine_quadratic() -> None:
    # Из 2 наблюдений квадратичную не определить — подойдёт лишь аффинная (честно).
    p = invent_primitive([(2, 6), (3, 12)])                # x²+x даёт 6,12 — но и 6x-6 даёт
    assert p is not None and p.fn(2) == 6 and p.fn(3) == 12
    assert p.fn(4) != 20                                    # на новом входе расходится с x²+x


def test_identity_and_nonpattern_return_none() -> None:
    assert invent_primitive([(1, 1), (2, 2), (3, 3)]) is None      # тождество — не операция
    assert invent_primitive([(0, 1), (1, 2), (2, 4), (3, 8)]) is None  # 2^x — вне шаблонов


def test_system_solve_falls_back_to_invention_and_grows_library() -> None:
    ts = ThinkingSystem()
    assert ts.solve([(2, 6), (3, 12), (4, 20)], invent=False) is None   # композиции seed нет
    prog = ts.solve([(2, 6), (3, 12), (4, 20)])                          # изобретает из данных
    assert prog is not None and prog(5) == 30
    assert "x²+x" in ts.library.abstractions                             # добавлено в библиотеку


def test_name_skill_can_name_an_invented_operation_for_language() -> None:
    ts = ThinkingSystem()
    assert ts.name_skill("парабола", [(2, 6), (3, 12), (4, 20)])         # x²+x из наблюдений
    assert ts.understand("парабола 5")["answer"] == 30                   # язык владеет новой операцией
