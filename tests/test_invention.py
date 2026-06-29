"""Тесты ИЗОБРЕТЕНИЯ операций из наблюдений: новое из регулярности, не из воздуха."""

from __future__ import annotations

from thinking_system.reasoning.invention import invent_primitive, invent_conditional
from thinking_system.system import ThinkingSystem


def test_invents_affine_from_observations() -> None:
    p = invent_primitive([(2, 11), (4, 17), (5, 20)])      # 3x+5 (3 разных x — подтверждено)
    assert p is not None and p.fn(10) == 35 and p.fn(0) == 5


def test_invents_quadratic_not_composable_from_seed() -> None:
    p = invent_primitive([(2, 6), (3, 12), (4, 20), (5, 30)])   # x²+x (4 точки — подтверждено)
    assert p is not None and p.fn(6) == 42 and p.fn(7) == 56


def test_invents_elementwise_over_lists() -> None:
    p = invent_primitive([([1, 2], [5, 8]), ([0, 4], [2, 14])])  # each 3e+2 (4 пары)
    assert p is not None and p.fn([10]) == [32]


def test_too_few_points_invent_nothing() -> None:
    # Честная граница: из 2 точек — ничего (любые 2 на какой-то прямой), 3 для квадратичной мало.
    assert invent_primitive([(2, 6), (3, 12)]) is None
    assert invent_primitive([(2, 6), (3, 12), (4, 20)]) is None     # 3 точки квадратичную не подтверждают


def test_identity_and_nonpattern_return_none() -> None:
    assert invent_primitive([(1, 1), (2, 2), (3, 3)]) is None      # тождество — не операция
    assert invent_primitive([(0, 1), (1, 2), (2, 4), (3, 8)]) is None  # 2^x — вне шаблонов


def test_invents_conditional_abs_from_observations() -> None:
    # СЛЕДУЮЩИЙ СЛОЙ: ветвящаяся операция (модуль) из наблюдений, обобщает на новый вход.
    p = invent_conditional([(-3, 3), (-2, 2), (-5, 5), (1, 1), (4, 4), (6, 6)])
    assert p is not None and p.fn(-7) == 7 and p.fn(5) == 5


def test_invents_conditional_branch_transform() -> None:
    # «удвоить чётные, нечётные оставить» — условие И преобразование выведены из данных.
    p = invent_conditional([(2, 4), (4, 8), (6, 12), (3, 3), (5, 5), (7, 7)])
    assert p is not None and p.fn(8) == 16 and p.fn(9) == 9


def test_conditional_rejects_arbitrary_data() -> None:
    # 4 произвольные точки нельзя честно объяснить ветвлением (ветки не подтверждены).
    assert invent_conditional([(1, 5), (2, 9), (3, 2), (4, 7)]) is None


def test_system_solve_falls_back_to_invention_and_grows_library() -> None:
    ts = ThinkingSystem()
    ex = [(2, 6), (3, 12), (4, 20), (5, 30)]                             # x²+x
    assert ts.solve(ex, invent=False) is None                           # композиции seed нет
    prog = ts.solve(ex)                                                  # изобретает из данных
    assert prog is not None and prog(6) == 42
    assert "x²+x" in ts.library.abstractions                            # добавлено в библиотеку


def test_name_skill_can_name_an_invented_operation_for_language() -> None:
    ts = ThinkingSystem()
    assert ts.name_skill("парабола", [(2, 6), (3, 12), (4, 20), (5, 30)])   # x²+x из наблюдений
    assert ts.understand("парабола 6")["answer"] == 42                      # язык владеет новой операцией
