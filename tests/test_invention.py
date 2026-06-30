"""Тесты ИЗОБРЕТЕНИЯ операций из наблюдений: новое из регулярности, не из воздуха."""

from __future__ import annotations

from thinking_system.reasoning.invention import (invent_primitive, invent_conditional,
                                                 invent_structural, invent_multibranch,
                                                 invent_window, invent_grid, invent)
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


def test_invents_structural_rotate() -> None:
    # Длино-относительный циклический сдвиг (не значение, а ПОЗИЦИЯ), на ≥2 длинах.
    p = invent_structural([([1, 2, 3], [2, 3, 1]), ([4, 5, 6, 7], [5, 6, 7, 4])])
    assert p is not None and p.fn([9, 8]) == [8, 9]


def test_invents_structural_repeat_period() -> None:
    p = invent_structural([([1, 2], [1, 2, 1, 2]), ([3, 4, 5], [3, 4, 5, 3, 4, 5])])  # повтор×2
    assert p is not None and p.fn([7]) == [7, 7]


def test_invents_structural_stride() -> None:
    p = invent_structural([([1, 2, 3, 4], [1, 3]), ([5, 6, 7, 8, 9, 10], [5, 7, 9])])  # каждый 2-й
    assert p is not None and p.fn([0, 1, 2, 3, 4]) == [0, 2, 4]


def test_structural_needs_two_lengths_to_corroborate() -> None:
    # Параметрическое правило на ОДНОЙ длине не подтверждено → честное None.
    assert invent_structural([([1, 2, 3], [2, 3, 1]), ([4, 5, 6], [5, 6, 4])]) is None
    assert invent_structural([([1, 2, 3], [9, 9, 9]), ([4, 5], [1, 2])]) is None       # вне шаблонов


def test_invents_multibranch_sign() -> None:
    # МНОГОВЕТОЧНОЕ (3 ветки): знак числа, выведенный из наблюдений.
    p = invent_multibranch([(-3, -1), (-2, -1), (5, 1), (7, 1), (0, 0)])
    assert p is not None and p.fn(-9) == -1 and p.fn(4) == 1 and p.fn(0) == 0


def test_multibranch_rejects_arbitrary_data() -> None:
    assert invent_multibranch([(1, 5), (2, 9), (3, 2), (4, 7)]) is None


def test_invents_window_differences() -> None:
    # Свёрточное: out[i] = in[i+1]-in[i], длино-относительно, на ≥2 длинах.
    p = invent_window([([1, 3, 6, 10], [2, 3, 4]), ([2, 5, 9], [3, 4])])
    assert p is not None and p.fn([10, 12, 20]) == [2, 8]


def test_invents_window_prefix_sum() -> None:
    p = invent_window([([1, 2, 3], [1, 3, 6]), ([4, 5], [4, 9])])
    assert p is not None and p.fn([2, 2, 2, 2]) == [2, 4, 6, 8]


def test_window_single_length_not_corroborated() -> None:
    assert invent_window([([1, 2, 3], [3, 5]), ([4, 5, 6], [9, 11])]) is None


def test_invents_grid_transform_from_observation() -> None:
    # Преобразование сетки подобрано поиском по grid-примитивам под наблюдаемые пары.
    p = invent_grid([([[1, 2], [3, 4]], [[2, 1], [4, 3]]),
                     ([[5, 6], [7, 8]], [[6, 5], [8, 7]])])      # flip_h
    assert p is not None and p.fn(((9, 0), (1, 2))) == ((0, 9), (2, 1))


def test_grid_rejects_arbitrary() -> None:
    assert invent_grid([([[1, 2], [3, 4]], [[9, 9], [9, 9]]),
                        ([[1, 1], [1, 1]], [[2, 2], [2, 2]])]) is None


def test_nested_rule_structural_branch_prefers_flat() -> None:
    # ВЛОЖЕННОЕ/композиционное: структурная операция в ветке под выученным предикатом;
    # выбирается ПЛОСКАЯ гипотеза по длине (Оккам), а не запутанная по чётности суммы.
    p = invent([([1, 2], [1, 2]), ([2, 1], [2, 1]), ([1, 2, 3], [3, 2, 1]),
                ([3, 2, 1], [1, 2, 3]), ([2, 4, 6, 8], [8, 6, 4, 2]), ([1, 3], [1, 3])])
    assert p is not None
    assert p.fn([5, 6, 7]) == [7, 6, 5] and p.fn([8, 9]) == [8, 9]


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
