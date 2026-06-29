"""Тесты индукции ПРЕДИКАТА из размеченных примеров: условие выучено, не захардкожено."""

from __future__ import annotations

from thinking_system.reasoning.predicates import induce_predicate


def test_induces_parity_predicate() -> None:
    name, fn = induce_predicate([(2, True), (3, False), (8, True), (5, False)])
    assert name == "x чётно"
    assert fn(10) is True and fn(7) is False


def test_induces_integer_threshold() -> None:
    name, fn = induce_predicate([(1, False), (2, False), (5, True), (6, True)])
    assert all(fn(x) == l for x, l in [(1, False), (2, False), (5, True), (6, True)])
    assert fn(10) is True and fn(0) is False        # порог обобщает


def test_induces_list_predicate() -> None:
    name, fn = induce_predicate([([1, 2], True), ([1, -1], False), ([3, 4], True), ([0, 5], False)])
    assert name == "все>0"
    assert fn([2, 3]) is True and fn([2, -1]) is False


def test_single_class_returns_none() -> None:
    assert induce_predicate([(1, True), (2, True)]) is None       # нечего различать


def test_unseparable_returns_none() -> None:
    # Метки, которые ни один шаблон не разделяет точно → честное None.
    assert induce_predicate([(1, True), (2, False), (3, True), (4, False), (1, False)]) is None
