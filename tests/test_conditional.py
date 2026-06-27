"""Тесты условных правил: операции из примеров, выбор правила по свойству данных."""

from __future__ import annotations

from thinking_system.language.conditional import ConditionalReader

TEXTBOOK = """
Пример: очисти [1, -2, 3] = [1, 3]
Пример: очисти [-5, 4, -1, 2] = [4, 2]
Пример: сумма [1, 2, 3] = 6
Пример: сумма [4, 5] = 9
Пример: максимум [3, 1, 2] = 3
Пример: максимум [5, 9, 2] = 9
если в списке есть отрицательные, очисти
если список длинный, максимум
если список короткий, сумма
Задача: обработай [1, -2, 3]
Задача: обработай [3, 1, 9, 2, 7]
Задача: обработай [4, 5]
"""


def _cr():
    cr = ConditionalReader()
    cr.study(TEXTBOOK)
    return cr


def test_learns_operations_and_conditional_rules() -> None:
    cr = _cr()
    assert str(cr.lex.words["очисти"]) == "positives"
    assert ("отрицательные", "очисти") in cr.rules
    assert len(cr.rules) == 3


def test_selects_rule_by_data_property() -> None:
    cr = _cr()
    assert cr.solve("обработай [1, -2, 3]")["answer"] == [1, 3]     # есть отрицательные → очисти
    assert cr.solve("обработай [3, 1, 9, 2, 7]")["answer"] == 9     # длинный → максимум
    assert cr.solve("обработай [4, 5]")["answer"] == 9              # короткий → сумма


def test_same_task_different_data_different_rule() -> None:
    cr = _cr()
    a = cr.solve("обработай [-1, 5, 6, 7, 8]")                      # есть отрицательные (важнее длины)
    b = cr.solve("обработай [5, 6, 7, 8]")                          # длинный, без отрицательных
    assert a["rule"].startswith("если отрицательные") and b["rule"].startswith("если длинный")


def test_honest_when_no_condition_matches() -> None:
    cr = _cr()
    assert cr.solve("обработай []")["solved"] is False             # пустой — ни одно условие не подходит
