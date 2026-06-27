"""Тесты учёбы по учебнику: операции из примеров, применимость из теории, решение задач."""

from __future__ import annotations

from thinking_system.language.study import Textbook

TEXTBOOK = """
# учебник
Пример: разверни [1,2,3] = [3,2,1]
Пример: разверни [4,5] = [5,4]
Пример: удвой [1,2,3] = [2,4,6]
Пример: удвой [5,1] = [10,2]
Пример: сумма [1,2,3] = 6
Пример: сумма [4,5] = 9
Пример: максимум [3,1,2] = 3
Пример: максимум [5,9,2] = 9
сумма нужна когда просят итог или общее
максимум нужен когда просят наибольшее
разверни применяй когда нужно перевернуть порядок
удвой применяй когда нужно увеличить вдвое
Задача: найди общее [10,20,30]
Задача: наибольшее [4,2,9,1]
"""


def _tb():
    tb = Textbook()
    tb.study(TEXTBOOK)
    return tb


def test_learns_operations_and_applicability() -> None:
    tb = _tb()
    assert str(tb.lex.words["сумма"]) == "sum"            # ЧТО делает — из примеров
    assert tb.applicability["общее"] == "сумма"           # КОГДА применять — из теории
    assert tb.applicability["наибольшее"] == "максимум"
    assert len(tb.exercises) == 2


def test_solves_by_selecting_rule_from_meaning() -> None:
    tb = _tb()
    assert tb.solve("найди общее [10,20,30]") == {"solved": True, "answer": 60, "used": ["сумма"]}
    assert tb.solve("наибольшее [4,2,9,1]")["answer"] == 9         # выбрал максимум по примете
    assert tb.solve("перевернуть порядок [1,2,3,4]")["answer"] == [4, 3, 2, 1]


def test_composes_rule_by_name_and_cue() -> None:
    tb = _tb()
    r = tb.solve("удвой и общее [1,2,3]")                          # удвой (имя) + общее (примета→сумма)
    assert r["answer"] == 12 and r["used"] == ["удвой", "сумма"]


def test_honest_when_no_rule_applies() -> None:
    tb = _tb()
    assert tb.solve("посчитай среднее [1,2,3]")["solved"] is False  # «среднее» не объяснено в учебнике
