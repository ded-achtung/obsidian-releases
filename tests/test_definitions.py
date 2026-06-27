"""Тесты роста языка из чтения: определения новых операций через известные."""

from __future__ import annotations

from thinking_system.language.definitions import DefinitionReader, parse_definition

TEXTBOOK = """
Пример: удвой [1,2,3] = [2,4,6]
Пример: удвой [5,1] = [10,2]
Пример: сумма [1,2,3] = 6
Пример: сумма [4,5] = 9
Пример: разверни [1,2,3] = [3,2,1]
Пример: разверни [4,5] = [5,4]
обработай это сначала удвой потом сумма
итог это сначала разверни потом сумма
Задача: обработай [1,2,3]
Задача: итог [10,20,30]
"""


def test_parse_definition() -> None:
    r = DefinitionReader()
    r.lex.learn("удвой", [([1, 2, 3], [2, 4, 6]), ([5, 1], [10, 2])])
    r.lex.learn("сумма", [([1, 2, 3], 6), ([4, 5], 9)])
    assert parse_definition("обработай это сначала удвой потом сумма", r.lex) == ("обработай", ["удвой", "сумма"])
    assert parse_definition("сократ это человек", r.lex) is None      # это факт, не определение операций


def test_learns_definitions_from_text() -> None:
    r = DefinitionReader()
    stats = r.study(TEXTBOOK)
    assert stats["определено"] == 2
    assert r.definitions["обработай"] == ["удвой", "сумма"]
    assert str(r.lex.words["обработай"]) == "each*2 ▸ sum"            # композиция усвоена


def test_solves_with_defined_operations() -> None:
    r = DefinitionReader()
    r.study(TEXTBOOK)
    assert r.solve("обработай [1,2,3]")["answer"] == 12               # each*2 ▸ sum
    assert r.solve("итог [10,20,30]")["answer"] == 60                 # reverse ▸ sum


def test_definition_with_unknown_op_is_ignored() -> None:
    r = DefinitionReader()
    r.lex.learn("удвой", [([1, 2, 3], [2, 4, 6]), ([5, 1], [10, 2])])
    assert parse_definition("штука это сначала удвой потом несуществующее", r.lex) is None  # <2 известных операций
