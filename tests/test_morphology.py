"""Тесты морфологии: стемминг форм и устойчивость понимания к формулировкам."""

from __future__ import annotations

from thinking_system.language.morphology import stem, normalize
from thinking_system.language.study import Textbook
from thinking_system.language.understanding import GroundedLexicon

BOOK = """
Пример: максимум [3,1,2] = 3
Пример: максимум [5,9,2] = 9
Пример: сумма [1,2,3] = 6
Пример: сумма [4,5] = 9
максимум нужен когда просят наибольший
сумма нужна когда просят общий итог
"""


def test_stem_unifies_word_forms() -> None:
    for forms in [["наибольший", "наибольшее", "наибольшие", "наибольшего"],
                  ["сумма", "сумму", "суммой", "суммы"],
                  ["отрицательный", "отрицательные", "отрицательными"],
                  ["длинный", "длинные", "длинному"]]:
        assert len({stem(w) for w in forms}) == 1          # все формы → одна основа
    assert len({stem(w) for w in ["сумма", "максимум", "разверни"]}) == 3  # разные слова различимы


def test_lexicon_recognizes_inflected_forms() -> None:
    lex = GroundedLexicon(normalize=stem)
    lex.learn("сложи", [([1, 2, 3], 6), ([4, 5], 9)])       # форма «сложи»
    assert lex.solve("сложить [10, 20]")["answer"] == 30    # другая форма «сложить» — узнал
    assert lex.solve("сложил [1, 1, 1]")["answer"] == 3


def test_study_robust_to_phrasing() -> None:
    plain = Textbook(); plain.study(BOOK)
    rob = Textbook(normalize=stem); rob.study(BOOK)
    # «наибольшее» — иная форма обученной приметы «наибольший»
    assert plain.solve("найди наибольшее [4,2,9,1]")["solved"] is False   # без морфологии не узнал
    assert rob.solve("найди наибольшее [4,2,9,1]")["answer"] == 9         # с морфологией — узнал
    assert rob.solve("посчитай общую сумму [10,20,30]")["answer"] == 60


def test_default_textbook_unchanged() -> None:
    tb = Textbook()                                         # без normalize — прежнее поведение
    tb.study(BOOK)
    assert tb.solve("наибольший [5,9,2]")["answer"] == 9
