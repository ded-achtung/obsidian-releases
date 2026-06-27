"""Тесты понимания задачи: смысл из показа, обобщение, композиция, честность."""

from __future__ import annotations

from thinking_system.language.understanding import GroundedLexicon, tokenize, extract_arg


def _lex():
    lex = GroundedLexicon()
    lex.learn("разверни", [([1, 2, 3], [3, 2, 1]), ([4, 5], [5, 4])])
    lex.learn("сложи", [([1, 2, 3], 6), ([4, 5], 9)])
    lex.learn("удвой", [([1, 2, 3], [2, 4, 6]), ([5, 1], [10, 2])])
    lex.learn("максимум", [([3, 1, 2], 3), ([5, 9, 2], 9)])
    return lex


def test_parsing_helpers() -> None:
    assert tokenize("Разверни список [1,2,3]") == ["разверни", "список"]
    assert extract_arg("сложи [10, 20, 30]") == [10, 20, 30]
    assert extract_arg("квадрат 7") == 7


def test_learns_meaning_from_use_and_generalizes() -> None:
    lex = _lex()
    assert lex.solve("разверни [7,8,9]")["answer"] == [9, 8, 7]      # смысл слова обобщён на новый вход
    assert lex.solve("максимум [8,1]")["answer"] == 8
    assert lex.solve("сложи [10,20,30]")["answer"] == 60


def test_composes_known_words_into_new_sentences() -> None:
    lex = _lex()
    r = lex.solve("удвой и сложи [1,2,3]")                          # невиданная комбинация
    assert r["understood"] and r["answer"] == 12                    # each*2 ▸ sum = 12 (системность)
    assert lex.solve("разверни и сложи [4,5,6]")["answer"] == 15


def test_honest_about_unknown_words() -> None:
    lex = _lex()
    r = lex.solve("заверни [5,6]")
    assert r["understood"] is False and "заверни" in r["unknown"]   # признаёт незнание, не выдумывает


def test_learns_new_word_from_one_demonstration() -> None:
    lex = _lex()
    w = lex.learn_from_demo("переверни [1,2,3]", [1, 2, 3], [3, 2, 1])
    assert w == "переверни"
    assert lex.solve("переверни [9,9,1]")["answer"] == [1, 9, 9]    # выучил из показа и применил к новому
