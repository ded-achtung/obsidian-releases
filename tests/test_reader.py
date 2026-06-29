"""Тесты чтения смысла из текста: извлечение по типам, понимание и использование."""

from __future__ import annotations

from thinking_system.language.reader import LessonReader, parse_demonstration

LESSON = """
# урок
Развернуть список переворачивает порядок.
Пример: разверни [1, 2, 3] = [3, 2, 1]
Ещё пример: разверни [4, 5] = [5, 4]
Пример: удвой [1, 2, 3] = [2, 4, 6]
Пример: удвой [5, 1] = [10, 2]
Пример: сумма [1, 2, 3] = 6
Пример: сумма [4, 5] = 9
Сократ это человек.
человек это млекопитающее.
млекопитающее это животное.
каждый человек смертный.
дождь вызывает мокрый.
мокрый вызывает скользкий.
просто повествование без структуры
"""


def test_parse_demonstration() -> None:
    assert parse_demonstration("Пример: разверни [1, 2, 3] = [3, 2, 1]") == ("разверни", [1, 2, 3], [3, 2, 1])
    assert parse_demonstration("сумма [4, 5] = 9") == ("сумма", [4, 5], 9)
    assert parse_demonstration("просто текст без примера") is None


def test_parse_demonstration_skips_filler_words() -> None:
    # Имя операции, а не служебное слово-филлер рядом с аргументом (англ. «on»/«apply»).
    assert parse_demonstration("apply reverse on [1, 2] = [2, 1]") == ("reverse", [1, 2], [2, 1])


def test_malformed_lines_are_skipped_not_crash() -> None:
    # Неполные причинные/фактовые строки честно пропускаются, без падения.
    r = LessonReader()
    stats = r.read("дождь вызывает\nэто\nA это\nкаждый\n")
    assert stats["причины"] == 0 and stats["факты"] == 0
    assert stats["пропущено"] == 4


def test_reads_mixed_material_and_learns() -> None:
    r = LessonReader()
    stats = r.read(LESSON)
    assert stats["слова"] == 3                              # разверни, удвой, сумма выучены из текста
    assert r.lex.words["разверни"].__str__() == "reverse"
    assert stats["факты"] >= 4 and stats["причины"] == 2
    assert stats["пропущено"] >= 1                          # проза честно пропущена


def test_uses_what_it_read() -> None:
    r = LessonReader()
    r.read(LESSON)
    assert r.solve("разверни [9,8,7]")["answer"] == [7, 8, 9]        # операция из текста на новом входе
    assert r.solve("удвой и сумма [1,2,3]")["answer"] == 12          # композиция двух слов из текста
    assert r.ask_fact("Сократ это животное?") == "да"               # дедукция по фактам из текста
    assert r.ask_fact("Сократ смертный?") == "да"
    assert r.ask_causal("если включить мокрый, будет скользкий?") == "да"   # причина из текста
