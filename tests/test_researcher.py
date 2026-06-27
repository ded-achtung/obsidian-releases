"""Тесты исследовательского круга: вывод нового, проверка гипотез, пробелы, интеграция."""

from __future__ import annotations

from thinking_system.language.reader import LessonReader
from thinking_system.agent.researcher import Researcher

LESSON = """
Пример: разверни [1,2,3] = [3,2,1]
Пример: разверни [4,5] = [5,4]
Пример: удвой [1,2,3] = [2,4,6]
Пример: удвой [5,1] = [10,2]
Сократ это человек.
человек это млекопитающее.
млекопитающее это животное.
каждый человек смертный.
каждое животное дышит.
дождь вызывает мокрый.
спринклер вызывает мокрый.
мокрый вызывает скользкий.
дождь вызывает холодный.
"""


def _sci():
    r = LessonReader()
    r.read(LESSON)
    return Researcher(r)


def test_derives_unstated_facts() -> None:
    sci = _sci()
    derived = sci.derive_new_facts()
    assert ("is_a", "сократ", "животное") in derived          # 3-хоповый вывод, не сказан
    assert ("свойство", "сократ", "смертный") in derived      # силлогизм
    assert ("свойство", "сократ", "дышит") in derived         # наследование через цепочку


def test_discovers_operation_properties_by_execution() -> None:
    sci = _sci()
    props = sci.operation_properties([[1, 2, 3], [4, 5, 6]])
    assert "инволюция" in props["разверни"]                   # развернуть дважды = исходное (проверено)
    assert props["удвой"] == "не инволюция"


def test_analyzes_causality_and_finds_gaps() -> None:
    sci = _sci()
    cf = sci.causal_findings()
    assert ("спринклер", "скользкий") in cf["causes"]         # причинный путь
    assert ("мокрый", "холодный") in cf["correlated_not_causal"]  # связаны, но не причина
    assert "дождь" in cf["roots"]                             # явление без объяснённой причины
    assert any("дождь" in q for q in sci.open_questions())


def test_closing_the_loop_integrates_and_reopens() -> None:
    sci = _sci()
    assert sci.learn_more("облака вызывают дождь") == "причина"
    assert sci.r.ask_causal("облака вызывают скользкий?") == "да"   # новая выведенная цепочка
    assert any("облака" in q for q in sci.open_questions())         # поднят следующий вопрос
    assert not any("дождь" in q for q in sci.open_questions())      # прежний пробел закрыт
