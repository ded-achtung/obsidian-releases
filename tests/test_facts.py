"""Тесты понимания утверждений: разбор в факты, ответ дедукцией, честность."""

from __future__ import annotations

import pytest

from thinking_system.language.facts import FactReader


def _reader():
    fr = FactReader()
    for s in ["Сократ это человек", "человек это млекопитающее", "млекопитающее это животное",
              "каждый человек смертный", "каждое животное дышит"]:
        fr.tell(s)
    return fr


def test_parses_statements_into_facts() -> None:
    fr = FactReader()
    assert fr.tell("Сократ это человек") == ("is_a", "сократ", "человек")
    assert fr.tell("каждый человек смертный") == ("свойство", "человек", "смертный")


def test_incomplete_statement_raises_not_crashes() -> None:
    # «это» с краю / пустое / неполное универсальное → ValueError, не IndexError.
    fr = FactReader()
    for bad in ["A это", "это человек", "это", "каждый"]:
        with pytest.raises(ValueError):
            fr.tell(bad)


def test_answers_by_multi_hop_deduction() -> None:
    fr = _reader()
    assert fr.ask("Сократ это животное?")        # транзитивность: сократ→человек→млекопитающее→животное
    assert fr.ask("человек это животное?")


def test_answers_by_inheritance() -> None:
    fr = _reader()
    assert fr.ask("Сократ смертный?")            # классический силлогизм (не сказано прямо)
    assert fr.ask("Сократ дышит?")               # наследование через транзитивный is_a


def test_honest_about_underivable() -> None:
    fr = _reader()
    assert not fr.ask("Сократ летает?")          # не следует — честное «нет», не выдумка
    assert fr.answer("Сократ летает?") == "нет"
