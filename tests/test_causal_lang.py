"""Тесты понимания причинных утверждений: граф, путь, видеть≠делать, do, контрфактика."""

from __future__ import annotations

from thinking_system.language.causal_lang import CausalReader


def _reader():
    cr = CausalReader()
    for s in ["дождь вызывает мокрый", "спринклер вызывает мокрый",
              "мокрый вызывает скользкий", "дождь вызывает холодный"]:
        cr.tell(s)
    return cr


def test_parses_causal_claims() -> None:
    cr = CausalReader()
    assert cr.tell("дождь вызывает мокрый") == ("дождь", "мокрый")
    assert "дождь" in cr.parents["мокрый"]


def test_causal_path_queries() -> None:
    cr = _reader()
    assert cr.ask("спринклер вызывает скользкий?")          # путь спринклер→мокрый→скользкий
    assert not cr.ask("спринклер вызывает дождь?")          # пути нет


def test_correlation_is_not_causation() -> None:
    cr = _reader()
    assert not cr.causes("мокрый", "холодный")              # причинного пути нет
    assert cr.correlated("мокрый", "холодный")              # но связаны через общий предок «дождь»


def test_intervention_and_counterfactual() -> None:
    cr = _reader()
    assert cr.would("спринклер", "скользкий")               # do(спринклер) → скользко
    assert not cr.would("спринклер", "дождь")               # do(спринклер) дождь не вызывает
    ev = {"спринклер": 1, "дождь": 0, "мокрый": 1, "скользкий": 1, "холодный": 0}
    assert cr.counterfactual(ev, {"спринклер": 0}, "мокрый") == {0: 1.0}   # без спринклера было бы сухо
