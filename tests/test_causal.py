"""Тесты причинного рассуждения: корреляция≠причинность, do(), контрфактика, план."""

from __future__ import annotations

from thinking_system.reasoning.causal import SCM


def _sprinkler() -> SCM:
    m = SCM()
    m.root("Se", {0: 0.5, 1: 0.5})
    m.eq("Rain", ["Se"], lambda p: p["Se"])
    m.eq("Spr", ["Se"], lambda p: 1 - p["Se"])
    m.eq("Wet", ["Rain", "Spr"], lambda p: int(p["Rain"] or p["Spr"]))
    m.eq("Slip", ["Wet"], lambda p: p["Wet"])
    return m


def test_correlation_without_causation() -> None:
    m = _sprinkler()
    assert abs(m.correlation("Rain", "Spr")) > 0.5         # видеть: сильно связаны (общий предок)
    assert abs(m.affects("Rain", "Spr")) < 1e-9            # делать: причинного влияния нет


def test_intervention_effect_of_true_cause() -> None:
    m = _sprinkler()
    assert m.prob("Slip", 1, do={"Spr": 1}) == 1.0         # do(поливалка) → точно скользко
    assert m.affects("Spr", "Slip") > 0.0                  # есть причинный эффект


def test_counterfactual_individual_case() -> None:
    m = _sprinkler()
    dry = {"Se": 0, "Spr": 1, "Rain": 0, "Wet": 1, "Slip": 1}
    assert m.counterfactual(dry, {"Spr": 0}, "Wet") == {0: 1.0}   # в сухой день поливалка была причиной
    wet = {"Se": 1, "Spr": 0, "Rain": 1, "Wet": 1, "Slip": 1}
    assert m.counterfactual(wet, {"Spr": 1}, "Wet") == {1: 1.0}   # в дождь поливалка ничего не меняет


def test_planning_picks_a_real_cause() -> None:
    m = _sprinkler()
    (var, val), p = m.plan("Slip", 1, [("Spr", 1), ("Rain", 1), ("Se", 1)])
    assert p == 1.0 and var in ("Spr", "Rain")             # вмешательство в настоящую причину достигает цели
