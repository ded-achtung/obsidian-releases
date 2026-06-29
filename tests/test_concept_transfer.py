"""Тест переноса понятий A→B с контролем на нерелевантном C."""

from __future__ import annotations

from thinking_system.reasoning.induction import default_primitives
from run_concept_transfer import FAMILY_B, learn_concepts, solved_at_depth, FAMILY_A, FAMILY_C


def test_transfer_helps_and_control_does_not() -> None:
    base = default_primitives()
    a = learn_concepts(FAMILY_A)
    c = learn_concepts(FAMILY_C)

    n = len(FAMILY_B)
    s_base, _ = solved_at_depth(base, 2)
    s_a, _ = solved_at_depth(a.to_library().prims, 2)
    s_c, _ = solved_at_depth(c.to_library().prims, 2)

    assert s_base == 0           # базовый язык не берёт B при глубине 2
    assert s_a == n              # понятия из A переносятся → B решается полностью
    assert s_c == 0              # нерелевантные понятия из C не помогают (контроль)


def test_control_actually_learned_concepts() -> None:
    """Контроль осмыслен только если C реально выучил (но нерелевантные) понятия."""
    c = learn_concepts(FAMILY_C)
    a = learn_concepts(FAMILY_A)
    assert len(c.abstractions) >= 1
    # и эти понятия НЕ пересекаются с релевантными из A
    assert set(c.abstractions) & set(a.abstractions) == set()
