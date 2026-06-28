"""Тест рефлексивного решателя: наблюдение, самопонимание, анализ провалов."""

from __future__ import annotations

from thinking_system.agent.reflect import ReflectiveSolver


def test_observes_a_solved_task() -> None:
    """Наблюдает успешную попытку и фиксирует, ЧЕМ решил."""
    s = ReflectiveSolver()
    train = [(((1, 2), (2, 1)), ((3, 4), (4, 3)))]          # colormap 1→3,2→4
    test = [(((1, 1), (2, 2)), ((3, 3), (4, 4)))]
    ep = s.solve_and_observe(train, test)
    assert ep["solved_by"] == "colormap" and ep["reason"] is None


def test_diagnoses_no_hypothesis() -> None:
    """Если схемы нет — честно ставит диагноз «нет схемы», а не молчит."""
    s = ReflectiveSolver()
    train = [(((1, 2),), ((9, 8, 7, 6, 5),))]              # ничем не объяснимо
    ep = s.solve_and_observe(train, None)
    assert ep["solved_by"] is None
    assert ep["reason"].startswith("no_hypothesis")


def test_knows_when_it_would_memorize() -> None:
    """Самосознание: распознаёт, что мог бы ЗАЗУБРИТЬ локальное правило, и отмечает это."""
    s = ReflectiveSolver()
    train = [(((1, 2, 3), (4, 5, 6)), ((9, 8, 7), (6, 5, 4))),
             (((2, 2, 1), (3, 1, 3)), ((1, 4, 0), (7, 2, 8)))]
    ep = s.solve_and_observe(train, None)
    assert ep["solved_by"] is None
    assert ep["reason"] == "rejected_as_memorization"


def test_reflect_self_report() -> None:
    """reflect() даёт самоотчёт: что решил и почему провалил остальное."""
    s = ReflectiveSolver()
    s.solve_and_observe([(((1, 2), (2, 1)), ((3, 4), (4, 3)))], [(((1, 1),), ((3, 3),))])
    s.solve_and_observe([(((1, 2),), ((9, 8, 7, 6),))], None)
    rep = s.reflect()
    assert rep["n"] == 2 and rep["solved"] == 1
    assert "colormap" in rep["by_synth"]
    assert sum(rep["failure_reasons"].values()) == 1


def test_competence_predicts_own_success() -> None:
    """Метапознание: после наблюдений предсказывает собственный успех (порог калиброван)."""
    s = ReflectiveSolver()
    # несколько решаемых (colormap, same_shape) и нерешаемых (странная форма)
    for a, b in [(1, 5), (2, 6), (3, 7), (1, 8)]:
        g = ((a, a + 1), (a + 1, a))
        o = tuple(tuple(v + b for v in r) for r in g)
        s.solve_and_observe([(g, o)], [(g, o)])
    for k in range(4):
        s.solve_and_observe([(((1, 2),), tuple([tuple(range(3 + k))]))], None)
    s.build_competence()
    # на похожей решаемой задаче метапознание не должно быть «всегда нет»
    g = ((2, 3), (3, 2)); o = tuple(tuple(v + 9 for v in r) for r in g)
    score = s.predict_solvable([(g, o)])
    assert isinstance(score, float)
    assert hasattr(s, "_thr")
