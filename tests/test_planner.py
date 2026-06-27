"""Тесты многошагового планирования: разбор цели, поиск цепочки операций."""

from __future__ import annotations

from thinking_system.language.understanding import GroundedLexicon
from thinking_system.language.planner import TaskPlanner, parse_goal_task, plan_ops


def _lex():
    lex = GroundedLexicon()
    lex.learn("разверни", [([1, 2, 3], [3, 2, 1]), ([4, 5], [5, 4])])
    lex.learn("удвой", [([1, 2, 3], [2, 4, 6]), ([5, 1], [10, 2])])
    lex.learn("сумма", [([1, 2, 3], 6), ([4, 5], 9)])
    lex.learn("максимум", [([3, 1, 2], 3), ([5, 9, 2], 9)])
    return lex


def test_parse_goal_task() -> None:
    assert parse_goal_task("получи 12 из [1, 2, 3]") == ([1, 2, 3], 12)
    assert parse_goal_task("преврати [1, 2, 3] в 12") == ([1, 2, 3], 12)


def test_plan_finds_multi_step_sequence() -> None:
    words = _lex().words
    assert plan_ops([1, 2, 3], 12, words) == ["удвой", "сумма"]      # двухшаговый план
    assert plan_ops([1, 2, 3], [6, 4, 2], words) is not None         # удвой+разверни
    assert plan_ops([1, 2, 3], 6, words) == ["сумма"]                # один шаг
    assert plan_ops([1, 2, 3], 100, words) is None                   # недостижимо


def test_task_planner_solves_and_verifies() -> None:
    p = TaskPlanner(_lex())
    r = p.plan("получи 12 из [1, 2, 3]")
    assert r["solved"] and r["plan"] == ["удвой", "сумма"] and r["answer"] == 12
    assert p.plan("получи 100 из [1, 2, 3]")["solved"] is False      # честно: плана нет
