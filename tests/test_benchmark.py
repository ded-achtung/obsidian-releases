"""Тест честного замера покрытия: обобщение на held-out + анти-галлюцинация.

Это «настоящий тест системы», а не отдельная демка: каждое «решено» — верный ответ
на ОТЛОЖЕННОМ входе (не использованном при индукции), а негативные контроли вне
шаблонов не должны давать обобщающего правила.
"""

from __future__ import annotations

import pytest

from thinking_system.benchmark import evaluate, suite


def test_full_coverage_on_in_scope_tasks() -> None:
    res = evaluate()
    assert res["in_scope_solved"] == res["in_scope_total"]      # все обобщают на held-out


def test_no_spurious_generalization_on_negatives() -> None:
    res = evaluate()
    assert res["negative_spurious"] == 0                         # вне шаблонов — ничего не выдумала


@pytest.mark.parametrize("task", [t for t in suite() if t.in_scope], ids=lambda t: t.name)
def test_each_in_scope_task_generalizes(task) -> None:
    from thinking_system.system import ThinkingSystem
    prog = ThinkingSystem().solve(task.train)
    assert prog is not None, f"{task.name}: правило не выведено"
    assert prog(task.test_in) == task.test_out, f"{task.name}: не обобщилось на held-out"


@pytest.mark.parametrize("task", [t for t in suite() if not t.in_scope], ids=lambda t: t.name)
def test_each_negative_is_rejected(task) -> None:
    from thinking_system.system import ThinkingSystem
    prog = ThinkingSystem().solve(task.train)
    got = None
    if prog is not None:
        try:
            got = prog(task.test_in)
        except Exception:  # noqa: BLE001
            got = None
    assert got != task.test_out, f"{task.name}: выдала обобщение там, где его нет"
