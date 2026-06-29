"""Тест единого агента: маршрутизация по модулям, сцепка, абляция смысла."""

from __future__ import annotations

from thinking_system.agent.mind import MindAgent
from thinking_system.world.rooms import rooms_world
from run_mind import COMMANDS, START


def _composite_score(semantic: bool) -> int:
    grid = rooms_world()
    agent = MindAgent(grid, semantic=semantic, seed=0)
    ok = 0
    for cmd, goal in COMMANDS:
        r = agent.solve_navigation(cmd, START)
        ok += int(r["reached"] and r["goal"] == goal)
    return ok


def test_unified_agent_solves_composite() -> None:
    """Единый агент связывает рассуждение→смысл→план и решает все композиты."""
    assert _composite_score(semantic=True) == len(COMMANDS)


def test_ablation_proves_meaning_is_load_bearing() -> None:
    """Без модуля смысла (новые словоформы) цепочка рвётся — интеграция настоящая."""
    full = _composite_score(semantic=True)
    ablated = _composite_score(semantic=False)
    assert full == len(COMMANDS)
    assert ablated <= 1                     # мешок слов не понимает новые словоформы
    assert full - ablated >= 3


def test_shared_memory_and_routing() -> None:
    """Доска накапливает результаты модулей; маршрут отражает сцепку."""
    grid = rooms_world()
    agent = MindAgent(grid, semantic=True, seed=0)
    agent.solve_navigation(COMMANDS[0][0], START)
    assert "world_model" in agent.bb and "goal" in agent.bb       # общая память заполнена
    assert any("reasoning" in t for t in agent.trace)
    assert any("meaning" in t for t in agent.trace)
    assert any("planning" in t for t in agent.trace)


def test_same_agent_does_deduction() -> None:
    """Тот же агент отвечает на факт-вопрос дедукцией (другой тип рассуждения)."""
    grid = rooms_world()
    agent = MindAgent(grid, semantic=True, seed=0)
    ans = agent.ask(["сократ это человек", "все человек смертен"], "сократ смертен?")
    assert ans == "да"
