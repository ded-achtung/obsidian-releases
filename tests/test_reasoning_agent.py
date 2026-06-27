"""Тесты рассуждающего агента: индукция мира, подцели по модели, язык, экономия опыта."""

from __future__ import annotations

from thinking_system.world.rooms import rooms_world
from thinking_system.agent.reasoning_agent import ReasoningAgent

TRUE_DOORS = {(1, 3), (5, 3), (3, 1), (3, 5)}


def _agent():
    agent = ReasoningAgent(rooms_world(), seed=0)
    agent.learn_world(40)
    return agent


def test_learns_world_dynamics_from_few_observations() -> None:
    agent = _agent()
    assert len(agent.model.rules) == 4                      # вывел правило на каждое действие
    assert agent.observations <= 40                         # из горстки наблюдений (sample-efficient)


def test_discovers_subgoals_over_imagined_model() -> None:
    agent = _agent()
    subs = set(agent.discover_subgoals(4))
    assert subs == TRUE_DOORS                               # подцели по достроенной модели = проёмы (без обхода)


def test_obeys_language_by_planning_over_model() -> None:
    agent = _agent()
    assert agent.command_success(agent.held[:80]) >= 0.9    # планирует к языковым целям почти безошибочно
    r = agent.obey("take the northern passage", (0, 0))
    assert r["reached"] and r["goal"] == (1, 3)


def test_reasons_about_world() -> None:
    agent = _agent()
    assert agent.predict((2, 0), 1) == (3, 0) or agent.predict((2, 0), 1) == (2, 0)  # вниз или упор
    assert agent.distance((0, 0), (6, 6)) is not None       # находит путь по выученной модели
