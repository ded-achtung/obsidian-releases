"""Тесты мета-агента: арбитраж рассуждение↔обучение, общие подцели и язык."""

from __future__ import annotations

from thinking_system.world.rooms import rooms_world
from thinking_system.agent.meta_agent import GridEnv, MetaAgent

TRUE_DOORS = {(1, 3), (5, 3), (3, 1), (3, 5)}


def test_reasons_cheaply_when_rule_fits() -> None:
    agent = MetaAgent(GridEnv(rooms_world()), seed=0)
    r = agent.live()
    assert r["mode"] == "reasoning"                         # правило подошло → выбрал рассуждение
    assert r["cost"] < 300                                  # почти даром (десятки наблюдений)
    assert set(r["subgoals"]) == TRUE_DOORS and r["commands"] >= 0.9


def test_falls_back_to_rl_when_rule_does_not_fit() -> None:
    agent = MetaAgent(GridEnv(rooms_world(), scramble_seed=1), seed=0)
    r = agent.live(rl_steps=4000, rl_episodes=1500)
    assert r["mode"] == "trial-and-error"                   # правило не прошло проверку → откат к RL
    assert agent.diag["verify_acc"] < 0.95                  # модель не описала перепутанный мир
    assert r["cost"] > 1000                                 # обучение из опыта дороже
    assert set(r["subgoals"]) == TRUE_DOORS                 # подцели всё равно найдены (общая способность)
    assert r["commands"] >= 0.8                             # навыки методом проб выполняют команды


def test_same_agent_class_handles_both_worlds() -> None:
    easy = MetaAgent(GridEnv(rooms_world()), seed=0); easy.live()
    hard = MetaAgent(GridEnv(rooms_world(), scramble_seed=2), seed=0); hard.live(rl_steps=4000, rl_episodes=1500)
    assert easy.mode != hard.mode                           # один класс, разный выбор по миру
    assert easy.cost < hard.cost                            # рассуждение экономнее обучения
