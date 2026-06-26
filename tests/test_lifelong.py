"""Тесты долгоживущего агента: рост библиотеки навыков, язык, частичная компетенция."""

from __future__ import annotations

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.agent.latent_qoption import TileEncoder
from thinking_system.agent.lifelong import LifelongAgent
from thinking_system.language.doorways import doorway_cell, DOORWAYS


def _agent():
    grid = rooms_world()
    enc = TileEncoder(grid.size, n_tiles=7, seed=0)          # восприятие — тайлы (быстро и надёжно)
    return LifelongAgent(grid, enc, enc.dim, alpha=0.05, seed=0)


def _live(agent, phases=4, episodes=4000):
    return [agent.live_phase(explore_steps=3000, episodes=episodes, max_new=1) for _ in range(phases)]


def test_lifelong_accumulates_skills_and_competence() -> None:
    agent = _agent()
    stats = _live(agent)
    assert [s["skills"] for s in stats] == [1, 2, 3, 4]      # по навыку за фазу — библиотека растёт
    cov = [s["coverage"] for s in stats]
    assert all(cov[i] <= cov[i + 1] + 1e-9 for i in range(3))  # покрытие мира не падает со временем
    assert cov[-1] >= 0.95                                   # зрелый агент покрывает мир
    assert stats[-1]["obey"] >= 0.8                          # …и выполняет большинство команд


def test_lifelong_obeys_language_and_generalizes() -> None:
    agent = _agent()
    _live(agent)
    rng = np.random.default_rng(1)
    free = agent.disc.free
    succ = np.mean([agent.obey(c, int(rng.choice(free)), seed=int(rng.integers(10 ** 6)))["reached"] for c, _ in agent.held[:80]])
    assert succ >= 0.8                                       # исполняет НЕВИДАННЫЕ формулировки команд


def test_young_agent_partial_and_reports_unknown() -> None:
    agent = _agent()
    st = agent.live_phase(explore_steps=3000, episodes=3000, max_new=1)
    assert st["skills"] == 1 and st["obey"] < 1.0           # молодой агент умеет не всё
    learned = set(agent.skills)
    idx_unknown = next(i for i in range(len(DOORWAYS)) if agent.g.sid(doorway_cell(i)) not in learned)
    cmd = f"go to the {DOORWAYS[idx_unknown][2][0]} doorway"
    r = agent.obey(cmd, agent.g.sid((0, 0)))
    assert r["doorway"] == idx_unknown and not r["known"]   # честно сообщает «навыка ещё нет»
