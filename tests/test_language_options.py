"""Тесты «язык → выбор выученной опции»: грунтинг, выбор навыка, композиция."""

from __future__ import annotations

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.agent.qoption import QOptionLibrary
from thinking_system.agent.language_options import LanguageOptionAgent
from thinking_system.language.doorways import train_doorway_classifier, accuracy, DOORWAYS


def _agent():
    grid = rooms_world()
    doorways = [grid.sid(DOORWAYS[i][0]) for i in range(4)]
    lib = QOptionLibrary(grid, doorways, seed=0)
    lib.train(episodes_each=1500)
    clf, _, held = train_doorway_classifier(seed=0)
    return grid, LanguageOptionAgent(lib, clf), held


def test_doorway_grounding_generalizes() -> None:
    clf, _, held = train_doorway_classifier(seed=0)
    assert accuracy(clf, held) >= 0.9                       # обобщает на невиданные формулировки


def test_command_selects_correct_option_and_reaches() -> None:
    grid, agent, held = _agent()
    free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    rng = np.random.default_rng(1)
    corr = reached = 0
    for cmd, y in held[:120]:
        r = agent.obey(cmd, int(rng.choice(free)), seed=int(rng.integers(10 ** 6)))
        corr += int(r["doorway"] == y)
        reached += int(r["reached"] and r["doorway"] == y)
    assert corr >= 114                                      # верный навык по команде (≥95%)
    assert reached >= 114                                   # и навык доводит до проёма


def test_command_chain_composes_options() -> None:
    grid, agent, _ = _agent()
    r = agent.obey_sequence(["go through the upper doorway", "reach the eastern passage"], grid.sid((0, 0)), seed=0)
    assert r["reached"]                                     # цепочка команд → цепочка опций доходит
