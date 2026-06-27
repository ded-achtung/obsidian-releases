"""Тесты частичного рассуждения: правило большинства + исключения, обобщение, план."""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.agent.meta_agent import GridEnv
from thinking_system.reasoning.hybrid import HybridWorldModel, induce_majority
from thinking_system.reasoning.grounded import induce_dynamics, WorldRule, pair_primitives


def _setup(budget=400):
    grid = GridWorld(11, set(), start=(0, 0), goal=(10, 10))
    env = GridEnv(grid, scramble_seed=3, n_scramble=12)
    rng = np.random.default_rng(0)
    s = grid.start
    obs = []
    for _ in range(budget):
        a = int(rng.integers(4)); sp = env.transition(s, a); obs.append((s, a, sp)); s = sp
    free = [(r, c) for r in range(11) for c in range(11)]
    return grid, env, free, obs


def test_induce_majority_returns_rule_and_exceptions() -> None:
    ex = [((1, 1), (1, 2)), ((2, 2), (2, 3)), ((3, 3), (3, 4)), ((4, 4), (5, 4))]  # 3 «col+1» + 1 исключение
    prog, miss = induce_majority(ex, pair_primitives(), max_depth=1)
    assert str(prog) == "col+1"                            # правило большинства
    assert miss == [3]                                     # и индекс исключения


def test_hybrid_beats_pure_rule_and_tabular() -> None:
    grid, env, free, obs = _setup(400)
    hyb = HybridWorldModel(grid).fit(obs)
    pure = WorldRule(grid, induce_dynamics(obs))
    acc = lambda m: np.mean([m.predict(s, a) == env.transition(s, a) for s in free for a in range(4)])
    tab = {(s, a): sp for s, a, sp in obs}
    tab_acc = np.mean([tab.get((s, a), s) == env.transition(s, a) for s in free for a in range(4)])
    assert acc(hyb) >= 0.9                                 # гибрид почти точен
    assert acc(pure) <= 0.2                                # чистое правило рушится от исключений
    assert acc(hyb) > tab_acc                              # и бьёт табличную память при том же бюджете


def test_hybrid_generalizes_to_unseen_regular_cells() -> None:
    grid, env, free, obs = _setup(400)
    hyb = HybridWorldModel(grid).fit(obs)
    seen = {(s, a) for s, a, _ in obs}
    regular = [(s, a) for s in free for a in range(4) if s not in env.perm and (s, a) not in seen]
    gen = np.mean([hyb.predict(s, a) == env.transition(s, a) for s, a in regular])
    assert gen >= 0.95                                     # правило переносит регулярное на невиданное


def test_hybrid_closed_loop_reaches_goal() -> None:
    grid, env, free, obs = _setup(200)                     # даже при неполной модели (бюджет мал)
    hyb = HybridWorldModel(grid).fit(obs)
    reached, _ = hyb.reach(env, (0, 0), (10, 10))
    assert reached                                         # замкнутый цикл доходит, доучивая исключения по ходу
