"""Тесты накопления DSL: базовый язык не выражает, синтез из данных, реюз и рост."""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.dsl_growth import ShiftEnv, GrowingLanguage, synthesize_shift
from thinking_system.reasoning.induction import induce
from thinking_system.reasoning.grounded import pair_primitives

GRID = GridWorld(9, set(), start=(0, 0), goal=(8, 8))
FREE = [(r, c) for r in range(9) for c in range(9)]


def _walk(env, steps=1500, seed=0):
    rng = np.random.default_rng(seed); s = GRID.start; obs = []
    for _ in range(steps):
        a = int(rng.integers(4)); sp = env.transition(s, a); obs.append((s, a, sp)); s = sp
    return obs


def _acc(model, env):
    return np.mean([model.predict(s, a) == env.transition(s, a) for s in FREE for a in range(4)])


def test_base_dsl_cannot_express_knight_but_synthesis_can() -> None:
    mv = [((1, 1), (3, 2)), ((2, 2), (4, 3))]              # Δ=(2,1)
    assert induce(mv, pair_primitives(), max_depth=2) is None   # базовый язык на глубине 2 не выражает
    prim = synthesize_shift("Δ", mv)
    assert prim is not None and prim.fn((0, 0)) == (2, 1)  # синтез восстановил сдвиг из данных


def test_growing_language_learns_unexpressible_world() -> None:
    env = ShiftEnv(GRID, [(2, 1), (1, 2), (-2, -1), (-1, -2)])
    lang = GrowingLanguage()
    model, n = lang.learn_world(GRID, _walk(env))
    assert n == 4 and len(lang.prims) == 8                 # синтезировал 4 примитива, язык вырос
    assert _acc(model, env) == 1.0                         # и точно описал мир


def test_accumulated_primitives_are_reused_without_synthesis() -> None:
    lang = GrowingLanguage()
    lang.learn_world(GRID, _walk(ShiftEnv(GRID, [(2, 1), (1, 2), (-2, -1), (-1, -2)])))   # синтез 4
    size_after_first = len(lang.prims)
    env2 = ShiftEnv(GRID, [(2, 1), (-2, -1), (1, 2), (-1, -2)])                            # те же ходы
    model2, n2 = lang.learn_world(GRID, _walk(env2, seed=1))
    assert n2 == 0                                         # всё переиспользовано — синтеза нет
    assert len(lang.prims) == size_after_first            # язык не вырос
    assert _acc(model2, env2) == 1.0
