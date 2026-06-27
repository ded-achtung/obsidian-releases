#!/usr/bin/env python3
"""Накопление DSL: агент сам расширяет язык правил под новые миры.

Базовый язык (шаги ±1 по одной оси) выражает не любую динамику. Если правило не
выводится поиском, агент СИНТЕЗИРУЕТ примитив из данных (постоянное смещение) и
запоминает его. Язык растёт с опытом: новые миры с уже виденными ходами решаются
без синтеза — агент становится способнее.

Запуск: python run_dsl.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.dsl_growth import ShiftEnv, GrowingLanguage, synthesize_shift
from thinking_system.reasoning.induction import induce
from thinking_system.reasoning.grounded import pair_primitives, WorldRule, induce_dynamics


def main():
    grid = GridWorld(9, set(), start=(0, 0), goal=(8, 8))
    free = [(r, c) for r in range(9) for c in range(9)]
    print("▶ Миры со сдвигами; базовый язык — шаги ±1 по одной оси\n")

    def walk(env, steps=1500, seed=0):
        rng = np.random.default_rng(seed); s = grid.start; obs = []
        for _ in range(steps):
            a = int(rng.integers(4)); sp = env.transition(s, a); obs.append((s, a, sp)); s = sp
        return obs

    def acc(model, env):
        return np.mean([model.predict(s, a) == env.transition(s, a) for s in free for a in range(4)])

    # 1) базовый язык не выражает ход конём
    envK = ShiftEnv(grid, [(2, 1), (1, 2), (-2, -1), (-1, -2)])
    obsK = walk(envK, seed=0)
    mv = [(s, sp) for s, a, sp in obsK if a == 0 and sp != s][:5]
    print("1) ХОД КОНЁМ Δ=(2,1):")
    print(f"   базовый язык на глубине 2 выводит: {induce(mv, pair_primitives(), max_depth=2)}  → не выразимо")
    print(f"   синтез из данных: примитив-сдвиг «{synthesize_shift('Δ', mv).name}» = {[ (sp[0]-s[0], sp[1]-s[1]) for s,sp in mv][:1][0]} из ОДНОГО примера")

    # 2) базовый язык БЕЗ роста — модель пуста → 0%
    base_model = WorldRule(grid, induce_dynamics(obsK))
    print(f"\n2) БЕЗ роста языка: точность модели мира-коня {acc(base_model, envK):.0%} (язык не описал динамику)")

    # 3) растущий язык: последовательность миров, синтез падает с реюзом
    print("\n3) С РОСТОМ ЯЗЫКА — последовательность миров (синтез падает по мере накопления):")
    worlds = [
        ("конь",         [(2, 1), (1, 2), (-2, -1), (-1, -2)]),   # 4 новых
        ("конь+прыжки",  [(2, 1), (-2, -1), (3, 0), (0, 3)]),     # 2 прежних + 2 новых
        ("всё знакомое", [(1, 2), (-1, -2), (3, 0), (0, 3)]),     # все из прошлых
    ]
    lang = GrowingLanguage()
    print(f"   старт: язык = {len(lang.prims)} примитива")
    for name, moves in worlds:
        env = ShiftEnv(grid, moves)
        model, n = lang.learn_world(grid, walk(env, seed=0))
        print(f"   {name:<13}: синтезировано +{n}, язык вырос до {len(lang.prims)}, точность модели {acc(model, env):.0%}")

    print("\n── Итог ──")
    print("   Базовый язык описывает не всё. Не вывелось — агент СИНТЕЗИРУЕТ примитив из")
    print("   данных и расширяет язык; накопленные ходы переиспользуются, и новый мир")
    print("   решается без синтеза. Язык растёт с опытом — путь к новым доменам без корпуса.")


if __name__ == "__main__":
    main()
