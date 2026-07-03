#!/usr/bin/env python3
"""Стохастический мир: вероятностная модель + планирование под неопределённостью.

Когда действие иногда соскальзывает, детерминированное правило не вывести (одна и
та же пара клетка×действие даёт разные исходы). Агент оценивает РАСПРЕДЕЛЕНИЕ
переходов и планирует под неопределённостью — получается РЕАКТИВНАЯ политика,
устойчивая к сбоям, тогда как фиксированный план разваливается.

Запуск: python run_stochastic.py
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.stochastic import StochasticGridEnv, ProbabilisticModel
from thinking_system.reasoning.grounded import induce_dynamics


def bfs_plan(grid, start, goal):
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            acts = []
            while prev[u] is not None:
                p, a = prev[u]; acts.append(a); u = p
            return acts[::-1]
        for a in range(4):
            r, c = u; dr, dc = GridWorld.MOVES[a]; v = (r + dr, c + dc)
            if 0 <= v[0] < grid.size and 0 <= v[1] < grid.size and v not in grid.walls and v not in prev:
                prev[v] = (u, a); q.append(v)
    return []


def main():
    grid = rooms_world(); goal = (6, 6); slip = 0.25
    free = grid.free_cells()
    env = StochasticGridEnv(grid, slip=slip, seed=0)
    print(f"▶ Скользкий мир-комнаты {grid.size}×{grid.size}, соскальзывание {slip:.0%}; цель {goal}\n")

    rng = np.random.default_rng(0); s = grid.start; obs = []
    for _ in range(8000):
        a = int(rng.integers(4)); sp = env.transition(s, a); obs.append((s, a, sp)); s = sp

    rules = induce_dynamics(obs)
    print(f"1) ДЕТЕРМИНИРОВАННАЯ ИНДУКЦИЯ не работает: выведено {len(rules)}/4 правил (шум ломает консистентность)")

    pm = ProbabilisticModel(grid).fit(obs)
    dist = pm.transitions((0, 0), 3)
    print("\n2) ВЕРОЯТНОСТНАЯ МОДЕЛЬ из наблюдений — P(s'|(0,0), →):")
    print("   " + ", ".join(f"{k}:{v:.0%}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1])))

    pm.value_iteration(goal)
    starts = [c for c in free if c != goal]
    rng = np.random.default_rng(1)
    vi_ok = vi_steps = det_ok = 0
    N = 300
    for _ in range(N):
        st = starts[int(rng.integers(len(starts)))]
        e1 = StochasticGridEnv(grid, slip=slip, seed=int(rng.integers(10 ** 6)))
        ok, t = pm.reach(e1, st, goal); vi_ok += ok; vi_steps += t if ok else 0
        e2 = StochasticGridEnv(grid, slip=slip, seed=int(rng.integers(10 ** 6)))
        s = st
        for a in bfs_plan(grid, st, goal):
            s = e2.transition(s, a)
        det_ok += s == goal
    print(f"\n3) ДОСТИЖЕНИЕ ЦЕЛИ под шумом ({N} запусков):")
    print(f"   реактивная политика (планир. под неопределённостью): {100 * vi_ok / N:.0f}% дошли, ~{vi_steps / max(vi_ok,1):.0f} шагов")
    print(f"   фиксированный план (кратчайший путь, без коррекции):  {100 * det_ok / N:.0f}% — разваливается от сбоев")

    print("\n── Итог ──")
    print("   В шумном мире детерминированного правила нет: агент оценивает РАСПРЕДЕЛЕНИЕ")
    print("   переходов и планирует под неопределённостью. Нужна реактивная политика (выбор")
    print("   по текущей клетке), а не жёсткий план. Шум — экзогенная случайность механизма")
    print("   (смычка с причинной моделью), которую агент учитывает.")


if __name__ == "__main__":
    main()
