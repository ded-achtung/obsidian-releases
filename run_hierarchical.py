#!/usr/bin/env python3
"""П.2: иерархическая память — эпизод → семантика (ориентиры) → навыки.

Агент исследует мир-комнаты (эпизодически), консолидирует опыт: ориентиры
(дверные проёмы всплывают как часто-посещаемые) + навыки (маршруты между ними).
Затем планирует ИЕРАРХИЧЕСКИ — по ориентирам, переиспользуя навыки.

Запуск: python run_hierarchical.py
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.memory.hierarchical import HierarchicalMemory
from thinking_system.world.rooms import rooms_world


def bfs_dist(grid, start, goal):
    from thinking_system.world.gridworld import GridWorld
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            d = 0
            while prev[u] is not None:
                u = prev[u]
                d += 1
            return d
        r, c = divmod(u, grid.size)
        for dr, dc in GridWorld.MOVES:
            nxt = (r + dr, c + dc)
            if grid.is_free(nxt):
                v = grid.sid(nxt)
                if v not in prev:
                    prev[v] = u
                    q.append(v)
    return -1


def main():
    grid = rooms_world()
    free = grid.free_sids()
    mem = HierarchicalMemory(grid)
    mem.explore(20000, seed=0)
    mem.consolidate(n_landmarks=4)

    print(f"▶ Мир-комнаты {grid.size}×{grid.size}: {len(free)} клеток, 4 комнаты через проёмы\n")

    print("1) КОНСОЛИДАЦИЯ: ориентиры = клетки наибольшего транзита (хабы у проёмов):")
    for L in mem.landmarks:
        print(f"   клетка {divmod(L, grid.size)}  (через неё кратчайших путей: {mem.betweenness[L]})")
    print(f"\n2) СЕМАНТИЧЕСКАЯ КОМПРЕССИЯ: {len(mem.landmarks)} ориентиров вместо {len(free)} клеток")
    print(f"   НАВЫКИ (маршруты между ориентирами): {len(mem.skills)} штук")

    # иерархическое планирование на множестве целей
    rng = np.random.default_rng(1)
    n_trials = 60
    succ = 0
    high_lens, flat_lens = [], []
    used_skills = set()
    for _ in range(n_trials):
        start, goal = (int(x) for x in rng.choice(free, 2, replace=False))
        plan = mem.plan(start, goal)
        if plan is None:
            continue
        s = start  # проверка: выполняем план и приходим ли в цель
        for a in plan["actions"]:
            s = mem._move(s, a)
        if s == goal:
            succ += 1
            high_lens.append(plan["n_high"])
            flat_lens.append(bfs_dist(grid, start, goal))
            for li, lj in zip(plan["landmarks"], plan["landmarks"][1:]):
                used_skills.add((li, lj))

    print(f"\n3) ИЕРАРХИЧЕСКОЕ ПЛАНИРОВАНИЕ на {n_trials} случайных целях:")
    print(f"   дошёл до цели: {succ}/{n_trials}")
    print(f"   высокоуровневый план: ~{np.mean(high_lens):.1f} ОРИЕНТИРОВ против ~{np.mean(flat_lens):.1f} примитивных шагов")
    print(f"   переиспользовано НАВЫКОВ: {len(used_skills)} на все {succ} задач (procedural reuse)")

    print("\n── Итог ──")
    print("   Память иерархична: конкретный опыт (эпизоды) консолидируется в компактные")
    print("   ориентиры (семантика) и переиспользуемые навыки-маршруты (процедуры).")
    print("   Планирование идёт по нескольким ориентирам, а не десяткам клеток.")


if __name__ == "__main__":
    main()
