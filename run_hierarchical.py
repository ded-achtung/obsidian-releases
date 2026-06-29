#!/usr/bin/env python3
"""П.2: иерархическая память — эпизод → семантика (ориентиры) → навыки.

Агент исследует мир-комнаты (эпизодически), консолидирует опыт: ориентиры
(клетки наибольшего транзита по betweenness) + навыки (маршруты между ними).
Затем планирует ИЕРАРХИЧЕСКИ — по ориентирам, переиспользуя навыки.

Честно: ориентиры здесь — клетки с высоким betweenness (часто лежат на путях), а
НЕ обязательно дверные проёмы (бутылочные горлышки). Дверные проёмы как таковые
находят отдельные модули — option_discovery / spectral_options. И иерархия
сокращает число УЗЛОВ планирования, а не длину итогового пути (см. п.3).

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
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls:
                v = grid.sid((nr, nc))
                if v not in prev:
                    prev[v] = u
                    q.append(v)
    return -1


def main():
    grid = rooms_world()
    free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    mem = HierarchicalMemory(grid)
    mem.explore(20000, seed=0)
    mem.consolidate(n_landmarks=4)

    print(f"▶ Мир-комнаты {grid.size}×{grid.size}: {len(free)} клеток, 4 комнаты через проёмы\n")

    print("1) КОНСОЛИДАЦИЯ: ориентиры = клетки наибольшего транзита (betweenness):")
    for L in mem.landmarks:
        print(f"   клетка {divmod(L, grid.size)}  (через неё кратчайших путей: {mem.betweenness[L]})")
    print(f"\n2) СЕМАНТИЧЕСКАЯ КОМПРЕССИЯ: {len(mem.landmarks)} ориентиров вместо {len(free)} клеток")
    print(f"   НАВЫКИ (маршруты между ориентирами): {len(mem.skills)} штук")

    # иерархическое планирование на множестве целей
    rng = np.random.default_rng(1)
    n_trials = 60
    succ = 0
    high_lens, flat_lens, prim_lens = [], [], []
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
            prim_lens.append(plan["n_steps"])
            for li, lj in zip(plan["landmarks"], plan["landmarks"][1:]):
                used_skills.add((li, lj))

    print(f"\n3) ИЕРАРХИЧЕСКОЕ ПЛАНИРОВАНИЕ на {n_trials} случайных целях:")
    print(f"   дошёл до цели: {succ}/{n_trials}")
    print(f"   планирование идёт по ~{np.mean(high_lens):.1f} ОРИЕНТИРАМ (узлов в графе), а не по")
    print(f"     ~{np.mean(flat_lens):.1f} клеткам — компрессия ПОИСКА плана (меньше узлов для перебора)")
    print(f"   но длина итогового пути: ~{np.mean(prim_lens):.1f} примитивных шагов против ~{np.mean(flat_lens):.1f} оптимума")
    print(f"     (≈{np.mean(prim_lens) / max(np.mean(flat_lens), 1e-9):.2f}× — абстракция упрощает планирование, не укорачивает путь)")
    print(f"   переиспользовано НАВЫКОВ: {len(used_skills)} на все {succ} задач (procedural reuse)")

    print("\n── Итог ──")
    print("   Память иерархична: конкретный опыт (эпизоды) консолидируется в компактные")
    print("   ориентиры (семантика) и переиспользуемые навыки-маршруты (процедуры).")
    print("   Планирование идёт по нескольким ориентирам, а не десяткам клеток — это")
    print("   упрощает ПОИСК плана; длина итогового пути при этом близка к оптимуму, но не короче.")


if __name__ == "__main__":
    main()
