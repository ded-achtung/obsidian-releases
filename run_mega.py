#!/usr/bin/env python3
"""Полное слияние: JEPA-модель мира + иерархическая память + язык в одном агенте.

Агент в зашумлённом мире-комнатах: понимает языковую команду, воспринимает свою
позицию обученной JEPA-моделью (сквозь шум), идёт к цели по иерархической памяти
(карта/ориентиры/навыки). Все три блока работают вместе.

Запуск: python run_mega.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.agent.mega import MegaAgent
from thinking_system.language.grounding import BagOfWords, GoalClassifier, PLACES, generate_commands, goal_cell
from thinking_system.memory.hierarchical import HierarchicalMemory
from thinking_system.world.features import CellFeatures, FeatureWorld
from thinking_system.world.gridworld import GridWorld
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.world.rooms import rooms_world


def collect(grid, feats, n, *, seed):
    rng = np.random.default_rng(seed)
    s = grid.sid(grid.start)
    O, A, O2 = [], [], []
    for _ in range(n):
        a = int(rng.integers(4))
        r, c = divmod(s, grid.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        sp = grid.sid((nr, nc)) if (0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls) else s
        O.append(feats.observe(s)); A.append(a); O2.append(feats.observe(sp)); s = sp
    return np.array(O), np.array(A), np.array(O2)


def main():
    grid = rooms_world()
    size = grid.size
    free = [s for s in range(grid.n_states) if (s // size, s % size) not in grid.walls]
    dim = 24
    feats = CellFeatures(grid.n_states, dim, noise=0.3, seed=0)

    # — JEPA-восприятие —
    O, A, O2 = collect(grid, feats, 12000, seed=1)
    jepa = LatentWorldModel(dim, 4, latent_dim=16, hidden=64, lr=5e-4, var_coef=0.02, seed=0)
    rng = np.random.default_rng(2)
    for _ in range(15000):
        b = rng.integers(0, len(O), 128)
        jepa.update(O[b], A[b], O2[b])
    protos = np.array([jepa.encode(np.array([feats.observe(s) for _ in range(40)])).mean(0) for s in free])
    pacc = np.mean([free[int(np.argmin(((protos - jepa.encode(feats.observe(s))[0]) ** 2).sum(1)))] == s for s in free for _ in range(10)])
    print(f"▶ Зашумлённый мир-комнаты {size}×{size} (dim={dim}); {len(free)} клеток")
    print(f"  ВОСПРИЯТИЕ (JEPA): позиция сквозь шум — точность {pacc * 100:.0f}%")

    # — иерархическая память —
    mem = HierarchicalMemory(grid)
    mem.explore(20000, seed=0)
    mem.consolidate(n_landmarks=4)
    print(f"  ПАМЯТЬ (иерархия): {len(mem.landmarks)} ориентира, {len(mem.skills)} навыков")

    # — язык —
    gc = GoalClassifier(BagOfWords([t for t, _ in generate_commands()]))
    gc.fit(generate_commands(), epochs=300)
    print("  ЯЗЫК: классификатор команд обучен\n")

    agent = MegaAgent(grid, jepa, protos, free, mem, gc)
    commands = [
        ("go to the top right corner", 1),
        ("head to the bottom left", 2),
        ("reach the upper left", 0),
        ("walk to the lower right corner", 3),
    ]
    print("КОМАНДА → (язык+JEPA+иерархия) → ДЕЙСТВИЕ в шумном мире:")
    print(f"   {'команда':<32}{'цель':<14}{'итог':>26}")
    rng2 = np.random.default_rng(3)
    for text, expected in commands:
        goal_sid = grid.sid(goal_cell(expected, size))
        starts = [s for s in free if s != goal_sid]
        succ, steps_sum, pacc_sum = 0, 0, 0.0
        for _ in range(6):
            world = FeatureWorld(grid, feats)
            res = agent.navigate(text, world, int(rng2.choice(starts)), size)
            succ += res["reached"]
            steps_sum += res["steps"]
            pacc_sum += res["perception_acc"]
        ok = "✓" if res["cls"] == expected else "✗"
        print(f"   {text:<32}{PLACES[res['cls']][0]:<14}{f'{succ}/6 дошёл, ~{steps_sum / 6:.0f} шаг':>26}  {ok}")

    print("\n── Итог ──")
    print("   Один агент: понимает язык, воспринимает позицию обученной моделью мира сквозь")
    print("   шум и доходит до цели по иерархической памяти — JEPA + память + язык вместе.")


if __name__ == "__main__":
    main()
