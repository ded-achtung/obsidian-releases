#!/usr/bin/env python3
"""П.2: любопытство движет всем поведением — автотелический агент в шумном мире.

Без внешних целей агент сам ставит задачи (фронтир) И достигает их, воспринимая
позицию через JEPA сквозь шум. Сравнение со случайными самоцелями.

Запуск: python run_autotelic.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.agent.autotelic import AutotelicAgent
from thinking_system.world.features import CellFeatures, FeatureWorld
from thinking_system.world.gridworld import GridWorld
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.world.rooms import rooms_world
from thinking_system.viz import sparkline


def collect(grid, feats, n, *, seed):
    rng = np.random.default_rng(seed)
    s = grid.sid(grid.start)
    O, A, O2 = [], [], []
    for _ in range(n):
        a = int(rng.integers(4))
        sp = grid.move_sid(s, a)
        O.append(feats.observe(s)); A.append(a); O2.append(feats.observe(sp)); s = sp
    return np.array(O), np.array(A), np.array(O2)


def main():
    grid = rooms_world()
    free = grid.free_sids()
    dim = 24
    feats = CellFeatures(grid.n_states, dim, noise=0.3, seed=0)

    O, A, O2 = collect(grid, feats, 10000, seed=1)
    jepa = LatentWorldModel(dim, 4, latent_dim=16, hidden=64, lr=5e-4, var_coef=0.02, seed=0)
    rng = np.random.default_rng(2)
    for _ in range(12000):
        b = rng.integers(0, len(O), 128)
        jepa.update(O[b], A[b], O2[b])
    protos = np.array([jepa.encode(np.array([feats.observe(s) for _ in range(40)])).mean(0) for s in free])
    print(f"▶ Шумный мир-комнаты {grid.size}×{grid.size}; {len(free)} клеток; JEPA-восприятие обучено\n")

    def run(intrinsic):
        ag = AutotelicAgent(grid, jepa, protos, free, intrinsic=intrinsic, seed=0)
        reach, pa = [], []
        for _ in range(120):
            r = ag.episode(FeatureWorld(grid, feats))
            reach.append(r["reachable"]); pa.append(r["perception_acc"])
        return reach, pa

    ri, pai = run(True)
    rr, _ = run(False)
    first = lambda r: next((i + 1 for i, x in enumerate(r) if x >= len(free)), -1)

    print("1) АВТОНОМНОЕ ОСВОЕНИЕ шумного мира (достижимых клеток; цели ставит сам):")
    idx = np.linspace(0, 119, 50).astype(int)
    print("   любопытство: " + sparkline([ri[i] for i in idx]) + f"   полностью к эпизоду {first(ri)}")
    print("   случайно:    " + sparkline([rr[i] for i in idx]) + f"   полностью к эпизоду {first(rr)}")
    print(f"\n2) ВОСПРИЯТИЕ сквозь шум по ходу автономной жизни: {100 * np.mean(pai):.0f}%")

    print("\n── Итог ──")
    print("   Без единой внешней цели агент сам ставит задачи и достигает их в шумном мире,")
    print("   воспринимая позицию своей моделью мира — любопытство движет всем поведением.")


if __name__ == "__main__":
    main()
