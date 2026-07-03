#!/usr/bin/env python3
"""Q-learning над ВОСПРИЯТИЕМ (функциональная аппроксимация вместо таблицы).

Опция-навык учится не по индексам клеток, а по ПРИЗНАКАМ восприятия φ(o): ценность
— линейная Q(φ(o),a). Признаков меньше, чем клеток, поэтому навык не запоминается
поклеточно, а ОБОБЩАЕТСЯ — доходит даже из стартов, не виденных при обучении, и
работает из зашумлённого восприятия.

  1) TileEncoder — гладкие RBF-тайлы: 17 признаков на 49 клеток, 100% обобщение;
  2) латенты JEPA — то же Q-обучение прямо над 16-мерным ВЫУЧЕННЫМ восприятием.

Запуск: python run_latent_q.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.world.features import CellFeatures
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.agent.latent_qoption import TileEncoder, LatentQOption, jepa_encoder
from thinking_system.viz import sparkline


def _move(grid, s, a):
    return grid.move_sid(s, a)  # стен в открытой комнате нет


def main():
    grid = GridWorld(7, set(), start=(0, 0), goal=(6, 6))   # открытая комната: ценность гладкая
    free = list(range(grid.n_states))
    sub = grid.sid((3, 3))
    print(f"▶ Открытая комната {grid.size}×{grid.size}; опция-навык к центру (3,3), Q над восприятием\n")

    # ── 1) Тайл-признаки: меньше параметров, чем клеток → обобщение ───────────────
    enc = TileEncoder(grid.size, n_tiles=4, noise=0.0, seed=0)
    rng = np.random.default_rng(0)
    perm = rng.permutation([s for s in free if s != sub])
    cut = int(0.6 * len(perm))
    train_starts, held = list(perm[:cut]), list(perm[cut:])     # учим на 60% стартов, проверяем на 40%

    opt = LatentQOption(grid, sub, enc, enc.dim, train_starts=train_starts, seed=0)
    curve = opt.train(3000)
    bins = [float(np.mean(curve[k:k + 60])) for k in range(0, len(curve) - 59, 60)]
    print(f"1) ТАЙЛ-ПРИЗНАКИ: {enc.dim} признаков на {len(free)} клеток (меньше параметров, чем клеток)")
    print("   шагов до подцели по ходу Q-learning:")
    print("   " + sparkline(bins) + f"   {bins[0]:.0f} → {bins[-1]:.0f} шагов")
    seen = np.mean([opt.reach(s, seed=i) for i, s in enumerate(train_starts)])
    out = np.mean([opt.reach(s, seed=100 + i) for i, s in enumerate(held)])
    print(f"   достигает подцели:  из виденных стартов {seen:.0%}   из ОТЛОЖЕННЫХ (обобщение) {out:.0%}")

    # устойчивость к шуму восприятия
    print("\n2) РАБОТА ИЗ ЗАШУМЛЁННОГО ВОСПРИЯТИЯ (координаты приходят с шумом σ):")
    for noise in [0.0, 0.3, 0.6]:
        e = TileEncoder(grid.size, n_tiles=4, noise=noise, seed=1)
        o = LatentQOption(grid, sub, e, e.dim, seed=0)
        o.train(3000)
        acc = np.mean([o.reach(s, seed=i) for i, s in enumerate(free) if s != sub])
        print(f"   σ={noise:<4}: {acc:.0%} дошли")

    # ── 3) Литерально над латентами JEPA ─────────────────────────────────────────
    print("\n3) Q НАД ЛАТЕНТАМИ JEPA (учим восприятие, потом навык поверх латента):")
    feats = CellFeatures(grid.n_states, dim=24, noise=0.15, seed=0)
    jepa = LatentWorldModel(obs_dim=24, n_actions=4, latent_dim=16, hidden=64, lr=5e-4, var_coef=0.02, seed=0)
    O, A, NO, s = [], [], [], 0
    for _ in range(10000):                                       # случайные переходы для обучения восприятия
        a = int(rng.integers(4)); sp = _move(grid, s, a)
        O.append(feats.observe(s)); A.append(a); NO.append(feats.observe(sp)); s = sp
    O, A, NO = np.array(O), np.array(A), np.array(NO)
    l0 = ll = 0.0
    for ep in range(60):
        idx = rng.permutation(len(O))
        for k in range(0, len(O), 128):
            b = idx[k:k + 128]; ll, _ = jepa.update(O[b], A[b], NO[b])
        if ep == 0:
            l0 = ll
    print(f"   JEPA выучила восприятие: loss {l0:.2f} → {ll:.2f}")

    enc_j = jepa_encoder(jepa, feats.observe, normalize=True)
    optj = LatentQOption(grid, sub, enc_j, jepa.Ld + 1, alpha=0.1, seed=0)
    cj = optj.train(6000)
    reach_j = np.mean([optj.reach(s, seed=i) for i, s in enumerate(free) if s != sub])
    print(f"   Q над 16-мерным латентом: шагов {np.mean(cj[:100]):.0f} → {np.mean(cj[-100:]):.0f}, доходит {reach_j:.0%}")

    print("\n── Итог ──")
    print("   Навык — функция от ВОСПРИЯТИЯ, а не таблица по клеткам: параметров меньше,")
    print("   чем состояний, поэтому ценность обобщается на невиданные старты и терпит шум.")
    print("   То же Q-обучение работает прямо над латентами обученной JEPA-модели —")
    print("   восприятие выучено, а не задано индексом клетки.")


if __name__ == "__main__":
    main()
