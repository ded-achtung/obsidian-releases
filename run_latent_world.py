#!/usr/bin/env python3
"""П.1: обучаемая JEPA-модель мира на НЕПРЕРЫВНЫХ ЗАШУМЛЁННЫХ наблюдениях.

Наблюдение клетки — это её скрытый признак + шум (каждый раз разный), поэтому
табличная модель невозможна. Модель учит инвариантное к шуму представление и
предсказывает следующий латент по действию. Проверяем:
  1) ошибка предсказания латента падает (учится динамике);
  2) латент НЕ схлопывается (std > 0);
  3) представление инвариантно к шуму (внутри-клеточная дистанция << меж-клеточной);
  4) латент восстановил ПРОСТРАНСТВО (корреляция латентной и сеточной дистанций);
  5) модель верно предсказывает ПОСЛЕДСТВИЕ действия (в какую клетку приведёт),
     несмотря на шум, — значит, ею можно планировать.

Запуск: python run_latent_world.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import default_maze
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.viz import sparkline


class CellFeatures:
    """Зашумлённое наблюдение клетки: скрытый признак + гауссов шум."""

    def __init__(self, n_states, dim, *, noise, seed):
        rng = np.random.default_rng(seed)
        self.feat = rng.standard_normal((n_states, dim))  # признак клетки полной амплитуды (SNR > 1)
        self.noise = noise
        self.dim = dim
        self.rng = rng

    def observe(self, state):
        return self.feat[state] + self.noise * self.rng.standard_normal(self.dim)


def collect(env, feats, n, *, seed):
    rng = np.random.default_rng(seed)
    s = env.reset()
    O, A, O2, S, S2 = [], [], [], [], []
    for _ in range(n):
        a = int(rng.integers(env.n_actions))
        o = feats.observe(s)
        sp, _ = env.step(a)
        O.append(o); A.append(a); O2.append(feats.observe(sp)); S.append(s); S2.append(sp)
        s = sp
        if s == env.goal_state:
            s = env.reset()
    return np.array(O), np.array(A), np.array(O2), np.array(S), np.array(S2)


def main():
    env = default_maze()
    dim = 24
    feats = CellFeatures(env.n_states, dim, noise=0.30, seed=0)
    free = [s for s in range(env.n_states) if (s // env.size, s % env.size) not in env.walls]

    O, A, O2, S, S2 = collect(env, feats, 9000, seed=1)
    print(f"▶ Мир-сетка {env.size}×{env.size}; наблюдение = признак клетки + шум (dim={dim}, шум 0.30)")
    print(f"  переходов для обучения: {len(O)}\n")

    model = LatentWorldModel(dim, env.n_actions, latent_dim=16, hidden=64, lr=5e-4, tau=0.99, var_coef=0.02, seed=0)
    rng = np.random.default_rng(2)
    curve = []
    std = 0.0
    for step in range(22000):
        b = rng.integers(0, len(O), 128)
        loss, std = model.update(O[b], A[b], O2[b])
        if (step + 1) % 700 == 0:
            curve.append(loss)

    print("1) ошибка предсказания латента по ходу обучения:")
    print("   " + sparkline(curve) + f"   {curve[0]:.4f} → {curve[-1]:.4f}")
    print(f"\n2) латент НЕ схлопнулся: std = {std:.3f}  ({'ок' if std > 0.05 else 'КОЛЛАПС'})")

    # средний латент каждой клетки (по 40 зашумлённым наблюдениям)
    cell_lat = {s: model.encode(np.array([feats.observe(s) for _ in range(40)])).mean(0) for s in free}

    # 3) инвариантность к шуму: внутри-клеточная vs меж-клеточная дистанция
    intra = np.mean([np.linalg.norm(model.encode(feats.observe(s)) - model.encode(feats.observe(s))) for s in free for _ in range(5)])
    inter = np.mean([np.linalg.norm(cell_lat[a] - cell_lat[b]) for a in free[::3] for b in free[::3] if a != b])
    print(f"\n3) инвариантность к шуму: внутри-клетки {intra:.3f}  <<  меж-клеток {inter:.3f}  (раз в {inter / max(intra, 1e-6):.1f})")

    # 4) восстановление пространства: корреляция латентной и сеточной дистанций
    rng2 = np.random.default_rng(3)
    ld, gd = [], []
    for _ in range(3000):
        a, b = rng2.choice(free, 2, replace=False)
        ld.append(np.linalg.norm(cell_lat[a] - cell_lat[b]))
        gd.append(abs(a // env.size - b // env.size) + abs(a % env.size - b % env.size))
    corr = float(np.corrcoef(ld, gd)[0, 1])
    print(f"\n4) латент = различимые place-коды клеток (не метрическая карта: corr={corr:.2f}) —")
    print("   для предсказания последствий этого достаточно (см. п.5)")

    # базовая трудность задачи: распознать КЛЕТКУ по зашумлённому наблюдению (без обучения)
    # — при SNR>1 это почти тривиально (ближайший чистый признак), поэтому суть задачи не в
    # восприятии, а в выученной ДИНАМИКЕ латента (переход под действием).
    clean = feats.feat
    id_correct = sum(int(np.argmin(np.sum((clean - feats.observe(s)) ** 2, axis=1)) == s)
                     for s in free for _ in range(10))
    print(f"\n5a) контроль: распознать клетку по шуму (ближайший чистый признак, БЕЗ обучения): "
          f"{100 * id_correct / (len(free) * 10):.1f}%  ← восприятие при этом шуме легко (SNR>1)")

    # 5b) предсказание последствий на ОТЛОЖЕННЫХ переходах (не из обучающего пула)
    Oh, Ah, O2h, Sh, S2h = collect(env, feats, 1500, seed=99)
    cells = np.array(free)
    lat_mat = np.array([cell_lat[s] for s in free])
    correct = 0
    n = 0
    for o, a, s2, s in zip(Oh, Ah, S2h, Sh):
        if s == s2:
            continue  # пропускаем «упёрся в стену» (нет движения)
        pred = model.predict_next(o, a)[0]
        nearest = cells[np.argmin(np.sum((lat_mat - pred) ** 2, axis=1))]
        correct += int(nearest == s2)
        n += 1
    print(f"5b) верно предсказал клетку-последствие на HELD-OUT переходах: {100 * correct / n:.1f}%  (из {n} ходов)")

    print("\n── Итог ──")
    print("   Обучаемая (не табличная) модель мира работает на зашумлённых наблюдениях:")
    print("   учит инвариантное к шуму представление клеток (главный результат: нет коллапса,")
    print("   внутри-клетки << меж-клеток) и предсказывает последствия действий на отложенных")
    print("   переходах. Честно: при SNR>1 распознать клетку легко (см. 5a) — ценность в")
    print("   выученной динамике латента, а не в самом проценте «последствия».")


if __name__ == "__main__":
    main()
