#!/usr/bin/env python3
"""Спектральные eigenoptions: горлышки из геометрии графа, а не из правила «узкая клетка».

  1) по графу опыта — нормированный лапласиан, вектор Фидлера разделяет комнаты,
     спектральный счёт находит горлышки (проёмы);
  2) по ЛАТЕНТАМ JEPA — кластеризуем латенты в прото-состояния, связываем по
     временным переходам и находим горлышки в латентном пространстве (без карты).

Запуск: python run_spectral_options.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.world.gridworld import GridWorld
from thinking_system.world.features import CellFeatures
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.agent.spectral_options import GridSpectralDiscoverer, SpectralOptions, kmeans


def main():
    grid = rooms_world()
    rc = lambda s: (s // grid.size, s % grid.size)
    true_doors = {grid.sid(c) for c in [(1, 3), (5, 3), (3, 1), (3, 5)]}
    print(f"▶ Мир-комнаты {grid.size}×{grid.size}; подцели из СПЕКТРА графа (eigenoptions)\n")

    # 1) спектр графа опыта
    disc = GridSpectralDiscoverer(grid, seed=0)
    disc.explore(8000)
    bott = disc.bottleneck_cells(4)
    print("1) ПО ГРАФУ ОПЫТА — нормированный лапласиан, спектральный счёт горлышка:")
    print(f"   найдены: {sorted(rc(s) for s in bott)}  → совпало с проёмами {len(set(bott) & true_doors)}/4")
    print(f"   узкий разрез Фидлера: {disc.so.cut_edges()} рёбер пересекают границу (бутылочное горло)")

    fb = disc.fiedler_by_cell()
    quad = {"TL": (lambda r, c: r < 3 and c < 3), "TR": (lambda r, c: r < 3 and c > 3),
            "BL": (lambda r, c: r > 3 and c < 3), "BR": (lambda r, c: r > 3 and c > 3)}
    print("   средний Фидлер по комнатам (медленная мода разделяет мир):")
    print("   " + "  ".join(f"{n}={np.mean([fb[s] for s in fb if q(*rc(s))]):+.2f}" for n, q in quad.items()))
    print(f"   полюса диффузионной моды (цели eigenoption): {sorted(rc(s) for s in disc.pole_cells(1))}")

    # 2) спектр в латентном пространстве JEPA (без карты)
    print("\n2) ПО ЛАТЕНТАМ JEPA — кластеризуем латенты, связываем временными переходами:")
    feats = CellFeatures(grid.n_states, dim=24, noise=0.15, seed=0)
    jepa = LatentWorldModel(24, 4, latent_dim=16, lr=5e-4, var_coef=0.02, seed=0)
    rng = np.random.default_rng(0)

    def move(s, a):
        r, c = divmod(s, grid.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        return grid.sid((nr, nc)) if 0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls else s

    O, A, NO, s = [], [], [], grid.sid(grid.start)
    for _ in range(10000):
        a = int(rng.integers(4)); sp = move(s, a)
        O.append(feats.observe(s)); A.append(a); NO.append(feats.observe(sp)); s = sp
    O, A, NO = np.array(O), np.array(A), np.array(NO)
    for _ in range(60):
        ii = rng.permutation(len(O))
        for k in range(0, len(O), 128):
            b = ii[k:k + 128]; jepa.update(O[b], A[b], NO[b])

    s, lat, truth = grid.sid(grid.start), [], []
    for _ in range(12000):                                   # агент видит только латенты, не id клеток
        truth.append(s); lat.append(jepa.encode(feats.observe(s))[0]); s = move(s, int(rng.integers(4)))
    lat = np.array(lat)
    lab, _ = kmeans(lat, 49, seed=1)                         # прото-состояния из латентов
    so = SpectralOptions(49)
    so.add_path(lab)                                         # рёбра — временные переходы латентов
    proto_cell = {j: max((truth[i] for i in np.where(lab == j)[0]), key=list(truth[i] for i in np.where(lab == j)[0]).count)
                  for j in range(49) if (lab == j).any()}
    found = {proto_cell[j] for j in so.bottlenecks(4) if j in proto_cell}
    near = sum(any(abs(rc(s)[0] - rc(d)[0]) + abs(rc(s)[1] - rc(d)[1]) <= 1 for d in true_doors) for s in found)
    print(f"   горлышки в латентном пространстве: {sorted(rc(s) for s in found)}")
    print(f"   из них у проёма (≤1 клетки): {near}/{len(found)} — спектр локализует горло и без карты")

    print("\n── Итог ──")
    print("   Подцели приходят из СПЕКТРА мира: вектор Фидлера разделяет комнаты, а счёт")
    print("   разноса соседей в собственном вложении указывает на горлышки. По графу опыта")
    print("   — точно проёмы; по латентам JEPA — там же, без дискретной карты.")


if __name__ == "__main__":
    main()
