"""Тесты спектральных eigenoptions: горлышки из спектра, Фидлер, полюса, латенты."""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.world.gridworld import GridWorld
from thinking_system.world.features import CellFeatures
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.agent.spectral_options import GridSpectralDiscoverer, SpectralOptions, kmeans


TRUE_DOORS = [(1, 3), (5, 3), (3, 1), (3, 5)]


def _disc():
    disc = GridSpectralDiscoverer(rooms_world(), seed=0)
    disc.explore(8000)
    return disc


def test_spectral_bottlenecks_are_doorways() -> None:
    disc = _disc()
    true = {disc.g.sid(c) for c in TRUE_DOORS}
    assert set(disc.bottleneck_cells(4)) == true            # спектр находит ровно проёмы (без правила «узкая клетка»)


def test_fiedler_splits_world_into_balanced_halves() -> None:
    disc = _disc()
    f = disc.so.fiedler()
    pos = int((f > 0).sum())
    assert min(pos, len(f) - pos) >= 0.3 * len(f)           # медленная мода делит мир на две сопоставимые части
    assert disc.so.cut_edges() <= 0.25 * (disc.so.A.sum() / 2)  # разрез — меньшинство рёбер (бутылочное горло)


def test_eigenoption_poles_are_diffusion_distant() -> None:
    disc = _disc()
    poles = [disc.idx[c] for c in disc.pole_cells(1)]        # экстремумы Фидлера
    # BFS-расстояние по графу опыта между полюсами
    src, dst = poles[0], poles[1]
    seen = {src}
    q = deque([(src, 0)])
    dist = -1
    while q:
        u, d = q.popleft()
        if u == dst:
            dist = d
            break
        for v in np.where(disc.so.A[u] > 0)[0]:
            if v not in seen:
                seen.add(v); q.append((int(v), d + 1))
    assert dist >= 6                                         # полюса — далёкие по диффузии области (цели eigenoption)


def test_spectral_in_latent_space_localizes_bottlenecks() -> None:
    grid = rooms_world()
    rc = lambda s: (s // grid.size, s % grid.size)
    true = [grid.sid(c) for c in TRUE_DOORS]
    feats = CellFeatures(grid.n_states, dim=24, noise=0.15, seed=0)
    jepa = LatentWorldModel(24, 4, latent_dim=16, lr=5e-4, var_coef=0.02, seed=0)
    rng = np.random.default_rng(0)

    def move(s, a):
        r, c = divmod(s, grid.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        return grid.sid((nr, nc)) if 0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls else s

    O, A, NO, s = [], [], [], grid.sid(grid.start)
    for _ in range(8000):
        a = int(rng.integers(4)); sp = move(s, a)
        O.append(feats.observe(s)); A.append(a); NO.append(feats.observe(sp)); s = sp
    O, A, NO = np.array(O), np.array(A), np.array(NO)
    for _ in range(50):
        ii = rng.permutation(len(O))
        for k in range(0, len(O), 128):
            b = ii[k:k + 128]; jepa.update(O[b], A[b], NO[b])

    s, lat, truth = grid.sid(grid.start), [], []
    for _ in range(12000):
        truth.append(s); lat.append(jepa.encode(feats.observe(s))[0]); s = move(s, int(rng.integers(4)))
    lab, _ = kmeans(np.array(lat), 49, seed=1)
    so = SpectralOptions(49)
    so.add_path(lab)
    proto_cell = {j: max((truth[i] for i in np.where(lab == j)[0]), key=list(truth[i] for i in np.where(lab == j)[0]).count)
                  for j in range(49) if (lab == j).any()}
    found = [proto_cell[j] for j in so.bottlenecks(4) if j in proto_cell]
    near = sum(any(abs(rc(s)[0] - rc(d)[0]) + abs(rc(s)[1] - rc(d)[1]) <= 1 for d in true) for s in found)
    assert near >= 3                                         # горлышки в латентном пространстве — у проёмов (без карты)
