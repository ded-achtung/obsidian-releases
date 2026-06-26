"""Спектральное открытие подцелей (eigenoptions): горлышки из геометрии графа опыта.

Вместо дискретного правила «узкая клетка» (option_discovery.py) — СПЕКТР графа. По
опыту строится граф переходов; нормированный лапласиан даёт собственные векторы —
диффузионные моды мира. Низкочастотные моды (вектор Фидлера) медленно меняются
ВНУТРИ комнат и резко — на ГОРЛЫШКАХ. Подцель-горлышко = клетка, чьи соседи далеко
разнесены в спектральном вложении (мост между кластерами). Экстремумы мод —
«полюса», естественные цели eigenoption (Machado et al., 2017).

Граф можно строить не из карты, а из ЛАТЕНТОВ: кластеризовать латенты JEPA в
прото-состояния и связать их ВРЕМЕННЫМИ переходами (см. run_spectral_options.py) —
тогда горлышки находятся прямо в латентном пространстве, без дискретной карты.
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld


class SpectralOptions:
    """Граф опыта + спектральный анализ (лапласово вложение, Фидлер, горлышки, полюса)."""

    def __init__(self, n_nodes: int) -> None:
        self.n = n_nodes
        self.A = np.zeros((n_nodes, n_nodes))

    def add_undirected(self, u: int, v: int) -> None:
        if u != v:
            self.A[u, v] = self.A[v, u] = 1.0

    def add_path(self, seq) -> None:
        """Рёбра из ВРЕМЕННОЙ последовательности состояний (поток опыта)."""
        for u, v in zip(seq, seq[1:]):
            self.add_undirected(int(u), int(v))

    def embedding(self, k: int = 3) -> np.ndarray:
        """Лапласово вложение: k нетривиальных собственных векторов норм. лапласиана."""
        d = self.A.sum(1)
        Dm = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-9)))
        L = np.eye(self.n) - Dm @ self.A @ Dm
        w, V = np.linalg.eigh(L)
        self.evals = w
        return V[:, 1:1 + k]

    def fiedler(self) -> np.ndarray:
        """Вектор Фидлера (2-й собственный) — медленная мода, разделяющая кластеры."""
        return self.embedding(1)[:, 0]

    def _adj(self) -> list[np.ndarray]:
        return [np.where(self.A[i] > 0)[0] for i in range(self.n)]

    def bottleneck_scores(self, *, k_emb: int = 3) -> np.ndarray:
        """Счёт горлышка = разнос соседей в спектральном вложении (мост между кластерами)."""
        emb = self.embedding(k_emb)
        adj = self._adj()
        score = np.zeros(self.n)
        for i in range(self.n):
            nb = adj[i]
            if len(nb) >= 2:
                score[i] = max(np.linalg.norm(emb[a] - emb[b]) for a in nb for b in nb if a < b)
        return score

    def bottlenecks(self, k: int = 4, *, k_emb: int = 3) -> list[int]:
        return list(np.argsort(-self.bottleneck_scores(k_emb=k_emb))[:k])

    def eigenoption_poles(self, m: int = 2) -> list[int]:
        """Полюса диффузионных мод: экстремумы первых m собственных векторов (цели eigenoption)."""
        emb = self.embedding(max(m, 1))
        poles: list[int] = []
        for j in range(m):
            poles += [int(emb[:, j].argmax()), int(emb[:, j].argmin())]
        return poles

    def cut_edges(self, vec: np.ndarray | None = None) -> int:
        """Число рёбер, пересекающих знаковую границу вектора (узкий разрез у Фидлера)."""
        v = self.fiedler() if vec is None else vec
        return int(sum(1 for i in range(self.n) for j in range(i + 1, self.n) if self.A[i, j] > 0 and v[i] * v[j] < 0))


class GridSpectralDiscoverer:
    """Спектральное открытие горлышек в сетке: блуждание → граф опыта → eigenoptions."""

    def __init__(self, grid: GridWorld, *, seed: int = 0) -> None:
        self.g = grid
        self.free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
        self.idx = {s: i for i, s in enumerate(self.free)}
        self.so = SpectralOptions(len(self.free))
        self.rng = np.random.default_rng(seed)

    def _move(self, s: int, a: int) -> int:
        r, c = divmod(s, self.g.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.g.size and 0 <= nc < self.g.size and (nr, nc) not in self.g.walls:
            return self.g.sid((nr, nc))
        return s

    def explore(self, steps: int) -> None:
        s = self.g.sid(self.g.start)
        for _ in range(steps):
            sp = self._move(s, int(self.rng.integers(4)))
            if sp != s:
                self.so.add_undirected(self.idx[s], self.idx[sp])
            s = sp

    def bottleneck_cells(self, k: int = 4) -> list[int]:
        return [self.free[i] for i in self.so.bottlenecks(k)]

    def pole_cells(self, m: int = 2) -> list[int]:
        return [self.free[i] for i in self.so.eigenoption_poles(m)]

    def fiedler_by_cell(self) -> dict[int, float]:
        f = self.so.fiedler()
        return {self.free[i]: float(f[i]) for i in range(len(self.free))}


def kmeans(X: np.ndarray, k: int, *, iters: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Простейший k-means (для дискретизации латентов в прото-состояния)."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab, C
