"""Иерархическая память: эпизодическая → семантическая → процедурная (навыки).

  • ЭПИЗОДИЧЕСКАЯ — конкретные переходы опыта (s, a, s') и счётчики посещений;
  • СЕМАНТИЧЕСКАЯ — консолидированная карта мира + ОРИЕНТИРЫ (бутылочные горлышки,
    всплывающие как часто-посещаемые при исследовании) и граф ориентиров;
  • ПРОЦЕДУРНАЯ — НАВЫКИ: переиспользуемые маршруты (последовательности действий)
    между соседними ориентирами.

Планирование иерархическое: высокоуровнево — по ориентирам (несколько шагов),
низкоуровнево — выполнить навыки. Небольшой набор навыков покрывает все цели.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from thinking_system.world.gridworld import GridWorld

_REV = {0: 1, 1: 0, 2: 3, 3: 2}  # обратные действия (↑↔↓, ←↔→)


class HierarchicalMemory:
    """Трёхуровневая память с консолидацией опыта в ориентиры и навыки."""

    def __init__(self, grid: GridWorld) -> None:
        self.g = grid
        self.episodic: list[tuple[int, int, int]] = []   # (s, a, s')
        self.visits: Counter = Counter()                  # клетка → посещений
        self.map: dict[tuple[int, int], int] = {}         # (s, a) → s'  (семантическая карта)
        self.landmarks: list[int] = []
        self.skills: dict[tuple[int, int], list[int]] = {}  # (Li, Lj) → маршрут действий
        self.adj: dict[int, dict[int, int]] = {}            # граф ориентиров

    def _move(self, s: int, a: int) -> int:
        return self.g.move_sid(s, a)

    def explore(self, steps: int, *, seed: int = 0) -> None:
        """ЭПИЗОДИЧЕСКИ: случайно блуждать, накапливая переходы и посещения."""
        rng = np.random.default_rng(seed)
        s = self.g.sid(self.g.start)
        for _ in range(steps):
            a = int(rng.integers(4))
            sp = self._move(s, a)
            self.episodic.append((s, a, sp))
            self.map[(s, a)] = sp
            self.visits[sp] += 1
            s = sp

    @staticmethod
    def _route(prev: dict, target: int) -> list[int]:
        acts: list[int] = []
        u = target
        while prev[u] is not None:
            p, a = prev[u]
            acts.append(a)
            u = p
        return acts[::-1]

    def _bfs_to(self, start: int, is_target) -> tuple[int | None, list[int]]:
        if is_target(start):
            return start, []
        prev = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            for a in range(4):
                v = self.map.get((u, a), u)
                if v != u and v not in prev:
                    prev[v] = (u, a)
                    if is_target(v):
                        return v, self._route(prev, v)
                    q.append(v)
        return None, []

    def _betweenness(self) -> Counter:
        """Центральность по посредничеству: через сколько кратчайших путей идёт клетка."""
        cells = sorted({s for s, _ in self.map} | set(self.map.values()))
        adj: dict[int, list[int]] = {c: [] for c in cells}
        for (s, a), sp in self.map.items():
            if sp != s and sp not in adj[s]:
                adj[s].append(sp)
        bc: Counter = Counter()
        for src in cells:
            prev: dict[int, int | None] = {src: None}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in prev:
                        prev[v] = u
                        q.append(v)
            for dst in cells:
                if dst == src or dst not in prev:
                    continue
                u = prev[dst]
                while u is not None and u != src:
                    bc[u] += 1  # промежуточная клетка кратчайшего пути
                    u = prev[u]
        return bc

    def consolidate(self, *, n_landmarks: int = 4) -> None:
        """СЕМАНТИЧЕСКИ: выделить ориентиры (бутылочные горлышки) и навыки между ними."""
        self.betweenness = self._betweenness()
        self.landmarks = [c for c, _ in self.betweenness.most_common(n_landmarks)]
        lset = set(self.landmarks)
        self.adj = {L: {} for L in self.landmarks}
        self.skills = {}
        for L in self.landmarks:  # рёбра графа: BFS с барьером на других ориентирах
            prev = {L: None}
            q = deque([L])
            while q:
                u = q.popleft()
                for a in range(4):
                    v = self.map.get((u, a), u)
                    if v != u and v not in prev:
                        prev[v] = (u, a)
                        if v in lset:
                            route = self._route(prev, v)
                            self.adj[L][v] = len(route)
                            self.skills[(L, v)] = route  # ПРОЦЕДУРНО: навык-маршрут
                        else:
                            q.append(v)

    def _dijkstra(self, src: int, dst: int) -> list[int] | None:
        import heapq

        dist = {src: 0}
        prev: dict[int, int] = {}
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                seq = [dst]
                while seq[-1] != src:
                    seq.append(prev[seq[-1]])
                return seq[::-1]
            if d > dist.get(u, 1e9):
                continue
            for v, w in self.adj.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, 1e9):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return None

    def plan(self, start: int, goal: int) -> dict | None:
        """Иерархический план: вход → ближний ориентир → навыки по графу → цель."""
        lset = set(self.landmarks)
        Ls, r_in = self._bfs_to(start, lambda c: c in lset)
        Lg, r_out_rev = self._bfs_to(goal, lambda c: c in lset)
        if Ls is None or Lg is None:
            return None
        r_out = [_REV[a] for a in reversed(r_out_rev)]  # маршрут Lg → цель
        lm_seq = self._dijkstra(Ls, Lg)
        if lm_seq is None:
            return None
        actions = list(r_in)
        for li, lj in zip(lm_seq, lm_seq[1:]):
            actions += self.skills[(li, lj)]
        actions += r_out
        return {"actions": actions, "landmarks": lm_seq, "n_high": len(lm_seq), "n_steps": len(actions)}
