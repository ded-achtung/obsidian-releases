"""LifelongAgent — единый долгоживущий агент, накапливающий библиотеку навыков.

Замыкает весь контур в одном существе, которое со временем становится умнее:

    исследует мир → СПЕКТРАЛЬНО открывает подцели-горлышки (из опыта) →
    учит к ним навыки ИЗ ВОСПРИЯТИЯ (Q над латентами/признаками) →
    принимает команды на ЯЗЫКЕ и исполняет нужный навык.

«Жизнь» идёт фазами. За фазу агент добирает опыта и консолидирует НОВЫЙ навык
(ограниченная ёмкость за раз). Библиотека навыков растёт, покрытие мира и доля
выполнимых команд поднимаются — система тем способнее, чем дольше живёт.

Восприятие ВНЕДРЯЕТСЯ извне (encode, n_features): тайлы или латенты JEPA — навык
в любом случае функция от восприятия, а не от привилегированного id клетки.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.agent.latent_qoption import LatentQOption
from thinking_system.agent.spectral_options import GridSpectralDiscoverer
from thinking_system.language.doorways import doorway_cell, train_doorway_classifier, DOORWAYS


class LifelongAgent:
    """Долгоживущий агент: исследование → открытие подцелей → навыки → язык.

    Args:
        grid: мир.
        encode: восприятие (state → вектор признаков); n_features: его размер.
        alpha: скорость Q-обучения навыка; seed: зерно.
    """

    def __init__(self, grid: GridWorld, encode, n_features: int, *, alpha: float = 0.1, seed: int = 0) -> None:
        self.g = grid
        self.encode = encode
        self.nf = n_features
        self.alpha = alpha
        self.seed = seed
        self.skills: dict[int, LatentQOption] = {}      # библиотека: клетка-подцель → навык
        self.disc = GridSpectralDiscoverer(grid, seed=seed)
        self.clf, _, self.held = train_doorway_classifier(seed=seed)  # язык: команда → проём
        self.experience = 0

    # ── жизнь по фазам ───────────────────────────────────────────────────────────
    def live_phase(self, *, explore_steps: int = 3000, episodes: int = 5000, max_new: int = 1) -> dict:
        """Фаза жизни: добрать опыт, открыть горлышки, консолидировать новый навык."""
        self.disc.explore(explore_steps)
        self.experience += explore_steps
        new = 0
        for sub in self.disc.bottleneck_cells(4):           # спектральные подцели из накопленного опыта
            if sub not in self.skills and new < max_new:
                opt = LatentQOption(self.g, sub, self.encode, self.nf, alpha=self.alpha, seed=self.seed + len(self.skills))
                opt.train(episodes)                          # навык ИЗ ВОСПРИЯТИЯ (Q над encode)
                self.skills[sub] = opt
                new += 1
        return {"experience": self.experience, "skills": len(self.skills),
                "coverage": self.coverage(), "obey": self.command_success()}

    # ── использование выученного ─────────────────────────────────────────────────
    def obey(self, command: str, start: int, *, seed: int = 0) -> dict:
        """Понять команду → выбрать навык из библиотеки → исполнить (или признать незнание)."""
        idx, conf = self.clf.predict(command)
        cell = self.g.sid(doorway_cell(idx))
        known = cell in self.skills
        reached = bool(known and self.skills[cell].reach(start, seed=seed))
        return {"command": command, "doorway": idx, "known": known, "reached": reached, "confidence": conf}

    def _bfs(self, src: int) -> dict[int, int]:
        A = self.disc.so.A
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in np.where(A[u] > 0)[0]:
                if int(v) not in d:
                    d[int(v)] = d[u] + 1
                    q.append(int(v))
        return d

    def coverage(self, *, local: int = 4) -> float:
        """Доля клеток, достижимых через ближайший навык-горлышко + короткий доход."""
        subs = list(self.skills)
        if not subs:
            return 0.0
        idx = self.disc.idx
        dist = {s: self._bfs(idx[s]) for s in subs}
        ok = tot = 0
        for g in self.disc.free:
            if g in subs:
                continue
            tot += 1
            v = min(subs, key=lambda s: dist[s].get(idx[g], 10 ** 9))
            if dist[v].get(idx[g], 10 ** 9) <= local:
                ok += 1
        return ok / tot

    def command_success(self, *, trials: int = 6, seed: int = 0) -> float:
        """Доля языковых команд (по 4 направлениям), которые агент уже умеет выполнять."""
        rng = np.random.default_rng(seed)
        free = self.disc.free
        ok = tot = 0
        for idx in range(len(DOORWAYS)):
            cell = self.g.sid(doorway_cell(idx))
            for _ in range(trials):
                tot += 1
                start = int(rng.choice([f for f in free if f != cell]))
                ok += int(cell in self.skills and self.skills[cell].reach(start, seed=int(rng.integers(10 ** 6))))
        return ok / tot
