"""MetaAgent — один агент, смыкающий ОБА капстоуна: рассуждение И обучение из опыта.

Брейн-подобный арбитраж model-based ↔ model-free (ср. Daw et al., 2005): дешёвое
рассуждение, когда оно описывает мир, и дорогое обучение методом проб, когда нет.

    наблюдает горстку шагов → ПРОБУЕТ вывести правило динамики (рассуждение) →
      • правило ОПИСЫВАЕТ мир → планирует по модели (почти даром, ~десятки шагов);
      • правило НЕ подходит   → откатывается к RL: открывает подцели и учит
                                навыки методом проб (надёжно, но дорого).
Язык и открытие подцелей общие для обоих режимов. Агент сам выбирает путь по тому,
прошла ли проверка выученной модели на свежих наблюдениях.

GridEnv даёт переходы клетка×действие; «перепутанная» среда (поклеточная
перестановка действий) — мир, который простым DSL описать нельзя → нужен RL.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.grounded import induce_dynamics, WorldRule
from thinking_system.language.doorways import train_doorway_classifier, doorway_cell


class GridEnv:
    """Среда переходов клетка×действие; опционально — поклеточная перестановка действий."""

    def __init__(self, grid: GridWorld, *, scramble_seed: int | None = None, n_scramble: int | None = None) -> None:
        self.g = grid
        self.free = grid.free_cells()
        self.perm: dict[tuple[int, int], np.ndarray] | None = None
        if scramble_seed is not None:
            rng = np.random.default_rng(scramble_seed)
            cells = self.free if n_scramble is None else \
                [self.free[i] for i in rng.choice(len(self.free), n_scramble, replace=False)]
            self.perm = {c: rng.permutation(4) for c in cells}      # перепутанные клетки (все или часть)

    def transition(self, s: tuple[int, int], a: int) -> tuple[int, int]:
        if self.perm is not None and s in self.perm:
            a = int(self.perm[s][a])
        return self.g.move_from(s, GridWorld.MOVES[a])


class MetaAgent:
    """Агент, выбирающий между рассуждением (model-based) и обучением (model-free)."""

    def __init__(self, env: GridEnv, *, seed: int = 0) -> None:
        self.env = env
        self.g = env.g
        self.free = env.free
        self.seed = seed
        self.clf, _, self.held = train_doorway_classifier(seed=seed)
        self.mode: str | None = None
        self.cost = 0
        self.model: WorldRule | None = None
        self.skills: dict[tuple[int, int], np.ndarray] = {}
        self.subgoals: list[tuple[int, int]] = []

    def _walk(self, steps: int, seed: int):
        rng = np.random.default_rng(seed)
        s = self.g.start
        obs = []
        for _ in range(steps):
            a = int(rng.integers(4))
            sp = self.env.transition(s, a)
            obs.append((s, a, sp))
            s = sp
        return obs

    # ── арбитраж: сначала пробуем рассуждение, потом откатываемся к RL ────────────
    def live(self, *, probe: int = 24, verify: int = 60, rl_steps: int = 4000, rl_episodes: int = 1500) -> dict:
        rules = induce_dynamics(self._walk(probe, self.seed))            # попытка вывести правило мира
        model = WorldRule(self.g, rules)
        test = self._walk(verify, self.seed + 1)
        acc = float(np.mean([model.predict(s, a) == sp for s, a, sp in test])) if rules else 0.0
        self.cost = probe + verify
        self.diag = {"rules": len(rules), "verify_acc": acc}            # почему выбран режим (прозрачность)
        if len(rules) == 4 and acc >= 0.95:                             # правило ОПИСЫВАЕТ мир → рассуждаем
            self.mode = "reasoning"
            self.model = model
            self.subgoals = self._discover_from_model()
            return self.report()

        self.mode = "trial-and-error"                                   # правило не подходит → учимся из опыта
        self.subgoals = self._discover_by_walk(rl_steps)
        self.cost += rl_steps
        for sub in self.subgoals:
            self.skills[sub] = self._qlearn(sub, episodes=rl_episodes, seed=self.seed)
        return self.report()

    # ── открытие подцелей (общая способность) ────────────────────────────────────
    @staticmethod
    def _bottlenecks(adj: dict, free: list, k: int) -> list:
        bc: Counter = Counter()
        for src in free:
            prev = {src: None}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in prev:
                        prev[v] = u
                        q.append(v)
            for dst in free:
                if dst == src or dst not in prev:
                    continue
                u = prev[dst]
                while u is not None and u != src:
                    bc[u] += 1
                    u = prev[u]
        narrow = [c for c in free if len(adj[c]) == 2]
        return sorted(narrow, key=lambda c: -bc[c])[:k]

    def _discover_from_model(self, k: int = 4) -> list:
        adj = {c: set() for c in self.free}                            # граф из ВЫУЧЕННОГО правила (без обхода)
        for s in self.free:
            for a in self.model.rules:
                sp = self.model.predict(s, a)
                if sp != s:
                    adj[s].add(sp); adj[sp].add(s)
        return self._bottlenecks(adj, self.free, k)

    def _discover_by_walk(self, steps: int, k: int = 4) -> list:
        rng = np.random.default_rng(self.seed)
        adj = {c: set() for c in self.free}
        s = self.g.start
        for _ in range(steps):                                         # граф из реальных переходов среды
            sp = self.env.transition(s, int(rng.integers(4)))
            if sp != s:
                adj[s].add(sp); adj[sp].add(s)
            s = sp
        return self._bottlenecks(adj, self.free, k)

    # ── обучение навыка методом проб (model-free), через переходы среды ───────────
    def _qlearn(self, goal: tuple[int, int], *, episodes: int, eps: float = 0.2, alpha: float = 0.5,
                gamma: float = 0.95, max_steps: int = 100, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed + hash(goal) % 1000)
        Q = np.zeros((self.g.n_states, 4))
        starts = [c for c in self.free if c != goal]
        for _ in range(episodes):
            s = starts[int(rng.integers(len(starts)))]
            for _ in range(max_steps):
                sid = self.g.sid(s)
                a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[sid]))
                sp = self.env.transition(s, a)
                done = sp == goal
                r = 1.0 if done else -0.01
                Q[sid, a] += alpha * (r + (0.0 if done else gamma * float(np.max(Q[self.g.sid(sp)]))) - Q[sid, a])
                s = sp
                if done:
                    break
        return Q

    # ── использование: язык → цель, исполнение в любом режиме ────────────────────
    def obey(self, command: str, start: tuple[int, int], *, max_steps: int = 120) -> dict:
        idx, conf = self.clf.predict(command)
        goal = doorway_cell(idx)
        if self.mode == "reasoning":
            path = self.model.plan(start, goal)
            s = start
            if path is not None:
                for a in path:
                    s = self.env.transition(s, a)
            return {"command": command, "goal": goal, "reached": s == goal and path is not None,
                    "steps": len(path) if path else None}
        Q = self.skills.get(goal)
        if Q is None:
            return {"command": command, "goal": goal, "reached": False, "steps": None}
        s = start
        for t in range(max_steps):
            if s == goal:
                return {"command": command, "goal": goal, "reached": True, "steps": t}
            s = self.env.transition(s, int(np.argmax(Q[self.g.sid(s)])))
        return {"command": command, "goal": goal, "reached": s == goal, "steps": max_steps}

    def command_success(self, commands) -> float:
        rng = np.random.default_rng(self.seed + 7)
        ok = 0
        for cmd, _ in commands:
            start = self.free[int(rng.integers(len(self.free)))]
            ok += int(self.obey(cmd, start)["reached"])
        return ok / len(commands)

    def report(self) -> dict:
        return {"mode": self.mode, "cost": self.cost, "subgoals": self.subgoals,
                "commands": self.command_success(self.held[:120])}
