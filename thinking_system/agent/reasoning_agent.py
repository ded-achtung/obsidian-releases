"""ReasoningAgent — агент, который УЧИТ свой мир РАССУЖДЕНИЕМ, а не перебором.

Капстоун, смыкающий рассуждение с остальной системой. Вместо тысяч проб RL агент:

    наблюдает горстку шагов → ИНДУЦИРУЕТ правило динамики (reasoning/grounded) →
    мысленно прогоняет правило по ВСЕМУ миру (полная модель без посещения клеток) →
    СПЕКТРАЛЬНО открывает подцели по воображённому графу (reasoning ⊕ discovery) →
    ПЛАНИРУЕТ к языковым целям по выученной модели.

Из ~20 наблюдений он строит модель, эквивалентную истинной динамике, и сразу
планирует куда угодно — на порядки меньше опыта, чем у чисто-RL агента
(LifelongAgent: ~12000 шагов + тысячи Q-эпизодов). Это «учиться многому из малого»
структурой и выводом, сомкнутое с восприятием, открытием подцелей и языком.
"""

from __future__ import annotations

from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.grounded import observe, induce_dynamics, WorldRule
from thinking_system.agent.spectral_options import SpectralOptions
from thinking_system.language.doorways import train_doorway_classifier, doorway_cell


class ReasoningAgent:
    """Долгоживущий агент: индукция динамики → модель → подцели → план к языку.

    Args:
        grid: мир (агент знает препятствия из восприятия, но НЕ знает динамику).
        seed: зерно блуждания/языка.
    """

    def __init__(self, grid: GridWorld, *, seed: int = 0) -> None:
        self.g = grid
        self.free = grid.free_cells()
        self.idx = {c: i for i, c in enumerate(self.free)}
        self.model: WorldRule | None = None
        self.observations = 0
        self.clf, _, self.held = train_doorway_classifier(seed=seed)   # язык: команда → проём
        self.seed = seed

    # ── 1) выучить мир рассуждением ──────────────────────────────────────────────
    def learn_world(self, steps: int = 20) -> int:
        """Понаблюдать несколько шагов и ИНДУЦИРОВАТЬ правило динамики на каждое действие."""
        obs = observe(self.g, steps, seed=self.seed)
        self.observations += steps
        self.model = WorldRule(self.g, induce_dynamics(obs))
        return len(self.model.rules)

    # ── 2) полная модель из малого: прогнать правило по всему миру ────────────────
    def imagined_transitions(self):
        """Все переходы (клетка, действие, след.) по ВЫУЧЕННОМУ правилу — без посещения."""
        assert self.model is not None
        for s in self.free:
            for a in self.model.rules:
                sp = self.model.predict(s, a)
                if sp != s:
                    yield s, a, sp

    # ── 3) открыть подцели по воображённому графу (reasoning ⊕ spectral) ──────────
    def discover_subgoals(self, k: int = 4) -> list[tuple[int, int]]:
        """Спектральные горлышки по графу, ПОРОЖДЁННОМУ моделью (а не блужданием)."""
        so = SpectralOptions(len(self.free))
        for s, _, sp in self.imagined_transitions():
            so.add_undirected(self.idx[s], self.idx[sp])
        return [self.free[i] for i in so.bottlenecks(k)]

    # ── 4) планировать и понимать язык по выученной модели ───────────────────────
    def plan_to(self, start: tuple[int, int], goal: tuple[int, int]) -> list[int] | None:
        assert self.model is not None
        return self.model.plan(start, goal)

    def _follow(self, start: tuple[int, int], path: list[int]) -> tuple[int, int]:
        s = start
        for a in path:
            s = self.model.predict(s, a)
        return s

    def obey(self, command: str, start: tuple[int, int]) -> dict:
        """Понять команду → проём → СПЛАНИРОВАТЬ путь по модели → исполнить."""
        idx, conf = self.clf.predict(command)
        goal = doorway_cell(idx)
        path = self.plan_to(start, goal)
        reached = path is not None and self._follow(start, path) == goal
        return {"command": command, "goal": goal, "reached": reached, "steps": len(path) if path else None, "confidence": conf}

    def command_success(self, commands) -> float:
        import numpy as np

        rng = np.random.default_rng(self.seed)
        ok = 0
        for cmd, _ in commands:
            start = self.free[int(rng.integers(len(self.free)))]
            ok += int(self.obey(cmd, start)["reached"])
        return ok / len(commands)

    # ── рассуждать о мире выученной моделью ───────────────────────────────────────
    def predict(self, s: tuple[int, int], a: int) -> tuple[int, int]:
        assert self.model is not None
        return self.model.predict(s, a)

    def distance(self, a: tuple[int, int], b: tuple[int, int]) -> int | None:
        path = self.plan_to(a, b)
        return len(path) if path is not None else None
