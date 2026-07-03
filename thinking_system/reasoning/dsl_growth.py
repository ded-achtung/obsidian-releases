"""Накопление DSL: агент САМ расширяет язык правил под новые миры.

Базовый язык (шаги ±1 по одной оси) описывает не любую динамику. Если правило не
выводится поиском в текущем языке, агент СИНТЕЗИРУЕТ новый примитив прямо из данных
(постоянное смещение Δ из наблюдений) и добавляет его в язык. Тогда:

  • мир с «прыжком конём» (Δ=(2,1)) базовый язык на малой глубине НЕ выразит, а
    синтез даёт примитив из ОДНОГО примера — без перебора;
  • в следующем мире с тем же ходом примитив уже в языке → решается за глубину 1.

Язык растёт с опытом — агент становится способнее (ср. рост библиотеки в induction).
Индуктивное смещение синтеза: «действия — это сдвиги»; недвижущуюся/непостоянную
динамику он не синтезирует (тогда — гибрид/обучение из опыта).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.induction import Primitive, Program, induce
from thinking_system.reasoning.grounded import pair_primitives, WorldRule


class ShiftEnv:
    """Мир, где действие сдвигает на фиксированный вектор moves[a] (диагональ/прыжок/конь)."""

    def __init__(self, grid: GridWorld, moves: list[tuple[int, int]]) -> None:
        self.g = grid
        self.moves = moves
        self.free = grid.free_cells()

    def transition(self, s: tuple[int, int], a: int) -> tuple[int, int]:
        return self.g.move_from(s, self.moves[a])


def synthesize_shift(name: str, moves: list[tuple[tuple[int, int], tuple[int, int]]]) -> Primitive | None:
    """Синтезировать примитив-сдвиг из наблюдений: доминирующее постоянное смещение Δ."""
    deltas = Counter((sp[0] - s[0], sp[1] - s[1]) for s, sp in moves)
    nz = [(d, n) for d, n in deltas.items() if d != (0, 0)]
    if not nz:
        return None
    (dr, dc), _ = max(nz, key=lambda x: x[1])
    return Primitive(name, lambda s, dr=dr, dc=dc: (s[0] + dr, s[1] + dc))


class GrowingLanguage:
    """Язык правил, который РАСТЁТ: не вывелось — синтезируем примитив и запоминаем."""

    def __init__(self, base: list[Primitive] | None = None) -> None:
        self.prims = list(base if base is not None else pair_primitives())
        self.synthesized: list[str] = []

    def learn_action(self, examples, *, max_depth: int = 2) -> tuple[Program | None, bool]:
        """Вывести правило действия текущим языком; иначе синтезировать примитив. (программа, синтез?)."""
        moves = [(s, sp) for s, sp in examples if sp != s]
        if not moves:
            return None, False
        prog = induce(moves, self.prims, max_depth=max_depth)
        if prog is not None:
            return prog, False                                # выразимо текущим языком (в т.ч. реюз синтезированного)
        prim = synthesize_shift(f"Δ{len(self.synthesized)}", moves)
        if prim is None:
            return None, False
        self.prims.append(prim)                               # язык вырос
        self.synthesized.append(prim.name)
        return Program([prim]), True

    def learn_world(self, grid: GridWorld, observations, *, max_depth: int = 2) -> tuple[WorldRule, int]:
        """Выучить динамику мира; вернуть (модель, сколько примитивов синтезировано здесь)."""
        by_a: dict[int, list] = defaultdict(list)
        for s, a, sp in observations:
            by_a[a].append((s, sp))
        rules: dict[int, Program] = {}
        n_synth = 0
        for a, ex in by_a.items():
            prog, synthed = self.learn_action(ex, max_depth=max_depth)
            if prog is not None:
                rules[a] = prog
                n_synth += int(synthed)
        return WorldRule(grid, rules), n_synth
