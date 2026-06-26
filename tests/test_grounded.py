"""Тесты зазёмленного рассуждения: индукция динамики из наблюдений, обобщение, план."""

from __future__ import annotations

from thinking_system.world.gridworld import GridWorld
from thinking_system.world.rooms import rooms_world
from thinking_system.reasoning.grounded import observe, induce_dynamics, WorldRule


def _true_next(grid, s, a):
    r, c = s
    dr, dc = GridWorld.MOVES[a]
    nr, nc = r + dr, c + dc
    return (nr, nc) if 0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls else s


def test_induces_dynamics_and_generalizes_to_whole_world() -> None:
    grid = rooms_world()
    obs = observe(grid, 40, seed=0)
    rules = induce_dynamics(obs)
    assert len(rules) == 4                                  # вывел правило на каждое действие из наблюдений
    wr = WorldRule(grid, rules)
    free = [(s // grid.size, s % grid.size) for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    ok = sum(wr.predict(s, a) == _true_next(grid, s, a) for s in free for a in range(4))
    assert ok == len(free) * 4                              # модель точна на ВСЕХ клетках (обобщение из малого)


def test_sample_efficiency_beats_tabular_coverage() -> None:
    grid = rooms_world()
    obs = observe(grid, 40, seed=0)
    free = [(s // grid.size, s % grid.size) for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    seen = {(s, a) for s, a, _ in obs}
    assert len(seen) < 0.3 * len(free) * 4                  # наблюдений мало (табличная память покрыла бы <30%)


def test_plans_with_learned_model() -> None:
    grid = rooms_world()
    wr = WorldRule(grid, induce_dynamics(observe(grid, 40, seed=0)))
    path = wr.plan((0, 0), (6, 6))
    assert path is not None                                 # по выученной модели находит путь к цели
    # путь действительно ведёт в цель по истинной динамике
    s = (0, 0)
    for a in path:
        s = _true_next(grid, s, a)
    assert s == (6, 6)
