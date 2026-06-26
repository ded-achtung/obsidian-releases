"""Тесты Q-learning опций: обучение из опыта, достижение подцели, устойчивость."""

from __future__ import annotations

import numpy as np

from thinking_system.agent.qoption import QOption, QOptionLibrary
from thinking_system.world.rooms import rooms_world


def _free(grid):
    return [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]


def test_qoption_learns_from_experience_and_reaches() -> None:
    grid = rooms_world()
    sub = grid.sid((3, 5))
    opt = QOption(grid, sub, seed=0)
    curve = opt.train(1500)
    assert np.mean(curve[:100]) > np.mean(curve[-100:])     # шагов до подцели меньше = учится из опыта

    free = [s for s in _free(grid) if s != sub]
    reached = sum(opt.reach(s, seed=0) for s in free[:15])
    assert reached >= 14                                    # доходит почти из всех стартов


def test_qoption_robust_to_perturbation() -> None:
    grid = rooms_world()
    sub = grid.sid((3, 1))
    opt = QOption(grid, sub, seed=0)
    opt.train(1500)
    free = [s for s in _free(grid) if s != sub]
    rng = np.random.default_rng(0)
    reached = sum(opt.reach(int(rng.choice(free)), perturb=0.15, seed=int(rng.integers(10 ** 6))) for _ in range(30))
    assert reached >= 26                                    # политика восстанавливается после сбоев


def test_qoption_library_composition() -> None:
    grid = rooms_world()
    doorways = [grid.sid(c) for c in [(1, 3), (3, 5)]]
    lib = QOptionLibrary(grid, doorways, seed=0)
    lib.train(1500)
    assert lib.compose(grid.sid((0, 0)), doorways, seed=0)  # цепочка опций доходит
