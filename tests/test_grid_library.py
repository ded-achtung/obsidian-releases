"""Тест роста библиотеки над полным перцептивным seed: имена для частых комбо."""

from __future__ import annotations

from thinking_system.reasoning.grid_seed import full_grid_seed
from thinking_system.reasoning.perception import bounding_box
from thinking_system.reasoning.structural import replicate_by_self
from thinking_system.reasoning.objects import trim_border, mirror_quad
from thinking_system.reasoning.induction import Library
from thinking_system.reasoning.library_learning import LibraryLearner


def _pad0(core):
    cols = len(core[0]); z = tuple([0] * (cols + 2))
    return (z,) + tuple(tuple([0] + list(r) + [0]) for r in core) + (z,)


def _frame(core, c):
    cols = len(core[0]); top = tuple([c] * (cols + 2))
    return (top,) + tuple(tuple([c] + list(r) + [c]) for r in core) + (top,)


def _combo1(g): return replicate_by_self(bounding_box(g))      # bbox ▸ fractal
def _combo2(g): return mirror_quad(trim_border(g))             # trim_border ▸ mirror_quad


def test_full_seed_has_all_layers() -> None:
    names = {p.name for p in full_grid_seed()}
    assert {"flip_h", "gravity", "fractal", "restore_sym"} <= names       # все четыре слоя присутствуют


def test_library_abstracts_perceptual_combos() -> None:
    seed = full_grid_seed()
    tasks = []
    for core in (((1, 2), (3, 4)), ((5, 0), (6, 7))):
        g = _pad0(core); tasks.append([(g, _combo1(g))])
    for core in (((2, 3), (3, 2)), ((8, 1), (1, 8))):
        g = _frame(core, 9); tasks.append([(g, _combo2(g))])

    learner = LibraryLearner(seed)
    learner.learn(tasks, rounds=2, max_depth=2, abstractions_per_round=1)
    abstr = set(learner.lib.abstractions)
    assert "bbox∘fractal" in abstr                                        # выделила комбо из решений
    assert "trim_border∘mirror_quad" in abstr


def test_abstraction_cuts_search_depth() -> None:
    seed = full_grid_seed()
    tasks = []
    for core in (((1, 2), (3, 4)), ((5, 0), (6, 7))):
        g = _pad0(core); tasks.append([(g, _combo1(g))])
    learner = LibraryLearner(seed)
    learner.learn(tasks, rounds=1, max_depth=2, abstractions_per_round=1)

    held = _pad0(((9, 4), (2, 7)))                                        # новое ядро, то же комбо
    ex = [(held, _combo1(held))]
    assert Library(seed).induce(ex, max_depth=1) is None                  # сырой seed на глубине 1 не берёт
    after = learner.solve(ex, max_depth=1)                                # после роста — берёт на глубине 1
    assert after is not None and after(held) == _combo1(held)
