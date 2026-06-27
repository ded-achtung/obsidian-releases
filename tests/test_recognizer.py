"""Тест распознавателя: меньше перебора на held-out, чем слепой BFS и униграммный приор."""

from __future__ import annotations

from thinking_system.reasoning.induction import default_primitives, Library
from thinking_system.reasoning.recognizer import Recognizer, task_features
from thinking_system.reasoning.search_prior import usage_prior, search_cost
from run_recognizer import TRAIN, TEST, recog_cost


def test_recognizer_reduces_search_on_heldout() -> None:
    prims = default_primitives()
    recog = Recognizer(prims).fit(TRAIN, max_depth=3)
    lib = Library(prims)
    uni = usage_prior([s for s in (lib.induce(ex, max_depth=3) for ex in TRAIN) if s], prims)

    s_b, c_b = search_cost(TEST, prims, None, max_depth=3)
    s_u, c_u = search_cost(TEST, prims, uni, max_depth=3)
    s_r, c_r = recog_cost(TEST, prims, recog, max_depth=3)

    assert s_b == s_u == s_r == len(TEST)        # все решают held-out
    assert c_r < c_b                              # распознаватель короче слепого
    assert c_r < c_u                              # и короче глобального униграммного приора


def test_features_separate_int_and_list_tasks() -> None:
    """Признаки отличают числовую задачу от списочной (основа обусловленного приора)."""
    fi = task_features([(2, 4), (3, 6)])         # int → int
    fl = task_features([([1, 2], 3), ([4], 4)])  # list → int
    assert fi[1] == 0.0 and fl[1] == 1.0          # признак is_list_in
