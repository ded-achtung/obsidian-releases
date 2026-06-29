"""Тест: MDL открывает понятия из решений (сжатие в битах) и они переиспользуются."""

from __future__ import annotations

from thinking_system.reasoning.induction import Library, default_primitives
from thinking_system.reasoning.mdl_abstraction import MDLLearner, description_length
from run_concepts import TASKS, HELD_OUT


def _corpus():
    lib = Library(default_primitives())
    corpus = []
    for ex in TASKS:
        prog = lib.induce(ex, max_depth=3)
        assert prog is not None
        corpus.append([s.name for s in prog.steps])
    return corpus


def test_mdl_compresses_and_discovers_both_concepts() -> None:
    base = default_primitives()
    corpus = _corpus()
    learner = MDLLearner(base)
    learner.compress(corpus, max_abstractions=6)
    # открыты оба ожидаемых понятия (по объективному критерию, не подсказке)
    assert "+1∘square" in learner.abstractions
    assert "each*2∘sum" in learner.abstractions
    # реальное сжатие корпуса в битах
    bits0 = description_length(corpus, {}, len(base))
    bits1 = description_length(corpus, learner.abstractions, len(base))
    assert bits1 < bits0


def test_mdl_is_principled_stops_when_no_gain() -> None:
    """MDL не абстрагирует ради абстракции: добавляет только то, что снижает биты."""
    base = default_primitives()
    corpus = _corpus()
    learner = MDLLearner(base)
    steps = learner.compress(corpus, max_abstractions=20)
    assert all(s["bits_saved"] > 0 for s in steps)         # каждый шаг реально сжимает
    # повторный прогон ничего не добавляет (сошлось)
    n = len(learner.abstractions)
    learner.compress(corpus, max_abstractions=20)
    assert len(learner.abstractions) == n


def test_discovered_concepts_enable_heldout_at_depth1() -> None:
    base = default_primitives()
    corpus = _corpus()
    learner = MDLLearner(base)
    learner.compress(corpus, max_abstractions=6)
    seed_lib = Library(base)
    grown = learner.to_library()
    for _, ex in HELD_OUT:
        assert seed_lib.induce(ex, max_depth=1) is None        # базовый язык не берёт на d=1
        prog = grown.induce(ex, max_depth=1)
        assert prog is not None                                # с понятием — берёт
        # и абстракция исполняется ВЕРНО на примерах
        for x, y in ex:
            assert prog(x) == y
