"""Тесты реляционного рассуждения: транзитивность, наследование, дедукция, аналогия."""

from __future__ import annotations

from thinking_system.reasoning.relational import KnowledgeBase


def test_transitive_closure_and_inheritance() -> None:
    kb = KnowledgeBase()
    for a, b in [("sparrow", "bird"), ("bird", "animal"), ("animal", "thing")]:
        kb.add("is_a", a, b)
    kb.add("can_fly", "bird", "yes")
    kb.transitive("is_a").inherits("is_a")
    assert kb.holds("is_a", "sparrow", "thing")            # транзитивный вывод
    assert kb.holds("can_fly", "sparrow", "yes")           # наследование свойства класса
    assert not kb.holds("can_fly", "animal", "yes")        # не выводит лишнего (animal не bird)


def test_forward_chaining_derives_family_relations() -> None:
    fam = KnowledgeBase()
    for p, c in [("ann", "bob"), ("bob", "cara"), ("cara", "dan")]:
        fam.add("parent", p, c)
    fam.rule(("grandparent", "?x", "?z"), [("parent", "?x", "?y"), ("parent", "?y", "?z")])
    fam.rule(("ancestor", "?x", "?y"), [("parent", "?x", "?y")]).transitive("ancestor")
    assert fam.holds("grandparent", "ann", "cara")         # многошаговый вывод
    assert fam.holds("ancestor", "ann", "dan")             # транзитивное замыкание предков
    assert not fam.holds("grandparent", "ann", "bob")      # bob — ребёнок, не внук


def test_analogy_finds_shared_relation() -> None:
    an = KnowledgeBase()
    for cap, co in [("Paris", "France"), ("Tokyo", "Japan")]:
        an.add("capital_of", cap, co)
    for m, f in [("king", "queen"), ("man", "woman")]:
        an.add("female_of", m, f)
    assert an.analogy("Paris", "France", "Tokyo") == ["Japan"]
    assert an.analogy("king", "queen", "man") == ["woman"]
    assert an.analogy("Paris", "France", "banana") == []   # нет общего отношения — честно пусто
