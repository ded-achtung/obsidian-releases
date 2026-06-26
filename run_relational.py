#!/usr/bin/env python3
"""Реляционно-аналоговое рассуждение: из нескольких фактов — много выводов.

Знания как отношения + правила Хорна; прямой вывод доводит до замыкания. Без
корпуса: транзитивность, наследование свойств, дедукция родственных связей и
аналогии A:B :: C:? — чистым выводом из малого.

Запуск: python run_relational.py
"""

from __future__ import annotations

from thinking_system.reasoning.relational import KnowledgeBase


def main():
    print("▶ Реляционно-аналоговое рассуждение: факты-отношения + правила, прямой вывод\n")

    # 1) таксономия (транзитивность) + наследование свойств
    kb = KnowledgeBase()
    for a, b in [("sparrow", "bird"), ("bird", "animal"), ("animal", "thing"), ("dog", "mammal"), ("mammal", "animal")]:
        kb.add("is_a", a, b)
    kb.add("can_fly", "bird", "yes").add("breathes", "animal", "yes")
    kb.transitive("is_a").inherits("is_a")
    print(f"1) ТАКСОНОМИЯ + НАСЛЕДОВАНИЕ: {len(kb.facts)} базовых факта → {len(kb.closure())} выведено")
    print(f"   sparrow → thing (транзитивно is_a):  {kb.holds('is_a', 'sparrow', 'thing')}")
    print(f"   sparrow умеет летать (от bird):       {kb.holds('can_fly', 'sparrow', 'yes')}")
    print(f"   dog дышит (от animal):                {kb.holds('breathes', 'dog', 'yes')}")
    print(f"   dog умеет летать (НЕ наследует bird):  {kb.holds('can_fly', 'dog', 'yes')}  ← честно False")

    # 2) дедукция родственных связей (многошаговый вывод)
    fam = KnowledgeBase()
    for p, c in [("ann", "bob"), ("bob", "cara"), ("cara", "dan")]:
        fam.add("parent", p, c)
    fam.rule(("grandparent", "?x", "?z"), [("parent", "?x", "?y"), ("parent", "?y", "?z")])
    fam.rule(("ancestor", "?x", "?y"), [("parent", "?x", "?y")]).transitive("ancestor")
    print("\n2) РОДСТВО ИЗ ПРАВИЛ (3 факта «родитель» + правила):")
    print(f"   дедушки/бабушки: {[(f[1], f[2]) for f in fam.query('grandparent')]}")
    print(f"   все потомки ann (предок ann→?): {[f[2] for f in fam.query('ancestor', 'ann')]}")

    # 3) аналогии A:B :: C:?
    an = KnowledgeBase()
    for cap, co in [("Paris", "France"), ("Tokyo", "Japan"), ("Rome", "Italy")]:
        an.add("capital_of", cap, co)
    for m, f in [("king", "queen"), ("man", "woman"), ("actor", "actress")]:
        an.add("female_of", m, f)
    print("\n3) АНАЛОГИИ A:B :: C:? (находит общее отношение и применяет):")
    print(f"   Paris:France :: Tokyo:?  →  {an.analogy('Paris', 'France', 'Tokyo')}")
    print(f"   king:queen   :: man:?    →  {an.analogy('king', 'queen', 'man')}")
    print(f"   Paris:France :: banana:? →  {an.analogy('Paris', 'France', 'banana')}  ← нет общего отношения")

    print("\n── Итог ──")
    print("   Из горстки отношений и правил вывод даёт множество новых фактов и решает")
    print("   аналогии — без корпуса. Это дедукция из приоров (правил), а не статистика:")
    print("   мало данных, но системная композиция знаний.")


if __name__ == "__main__":
    main()
