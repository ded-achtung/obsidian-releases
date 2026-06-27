#!/usr/bin/env python3
"""Условные правила: что применить — зависит от ДАННЫХ, а не от слов задачи.

Система учит операции из примеров и правила «если <условие на данных>, <операция>»
из теории. При решении она смотрит на сам список и выбирает подходящее правило.
Одна задача «обработай» решается по-разному в зависимости от данных.

Запуск: python run_conditional.py
"""

from __future__ import annotations

from thinking_system.language.conditional import ConditionalReader
from thinking_system.text.ingest import load_book


def main():
    print("▶ Условные правила: применимость зависит от данных\n")
    cr = ConditionalReader()
    stats = cr.study(load_book("books/textbook_conditional.md"))

    print(f"1) ВЫУЧИЛ: {stats['операции']} операции, {stats['правила']} условных правила")
    print(f"   операции (что делают): {', '.join(f'{w}={p}' for w, p in cr.lex.words.items())}")
    print("   правила (когда применять — по данным):")
    for cond, op in cr.rules:
        print(f"     если в списке «{cond}» → применить «{op}»")

    print("\n2) РЕШАЕТ — одна задача «обработай», правило выбрано ПО ДАННЫМ:")
    for r in cr.solve_exercises():
        print(f"   «{r['задача']:<24}» → {r['answer']}   ⟵ {r['rule']}")

    print("\n3) ЧЕСТНО, когда ни одно условие не подходит:")
    print(f"   «обработай []» → {cr.solve('обработай []')['reason'] if not cr.solve('обработай []')['solved'] else cr.solve('обработай []')}")

    print("\n── Что это значит ──")
    print("   Применимость теперь зависит не от слов задачи, а от СВОЙСТВА данных: система")
    print("   смотрит на сам список (есть отрицательные? длинный?) и выбирает правило. Это")
    print("   правило С ПРОВЕРКОЙ УСЛОВИЯ — ближе к тому, как условия формулируют в учебниках.")


if __name__ == "__main__":
    main()
