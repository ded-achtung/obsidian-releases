#!/usr/bin/env python3
"""Честный замер на ВНЕШНИХ задачах ARC (данные качаются отдельно, в репо не входят).

Правило выводится из `train` каждой задачи и проверяется на её `test`-входе (точное
совпадение всех выходов) — без утечки. Покрытие ожидаемо НИЗКОЕ: это узкий grid-движок
против разнообразного ARC. Низкая честная цифра показывает реальную ширину примитивов.

Данные (ARC-AGI, Apache-2.0):
    curl -sSL https://codeload.github.com/fchollet/ARC-AGI/tar.gz/refs/heads/master \\
        | tar xz && mkdir -p data/arc && mv ARC-AGI-master/data/* data/arc/

Запуск:
    python run_arc.py                      # data/arc/training
    python run_arc.py data/arc/evaluation  # другой набор
"""

from __future__ import annotations

import os
import sys

from thinking_system.arc import evaluate_arc


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/arc/training"
    if not os.path.isdir(path):
        print(f"✗ Нет папки {path!r}. Сначала скачайте ARC (см. докстринг run_arc.py):")
        print("   curl -sSL https://codeload.github.com/fchollet/ARC-AGI/tar.gz/refs/heads/master "
              "| tar xz && mkdir -p data/arc && mv ARC-AGI-master/data/* data/arc/")
        return

    print(f"▶ Честный замер на ВНЕШНИХ задачах ARC: {path}\n")
    res = evaluate_arc(path)
    total, solved = res["total"], res["solved"]

    print(f"── Решённые задачи (точное совпадение test-выхода) ──")
    for tid, prog in res["solved_tasks"]:
        print(f"  ✓ {tid}  «{prog}»")
    if not res["solved_tasks"]:
        print("  (ни одной — для этого набора узкого grid-движка не хватает)")

    print(f"\n── ИТОГ ──")
    print(f"  ПОКРЫТИЕ: {solved}/{total} = {100 * solved / total:.1f}%")

    print("\n── Что это значит (честно) ──")
    print("   Это РЕАЛЬНЫЕ внешние задачи, не свой набор. Каждое «решено» — точное совпадение")
    print("   на отложенном test-входе. Низкая цифра честна: grid-движок умеет flip/rot/")
    print("   transpose/перекраску/морфологию (композиция ≤2), а ARC требует сотни разных")
    print("   преобразований (объекты, счёт, замощение, рост…). Это меряет ШИРИНУ примитивов,")
    print("   а не подгонку. Поднять покрытие = добавлять восприятие/примитивы, а не шаблоны")
    print("   под свой бенчмарк. Так и должен выглядеть честный внешний замер «системы».")


if __name__ == "__main__":
    main()
