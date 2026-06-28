#!/usr/bin/env python3
"""Рефлексия: система наблюдает свои попытки, понимает что работает, анализирует провалы.

На реальном ARC: решатель НАБЛЮДАЕТ каждую попытку, строит модель собственной
компетентности (предсказывает «смогу ли я» на held-out — метапознание) и ВЫДАЁТ
самоотчёт с анализом, ПОЧЕМУ не сработало остальное.

Запуск: python run_reflect.py
"""

from __future__ import annotations

import numpy as np
import arckit

from thinking_system.reasoning.grids import to_grid
from thinking_system.agent.reflect import ReflectiveSolver


def grids(t):
    g = lambda a: to_grid(np.asarray(a).tolist())
    return [(g(i), g(o)) for i, o in t.train], [(g(i), g(o)) for i, o in t.test]


def main():
    tr, _ = arckit.load_data()
    tasks = list(tr)
    cut = 700
    print(f"▶ Рефлексивный решатель на ARC: наблюдает {cut} задач, метапознание на {len(tasks)-cut} held-out\n")

    solver = ReflectiveSolver()
    for t in tasks[:cut]:
        train, test = grids(t)
        solver.solve_and_observe(train, test, task_id=t.id)

    rep = solver.reflect()
    print("1) ЧТО Я ПОНЯЛ ПРО СЕБЯ (рефлексия над наблюдениями):")
    print(f"   решил {rep['solved']}/{rep['n']}; каким синтезатором: {rep['by_synth']}")
    print(f"   формы решённых задач: {rep['solved_shapes']}")
    print("\n2) ПОЧЕМУ НЕ СРАБОТАЛО остальное (анализ своих провалов):")
    for reason, k in sorted(rep["failure_reasons"].items(), key=lambda kv: -kv[1]):
        share = 100 * k / max(rep["n"] - rep["solved"], 1)
        human = {
            "no_hypothesis": "нет схемы под этот класс задач",
            "rejected_as_memorization": "мог зазубрить локальное правило — отказался (не обобщилось бы)",
            "fit_train_failed_test": "правило подошло на train, но не обобщилось на тест",
        }.get(reason, reason)
        print(f"   {k:>4} ({share:4.1f}%) — {human}")
    print(f"   формы нерешённых: {rep['failure_shapes']}")

    # 3) МЕТАПОЗНАНИЕ: предсказать собственный успех на held-out ДО решения
    solver.build_competence()
    held = tasks[cut:]
    tp = fp = fn = tn = 0
    for t in held:
        train, test = grids(t)
        pred = solver.thinks_it_can_solve(train)
        ep = ReflectiveSolver(); ep.solve_and_observe(train, test)   # фактический исход
        actual = ep.episodes[0]["solved_by"] is not None
        tp += pred and actual; fp += pred and not actual
        fn += (not pred) and actual; tn += (not pred) and not actual
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print("\n3) МЕТАПОЗНАНИЕ — предсказываю свой успех на held-out ДО попытки:")
    print(f"   решаемых на самом деле: {tp+fn}; предсказал решаемыми: {tp+fp}")
    print(f"   точность предсказания (precision) {prec*100:.0f}%, полнота (recall) {rec*100:.0f}%")
    print(f"   (база: верно отнёс нерешаемые {tn} — то есть знаю, чего НЕ умею)")

    print("\n── Что это значит ──")
    print("   Система теперь НАБЛЮДАЕТ свои попытки, строит понимание «что у меня работает» и")
    print("   предсказывает собственный успех заранее, и АНАЛИЗИРУЕТ, почему провалила остальное —")
    print("   честно видя свою границу (в основном «нет схемы»). Это рефлексия над собственным")
    print("   решением — то, чего не было. Честно: пока это самонаблюдение над перебором схем, а")
    print("   не понимание мира; но это шаг от слепого поиска к анализу своих действий.")


if __name__ == "__main__":
    main()
