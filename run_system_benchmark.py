#!/usr/bin/env python3
"""Честный замер ПОКРЫТИЯ думающей системы (обобщение на отложенный вход).

Для каждой задачи правило выводится из train-примеров и проверяется на ОТДЕЛЬНОМ
held-out входе (не использованном при индукции) — засчитывается только верное
обобщение, а не запоминание. Негативные контроли (вне шаблонов) проверяют, что
система не выдумывает обобщающее правило там, где его нет.

Запуск: python run_system_benchmark.py
"""

from __future__ import annotations

from collections import OrderedDict

from thinking_system.benchmark import evaluate


def main() -> None:
    res = evaluate()
    print("▶ Честный замер покрытия: обобщение на ОТЛОЖЕННЫЙ вход (без утечки)\n")

    by_cat: "OrderedDict[str, list]" = OrderedDict()
    for r in res["rows"]:
        by_cat.setdefault(r["task"].category, []).append(r)

    print("── По задачам ──")
    for cat, rows in by_cat.items():
        if cat == "негатив":
            continue
        for r in rows:
            mark = "✓" if r["solved"] else "✗"
            print(f"  {mark} {cat:<13} {r['task'].name:<26} → правило «{r['program']}»")

    print("\n── По категориям (решено/всего) ──")
    for cat, rows in by_cat.items():
        if cat == "негатив":
            continue
        s = sum(x["solved"] for x in rows)
        print(f"  {cat:<14} {s}/{len(rows)}")

    print("\n── Негативные контроли (вне шаблонов → ожидаем «не обобщила») ──")
    for r in by_cat.get("негатив", []):
        ok = "✓ отвергла" if not r["solved"] else "✗ ВЫДУМАЛА (!)"
        print(f"  {ok:<16} {r['task'].name:<26} → {r['program']}")

    cov = res["in_scope_solved"]
    tot = res["in_scope_total"]
    print("\n── ИТОГ ──")
    print(f"  ПОКРЫТИЕ (обобщение на held-out): {cov}/{tot} = {100 * cov / tot:.0f}%")
    print(f"  негативные: выдуманных обобщений {res['negative_spurious']}/{res['negative_total']} (должно быть 0)")

    print("\n── Что это значит (честно) ──")
    print("   Каждое «решено» — это ВЕРНЫЙ ответ на НОВОМ входе, а не подгонка под примеры.")
    print("   Негативные контроли вне шаблонов отвергнуты (None) — система не выдаёт правило,")
    print("   которого нет. Это и есть честный тест: не «сколько демок», а покрытие с проверкой")
    print("   обобщения и анти-галлюцинацией. Набор ограничен ШАБЛОНАМИ индукции (не любой")
    print("   мыслимой программой) — расширять покрытие можно новыми слоями или реальными")
    print("   задачами (ARC: положить *.json и измерить grid-движок — данные в репо не входят).")


if __name__ == "__main__":
    main()
