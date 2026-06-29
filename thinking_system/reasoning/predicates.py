"""Индукция ПРЕДИКАТА из размеченных примеров: учить УСЛОВИЕ из данных, не хардкодить.

Прежний conditional.py знал предикаты («отрицательные», «пустой», …) ЖЁСТКО. Здесь
система выводит условие из примеров (вход → да/нет): подбирает ПРОСТЕЙШИЙ предикат из
грунтованного набора шаблонов, ТОЧНО разделяющий метки. Это снимает аудит-замечание о
захардкоженных предикатах и даёт ветвление, выученное из наблюдаемого.

Честная граница (как и везде в few-shot): предикат — из набора ШАБЛОНОВ, и из малого
числа примеров условие может оказаться мнимым (проверяй на новых данных).
"""

from __future__ import annotations

from typing import Any, Callable

Pred = tuple[str, Callable[[Any], bool]]


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def int_predicates(values: list[int] | None = None) -> list[Pred]:
    """Шаблоны предикатов над целыми (+ пороги по наблюдаемым значениям)."""
    preds: list[Pred] = [
        ("x>0", lambda x: x > 0),
        ("x<0", lambda x: x < 0),
        ("x=0", lambda x: x == 0),
        ("x чётно", lambda x: x % 2 == 0),
        ("x нечётно", lambda x: x % 2 != 0),
    ]
    for k in sorted(set(values or [])):
        preds.append((f"x>{k}", lambda x, k=k: x > k))
        preds.append((f"x≥{k}", lambda x, k=k: x >= k))
    return preds


def list_predicates(lengths: list[int] | None = None) -> list[Pred]:
    """Шаблоны предикатов над списками целых (+ пороги по длине)."""
    preds: list[Pred] = [
        ("пусто", lambda xs: len(xs) == 0),
        ("непусто", lambda xs: len(xs) > 0),
        ("все>0", lambda xs: len(xs) > 0 and all(e > 0 for e in xs)),
        ("есть<0", lambda xs: any(e < 0 for e in xs)),
        ("все чётные", lambda xs: len(xs) > 0 and all(e % 2 == 0 for e in xs)),
        ("есть чётное", lambda xs: any(e % 2 == 0 for e in xs)),
        ("возрастает", lambda xs: all(a <= b for a, b in zip(xs, xs[1:]))),
        ("убывает", lambda xs: all(a >= b for a, b in zip(xs, xs[1:]))),
        ("есть повтор", lambda xs: len(set(xs)) != len(xs)),
        ("сумма чётна", lambda xs: sum(xs) % 2 == 0),
    ]
    for k in sorted(set(lengths or [])):
        preds.append((f"длина>{k}", lambda xs, k=k: len(xs) > k))
    return preds


def candidates_for(inputs: list[Any]) -> list[Pred]:
    """Подходящие шаблоны предикатов по типу наблюдаемых входов."""
    if inputs and all(_is_int(i) for i in inputs):
        return int_predicates(list(inputs))
    if inputs and all(isinstance(i, list) for i in inputs):
        return list_predicates([len(i) for i in inputs])
    return []


def induce_predicate(labeled: list[tuple[Any, bool]]) -> Pred | None:
    """Вывести простейший предикат, ТОЧНО разделяющий метки (или None).

    labeled: список (вход, да/нет). Требует, чтобы встречались оба класса меток.
    """
    inputs = [i for i, _ in labeled]
    labels = [bool(l) for _, l in labeled]
    if len(set(labels)) < 2:
        return None  # нечего различать — один класс
    for name, fn in candidates_for(inputs):
        try:
            if all(bool(fn(i)) == l for i, l in zip(inputs, labels)):
                return (name, fn)
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
    return None
