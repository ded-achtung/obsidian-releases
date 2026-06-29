"""Изобретение операции ИЗ НАБЛЮДЕНИЙ: не из воздуха, а из регулярности в данных.

Когда композиция известных примитивов не подходит, система НЕ сдаётся и НЕ выдумывает
из пустоты — она смотрит на сами пары вход→выход и подгоняет ГРУНТОВАННЫЕ шаблоны-
регулярности к тому, что видит:

  • над целыми:  аффинная  x ↦ a·x + b           (нужно ≥2 наблюдения с разными x)
  •              квадратичная x ↦ a·x² + b·x + c  (нужно ≥3 наблюдения с разными x)
  • над списками: поэлементно-аффинная  [e] ↦ [a·e + b]

Если шаблон ТОЧНО (над целыми, без округления) согласован со ВСЕМИ примерами —
рождается НОВАЯ именованная операция, выведенная из наблюдаемого, и её можно
добавить в библиотеку как примитив (далее переиспользуется языком и поиском).

Честная граница: это набор ШАБЛОНОВ (а не «любая операция из воздуха») и оно
ограничено тем, СКОЛЬКО система увидела — из 2 точек квадратичную не определить
(их бесконечно много), поэтому из малых данных правило может оказаться мнимым.
Это та же честность, что и в few-shot индукции: проверяй на новом входе.
"""

from __future__ import annotations

from fractions import Fraction as F
from typing import Any, Callable

from thinking_system.reasoning.induction import Primitive
from thinking_system.reasoning.predicates import candidates_for


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _fit_affine(pts: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Подогнать целочисленную аффинную a·x+b ко ВСЕМ точкам (или None).

    Требует ≥3 точек с РАЗНЫМ x: 2 определяют прямую, 3-я её ПОДТВЕРЖДАЕТ. Иначе это
    подгонка под данные (любые 2 точки лежат на какой-то прямой), а не правило.
    """
    if len({x for x, _ in pts}) < 3:
        return None
    base = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if pts[i][0] != pts[j][0]:
                base = (pts[i], pts[j])
                break
        if base:
            break
    if base is None:
        return None
    (x0, y0), (x1, y1) = base
    a = F(y1 - y0, x1 - x0)
    b = F(y0) - a * x0
    if a.denominator == 1 and b.denominator == 1 and all(a * x + b == y for x, y in pts):
        return int(a), int(b)
    return None


def _fit_quadratic(pts: list[tuple[int, int]]) -> tuple[int, int, int] | None:
    """Подогнать целочисленную a·x²+b·x+c ко ВСЕМ точкам (или None).

    Требует ≥4 точек с РАЗНЫМ x: 3 определяют параболу, 4-я её ПОДТВЕРЖДАЕТ (иначе из
    любых 3 точек выводится какая-нибудь парабола — это память, а не правило).
    """
    if len({x for x, _ in pts}) < 4:
        return None
    P: list[tuple[int, int]] = []
    seen: set[int] = set()
    for x, y in pts:
        if x not in seen:
            seen.add(x)
            P.append((x, y))
        if len(P) == 3:
            break
    if len(P) < 3:
        return None
    M = [[F(x * x), F(x), F(1), F(y)] for x, y in P]  # решаем систему точно (Fraction)
    for i in range(3):
        if M[i][i] == 0:
            for j in range(i + 1, 3):
                if M[j][i] != 0:
                    M[i], M[j] = M[j], M[i]
                    break
        if M[i][i] == 0:
            return None
        piv = M[i][i]
        M[i] = [v / piv for v in M[i]]
        for j in range(3):
            if j != i and M[j][i] != 0:
                f = M[j][i]
                M[j] = [M[j][k] - f * M[i][k] for k in range(4)]
    a, b, c = M[0][3], M[1][3], M[2][3]
    if a == 0:
        return None  # вырожденно (на деле линейно/тождественно) — это случай аффинной
    if all(v.denominator == 1 for v in (a, b, c)) and all(a * x * x + b * x + c == y for x, y in pts):
        return int(a), int(b), int(c)
    return None


def _term(coef: int, power: int) -> str:
    mag = abs(coef)
    if power == 0:
        return str(mag)
    sym = "x" if power == 1 else "x²"
    return sym if mag == 1 else f"{mag}·{sym}"


def _name_poly(a: int, b: int, c: int) -> str:
    """Читаемое имя многочлена: a·x²+b·x+c (опуская нули и единицы)."""
    parts: list[str] = []
    for coef, p in [(a, 2), (b, 1), (c, 0)]:
        if coef == 0:
            continue
        sign = "-" if coef < 0 else ("+" if parts else "")
        parts.append(sign + _term(coef, p))
    return "".join(parts) or "0"


def _mk_int_fn(fn: Callable[[int], int]) -> Callable[[Any], Any]:
    def f(x: Any) -> Any:
        if not _is_int(x):
            raise TypeError
        return fn(x)
    return f


def _mk_list_fn(fn: Callable[[list], list]) -> Callable[[Any], Any]:
    def f(x: Any) -> Any:
        if not isinstance(x, list):
            raise TypeError
        return fn(x)
    return f


def invent_primitive(examples: list[tuple[Any, Any]]) -> Primitive | None:
    """Изобрести примитив из наблюдаемых примеров (или None, если шаблон не подошёл).

    Пробует (по простоте, бритва Оккама): целочисленные аффинную → квадратичную над
    числами; поэлементно-аффинную над списками равной длины. Имя — читаемая формула.
    """
    ins = [i for i, _ in examples]
    outs = [o for _, o in examples]

    # 1) число → число
    if ins and all(_is_int(i) for i in ins) and all(_is_int(o) for o in outs):
        aff = _fit_affine(list(zip(ins, outs)))
        if aff is not None and aff != (1, 0):  # (1,0)=тождество — не операция
            a, b = aff
            return Primitive(_name_poly(0, a, b), _mk_int_fn(lambda x, a=a, b=b: a * x + b))
        quad = _fit_quadratic(list(zip(ins, outs)))
        if quad is not None:
            a, b, c = quad
            return Primitive(_name_poly(a, b, c), _mk_int_fn(lambda x, a=a, b=b, c=c: a * x * x + b * x + c))

    # 2) список → список равной длины: поэлементно-аффинная
    if ins and all(isinstance(i, list) for i in ins) and all(isinstance(o, list) for o in outs):
        flat: list[tuple[int, int]] = []
        ok = True
        for i, o in zip(ins, outs):
            if len(i) != len(o):
                ok = False
                break
            flat += list(zip(i, o))
        if ok and flat and all(_is_int(v) for pair in flat for v in pair):
            aff = _fit_affine(flat)
            if aff is not None and aff != (1, 0):
                a, b = aff
                return Primitive("each:" + _name_poly(0, a, b),
                                 _mk_list_fn(lambda xs, a=a, b=b: [a * e + b for e in xs]))

    return None


# ── СЛЕДУЮЩИЙ СЛОЙ: условная (ветвящаяся) операция «если P(x): f иначе g» из данных ──

def _eq(a: Any, b: Any) -> bool:
    try:
        return type(a) == type(b) and a == b
    except (TypeError, ValueError):
        return False


def _fit_branch(pairs: list[tuple[Any, Any]]) -> tuple[Callable[[Any], Any], str] | None:
    """Подогнать ветку: тождество, константа или изобретённая прямолинейная операция."""
    outs = [o for _, o in pairs]
    if all(_eq(i, o) for i, o in pairs):
        return (lambda x: x), "x"
    if len(pairs) >= 2 and all(_eq(o, outs[0]) for o in outs):  # постоянная ветка (≥2 — подтверждение)
        c = outs[0]
        return (lambda x, c=c: c), str(c)
    prim = invent_primitive(pairs)                       # аффинная/квадратичная/поэлементная
    if prim is not None:
        return prim.fn, prim.name
    return None


def invent_conditional(examples: list[tuple[Any, Any]]) -> Primitive | None:
    """Собрать операцию «если P(x): f иначе g», где P, f, g выведены из наблюдений.

    Перебирает грунтованные предикаты как РАЗДЕЛИТЕЛИ примеров; для каждой ветки
    подгоняет операцию (тождество/константа/аффинная…). Берёт первую гипотезу,
    ТОЧНО воспроизводящую все примеры. Так из данных рождается ветвление (abs,
    «обнулить отрицательные», «удвоить чётные» …), а не только прямая линия.
    """
    if len(examples) < 3:
        return None  # нужно ≥1 примера на ветку и разделение меток
    ins = [i for i, _ in examples]
    for pname, P in candidates_for(ins):
        try:
            true_pairs = [(i, o) for i, o in examples if P(i)]
            false_pairs = [(i, o) for i, o in examples if not P(i)]
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
        if not true_pairs or not false_pairs:
            continue  # предикат должен РАЗДЕЛЯТЬ примеры на обе ветки
        f = _fit_branch(true_pairs)
        g = _fit_branch(false_pairs)
        if f is None or g is None:
            continue
        ffn, fname = f
        gfn, gname = g

        def cond(x, P=P, ffn=ffn, gfn=gfn):
            return ffn(x) if P(x) else gfn(x)

        try:
            if all(_eq(cond(i), o) for i, o in examples):
                return Primitive(f"если {pname}: {fname} иначе {gname}", cond)
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
    return None


def invent(examples: list[tuple[Any, Any]]) -> Primitive | None:
    """Изобрести операцию из наблюдений: прямолинейную, иначе — условную (ветвящуюся)."""
    return invent_primitive(examples) or invent_conditional(examples)
