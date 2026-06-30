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

from thinking_system.reasoning.induction import Primitive, induce as _compose_induce
from thinking_system.reasoning.predicates import candidates_for
from thinking_system.reasoning.grids import grid_primitives, to_grid

_MAX_NEST = 2  # глубина вложенности условий внутри ветвей (дерево решений до ~3 уровней)


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


def _fit_leaf(pairs: list[tuple[Any, Any]], *, allow_singleton_const: bool = False) -> tuple[Callable[[Any], Any], str] | None:
    """Подогнать ЛИСТОВУЮ ветку — одиночную операцию без управления потоком.

    Тождество/константа или изобретённая прямолинейная/структурная/оконная/сеточная
    операция. Это «простые» ветки; вложенные условия — отдельно (см. _fit_branch).
    """
    outs = [o for _, o in pairs]
    if all(_eq(i, o) for i, o in pairs):
        return (lambda x: x), "x"
    min_const = 1 if allow_singleton_const else 2
    if len(pairs) >= min_const and all(_eq(o, outs[0]) for o in outs):  # постоянная ветка (подтверждение)
        c = outs[0]
        return (lambda x, c=c: c), str(c)
    for inv in (invent_primitive, invent_structural, invent_window, invent_grid,
                invent_grid_recolor, invent_grid_window):
        p = inv(pairs)
        if p is not None:
            return p.fn, p.name
    return None


def _fit_branch(pairs: list[tuple[Any, Any]], *, allow_singleton_const: bool = False, depth: int = 0, allow_nest: bool = True) -> tuple[Callable[[Any], Any], str] | None:
    """Ветка: листовая операция; иначе (если allow_nest) — ВЛОЖЕННОЕ условие (до _MAX_NEST)."""
    leaf = _fit_leaf(pairs, allow_singleton_const=allow_singleton_const)
    if leaf is not None:
        return leaf
    if allow_nest and depth < _MAX_NEST:                  # ВЛОЖЕННОСТЬ: условие внутри ветки
        nested = invent_multibranch(pairs, depth=depth + 1) or invent_conditional(pairs, depth=depth + 1)
        if nested is not None:
            return nested.fn, f"({nested.name})"
    return None


def invent_conditional(examples: list[tuple[Any, Any]], *, depth: int = 0) -> Primitive | None:
    """Собрать операцию «если P(x): f иначе g», где P, f, g выведены из наблюдений.

    Перебирает грунтованные предикаты как РАЗДЕЛИТЕЛИ примеров; для каждой ветки
    подгоняет операцию. Сперва ищет ПЛОСКУЮ гипотезу (обе ветки — листовые операции,
    бритва Оккама), и лишь если её нет — допускает ВЛОЖЕННОЕ условие в ветке. Так из
    данных рождается ветвление (abs, «обнулить отрицательные», «если длинный — развернуть»).
    """
    if len(examples) < 3:
        return None  # нужно ≥1 примера на ветку и разделение меток
    ins = [i for i, _ in examples]
    for allow_nest in (False, True):                     # сперва плоские гипотезы, потом вложенные
        for pname, P in candidates_for(ins):
            try:
                true_pairs = [(i, o) for i, o in examples if P(i)]
                false_pairs = [(i, o) for i, o in examples if not P(i)]
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                continue
            if not true_pairs or not false_pairs:
                continue  # предикат должен РАЗДЕЛЯТЬ примеры на обе ветки
            f = _fit_branch(true_pairs, depth=depth, allow_nest=allow_nest)
            g = _fit_branch(false_pairs, depth=depth, allow_nest=allow_nest)
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


# ── СЛЕДУЮЩИЙ СЛОЙ: СТРУКТУРНЫЕ операции над списками (перестановки/период/выбор) ──

def invent_structural(examples: list[tuple[Any, Any]]) -> Primitive | None:
    """Вывести длино-ОТНОСИТЕЛЬНОЕ структурное преобразование списка из наблюдений.

    Не значения элементов (это поэлементная), а ПОЗИЦИИ/длина: разворот, циклический
    сдвиг, повтор (период), прореживание, выбор префикса/суффикса. Параметрические
    правила (сдвиг k, повтор m, шаг s, n штук) подтверждаются на ≥2 РАЗНЫХ длинах —
    иначе это запоминание одной перестановки, а не правило.
    """
    ins = [i for i, _ in examples]
    outs = [o for _, o in examples]
    if not ins or not all(isinstance(i, list) for i in ins) or not all(isinstance(o, list) for o in outs):
        return None
    if len(examples) < 2:
        return None
    distinct_lens = len({len(i) for i in ins})
    maxlen = max((len(i) for i in ins), default=0)

    templates: list[tuple[str, Callable[[list], list], bool]] = [
        ("разворот", lambda xs: xs[::-1], False),
        ("сорт↑", lambda xs: sorted(xs), False),
        ("сорт↓", lambda xs: sorted(xs, reverse=True), False),
    ]
    for m in range(2, 5):
        templates.append((f"повтор×{m}", (lambda xs, m=m: xs * m), True))
    for k in range(1, maxlen):
        templates.append((f"сдвиг←{k}", (lambda xs, k=k: (xs[k % len(xs):] + xs[:k % len(xs)]) if xs else xs), True))
    for s in range(2, maxlen + 1):
        templates.append((f"каждый {s}-й", (lambda xs, s=s: xs[::s]), True))
    for n in range(1, maxlen):
        templates.append((f"первые {n}", (lambda xs, n=n: xs[:n]), True))
        templates.append((f"последние {n}", (lambda xs, n=n: xs[-n:]), True))
        templates.append((f"без первых {n}", (lambda xs, n=n: xs[n:]), True))

    for name, fn, parametric in templates:
        if parametric and distinct_lens < 2:
            continue  # параметрическое правило корроборируем на ≥2 разных длинах
        try:
            if all(fn(i) == o for i, o in examples) and not all(_eq(i, o) for i, o in examples):
                return Primitive(name, _mk_list_fn(fn))
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
    return None


# ── СЛЕДУЮЩИЙ СЛОЙ: МНОГОВЕТОЧНОЕ условие (например sign: <0 / =0 / >0) ─────────────

def _partition_families(ins: list[Any]) -> list[tuple[str, list[tuple[str, Callable[[Any], bool], bool]]]]:
    """Семейства ВЗАИМОИСКЛЮЧАЮЩИХ предикатов, покрывающих входы (для N ветвей)."""
    fams: list[tuple[str, list[tuple[str, Callable[[Any], bool], bool]]]] = []
    if ins and all(_is_int(i) for i in ins):
        fams.append(("знаку", [("<0", lambda x: x < 0, False),
                               ("=0", lambda x: x == 0, True),   # ячейка-одиночка
                               (">0", lambda x: x > 0, False)]))
        for m in (2, 3):
            fams.append((f"mod {m}", [(f"≡{r}", (lambda x, r=r, m=m: x % m == r), False) for r in range(m)]))
    return fams


def invent_multibranch(examples: list[tuple[Any, Any]], *, depth: int = 0) -> Primitive | None:
    """Собрать МНОГОВЕТОЧНУЮ операцию из взаимоисключающих предикатов (≥3 активных ветки).

    Каждая ветка выведена из своей подвыборки (с подтверждением). Так из данных
    рождается, например, знак числа: <0→-1, =0→0, >0→1.
    """
    if len(examples) < 4:
        return None
    ins = [i for i, _ in examples]
    for allow_nest in (False, True):                     # сперва плоские ветки, потом вложенные
        for fname, cells in _partition_families(ins):
            branches: list[tuple[str, Callable[[Any], bool], tuple[Callable[[Any], Any], str] | None]] = []
            active = 0
            ok = True
            for label, pred, singleton in cells:
                try:
                    pairs = [(i, o) for i, o in examples if pred(i)]
                except (TypeError, ValueError, ZeroDivisionError):
                    ok = False
                    break
                if not pairs:
                    branches.append((label, pred, None))
                    continue
                fb = _fit_branch(pairs, allow_singleton_const=singleton, depth=depth, allow_nest=allow_nest)
                if fb is None:
                    ok = False
                    break
                branches.append((label, pred, fb))
                active += 1
            if not ok or active < 3:                     # «много» = ≥3 активных ветки (2-way — отдельно)
                continue

            def fn(x, branches=branches):
                for _label, pred, fb in branches:
                    if fb is not None and pred(x):
                        return fb[0](x)
                raise TypeError  # вход вне покрытых случаев

            try:
                if all(_eq(fn(i), o) for i, o in examples):
                    body = "; ".join(f"{lab}→{fb[1]}" for lab, _p, fb in branches if fb)
                    return Primitive(f"по {fname}: {body}", fn)
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                continue
    return None


# ── СЛЕДУЮЩИЙ СЛОЙ: СКОЛЬЗЯЩЕЕ ОКНО / свёртка над списком (разности/префикс/сумма) ──

def invent_window(examples: list[tuple[Any, Any]]) -> Primitive | None:
    """Вывести оконное (свёрточное) преобразование списка из наблюдений.

    Выход — функция СОСЕДНИХ элементов: разности in[i+1]-in[i], префикс-сумма,
    скользящая сумма окна w. Длино-относительно → подтверждаем на ≥2 разных длинах.
    """
    ins = [i for i, _ in examples]
    outs = [o for _, o in examples]
    if not ins or not all(isinstance(i, list) for i in ins) or not all(isinstance(o, list) for o in outs):
        return None
    if len(examples) < 2 or not all(_is_int(v) for i in ins for v in i):
        return None
    if len({len(i) for i in ins}) < 2:
        return None  # длино-относительное окно подтверждаем на ≥2 длинах

    def _diff(xs):
        return [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]

    def _prefix(xs):
        out, s = [], 0
        for v in xs:
            s += v
            out.append(s)
        return out

    templates: list[tuple[str, Callable[[list], list]]] = [("разности", _diff), ("префикс-сумма", _prefix)]
    maxlen = max(len(i) for i in ins)
    for w in range(2, maxlen + 1):
        templates.append((f"скольз.сумма×{w}", (lambda xs, w=w: [sum(xs[i:i + w]) for i in range(len(xs) - w + 1)])))

    for name, fn in templates:
        try:
            if all(fn(i) == o for i, o in examples) and not all(_eq(i, o) for i, o in examples):
                return Primitive(name, _mk_list_fn(fn))
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
    return None


# ── СЛЕДУЮЩИЙ СЛОЙ: операции над СЕТКАМИ из наблюдений (поверх grid-примитивов) ─────

def _is_gridish(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(r, (list, tuple)) for r in v)


def invent_grid(examples: list[tuple[Any, Any]]) -> Primitive | None:
    """Вывести преобразование СЕТКИ из наблюдений: композиция grid-примитивов (≤2 шага).

    Те же flip/transpose/rot… что и в grid-домене, но подбираются ПОИСКОМ под
    наблюдаемые пары сеток (не заданы руками под задачу). ≥2 примеров.
    """
    ins = [i for i, _ in examples]
    outs = [o for _, o in examples]
    if not ins or not all(_is_gridish(i) for i in ins) or not all(_is_gridish(o) for o in outs):
        return None
    if len(examples) < 2:
        return None
    grids = [(to_grid([list(r) for r in i]), to_grid([list(r) for r in o])) for i, o in examples]
    if all(_eq(i, o) for i, o in grids):
        return None
    prog = _compose_induce(grids, grid_primitives(), max_depth=2)
    if prog is None or prog.length == 0:
        return None
    return Primitive(str(prog), prog.__call__)


def _grid(v: Any):
    return to_grid([list(r) for r in v])


def _mk_grid_fn(fn: Callable) -> Callable[[Any], Any]:
    def f(x: Any) -> Any:
        if not _is_gridish(x):
            raise TypeError
        return fn(_grid(x))
    return f


# ── СЛЕДУЮЩИЙ СЛОЙ: 2D-ОКНА над сетками — поэлементная перекраска и морфология ──────

def invent_grid_recolor(examples: list[tuple[Any, Any]]) -> Primitive | None:
    """Вывести ПЕРЕКРАСКУ сетки: согласованное отображение цвет→цвет (out[r][c]=m[in[r][c]]).

    Карта строится по всем клеткам всех примеров; если цвет ведёт в ≠ выходы — это не
    перекраска (None). Неизвестные при применении цвета остаются как есть.
    """
    ins = [i for i, _ in examples]
    outs = [o for _, o in examples]
    if not ins or not all(_is_gridish(i) for i in ins) or not all(_is_gridish(o) for o in outs):
        return None
    g_ex = [(_grid(i), _grid(o)) for i, o in examples]
    cmap: dict[Any, Any] = {}
    for gi, go in g_ex:
        if len(gi) != len(go) or any(len(ri) != len(ro) for ri, ro in zip(gi, go)):
            return None  # перекраска сохраняет форму
        for ri, ro in zip(gi, go):
            for a, b in zip(ri, ro):
                if a in cmap and cmap[a] != b:
                    return None  # неоднозначно → не функция-перекраска
                cmap[a] = b
    changes = {a: b for a, b in cmap.items() if a != b}
    if not changes:
        return None  # тождество — не операция

    def recolor(g, cmap=dict(cmap)):
        return tuple(tuple(cmap.get(v, v) for v in row) for row in g)

    name = "перекраска " + ",".join(f"{a}→{b}" for a, b in sorted(changes.items()))
    return Primitive(name, _mk_grid_fn(recolor))


def _morph(g, agg: Callable, diag: bool):
    rows, cols = len(g), len(g[0])
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    if diag:
        offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            vals = [g[r + dr][c + dc] for dr, dc in offs if 0 <= r + dr < rows and 0 <= c + dc < cols]
            row.append(agg(vals))
        out.append(tuple(row))
    return tuple(out)


def invent_grid_window(examples: list[tuple[Any, Any]]) -> Primitive | None:
    """Вывести 2D-ОКОННОЕ (морфологическое) правило: out[r][c] = агрегат по соседству.

    Дилатация (max) и эрозия (min) по 4- и 8-соседству — классические свёртки над
    сеткой (паттерны соседства). ≥2 примеров, форма сохраняется, не тождество.
    """
    ins = [i for i, _ in examples]
    outs = [o for _, o in examples]
    if not ins or not all(_is_gridish(i) for i in ins) or not all(_is_gridish(o) for o in outs):
        return None
    if len(examples) < 2:
        return None
    g_ex = [(_grid(i), _grid(o)) for i, o in examples]
    templates: list[tuple[str, Callable]] = [
        ("дилатация4", lambda g: _morph(g, max, False)),
        ("эрозия4", lambda g: _morph(g, min, False)),
        ("дилатация8", lambda g: _morph(g, max, True)),
        ("эрозия8", lambda g: _morph(g, min, True)),
    ]
    for name, fn in templates:
        try:
            if all(fn(gi) == go for gi, go in g_ex) and not all(_eq(gi, go) for gi, go in g_ex):
                return Primitive(name, _mk_grid_fn(fn))
        except (TypeError, ValueError, IndexError):
            continue
    return None


def _invent(examples: list[tuple[Any, Any]], *, depth: int = 0) -> Primitive | None:
    """Изобрести операцию из наблюдений (с возможной вложенностью внутри ветвей).

    Порядок — по простоте (Оккам): листовые (прямая/структура/окно/сетка) →
    многоветочное (плоское N-way) → условное (2-way, может вкладывать).
    """
    return (invent_primitive(examples) or invent_structural(examples) or invent_window(examples)
            or invent_grid(examples) or invent_grid_recolor(examples) or invent_grid_window(examples)
            or invent_multibranch(examples, depth=depth)
            or invent_conditional(examples, depth=depth))


def invent(examples: list[tuple[Any, Any]]) -> Primitive | None:
    """Изобрести операцию из наблюдений: прямолинейную → структурную → оконную →

    сеточную → условную → многоветочную (ветки могут быть вложенными правилами).
    """
    return _invent(examples, depth=0)
