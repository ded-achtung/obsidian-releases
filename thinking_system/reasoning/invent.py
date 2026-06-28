"""Изобретение примитива ИЗ ДАННЫХ задачи (а не из готового словаря).

Потолок фиксированного DSL: примитивы беспараметрические (`color+1`, фиксированный
`fractal`). Но многие ARC-задачи — это операция, ПАРАМЕТРЫ которой надо вывести из
самой задачи: какая именно перекраска, какой масштаб, какое замощение. Здесь —
СИНТЕЗАТОРЫ: каждый читает пары вход→выход, СТРОИТ конкретную функцию (новую теорию),
проверяет её на ВСЕХ обучающих парах и, если согласована, возвращает — это и есть
«построить теорию из данных и проверить её», а не перебор готового.

Идея — параметризованные операции из обзора решателей ARC (Recolor(map), Scale(k),
Tile) и ShapeCoder (абстракции из неструктурированных примитивов).
"""

from __future__ import annotations

from collections import Counter

Grid = tuple


def _shape(g: Grid) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def synth_colormap(train: list[tuple[Grid, Grid]]):
    """Вывести ПОКЛЕТОЧНУЮ перекраску цвет→цвет (если форма сохраняется)."""
    mapping: dict[int, int] = {}
    for inp, out in train:
        if _shape(inp) != _shape(out) or _shape(inp)[0] == 0:
            return None
        for ri, ro in zip(inp, out):
            for a, b in zip(ri, ro):
                if a in mapping and mapping[a] != b:
                    return None
                mapping[a] = b
    if all(v == k for k, v in mapping.items()):     # тождество — неинтересно
        return None
    m = dict(mapping)
    return lambda g: tuple(tuple(m.get(v, v) for v in r) for r in g)


def _factors(train, *, up):
    f = None
    for inp, out in train:
        ir, ic = _shape(inp); orr, oc = _shape(out)
        big, small = ((orr, oc), (ir, ic)) if up else ((ir, ic), (orr, oc))
        if small[0] == 0 or small[1] == 0 or big[0] % small[0] or big[1] % small[1]:
            return None
        a, b = big[0] // small[0], big[1] // small[1]
        if (a, b) == (1, 1) or a < 1 or b < 1:
            return None
        if f is None:
            f = (a, b)
        elif f != (a, b):
            return None
    return f


def synth_upscale(train):
    """Вывести целочисленный масштаб ВВЕРХ и правило заполнения (репликация / замощение)."""
    f = _factors(train, up=True)
    if f is None:
        return None
    a, b = f

    def replicate(g):                                # каждая клетка → блок a×b того же цвета
        out = []
        for row in g:
            br = [[] for _ in range(a)]
            for v in row:
                for k in range(a):
                    br[k].extend([v] * b)
            out += [tuple(x) for x in br]
        return tuple(out)

    def tile(g):                                     # периодическое замощение a×b копий
        ir, ic = _shape(g)
        return tuple(tuple(g[r % ir][c % ic] for c in range(b * ic)) for r in range(a * ir))

    for fn in (replicate, tile):
        if all(fn(i) == o for i, o in train):
            return fn
    return None


def synth_downscale(train):
    """Вывести масштаб ВНИЗ: каждый блок → один цвет (мажоритарный / единственный непустой)."""
    f = _factors(train, up=False)
    if f is None:
        return None
    a, b = f

    def reduce_block(pick):
        def fn(g):
            ir, ic = _shape(g); orr, oc = ir // a, ic // b
            res = []
            for R in range(orr):
                row = []
                for C in range(oc):
                    vals = [g[R * a + dr][C * b + dc] for dr in range(a) for dc in range(b)]
                    row.append(pick(vals))
                res.append(tuple(row))
            return tuple(res)
        return fn

    majority = reduce_block(lambda v: Counter(v).most_common(1)[0][0])
    nonzero = reduce_block(lambda v: next((x for x in v if x != 0), 0))
    for fn in (majority, nonzero):
        if all(fn(i) == o for i, o in train):
            return fn
    return None


def synth_mosaic(train):
    """Вывести мозаику a×b из копий входа, преобразованных {id, flip_h, flip_v, rot180}.

    Покрывает «вход рядом со своим зеркалом» (1×2, 2×1) и калейдоскоп (2×2): для каждой
    ячейки сетки a×b подбирается преобразование, согласованное со всеми обучающими парами.
    """
    from itertools import product

    f = _factors(train, up=True)
    if f not in {(1, 2), (2, 1), (2, 2)}:
        return None
    a, b = f
    fh = lambda g: tuple(r[::-1] for r in g)
    fv = lambda g: g[::-1]
    r180 = lambda g: tuple(r[::-1] for r in g[::-1])
    ops = {"id": lambda g: g, "fh": fh, "fv": fv, "r180": r180}
    names = list(ops)

    def make(layout):                                # layout[r][c] = имя операции для ячейки
        def fn(g):
            rows = []
            for r in range(a):
                strips = [ops[layout[r * b + c]](g) for c in range(b)]
                rows += [tuple(sum((s[i] for s in strips), ())) for i in range(len(strips[0]))]
            return tuple(rows)
        return fn

    for combo in product(names, repeat=a * b):
        fn = make(combo)
        if all(fn(i) == o for i, o in train):
            return fn
    return None


def synth_select_object(train):
    """ИЗОБРЕСТИ правило выбора объекта: вывести (фон, связность, свойство, режим) по данным.

    Пространство правил ПОРОЖДАЕТСЯ (фон×связность×свойство×{max,min,unique}); конкретное
    правило выбирается тем, что согласуется со всеми парами. Выход = объект, обрезанный
    до bbox. Это семейство «извлеки особый объект» — много правил, не одно.
    """
    from thinking_system.reasoning.objects_arc import objects, select_by, background, PROPS

    for bgm in ("common", "zero"):
        for cb in (False, True):
            for prop in PROPS:
                for mode in ("max", "min", "unique"):
                    def sel(g, bgm=bgm, cb=cb, prop=prop, mode=mode):
                        bg = background(g) if bgm == "common" else 0
                        o = select_by(objects(g, bg=bg, color_blind=cb), prop, mode)
                        return None if o is None else o["sub"]
                    try:
                        if all(sel(i) == o for i, o in train):
                            return sel
                    except Exception:  # noqa: BLE001
                        continue
    return None


def synth_object_recolor(train):
    """ИЗОБРЕСТИ перекраску объектов по свойству: вывести правило свойство-объекта→цвет."""
    from thinking_system.reasoning.objects_arc import objects, background, PROPS

    for bgm in ("common", "zero"):
        for cb in (False, True):
            for prop in PROPS:
                mapping, ok = {}, True
                for inp, out in train:
                    if _shape(inp) != _shape(out):
                        ok = False; break
                    bg = background(inp) if bgm == "common" else 0
                    for ob in objects(inp, bg=bg, color_blind=cb):
                        ocols = {out[r][c] for r, c in ob["cells"]}
                        if len(ocols) != 1:
                            ok = False; break
                        oc = next(iter(ocols)); key = ob[prop]
                        if key in mapping and mapping[key] != oc:
                            ok = False; break
                        mapping[key] = oc
                    if not ok:
                        break
                if not ok or not mapping or all(False for _ in [0]):
                    continue

                def fn(g, bgm=bgm, cb=cb, prop=prop, m=dict(mapping)):
                    bg = background(g) if bgm == "common" else 0
                    grid = [list(row) for row in g]
                    for ob in objects(g, bg=bg, color_blind=cb):
                        if ob[prop] in m:
                            for r, c in ob["cells"]:
                                grid[r][c] = m[ob[prop]]
                    return tuple(tuple(row) for row in grid)
                try:
                    if all(fn(i) == o for i, o in train):
                        return fn
                except Exception:  # noqa: BLE001
                    continue
    return None


INVENTORS = [
    ("colormap", synth_colormap),
    ("upscale", synth_upscale),
    ("downscale", synth_downscale),
    ("mosaic", synth_mosaic),
    ("select_object", synth_select_object),
    ("object_recolor", synth_object_recolor),
]


def invent(train: list[tuple[Grid, Grid]]):
    """Перебрать синтезаторы; вернуть (имя, функция) для ПЕРВОГО, согласованного со всеми парами."""
    for name, synth in INVENTORS:
        try:
            fn = synth(train)
        except Exception:  # noqa: BLE001
            fn = None
        if fn is not None and all(fn(i) == o for i, o in train):
            return name, fn
    return None, None
