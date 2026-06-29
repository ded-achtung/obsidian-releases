"""Композиционный синтез: система САМА строит схему из типизированных атомов.

Раньше каждую схему писал человек (synth_colormap, synth_select_object…). Здесь иначе:
даны АТОМЫ (детерминированные grid→grid преобразования), а ДВИЖОК сам ищет их
композицию (= схему) и подбирает параметры хвостовой операции из данных. Тогда цепочки
вроде «крупнейший_объект ▸ перекраска» или «поворот ▸ масштаб» РОЖДАЮТСЯ поиском, а не
пишутся руками — шаг к тому, чтобы система строила схемы сама.

Стратегия: префикс из детерминированных атомов (поиск) + хвост-операция с параметрами,
выведенными из данных (`invent`). Параметры хвоста выводятся между ВЫХОДОМ префикса и
истинным выходом — поэтому не нужны промежуточные цели.
"""

from __future__ import annotations

from itertools import product

from thinking_system.reasoning.grids import flip_h, flip_v, transpose, rot90, rot180
from thinking_system.reasoning.objects_arc import objects, background, select_by
from thinking_system.reasoning.invent import invent


def _crop_content(g):
    bg = background(g)
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != bg]
    if not cells:
        return None
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    return tuple(tuple(g[r][c] for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))


def _select_crop(prop, mode):
    def f(g):
        o = select_by(objects(g, bg=background(g), color_blind=True), prop, mode)
        return None if o is None else o["sub"]
    return f


# типизированные АТОМЫ grid→grid (детерминированные); композиции строит движок
ATOMS = [
    ("flip_h", flip_h), ("flip_v", flip_v), ("transpose", transpose),
    ("rot90", rot90), ("rot180", rot180),
    ("crop", _crop_content),
    ("largest", _select_crop("size", "max")),
    ("smallest", _select_crop("size", "min")),
]


def _apply_seq(seq, g):
    for _, fn in seq:
        try:
            g = fn(g)
        except Exception:  # noqa: BLE001
            return None
        if g is None:
            return None
    return g


def _compose(seq, tail):
    def fn(g):
        m = _apply_seq(seq, g)
        if m is None:
            return None
        return m if tail is None else tail(m)
    return fn


def synthesize(train, *, max_prefix: int = 2):
    """Найти композицию атомов (+ хвост-операцию из данных), согласованную со всеми парами.

    Возвращает (описание_программы, функция) или (None, None). Схему строит поиск, а не
    автор: префикс из атомов перебирается, хвостовые параметры выводятся `invent`.
    """
    inputs = [i for i, _ in train]
    outputs = [o for _, o in train]
    for d in range(0, max_prefix + 1):
        for combo in product(range(len(ATOMS)), repeat=d):
            seq = [ATOMS[k] for k in combo]
            mids = [_apply_seq(seq, i) for i in inputs]
            if any(m is None for m in mids):
                continue
            label = " ▸ ".join(n for n, _ in seq)
            if mids == outputs:                              # чистая детерминированная схема
                return (label or "id"), _compose(seq, None)
            name, tail = invent(list(zip(mids, outputs)))    # хвост: параметры из данных
            if tail is None:
                continue
            full = _compose(seq, tail)
            try:
                if all(full(i) == o for i, o in train):
                    return ((label + " ▸ " if label else "") + name), full
            except Exception:  # noqa: BLE001
                continue
    return None, None
