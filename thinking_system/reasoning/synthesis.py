"""Синтез примитивов ИЗ ДАННЫХ задачи — выход за рамки фиксированного словаря.

Находка аудита: на ARC система решала только короткие КОМПОЗИЦИИ из ~23 руками
закодированных примитивов; «рост библиотеки» лишь переименовывал комбо, а охват
упирался в ШИРИНУ словаря (глубина 3 и абстракции давали +0 задач). Чтобы потолок
перестал быть зафиксирован заранее, нужен механизм, который СОЗДАЁТ новые операции,
которых в seed нет и которые не выражаются его композицией.

Это перенос идеи dsl_growth (там Δ-сдвиги синтезировались из наблюдений мира) на домен
сеток: по TRAIN-парам задачи синтезируем КОНКРЕТНЫЕ параметрические операции —
  • colormap*   — выученная по данным замена цветов (не «color+1», а нужная таблица);
  • upscale*    — поблочное увеличение в k раз (фактор выведен из данных);
  • tile*       — простое размножение сетки a×b раз (фактор выведен из данных);
  • downscale*  — обратное прореживание в k раз.
Каждая операция ВЫВЕДЕНА из конкретной задачи (её не было в языке до встречи с данными)
и проверена end-to-end на ВСЕХ train-парах, затем — на ОТЛОЖЕННОЙ test-паре. Примитивы
добавляются к seed как обычные (поиск может их и компоновать), но поскольку они
проверяются как ЦЕЛОЕ преобразование вход→выход, на практике решают сами по себе на
глубине 1 (colormap — позиционно, поэтому не синтезируется из задач со сдвигом/поворотом).

Честно о границах: это не «любое новое понятие», а несколько семейств параметрических
операций. Но это уже настоящий рост ВЫРАЗИТЕЛЬНОСТИ из данных, а не рекомбинация — и он
решает задачи, недостижимые фиксированным словарём (см. замер в run_arc.py).
"""

from __future__ import annotations

from thinking_system.reasoning.induction import Primitive

Grid = tuple


def _dims(g) -> tuple[int, int]:
    return len(g), (len(g[0]) if g else 0)


def synth_colormap(train: list[tuple]) -> Primitive | None:
    """Согласованная по всем парам замена цвет→цвет (та же форма входа/выхода)."""
    mapping: dict[int, int] = {}
    for inp, out in train:
        if _dims(inp) != _dims(out) or _dims(inp)[0] == 0:
            return None
        for ri, ro in zip(inp, out):
            for a, b in zip(ri, ro):
                if a in mapping and mapping[a] != b:
                    return None  # не функция цвета → не colormap
                mapping[a] = b
    if not mapping or all(k == v for k, v in mapping.items()):
        return None  # тождество — смысла нет

    def fn(g, m=dict(mapping)):
        return tuple(tuple(m.get(v, v) for v in row) for row in g)

    return Primitive("colormap*", fn)


def _consistent_factor(train, *, mode: str) -> tuple[int, int] | None:
    """Единый целочисленный фактор (kr,kc) по всем парам для up/tile/down."""
    fr = fc = None
    for inp, out in train:
        ir, ic = _dims(inp)
        orr, oc = _dims(out)
        if 0 in (ir, ic, orr, oc):
            return None
        if mode == "down":
            big, small = (ir, ic), (orr, oc)
        else:
            big, small = (orr, oc), (ir, ic)
        if big[0] % small[0] or big[1] % small[1]:
            return None
        kr, kc = big[0] // small[0], big[1] // small[1]
        if (kr, kc) == (1, 1):
            return None
        if fr is None:
            fr, fc = kr, kc
        elif (kr, kc) != (fr, fc):
            return None
    return (fr, fc) if fr is not None else None


def _verify(train, fn) -> bool:
    try:
        return all(fn(i) == o for i, o in train)
    except Exception:
        return False


def synth_upscale(train: list[tuple]) -> Primitive | None:
    """Поблочное увеличение в kr×kc раз: каждая клетка → блок kr×kc (фактор из данных)."""
    f = _consistent_factor(train, mode="up")
    if f is None:
        return None
    fr, fc = f

    def fn(g, fr=fr, fc=fc):
        return tuple(tuple(g[r // fr][c // fc] for c in range(len(g[0]) * fc))
                     for r in range(len(g) * fr))

    return Primitive(f"upscale*{fr}x{fc}", fn) if _verify(train, fn) else None


def synth_tile(train: list[tuple]) -> Primitive | None:
    """Размножение сетки kr×kc раз (повтор содержимого, фактор из данных)."""
    f = _consistent_factor(train, mode="up")
    if f is None:
        return None
    fr, fc = f

    def fn(g, fr=fr, fc=fc):
        ir, ic = len(g), len(g[0])
        return tuple(tuple(g[r % ir][c % ic] for c in range(ic * fc)) for r in range(ir * fr))

    return Primitive(f"tile*{fr}x{fc}", fn) if _verify(train, fn) else None


def synth_downscale(train: list[tuple]) -> Primitive | None:
    """Прореживание в kr×kc раз: из каждого блока берём верхне-левую клетку (фактор из данных)."""
    f = _consistent_factor(train, mode="down")
    if f is None:
        return None
    fr, fc = f

    def fn(g, fr=fr, fc=fc):
        return tuple(tuple(g[r * fr][c * fc] for c in range(len(g[0]) // fc))
                     for r in range(len(g) // fr))

    return Primitive(f"downscale*{fr}x{fc}", fn) if _verify(train, fn) else None


_SYNTHS = (synth_colormap, synth_upscale, synth_tile, synth_downscale)


def synthesize(train: list[tuple]) -> list[Primitive]:
    """Синтезировать список параметрических примитивов, выведенных из TRAIN-пар задачи.

    Каждый кандидат проверен на всех train-парах. Возвращается набор, который дополняет
    seed и компонуется с ним при поиске. Имена синтезированных примитивов оканчиваются
    на «*» — по ним видно, что решение вышло за рамки исходного словаря.
    """
    out: list[Primitive] = []
    seen: set[str] = set()
    for f in _SYNTHS:
        try:
            p = f(train)
        except Exception:
            p = None
        if p is not None and p.name not in seen:
            out.append(p)
            seen.add(p.name)
    return out
