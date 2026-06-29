"""Объектный уровень для ARC: сегментация + порождённое пространство правил.

Шаг к «изобретению самих семейств», а не параметров: вместо пиксельных шаблонов
система работает с ОБЪЕКТАМИ (связные компоненты) и ИЩЕТ правило в порождённом
комбинаторном пространстве — «выбрать объект, экстремальный по свойству P», «перекрасить
объект по свойству→цвет». Конкретное правило («взять самый крупный», «уникальный по
цвету») выбирается ПО ДАННЫМ, и таких правил — много, а не четыре написанных руками.

Честно: схемы (селекция/перекраска) всё ещё авторские; но их КОНКРЕТНЫЕ инстансы
порождаются и отбираются данными — это шире, чем фиксированные синтезаторы.
"""

from __future__ import annotations

from collections import Counter

Grid = tuple


def _shape(g):
    return (len(g), len(g[0]) if g else 0)


def background(grid: Grid) -> int:
    return Counter(v for row in grid for v in row).most_common(1)[0][0]


def components(grid: Grid, *, bg: int, color_blind: bool) -> list[list[tuple[int, int]]]:
    """Связные компоненты (4-связность) не-фоновых клеток; color_blind — соединять любые цвета."""
    H, W = _shape(grid)
    seen = [[False] * W for _ in range(H)]
    comps = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == bg or seen[r][c]:
                continue
            col = grid[r][c]
            stack, cells = [(r, c)], []
            seen[r][c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] and grid[ny][nx] != bg \
                            and (color_blind or grid[ny][nx] == col):
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def obj_view(grid: Grid, cells: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(cells)
    sub = tuple(tuple(grid[r][c] for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))
    colors = {grid[r][c] for r, c in cells}
    return {"cells": cells, "bbox": (r0, c0, r1, c1), "sub": sub,
            "size": len(cells), "h": r1 - r0 + 1, "w": c1 - c0 + 1,
            "ncolors": len(colors), "color": Counter(grid[r][c] for r, c in cells).most_common(1)[0][0]}


PROPS = ["size", "h", "w", "ncolors"]


def objects(grid: Grid, *, bg: int, color_blind: bool) -> list[dict]:
    return [obj_view(grid, cells) for cells in components(grid, bg=bg, color_blind=color_blind)]


def select_by(objs: list[dict], prop: str, mode: str):
    """Выбрать объект: экстремум по свойству (max/min) или уникальный по значению."""
    if not objs:
        return None
    if mode == "max":
        return max(objs, key=lambda o: o[prop])
    if mode == "min":
        return min(objs, key=lambda o: o[prop])
    if mode == "unique":                                # объект с уникальным значением свойства
        cnt = Counter(o[prop] for o in objs)
        uniq = [o for o in objs if cnt[o[prop]] == 1]
        return uniq[0] if len(uniq) == 1 else None
    return None
