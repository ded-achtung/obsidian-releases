#!/usr/bin/env python3
"""Единая репрезентация: язык и мир в ОДНОМ пространстве, поиск работает в обе стороны.

Разрыв «нет общей валюты»: одна операция (близость в общем пространстве) служит и
грунтингу (язык→клетка), и описанию (клетка→язык) — чего классификатор не умеет
(он односторонний). Меряем кросс-модальный поиск на held-out фразах; бейзлайн —
без выравнивания (случайные проекции) ≈ случайность.

Запуск: python run_shared_space.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.language.grounding import CharNgram, PLACES, generate_commands
from thinking_system.language.shared_space import SharedSpace, region_of_cell


def split(seed=0, frac=0.7):
    data = generate_commands()
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    cut = int(frac * len(data))
    return data[:cut], data[cut:]


def lang_to_cell_acc(space, test, cells):
    """язык→клетка: ближайшая клетка в общем пространстве попадает в нужный регион."""
    ce = np.array([space.embed_cell(c) for c in cells])
    ok = 0
    for text, region in test:
        q = space.embed_text(text)
        nearest = cells[int(np.argmin(((ce - q) ** 2).sum(1)))]
        ok += int(region_of_cell(nearest, space.size) == region)
    return ok / len(test)


def cell_to_lang_acc(space, test, cells):
    """клетка→язык: ближайшая фраза в общем пространстве описывает регион клетки."""
    te = np.array([space.embed_text(t) for t, _ in test])
    treg = [r for _, r in test]
    ok = 0
    for c in cells:
        q = space.embed_cell(c)
        j = int(np.argmin(((te - q) ** 2).sum(1)))
        ok += int(treg[j] == region_of_cell(c, space.size))
    return ok / len(cells)


def main():
    size = 7
    cells = [(s // size, s % size) for s in range(size * size)]
    train, test = split()
    feat = CharNgram([t for t, _ in train])

    print("▶ Единое пространство: язык и мир — одна общая валюта, поиск в обе стороны\n")

    space = SharedSpace(dim=24, seed=0, trained=True)
    space.fit_language(train, feat).fit_world(cells, size)

    base = SharedSpace(dim=24, seed=0, trained=False)   # без выравнивания (контроль)
    base.fit_language(train, feat).fit_world(cells, size)

    print(f"   обучено на {len(train)} командах; общее пространство dim={space.dim}, {len(cells)} клеток\n")
    print("   кросс-модальный поиск (held-out фразы), доля попаданий в регион:")
    print(f"   {'направление':<26}{'без выравнивания':>18}{'единое пространство':>22}")
    l2c_b = lang_to_cell_acc(base, test, cells); l2c = lang_to_cell_acc(space, test, cells)
    c2l_b = cell_to_lang_acc(base, test, cells); c2l = cell_to_lang_acc(space, test, cells)
    print(f"   {'язык → клетка (грунтинг)':<26}{l2c_b * 100:>17.0f}%{l2c * 100:>21.0f}%")
    print(f"   {'клетка → язык (описание)':<26}{c2l_b * 100:>17.0f}%{c2l * 100:>21.0f}%")

    print("\n   пример описания клетки словами через ТО ЖЕ пространство:")
    te = np.array([space.embed_text(t) for t, _ in test])
    for c in [(0, 0), (0, 6), (3, 3), (6, 6)]:
        q = space.embed_cell(c)
        j = int(np.argmin(((te - q) ** 2).sum(1)))
        print(f"   клетка {str(c):<8} → «{test[j][0]}»  (регион {PLACES[region_of_cell(c, size)][0]})")

    print("\n── Что это значит ──")
    print("   Язык и мир живут в ОДНОМ пространстве: одна операция близости и грунтит команду")
    print("   в локацию, и описывает локацию словами — двунаправленно, чего классификатор не")
    print("   умеет. Без выравнивания (контроль) кросс-модальный поиск рушится до случайного.")
    print("   Честно: общая валюта здесь над фиксированными якорями регионов — шаг к единому")
    print("   «языку мысли», а не универсальный субстрат для всех модулей сразу.")


if __name__ == "__main__":
    main()
