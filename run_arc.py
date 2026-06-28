#!/usr/bin/env python3
"""ЧЕСТНЫЙ прогон системы на НАСТОЯЩЕМ ARC с held-out проверкой обобщения.

Это воспроизводимая замена прежним заявлениям про ARC (находка аудита C1: числа были
только в сообщении коммита, без данных и загрузчика). Скрипт сам скачивает датасет
ARC-AGI (Apache-2.0, fchollet/ARC-AGI) в локальный кэш и измеряет, сколько задач
система решает ПО-НАСТОЯЩЕМУ:

  «решено» = программа, выведенная поиском из TRAIN-пар, ВЕРНО предсказывает ОТЛОЖЕННУЮ
  test-пару (а не просто подошла к данным примерам).

Запуск:
    python run_arc.py                      # 400 training задач, глубины 1..2
    python run_arc.py --split evaluation   # 400 evaluation задач
    python run_arc.py --max-depth 3 --growth
    python run_arc.py --data-dir путь/к/ARC-AGI/data   # без скачивания

Воспроизведённый результат (training, глубина ≤2, held-out): 22/400 решено, 0 overfit,
8 многошаговых; повторяющиеся комбо «keep_largest ▸ bbox» ×2 и «flip_h ▸ flip_v» ×2.
Рост библиотеки даёт ЭФФЕКТИВНОСТЬ (эти семьи берутся на глубине 1), но +0 к ОХВАТУ;
глубина 3 тоже не расширяет охват — потолок задаёт ШИРИНА примитивов, не глубина поиска.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import signal
import time
import urllib.request

from thinking_system.reasoning.grids import to_grid
from thinking_system.reasoning.grid_seed import full_grid_seed
from thinking_system.reasoning.induction import Library, Primitive
from thinking_system.reasoning.library_learning import LibraryLearner
from thinking_system.reasoning.search_prior import best_first_induce

_API = "https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}?per_page=100&page={page}"
_CACHE = os.path.join(os.path.dirname(__file__), ".arc_cache")
_CELL_CAP = 2000  # держим сетки малыми при поиске (ARC ≤900 клеток; раздувание fractal/mirror_quad отсекаем)


# ── загрузка датасета ────────────────────────────────────────────────────────────
def ensure_data(split: str, data_dir: str | None) -> str:
    if data_dir:
        path = os.path.join(data_dir, split)
        if not glob.glob(os.path.join(path, "*.json")):
            raise SystemExit(f"нет .json в {path}")
        return path
    path = os.path.join(_CACHE, split)
    os.makedirs(path, exist_ok=True)
    if len(glob.glob(os.path.join(path, "*.json"))) >= 400:
        return path
    print(f"скачиваю ARC-AGI/{split} в {path} …", flush=True)
    urls: list[str] = []
    for page in range(1, 6):
        with urllib.request.urlopen(_API.format(split=split, page=page), timeout=30) as r:
            items = json.load(r)
        if not items:
            break
        urls += [it["download_url"] for it in items if it["name"].endswith(".json")]
    for i, u in enumerate(urls):
        dst = os.path.join(path, os.path.basename(u))
        if not os.path.exists(dst):
            with urllib.request.urlopen(u, timeout=30) as r:
                open(dst, "wb").write(r.read())
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(urls)}", flush=True)
    print(f"готово: {len(glob.glob(os.path.join(path, '*.json')))} задач", flush=True)
    return path


# ── решатель с защитой от раздувания сеток и таймаутом на задачу ────────────────────
def _cells(g) -> int:
    return len(g) * (len(g[0]) if g else 0)


def _guard(p: Primitive) -> Primitive:
    def fn(x, _f=p.fn):
        if isinstance(x, tuple) and x and isinstance(x[0], tuple) and _cells(x) > _CELL_CAP:
            raise ValueError("too big")
        y = _f(x)
        if isinstance(y, tuple) and y and isinstance(y[0], tuple) and _cells(y) > _CELL_CAP:
            raise ValueError("too big")
        return y
    return Primitive(p.name, fn, p.cost)


class _Timeout(Exception):
    ...


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))


def load_task(path):
    d = json.load(open(path))
    tr = [(to_grid(p["input"]), to_grid(p["output"])) for p in d["train"]]
    te = [(to_grid(p["input"]), to_grid(p["output"])) for p in d["test"]]
    return tr, te


def _heldout_ok(prog, test) -> bool:
    try:
        return all(prog(i) == o for i, o in test)
    except Exception:
        return False


def evaluate(files, prims, depth, *, budget, timeout):
    """Вернуть (решённые held-out, overfit train✓/test✗, программы, timeouts)."""
    prims = prims.prims if isinstance(prims, Library) else prims
    solved, overfit, programs, timeouts = [], [], {}, 0
    for path in files:
        name = os.path.basename(path)[:-5]
        try:
            tr, te = load_task(path)
        except Exception:
            continue
        signal.alarm(timeout)
        try:
            prog, _ = best_first_induce(tr, prims, None, max_depth=depth, budget=budget)
        except _Timeout:
            timeouts += 1; signal.alarm(0); continue
        except Exception:
            signal.alarm(0); continue
        signal.alarm(0)
        if prog is None:
            continue
        (solved if _heldout_ok(prog, te) else overfit).append(name)
        if name in solved:
            programs[name] = prog
    return solved, overfit, programs, timeouts


def main() -> None:
    ap = argparse.ArgumentParser(description="Честный held-out прогон на ARC")
    ap.add_argument("--split", choices=["training", "evaluation"], default="training")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--budget", type=int, default=2500)
    ap.add_argument("--timeout", type=int, default=4, help="сек на задачу")
    ap.add_argument("--data-dir", default=None, help="локальный ARC-AGI/data (иначе — скачать)")
    ap.add_argument("--growth", action="store_true", help="проверить рост библиотеки: охват vs эффективность")
    args = ap.parse_args()

    data = ensure_data(args.split, args.data_dir)
    files = sorted(glob.glob(os.path.join(data, "*.json")))
    seed = [_guard(p) for p in full_grid_seed()]
    print(f"\nARC {args.split}: {len(files)} задач | seed={len(seed)} прим. | budget={args.budget} | held-out\n")

    all_progs: dict = {}
    last_solved: list[str] = []
    for depth in range(1, args.max_depth + 1):
        t0 = time.time()
        solved, overfit, progs, tos = evaluate(files, seed, depth, budget=args.budget, timeout=args.timeout)
        all_progs.update(progs)
        last_solved = solved
        ms = sum(1 for p in progs.values() if p.length >= 2)
        print(f"depth {depth}: РЕШЕНО held-out {len(solved)}/{len(files)} | "
              f"overfit(train✓/test✗) {len(overfit)} | многошаг {ms} | timeouts {tos} | {time.time()-t0:.0f}s")

    combos = collections.Counter(str(p) for p in all_progs.values() if p.length >= 2)
    rep = {k: c for k, c in combos.items() if c >= 2}
    print(f"\nповторяющиеся многошаговые комбо (≥2): {rep or 'нет'}")

    if args.growth:
        learner = LibraryLearner(seed)
        added = learner.grow_from_solutions(list(all_progs.values()), top=10, min_count=2)
        print(f"\nрост библиотеки из найденных решений -> {added}")
        s1_raw, _, _, _ = evaluate(files, seed, 1, budget=args.budget, timeout=args.timeout)
        s1_grw, _, _, _ = evaluate(files, learner.lib, 1, budget=args.budget, timeout=args.timeout)
        s2_grw, _, _, _ = evaluate(files, learner.lib, args.max_depth, budget=args.budget, timeout=args.timeout)
        print(f"ЭФФЕКТИВНОСТЬ (глубина 1): сырой {len(s1_raw)} -> после роста {len(s1_grw)} "
              f"(+{len(set(s1_grw)-set(s1_raw))})")
        print(f"ОХВАТ (глубина {args.max_depth}): сырой {len(last_solved)} -> после роста {len(s2_grw)} "
              f"(новых: {sorted(set(s2_grw)-set(last_solved)) or 'НЕТ — рост даёт эффективность, не охват'})")


if __name__ == "__main__":
    main()
