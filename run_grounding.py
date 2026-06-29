#!/usr/bin/env python3
"""Заземление: представление, выученное из СЫРЫХ пикселей без меток, бьёт сырые пиксели.

Разрыв «символы вырезаны руками»: восприятие было оракулом (вектор-признак на клетку).
Здесь агент видит сырые зашумлённые глифы и учит представление БЕЗ меток (noise2noise:
предсказать один зашумлённый вид по другому). Ценность представления меряем честно —
по ДЕФИЦИТУ меток: классификатор ближайшего центроида получает лишь k примеров на тип.

  • сырые пиксели (baseline) — центроиды шумные → при малом k слабо;
  • выученный латент       — денойз убрал шум → при малом k уже хорошо;
  • чистый глиф (оракул)   — верхняя граница (идеальные признаки).

Honestly: это представление перцептов из сырого входа (шаг к заземлению), не полный
символ-грундинг; «объекты» здесь — типы клеток, рендеренные в фиксированные глифы.

Запуск: python run_grounding.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.encoders.learned import DenoisingAutoencoder
from thinking_system.world.gridworld import default_maze
from thinking_system.world.pixels import GlyphWorld


def centroid_accuracy(repr_fn, world: GlyphWorld, *, k: int, n_test: int = 1200, seed: int = 0) -> float:
    """Классификатор ближайшего центроида: k меток на тип; точность на свежих наблюдениях."""
    rng = np.random.default_rng(seed)
    by_type: dict[int, list[int]] = {}
    for s in world.free:
        by_type.setdefault(world.type_of[s], []).append(s)
    # центроиды из k размеченных наблюдений на тип
    cents = {}
    for t, cells in by_type.items():
        obs = np.array([world.observe(cells[int(rng.integers(len(cells)))]) for _ in range(k)])
        cents[t] = repr_fn(obs).mean(0)
    cm = np.array([cents[t] for t in sorted(cents)])
    types = sorted(cents)
    # тест на свежих наблюдениях
    ok = 0
    for _ in range(n_test):
        c = world.free[int(rng.integers(len(world.free)))]
        z = repr_fn(world.observe(c))[0]
        pred = types[int(np.argmin(((cm - z) ** 2).sum(1)))]
        ok += int(pred == world.type_of[c])
    return ok / n_test


def main():
    grid = default_maze()
    world = GlyphWorld(grid, patch=5, noise=0.8, seed=0)
    print(f"▶ Заземление: представление из сырых пикселей (глиф {world.dim}D + шум {world.noise}), "
          f"{world.n_types} типов клеток\n")

    # 1) обучаем энкодер БЕЗ меток: noise2noise на парах видов одной клетки
    A, B, _ = world.sample_pairs(12000)
    enc = DenoisingAutoencoder(world.dim, latent_dim=12, hidden=64, lr=5e-3, seed=0)
    enc.fit(A, B, epochs=60, batch=128)
    print("1) энкодер обучен на 12000 НЕразмеченных пар наблюдений (метки типов не использованы)\n")

    raw = lambda X: np.atleast_2d(X)
    learned = lambda X: enc.encode(X)
    oracle = lambda X: np.atleast_2d(X)  # для оракула передаём чистые глифы (см. ниже)

    # оракул считаем на чистых глифах: репрезентация без шума = верхняя граница
    class OracleWorld(GlyphWorld):
        def observe(self, cell):
            return self.clean(cell)
    oracle_world = OracleWorld(grid, patch=5, noise=0.0, seed=0)

    print("2) точность распознавания типа клетки при ДЕФИЦИТЕ меток (k на тип):")
    print(f"   {'k меток/тип':<14}{'сырые пиксели':>16}{'выученный латент':>18}{'оракул (чистый)':>18}")
    for k in (1, 2, 5):
        a_raw = centroid_accuracy(raw, world, k=k, seed=1)
        a_lat = centroid_accuracy(learned, world, k=k, seed=1)
        a_ora = centroid_accuracy(oracle, oracle_world, k=k, seed=1)
        print(f"   {k:<14}{a_raw * 100:>15.0f}%{a_lat * 100:>17.0f}%{a_ora * 100:>17.0f}%")

    print("\n── Что это значит ──")
    print("   Энкодер без единой метки убрал шум (noise2noise), и при дефиците меток выученный")
    print("   латент распознаёт тип клетки заметно лучше сырых пикселей, приближаясь к оракулу")
    print("   с идеальными признаками. Восприятие теперь ВЫУЧЕНО из сырого входа, а не задано.")
    print("   Честно: это шаг к заземлению (перцепты из пикселей), не полный символ-грундинг —")
    print("   «объекты» здесь = типы клеток; следующий шаг — растить и сами примитивы из перцептов.")


if __name__ == "__main__":
    main()
