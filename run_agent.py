#!/usr/bin/env python3
"""Единый когнитивный контур на реальных данных — всё вместе.

Агент сам читает три реальных источника (английский / русский / код), выбирая
куда смотреть (любопытство), предсказывая байты (модель мира), используя ошибку
предсказания (surprise) и запоминая опыт с консолидацией (replay).

Показывает:
  1) УЧИТСЯ        — общий surprise (бит/байт) падает по ходу;
  2) ЛЮБОПЫТСТВО   — как распределил внимание между источниками;
  3) ПАМЯТЬ        — с консолидацией удерживает все источники; без неё (маленькая
                     память) забывает менее свежие — аблация;
  4) ИСПОЛЬЗУЕТ    — сканирует новый код и по surprise находит опечатку.

Запуск: python run_agent.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from thinking_system.agent.cognitive import CognitiveAgent
from thinking_system.text.ingest import load_book
from thinking_system.text.vocab import ByteVocab
from thinking_system.viz import sparkline

VOCAB = ByteVocab()


def _bar(bits: np.ndarray, text: str) -> None:
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = bits.min(), max(bits.max(), bits.min() + 1e-6)
    line = "".join(blocks[min(7, int((b - lo) / (hi - lo) * 7))] for b in bits)
    safe = "".join(c if 32 <= ord(c) < 127 else "·" for c in text)
    print(f"   текст:    {safe}")
    print(f"   surprise: {line}")


def build_sources() -> list[tuple[str, np.ndarray]]:
    english = load_book("books/sample_mathematics.md")
    russian = load_book("research/brain-inspired-thinking-system.md")
    code = "\n\n".join(
        Path(p).read_text(encoding="utf-8")
        for p in sorted(glob.glob("thinking_system/predictors/*.py") + sorted(glob.glob("thinking_system/core/*.py")))
    )
    return [
        ("english", VOCAB.encode(english)),
        ("russian", VOCAB.encode(russian)),
        ("code", VOCAB.encode(code)),
    ]


def main() -> None:
    sources = build_sources()
    sizes = "  ".join(f"{n}:{len(d) // 1000}КБ" for n, d in sources)
    print(f"▶ Источники: {sizes}\n")

    STEPS = 45000

    print("обучаю единый контур (с консолидацией)…")
    agent = CognitiveAgent(sources, memory_capacity=200_000, seed=0).run(STEPS)

    # 1) УЧИТСЯ — общий surprise падает
    bits = np.array(agent.bits)
    curve = [float(bits[i : i + 1500].mean()) for i in range(0, len(bits) - 1500, len(bits) // 60)]
    print("\n1) УЧИТСЯ — общий surprise (бит/байт) по ходу:")
    print("   " + sparkline(curve) + f"   {curve[0]:.2f} → {curve[-1]:.2f}")

    # 2) ЛЮБОПЫТСТВО — распределение внимания
    print("\n2) ЛЮБОПЫТСТВО — доля внимания по источникам:")
    vf = agent.visits / agent.visits.sum()
    for k, n in enumerate(agent.names):
        print(f"   {n:8}: {vf[k] * 100:5.1f}%")

    # 3) ПАМЯТЬ — удержание всех источников vs аблация без консолидации
    print("\n3) ПАМЯТЬ — BPC по источникам сейчас (меньше = помнит):")
    print("   обучаю аблацию (маленькая память, без консолидации)…")
    ablation = CognitiveAgent(sources, memory_capacity=1500, seed=0).run(STEPS)
    print(f"   {'источник':10}{'с консолидацией':>18}{'без (recency)':>16}")
    for k, n in enumerate(agent.names):
        print(f"   {n:10}{agent.source_bpc(k):>18.2f}{ablation.source_bpc(k):>16.2f}")
    print("   (с консолидацией — ниже/ровнее по всем; без — забывает менее свежие)")

    # 4) ИСПОЛЬЗУЕТ — находит опечатку в новом коде
    print("\n4) ИСПОЛЬЗУЕТ выученное — находит аномалию в НОВОМ коде:")
    line = "        retunr self.W1"   # опечатка return→retunr
    _bar(agent.scan(line), line)
    good = "        return self.W1"
    print(f"   суммарно бит: верно {agent.scan(good).sum():.1f}  vs  опечатка {agent.scan(line).sum():.1f}")

    print("\n── Итог ──")
    print("   Один контур: выбор источника (любопытство) → восприятие+предсказание →")
    print("   surprise (детект аномалий) → память+replay. Тот же сигнал ошибки")
    print("   предсказания и учит, и направляет внимание, и находит аномалии.")


if __name__ == "__main__":
    main()
