#!/usr/bin/env python3
"""Демонстрация MVP: замкнутый цикл предсказание → ошибка → обучение.

Запуск:
    python run_mvp.py
    python run_mvp.py --steps 8000 --obs-dim 48 --latent-dim 24 --plot

Что показывает: скользящая ошибка предсказания падает со временем и опускается
ниже тривиального baseline (persistence) — измеримый признак того, что система
со временем всё лучше «понимает» временную структуру потока.
"""

from __future__ import annotations

import argparse

from thinking_system import (
    EpisodicBuffer,
    MetricsTracker,
    MLPPredictor,
    PredictiveLoop,
    RandomProjectionEncoder,
    SyntheticStream,
)
from thinking_system.viz import maybe_save_plot, sparkline


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Brain-inspired thinking system — MVP predictive loop")
    p.add_argument("--steps", type=int, default=6000, help="число шагов потока")
    p.add_argument("--obs-dim", type=int, default=32, help="размерность наблюдения")
    p.add_argument("--latent-dim", type=int, default=16, help="размерность латента")
    p.add_argument("--context-len", type=int, default=2, help="латентов в контексте")
    p.add_argument("--hidden-dim", type=int, default=128, help="ширина скрытого слоя предиктора")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--noise", type=float, default=0.02, help="шум потока")
    p.add_argument("--seed", type=int, default=0, help="зерно ГСЧ")
    p.add_argument("--plot", action="store_true", help="сохранить PNG (нужен matplotlib)")
    return p.parse_args()


def main() -> None:
    args = build_args()

    stream = SyntheticStream(obs_dim=args.obs_dim, noise=args.noise, seed=args.seed)
    encoder = RandomProjectionEncoder(args.obs_dim, args.latent_dim, seed=args.seed)
    predictor = MLPPredictor(
        context_dim=args.context_len * args.latent_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
    )
    memory = EpisodicBuffer(capacity=10_000, seed=args.seed)
    loop = PredictiveLoop(
        encoder,
        predictor,
        memory,
        context_len=args.context_len,
        tracker=MetricsTracker(window=200),
    )

    print(f"▶ Запуск цикла: {args.steps} шагов, latent_dim={args.latent_dim}, context_len={args.context_len}\n")
    tracker = loop.run(stream, n_steps=args.steps)
    s = tracker.summary()

    print("── Кривая ошибки предсказания (скользящее среднее) ──")
    print("  " + sparkline(tracker.rolling()))
    print("  baseline  " + sparkline(tracker.rolling(tracker.baselines)))
    print()
    print("── Сводка ──")
    print(f"  эпизодов в памяти       : {len(memory)}")
    print(f"  начальная ошибка        : {s['initial_error']:.5f}")
    print(f"  финальная ошибка        : {s['final_error']:.5f}")
    print(f"  baseline (persistence)  : {s['baseline_final']:.5f}")
    print(f"  baseline (лин. экстрап.) : {s['linear_final']:.5f}")
    print(f"  улучшение vs нач. ошибка : {s['improvement_pct']:.1f}%")
    print(f"  обходит persistence     : {'да' if s['beats_baseline'] else 'нет'}")
    print(f"  обходит лин. экстрап.   : {'да' if s['beats_linear'] else 'нет'}")
    halve = s["steps_to_halve_error"]
    print(f"  шагов до −50% ошибки     : {int(halve) if halve == halve else '—'}")
    print()
    print("  ⚠ persistence — слабейшая планка; «улучшение %» зависит от уровня шума")
    print("    (≈99% при noise=0.02, ~67% при noise=0.5) и от случайного старта сети.")
    print("    Линейная экстраполяция — честная планка без обучения.")

    if args.plot:
        path = maybe_save_plot(tracker)
        hint = 'matplotlib не установлен (pip install -e ".[viz]")'
        print(f"\n  график: {path if path else hint}")


if __name__ == "__main__":
    main()
