#!/usr/bin/env python3
"""Демонстрация шага 3: консолидация/replay против катастрофического забывания.

Последовательность из 4 различных задач (режимов) предъявляется по очереди одной
общей модели мира. Сравниваем два режима:
  • naive  — без повторения (память сбрасывается на каждой задаче);
  • replay — CLS-консолидация (память хранится между задачами, переигрывается).

Ожидаемо: naive катастрофически забывает старые задачи (ошибка на них растёт
после обучения новым), replay — почти нет.

Запуск:
    python run_step3.py
    python run_step3.py --tasks 5 --steps-per-task 4000
"""

from __future__ import annotations

import argparse

import numpy as np

from thinking_system.consolidation.continual import ContinualLearner
from thinking_system.encoders.random_projection import RandomProjectionEncoder
from thinking_system.memory.buffer import EpisodicBuffer
from thinking_system.metrics.continual import ContinualEvaluator
from thinking_system.predictors.mlp import MLPPredictor
from thinking_system.tasks.sequence import make_task_sequence


def _run(replay: bool, *, tasks, steps_per_task: int, obs_dim: int, latent_dim: int, context_len: int, seed: int) -> tuple[ContinualEvaluator, dict]:
    encoder = RandomProjectionEncoder(obs_dim, latent_dim, seed=seed)
    predictor = MLPPredictor(context_dim=context_len * latent_dim, latent_dim=latent_dim, hidden_dim=128, lr=1e-3, seed=seed)
    learner = ContinualLearner(
        encoder,
        predictor,
        memory_factory=lambda: EpisodicBuffer(capacity=50_000, seed=seed),
        replay=replay,
        context_len=context_len,
    )
    evaluator = ContinualEvaluator(encoder, tasks, context_len=context_len)
    learner.run(tasks, steps_per_task, evaluator)
    return evaluator, evaluator.metrics()


def _print_matrix(name: str, ev: ContinualEvaluator) -> None:
    m = ev.matrix()
    names = ev.task_names
    print(f"\n── {name}: ошибка по задачам после каждой обученной задачи ──")
    print("  после\\оценка " + "".join(f"{n:>9}" for n in names))
    labels = ["(случ.)"] + [f"+{n}" for n in names]
    for r, row in enumerate(m):
        print(f"  {labels[r]:<11} " + "".join(f"{v:>9.4f}" for v in row))


def main() -> None:
    p = argparse.ArgumentParser(description="Step 3 — consolidation / replay vs catastrophic forgetting")
    p.add_argument("--tasks", type=int, default=4)
    p.add_argument("--steps-per-task", type=int, default=4000)
    p.add_argument("--obs-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--context-len", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tasks = make_task_sequence(args.tasks, obs_dim=args.obs_dim, seed=args.seed)
    print(f"▶ {args.tasks} различных задач × {args.steps_per_task} шагов, одна общая модель мира")

    common = dict(tasks=tasks, steps_per_task=args.steps_per_task, obs_dim=args.obs_dim, latent_dim=args.latent_dim, context_len=args.context_len, seed=args.seed)
    naive_ev, naive_m = _run(False, **common)
    replay_ev, replay_m = _run(True, **common)

    _print_matrix("NAIVE (без replay)", naive_ev)
    _print_matrix("REPLAY (CLS-консолидация)", replay_ev)

    # Забывание задачи 1 наглядно: её ошибка по ходу обучения всем задачам (столбец 0).
    print("\n── Ошибка на ЗАДАЧЕ 1 по мере обучения задачам 1→N ──")
    print(f"  naive : {np.array2string(naive_ev.matrix()[1:, 0], precision=4, floatmode='fixed')}  ← растёт = забывает")
    print(f"  replay: {np.array2string(replay_ev.matrix()[1:, 0], precision=4, floatmode='fixed')}  ← держится = помнит")

    print("\n── Итог (continual-learning метрики) ──")
    print(f"  {'метрика':<26}{'naive':>10}{'replay':>10}")
    print(f"  {'ACC (фин. ошибка, ↓)':<26}{naive_m['acc']:>10.4f}{replay_m['acc']:>10.4f}")
    print(f"  {'forgetting (рост ошибки, ↓)':<26}{naive_m['forgetting']:>10.4f}{replay_m['forgetting']:>10.4f}")
    print(f"  {'BWT (↑ лучше)':<26}{naive_m['bwt']:>10.4f}{replay_m['bwt']:>10.4f}")
    print(f"  {'FWT (↑ лучше)':<26}{naive_m['fwt']:>10.4f}{replay_m['fwt']:>10.4f}")
    verdict = "да" if replay_m["forgetting"] < naive_m["forgetting"] and replay_m["acc"] < naive_m["acc"] else "нет"
    print(f"\n  replay-консолидация снизила забывание и финальную ошибку: {verdict}")


if __name__ == "__main__":
    main()
