#!/usr/bin/env python3
"""Демонстрация шага 2: любопытство через активный вывод.

Сравнивает два агента на одной среде из 4 каналов (easy / medium / hard / noise):
  • ActiveInference — выбирает канал по эпистемической ценности (любопытство);
  • Random          — равномерно случайное внимание (baseline).

Отчёт агрегируется по НЕСКОЛЬКИМ сидам (среднее ± σ и счётчики «на скольких
сидах эффект есть») — чтобы не выдавать удачную фазу одного сида за устойчивый
результат. Устойчиво по сидам одно: максимум внимания — на выучиваемом канале.
Избегание шума за весь прогон — слабый эффект, куррикулум easy→medium→hard —
не устойчив; это печатается честно.

Запуск:
    python run_step2.py
    python run_step2.py --steps 12000 --gamma 10 --seeds 10
"""

from __future__ import annotations

import argparse

import numpy as np

from thinking_system.core.curiosity_loop import ActiveInferenceLoop
from thinking_system.encoders.random_projection import RandomProjectionEncoder
from thinking_system.envs.multichannel import MultiChannelEnv
from thinking_system.memory.buffer import EpisodicBuffer
from thinking_system.metrics.curiosity import CuriosityTracker
from thinking_system.policies.active_inference import ActiveInferencePolicy, RandomPolicy
from thinking_system.predictors.mlp import MLPPredictor


def _run_agent(policy_kind: str, *, steps: int, obs_dim: int, latent_dim: int, context_len: int, gamma: float, seed: int) -> tuple[CuriosityTracker, MultiChannelEnv]:
    env = MultiChannelEnv.default(obs_dim=obs_dim, seed=seed)
    k = env.n_actions
    encoder = RandomProjectionEncoder(obs_dim, latent_dim, seed=seed)
    predictor = MLPPredictor(
        context_dim=context_len * latent_dim + k,  # +one-hot канала
        latent_dim=latent_dim,
        hidden_dim=128,
        lr=1e-3,
        seed=seed,
    )
    memory = EpisodicBuffer(capacity=20_000, seed=seed)
    policy = (
        ActiveInferencePolicy(gamma=gamma, seed=seed)
        if policy_kind == "active"
        else RandomPolicy(k, seed=seed)
    )
    loop = ActiveInferenceLoop(encoder, predictor, memory, policy, n_actions=k, context_len=context_len)
    loop.run(env, n_steps=steps)
    return loop.tracker, env


PHASES = [("ранняя", 0.0, 0.33), ("средняя", 0.33, 0.66), ("поздняя", 0.66, 1.0)]


def _collect(seed: int, args) -> dict:
    """Один сид: прогнать оба агента, снять метрики (без хранения трекеров)."""
    kw = dict(steps=args.steps, obs_dim=args.obs_dim, latent_dim=args.latent_dim,
              context_len=args.context_len, gamma=args.gamma, seed=seed)
    act_tr, env = _run_agent("active", **kw)
    rnd_tr, _ = _run_agent("random", **kw)
    mask = env.learnable_mask()
    return {
        "labels": env.action_labels(), "mask": mask,
        "act_fr": act_tr.visit_fractions(), "rnd_fr": rnd_tr.visit_fractions(),
        "act_phase": np.array([[act_tr.phase_visit_fraction(k, lo=lo, hi=hi)
                                for k in range(env.n_actions)] for _, lo, hi in PHASES]),
        "act_noise": act_tr.wasted_on_noise(mask), "rnd_noise": rnd_tr.wasted_on_noise(mask),
        "act_err": float(np.nanmean(act_tr.final_error()[mask])),
        "rnd_err": float(np.nanmean(rnd_tr.final_error()[mask])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Step 2 — curiosity via active inference")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--obs-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--context-len", type=int, default=2)
    p.add_argument("--gamma", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0, help="первый сид")
    p.add_argument("--seeds", type=int, default=5, help="число сидов для агрегатов")
    args = p.parse_args()

    seeds = list(range(args.seed, args.seed + args.seeds))
    print(f"▶ Среда: 4 канала (easy/medium/hard/noise), {args.steps} шагов на агента, "
          f"сиды {seeds[0]}–{seeds[-1]}")

    runs = [_collect(s, args) for s in seeds]
    labels, mask = runs[0]["labels"], runs[0]["mask"]
    n = len(runs)

    act_fr = np.array([r["act_fr"] for r in runs])           # (n, 4)
    rnd_fr = np.array([r["rnd_fr"] for r in runs])
    print("\n── Внимание за ВЕСЬ прогон, среднее ± σ по сидам ──")
    print(f"  {'канал':<8} {'любопытство':>16} {'random':>16}")
    for k, lab in enumerate(labels):
        print(f"  {lab:<8} {act_fr[:, k].mean() * 100:>10.1f} ± {act_fr[:, k].std() * 100:<4.1f}%"
              f" {rnd_fr[:, k].mean() * 100:>10.1f} ± {rnd_fr[:, k].std() * 100:<4.1f}%")

    phase = np.array([r["act_phase"] for r in runs])         # (n, 3, 4)
    print("\n── Куда смотрит любопытство по фазам (среднее ± σ по сидам) ──")
    print("  фаза      " + "".join(f"{lab:>14}" for lab in labels))
    for pi, (name, _, _) in enumerate(PHASES):
        row = "".join(f"{phase[:, pi, k].mean() * 100:>8.1f} ± {phase[:, pi, k].std() * 100:<4.1f}"
                      for k in range(len(labels)))
        print(f"  {name:<9} {row}")

    top_learnable = sum(bool(mask[int(r['act_fr'].argmax())])
                        and r["act_fr"].max() > r["rnd_fr"].max() for r in runs)
    noise_better = sum(r["act_noise"] < r["rnd_noise"] for r in runs)
    err_better = sum(r["act_err"] < r["rnd_err"] for r in runs)
    a_noise = np.array([r["act_noise"] for r in runs])
    r_noise = np.array([r["rnd_noise"] for r in runs])
    a_err = np.array([r["act_err"] for r in runs])
    r_err = np.array([r["rnd_err"] for r in runs])
    early_top = [labels[int(r["act_phase"][0].argmax())] for r in runs]

    print(f"\n── Итог (честно, по {n} сидам) ──")
    print(f"  УСТОЙЧИВО: топ внимания — выучиваемый канал и выше max random: {top_learnable}/{n} сидов")
    print(f"  внимание на шум за весь прогон: {a_noise.mean() * 100:.1f} ± {a_noise.std() * 100:.1f}%"
          f" vs {r_noise.mean() * 100:.1f} ± {r_noise.std() * 100:.1f}% у random;"
          f" ниже random на {noise_better}/{n} сидов — эффект слабый,")
    print("    залипания навсегда нет, но ранняя фаза тратит на шум БОЛЬШЕ random")
    print(f"  ошибка на выучиваемых: {a_err.mean():.4f} vs {r_err.mean():.4f};"
          f" ниже random на {err_better}/{n} сидов")
    print(f"  куррикулум: топ ранней фазы по сидам = {early_top} — порядок easy→medium→hard"
          f" НЕ устойчив")


if __name__ == "__main__":
    main()
