#!/usr/bin/env python3
"""Демонстрация шага 2: любопытство через активный вывод.

Сравнивает два агента на одной среде из 4 каналов (easy / medium / hard / noise):
  • ActiveInference — выбирает канал по эпистемической ценности (любопытство);
  • Random          — равномерно случайное внимание (baseline).

Ожидаемый результат: любопытный агент КОНЦЕНТРИРУЕТ внимание на ещё-выучиваемом
канале и тратит МЕНЬШЕ внимания на шумный (нередуцируемый «шумный телевизор»).
Утверждение об устойчивости проверяется ЧЕСТНО — прогоном по нескольким сидам со
сводкой «в скольких сидах из N свойство выполнено» (а не одним удачным сидом).
Замечание: это про РАСПРЕДЕЛЕНИЕ ВНИМАНИЯ; меньшая итоговая ошибка на выучиваемых
каналах из этого НЕ гарантируется и отдельно отмечается в сводке.

Запуск:
    python run_step2.py                 # 5 сидов (по умолчанию)
    python run_step2.py --seeds 10 --steps 12000 --gamma 10
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
from thinking_system.viz import sparkline


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


def _report(name: str, tracker: CuriosityTracker, env: MultiChannelEnv) -> None:
    labels = env.action_labels()
    fr = tracker.visit_fractions()
    err = tracker.final_error()
    print(f"\n── {name} ──")
    print(f"  {'канал':<8} {'внимание':>9} {'фин.ошибка':>12}")
    for k, lab in enumerate(labels):
        e = "—" if np.isnan(err[k]) else f"{err[k]:.4f}"
        print(f"  {lab:<8} {fr[k] * 100:>8.1f}% {e:>12}")
    waste = tracker.wasted_on_noise(env.learnable_mask())
    print(f"  внимание на шум: {waste * 100:.1f}%")


def _seed_metrics(seed: int, **kw) -> dict:
    """Прогнать оба агента на одном сиде и собрать сравнительные метрики."""
    act_tr, env = _run_agent("active", seed=seed, **kw)
    rnd_tr, _ = _run_agent("random", seed=seed, **kw)
    mask = env.learnable_mask()
    a_fr, r_fr = act_tr.visit_fractions(), rnd_tr.visit_fractions()
    a_top = int(a_fr.argmax())
    return {
        "env": env, "act_tr": act_tr, "rnd_tr": rnd_tr, "mask": mask,
        "act_top_frac": float(a_fr.max()), "rnd_top_frac": float(r_fr.max()),
        "act_top_learnable": bool(mask[a_top]),
        "act_noise": float(act_tr.wasted_on_noise(mask)),
        "rnd_noise": float(rnd_tr.wasted_on_noise(mask)),
        "act_err_learn": float(np.nanmean(act_tr.final_error()[mask])),
        "rnd_err_learn": float(np.nanmean(rnd_tr.final_error()[mask])),
    }


def _mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    s = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return m, s


def main() -> None:
    p = argparse.ArgumentParser(description="Step 2 — curiosity via active inference")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--obs-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--context-len", type=int, default=2)
    p.add_argument("--gamma", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0, help="стартовый сид")
    p.add_argument("--seeds", type=int, default=5, help="сколько сидов прогнать (для честной устойчивости)")
    args = p.parse_args()

    seeds = [args.seed + i for i in range(max(1, args.seeds))]
    print(f"▶ Среда: 4 канала (easy/medium/hard/noise), {args.steps} шагов на агента; сиды {seeds}")

    kw = dict(steps=args.steps, obs_dim=args.obs_dim, latent_dim=args.latent_dim,
              context_len=args.context_len, gamma=args.gamma)
    runs = [_seed_metrics(s, **kw) for s in seeds]
    first = runs[0]
    env, act_tr, rnd_tr, mask = first["env"], first["act_tr"], first["rnd_tr"], first["mask"]

    # ── подробный разбор одного (первого) сида ────────────────────────────────────
    print(f"\n(подробности для сида {seeds[0]})")
    _report("Любопытство (active inference)", act_tr, env)
    _report("Случайное внимание (baseline)", rnd_tr, env)

    noise_ch = int(np.where(~mask)[0][0])
    labels = env.action_labels()
    phases = [("ранняя", 0.0, 0.33), ("средняя", 0.33, 0.66), ("поздняя", 0.66, 1.0)]
    print("\n── Куда смотрит ЛЮБОПЫТСТВО по фазам (доля внимания) ──")
    print("  фаза      " + "".join(f"{lab:>9}" for lab in labels))
    for name, lo, hi in phases:
        row = "".join(f"{act_tr.phase_visit_fraction(k, lo=lo, hi=hi) * 100:>8.1f}%" for k in range(env.n_actions))
        print(f"  {name:<9} {row}")
    print("\n  для сравнения СЛУЧАЙНЫЙ агент (поздняя фаза):")
    print("           " + "".join(f"{rnd_tr.phase_visit_fraction(k, lo=0.66, hi=1.0) * 100:>8.1f}%" for k in range(env.n_actions)))
    print("\n  внимание к шуму во времени (любопытство):")
    print("  " + sparkline(act_tr.windowed_visit_fraction(noise_ch)))

    # ── честная сводка ПО ВСЕМ сидам ──────────────────────────────────────────────
    n = len(runs)
    concentrated = sum(1 for r in runs if r["act_top_frac"] > r["rnd_top_frac"] and r["act_top_learnable"])
    avoid_noise = sum(1 for r in runs if r["act_noise"] < r["rnd_noise"])
    lower_err = sum(1 for r in runs if r["act_err_learn"] < r["rnd_err_learn"])
    a_top_m, a_top_s = _mean_std([r["act_top_frac"] for r in runs])
    r_top_m, r_top_s = _mean_std([r["rnd_top_frac"] for r in runs])
    a_noi_m, a_noi_s = _mean_std([r["act_noise"] for r in runs])
    r_noi_m, r_noi_s = _mean_std([r["rnd_noise"] for r in runs])

    print(f"\n── Итог по {n} сидам (честная устойчивость, а не один сид) ──")
    print(f"  концентрация на выучиваемом канале (любопытство > случайного): {concentrated}/{n} сидов")
    print(f"    макс. внимание: любопытство {a_top_m * 100:.1f}±{a_top_s * 100:.1f}%  vs  случайно {r_top_m * 100:.1f}±{r_top_s * 100:.1f}%")
    print(f"  избегание шума (внимание на шум меньше случайного): {avoid_noise}/{n} сидов")
    print(f"    внимание на шум: любопытство {a_noi_m * 100:.1f}±{a_noi_s * 100:.1f}%  vs  случайно {r_noi_m * 100:.1f}±{r_noi_s * 100:.1f}%")
    print(f"  меньше ошибка на выучиваемых каналах: {lower_err}/{n} сидов")
    print("    (честно: это про РАСПРЕДЕЛЕНИЕ ВНИМАНИЯ; меньшая итоговая ошибка не гарантирована)")


if __name__ == "__main__":
    main()
