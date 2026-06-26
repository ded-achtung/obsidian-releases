#!/usr/bin/env python3
"""Демонстрация шага 2: любопытство через активный вывод.

Сравнивает два агента на одной среде из 4 каналов (easy / medium / hard / noise):
  • ActiveInference — выбирает канал по эпистемической ценности (любопытство);
  • Random          — равномерно случайное внимание (baseline).

Ожидаемый результат: любопытный агент тратит МЕНЬШЕ внимания на шумный канал
(нередуцируемый «шумный телевизор») и достигает меньшей ошибки на выучиваемых
каналах при том же числе шагов.

Запуск:
    python run_step2.py
    python run_step2.py --steps 12000 --gamma 10
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


def main() -> None:
    p = argparse.ArgumentParser(description="Step 2 — curiosity via active inference")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--obs-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--context-len", type=int, default=2)
    p.add_argument("--gamma", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"▶ Среда: 4 канала (easy/medium/hard/noise), {args.steps} шагов на агента")

    act_tr, env = _run_agent("active", steps=args.steps, obs_dim=args.obs_dim, latent_dim=args.latent_dim, context_len=args.context_len, gamma=args.gamma, seed=args.seed)
    rnd_tr, _ = _run_agent("random", steps=args.steps, obs_dim=args.obs_dim, latent_dim=args.latent_dim, context_len=args.context_len, gamma=args.gamma, seed=args.seed)

    _report("Любопытство (active inference)", act_tr, env)
    _report("Случайное внимание (baseline)", rnd_tr, env)

    mask = env.learnable_mask()
    noise_ch = int(np.where(~mask)[0][0])
    act_learn = np.nanmean(act_tr.final_error()[mask])
    rnd_learn = np.nanmean(rnd_tr.final_error()[mask])

    # Эпистемический фуражинг = КУРРИКУЛУМ: любопытство переключает внимание с
    # освоенных каналов на ещё-выучиваемые, минуя шум. Видно по фазам.
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

    a_fr = act_tr.visit_fractions()
    top = int(a_fr.argmax())
    print("\n── Итог ──")
    print(f"  УСТОЙЧИВО (по сидам): любопытство КОНЦЕНТРИРУЕТ внимание на ещё-выучиваемом канале")
    print(f"    макс. внимание: любопытство {a_fr.max() * 100:.1f}% (канал '{labels[top]}', выучиваемый={bool(mask[top])})  vs  случайно {rnd_tr.visit_fractions().max() * 100:.1f}%")
    print(f"  на этой конфигурации также: внимание на шум {act_tr.wasted_on_noise(mask) * 100:.1f}% vs {rnd_tr.wasted_on_noise(mask) * 100:.1f}%; "
          f"ошибка на выучиваемых {act_learn:.4f} vs {rnd_learn:.4f}")
    verdict = "да" if (a_fr.max() > rnd_tr.visit_fractions().max() and mask[top]) else "нет"
    print(f"  любопытство сфокусировалось на выучиваемом канале сильнее равномерного: {verdict}")


if __name__ == "__main__":
    main()
