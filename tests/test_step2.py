"""Тесты шага 2: среда, политика и устойчивое поведение любопытства."""

from __future__ import annotations

import numpy as np

from thinking_system import (
    ActiveInferenceLoop,
    ActiveInferencePolicy,
    EnsemblePredictor,
    EpisodicBuffer,
    MultiChannelEnv,
    RandomPolicy,
    RandomProjectionEncoder,
)
from thinking_system.core.curiosity_loop import ActiveInferenceLoop as Loop
from thinking_system.metrics.curiosity import CuriosityTracker
from thinking_system.predictors.mlp import MLPPredictor


def test_env_shapes_and_mask() -> None:
    env = MultiChannelEnv.default(obs_dim=20, seed=0)
    env.reset()
    assert env.n_actions == 4
    assert env.step(0).shape == (20,)
    assert env.learnable_mask().tolist() == [True, True, True, False]  # noise последний


def test_active_policy_prefers_high_epistemic() -> None:
    policy = ActiveInferencePolicy(gamma=8.0, seed=0)
    epi = np.array([0.0, 0.0, 1.0, 0.0])  # 3-й канал ценнее
    picks = [policy.select_action(epi) for _ in range(400)]
    counts = np.bincount(picks, minlength=4)
    assert counts.argmax() == 2  # чаще выбирает самый ценный


def test_random_policy_is_uniform() -> None:
    policy = RandomPolicy(4, seed=0)
    picks = [policy.select_action() for _ in range(4000)]
    counts = np.bincount(picks, minlength=4) / 4000
    assert np.all(np.abs(counts - 0.25) < 0.05)  # примерно равномерно


def test_ensemble_disagreement_nonnegative() -> None:
    pred = EnsemblePredictor(context_dim=10, latent_dim=4, n_members=3, seed=0)
    ctx = np.ones(10)
    assert pred.predict(ctx).shape == (4,)
    assert pred.disagreement(ctx) >= 0.0


def _run(policy_kind: str, *, steps: int, seed: int) -> tuple[CuriosityTracker, MultiChannelEnv]:
    env = MultiChannelEnv.default(obs_dim=24, seed=seed)
    k = env.n_actions
    enc = RandomProjectionEncoder(24, 12, seed=seed)
    pred = MLPPredictor(context_dim=2 * 12 + k, latent_dim=12, hidden_dim=96, lr=1e-3, seed=seed)
    mem = EpisodicBuffer(capacity=20_000, seed=seed)
    policy = ActiveInferencePolicy(gamma=8.0, seed=seed) if policy_kind == "active" else RandomPolicy(k, seed=seed)
    loop = Loop(enc, pred, mem, policy, n_actions=k, context_len=2)
    loop.run(env, n_steps=steps)
    return loop.tracker, env


def test_curiosity_concentrates_on_learnable_channel() -> None:
    # Устойчивый сигнал эпистемического фуражинга: любопытство концентрирует
    # внимание на самом «ещё-выучиваемом» канале, а не размазывает равномерно,
    # и эта концентрация — на ВЫУЧИВАЕМОМ канале, не на шуме.
    act, env = _run("active", steps=10_000, seed=0)
    rnd, _ = _run("random", steps=10_000, seed=0)
    mask = env.learnable_mask()
    a_fr = act.visit_fractions()

    assert a_fr.max() > rnd.visit_fractions().max() + 0.03  # концентрирует сильнее равномерного
    assert bool(mask[int(a_fr.argmax())])                   # и на выучиваемом канале, не на шуме


def test_imports_exposed() -> None:
    # экспорт публичного API шага 2
    assert ActiveInferenceLoop is Loop
