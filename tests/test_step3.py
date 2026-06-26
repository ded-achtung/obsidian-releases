"""Тесты шага 3: последовательность задач, оценщик и эффект replay-консолидации."""

from __future__ import annotations

import numpy as np

from thinking_system import (
    ContinualEvaluator,
    ContinualLearner,
    EpisodicBuffer,
    MLPPredictor,
    RandomProjectionEncoder,
    make_task_sequence,
)


def test_task_sequence_distinct_and_streams() -> None:
    tasks = make_task_sequence(3, obs_dim=16, seed=0)
    assert len(tasks) == 3
    assert [t.name for t in tasks] == ["task1", "task2", "task3"]
    obs = next(iter(tasks[0].stream()))
    assert obs.shape == (16,)


def test_evaluator_eval_shapes() -> None:
    tasks = make_task_sequence(3, obs_dim=16, seed=0)
    enc = RandomProjectionEncoder(16, 8, seed=0)
    ev = ContinualEvaluator(enc, tasks, context_len=2, n_eval=32)
    pred = MLPPredictor(context_dim=2 * 8, latent_dim=8, seed=0)
    errs = ev.eval_errors(pred)
    assert errs.shape == (3,)
    assert np.all(errs >= 0)


def _forgetting(replay: bool, *, seed: int) -> dict:
    tasks = make_task_sequence(3, obs_dim=24, seed=seed)
    enc = RandomProjectionEncoder(24, 12, seed=seed)
    pred = MLPPredictor(context_dim=2 * 12, latent_dim=12, hidden_dim=96, lr=1e-3, seed=seed)
    learner = ContinualLearner(
        enc, pred, memory_factory=lambda: EpisodicBuffer(30_000, seed=seed), replay=replay, context_len=2
    )
    ev = ContinualEvaluator(enc, tasks, context_len=2, n_eval=128)
    learner.run(tasks, steps_per_task=2500, evaluator=ev)
    return ev.metrics()


def test_replay_prevents_forgetting() -> None:
    naive = _forgetting(False, seed=0)
    replay = _forgetting(True, seed=0)

    assert naive["forgetting"] > 0.05                       # наивно реально забывает
    assert replay["forgetting"] < naive["forgetting"] * 0.5  # replay резко снижает забывание
    assert replay["acc"] < naive["acc"]                      # и финальная ошибка ниже


def test_matrix_includes_random_init_row() -> None:
    tasks = make_task_sequence(3, obs_dim=16, seed=0)
    enc = RandomProjectionEncoder(16, 8, seed=0)
    pred = MLPPredictor(context_dim=2 * 8, latent_dim=8, seed=0)
    learner = ContinualLearner(enc, pred, memory_factory=lambda: EpisodicBuffer(10_000, seed=0), replay=True, context_len=2)
    ev = ContinualEvaluator(enc, tasks, context_len=2, n_eval=32)
    learner.run(tasks, steps_per_task=600, evaluator=ev)
    assert ev.matrix().shape == (4, 3)  # строка случ. инициализации + 3 задачи
