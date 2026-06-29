"""Тест бизнес-правил: индукция интерпретируемого правила решения + честные метрики."""

from __future__ import annotations

import numpy as np

from thinking_system.business.rules import (
    balanced_accuracy, one_rule, rule_predict, decision_list,
)


def test_balanced_accuracy_handles_imbalance() -> None:
    """При дисбалансе «всегда 0» даёт balanced-acc 0.5, а не высокую accuracy."""
    y = np.array([0] * 95 + [1] * 5)
    pred_all0 = np.zeros(100, dtype=bool)
    assert abs(balanced_accuracy(y, pred_all0) - 0.5) < 1e-9


def test_one_rule_finds_separating_threshold() -> None:
    """OneR находит признак и порог, разделяющие классы."""
    rng = np.random.default_rng(0)
    n = 400
    drive = rng.normal(0, 1, n)                          # признак-драйвер
    noise = rng.normal(0, 1, n)                          # шумовой признак
    y = (drive > 0.5).astype(int)
    X = np.column_stack([noise, drive])                 # драйвер — второй столбец
    r = one_rule(X, y, ["noise", "drive"])
    assert r["feature"] == "drive"                       # выбрал правильный признак
    assert r["gt"] is True and 0.2 < r["thr"] < 0.8      # и верный порог
    assert balanced_accuracy(y, rule_predict(r, X)) > 0.9


def test_default_dataset_driver_is_balance() -> None:
    """На реальном ISLR Default система видит драйвер дефолта = BALANCE (не income/student)."""
    from run_business import load
    X, y = load("data/Default.csv")
    r = one_rule(X, y, ["balance", "income", "student"])
    assert r["feature"] == "balance"                     # верный бизнес-драйвер
    assert balanced_accuracy(y, rule_predict(r, X)) > 0.85


def test_decision_list_returns_precise_rules() -> None:
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(0, 1, n)
    y = (x > 1.0).astype(int)
    X = x.reshape(-1, 1)
    rules = decision_list(X, y, ["x"], k=2)
    assert len(rules) >= 1
    assert all(r["precision"] >= 0.5 for r in rules)
