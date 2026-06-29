"""Интерпретируемые правила решения: «как система ВИДИТ бизнес» — IF признак THEN класс.

Тот же дух индукции правил, что и в reasoning, но над табличными бизнес-признаками:
ищем короткое читаемое правило (порог по признаку), объясняющее решения. Метрика —
сбалансированная точность (важно при дисбалансе классов, типичном для бизнеса:
дефолт/отток — редкое событие).
"""

from __future__ import annotations

import numpy as np


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Среднее из recall по обоим классам — честно при дисбалансе (в отличие от accuracy)."""
    pos, neg = y == 1, y == 0
    rp = float(pred[pos].mean()) if pos.any() else 0.0
    rn = float((~pred[neg]).mean()) if neg.any() else 0.0
    return 0.5 * (rp + rn)


def one_rule(X: np.ndarray, y: np.ndarray, names: list[str], *, n_thr: int = 200) -> dict:
    """OneR: лучший единичный порог по признаку (максимум сбалансированной точности)."""
    best = None
    for j in range(X.shape[1]):
        col = X[:, j]
        for thr in np.unique(np.quantile(col, np.linspace(0.01, 0.99, n_thr))):
            for gt in (True, False):
                pred = (col > thr) if gt else (col <= thr)
                s = balanced_accuracy(y, pred)
                if best is None or s > best["score"]:
                    best = {"feature": names[j], "j": j, "thr": float(thr), "gt": bool(gt), "score": s}
    return best


def rule_predict(rule: dict, X: np.ndarray) -> np.ndarray:
    col = X[:, rule["j"]]
    return (col > rule["thr"]) if rule["gt"] else (col <= rule["thr"])


def rule_text(rule: dict) -> str:
    op = ">" if rule["gt"] else "≤"
    return f"ЕСЛИ {rule['feature']} {op} {rule['thr']:.0f} → класс 1 (иначе 0)"


def decision_list(X: np.ndarray, y: np.ndarray, names: list[str], *, k: int = 3, n_thr: int = 200) -> list[dict]:
    """Жадный список правил: пороги, выделяющие чистые положительные области (точность×покрытие)."""
    rules: list[dict] = []
    remaining = np.ones(len(y), dtype=bool)
    for _ in range(k):
        idx = np.where(remaining)[0]
        if len(idx) < 10 or y[idx].sum() == 0:
            break
        Xr, yr = X[idx], y[idx]
        best = None
        for j in range(X.shape[1]):
            col = Xr[:, j]
            for thr in np.unique(np.quantile(col, np.linspace(0.01, 0.99, n_thr))):
                for gt in (True, False):
                    pred = (col > thr) if gt else (col <= thr)
                    cov = int(pred.sum())
                    if cov < 5:
                        continue
                    prec = float(yr[pred].mean())
                    score = prec * np.sqrt(cov)            # точность × √покрытие
                    if best is None or score > best["_score"]:
                        best = {"feature": names[j], "j": j, "thr": float(thr), "gt": bool(gt),
                                "precision": prec, "coverage": cov, "_score": score}
        if best is None or best["precision"] < 0.5:
            break
        rules.append(best)
        covered = rule_predict(best, X) & remaining
        remaining &= ~covered                              # снять покрытые этим правилом
    return rules
