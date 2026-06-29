#!/usr/bin/env python3
"""Бизнес-бенчмарк: как система ВИДИТ бизнес — предсказание + интерпретируемое правило.

Датасет ISLR «Default» (10 000 клиентов): предсказать дефолт по кредитке из признаков
balance / income / student. Классика, бизнес-смысл, сильный дисбаланс (3.3% дефолтов —
как реальный отток/дефолт). Поэтому accuracy ОБМАНЧИВА (предсказывай «нет» → 96.7%); честно
смотрим AUC и recall по редкому классу. Плюс извлекаем читаемое правило «как система
видит бизнес».

Данные: data/Default.csv (из statsmodels get_rdataset('Default','ISLR')).
Запуск: python run_business.py
"""

from __future__ import annotations

import csv

import numpy as np

from thinking_system.agent.unified import SoftmaxClassifier
from thinking_system.business.rules import one_rule, decision_list, rule_predict, balanced_accuracy, rule_text

NAMES = ["balance", "income", "student"]


def load(path="data/Default.csv"):
    bal, inc, stu, y = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            bal.append(float(row["balance"])); inc.append(float(row["income"]))
            stu.append(1.0 if row["student"].strip() == "Yes" else 0.0)
            y.append(1 if row["default"].strip() == "Yes" else 0)
    X = np.column_stack([bal, inc, stu])
    return X, np.array(y)


def auc(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    alls = np.concatenate([pos, neg])
    ranks = np.empty(len(alls)); ranks[alls.argsort()] = np.arange(1, len(alls) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_precision(y, pred):
    pos = y == 1
    tp = float((pred & pos).sum())
    rec = tp / max(pos.sum(), 1)
    prec = tp / max(pred.sum(), 1)
    return rec, prec


def main():
    X, y = load()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    print(f"▶ Бизнес-бенчмарк ISLR Default: {len(y)} клиентов, дефолтов {100*y.mean():.1f}% (редкий класс)")
    print(f"  train {len(tr)} / held-out {len(te)}\n")

    # стандартизация для линейной модели (по train)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    # 1) бейзлайн «всегда не дефолт» — показать, почему accuracy обманчива
    maj_acc = float((yte == 0).mean())
    print("1) ПОЧЕМУ ACCURACY ОБМАНЧИВА:")
    print(f"   бейзлайн «предсказывай 'нет дефолта'»: accuracy {maj_acc*100:.1f}%, но recall дефолтов 0%")
    print("   → на дисбалансе смотрим AUC и recall, а не accuracy\n")

    # 2) классификатор системы (линейный softmax)
    clf = SoftmaxClassifier(3, 2, lr=0.3); clf.fit(Ztr, ytr, epochs=2000)
    proba = clf.proba(Zte)[:, 1]
    pred = proba >= 0.5
    rec, prec = recall_precision(yte, pred)
    # порог под баланс (Youden по train)
    ptr = clf.proba(Ztr)[:, 1]
    thrs = np.quantile(ptr, np.linspace(0.5, 0.99, 50))
    best_thr = max(thrs, key=lambda t: balanced_accuracy(ytr, ptr >= t))
    pred_b = proba >= best_thr
    rec_b, prec_b = recall_precision(yte, pred_b)
    print("2) КЛАССИФИКАТОР СИСТЕМЫ (линейный softmax = логрегрессия):")
    print(f"   AUC {auc(proba, yte):.3f};  accuracy {float((pred==yte).mean())*100:.1f}%")
    print(f"   при пороге 0.5:  recall дефолтов {rec*100:.0f}%, precision {prec*100:.0f}%")
    print(f"   при пороге под баланс ({best_thr:.2f}): recall {rec_b*100:.0f}%, precision {prec_b*100:.0f}%, "
          f"balanced-acc {balanced_accuracy(yte, pred_b)*100:.0f}%")
    w = clf.W[:, 1] - clf.W[:, 0]
    print(f"   ВЕС признаков (стандартизованные): " + ", ".join(f"{n}={wi:+.2f}" for n, wi in zip(NAMES, w)))
    print("   → система видит дефолт почти целиком как функцию BALANCE (income/student слабы)\n")

    # 3) интерпретируемое правило — «как система видит бизнес»
    r1 = one_rule(Xtr, ytr, NAMES)
    pr = rule_predict(r1, Xte)
    rec_r, prec_r = recall_precision(yte, pr)
    print("3) ИНТЕРПРЕТИРУЕМОЕ ПРАВИЛО (OneR — как система видит бизнес):")
    print(f"   {rule_text(r1)}")
    print(f"   на held-out: balanced-acc {balanced_accuracy(yte, pr)*100:.0f}%, "
          f"recall дефолтов {rec_r*100:.0f}%, precision {prec_r*100:.0f}%")
    dl = decision_list(Xtr, ytr, NAMES, k=3)
    print("   список правил (жадный, для повышения точности):")
    for r in dl:
        op = ">" if r["gt"] else "≤"
        print(f"      ЕСЛИ {r['feature']} {op} {r['thr']:.0f} → дефолт  (точность {r['precision']*100:.0f}%, охват {r['coverage']})")

    print("\n── Что это значит ──")
    print("   Система ВИДИТ этот бизнес верно по сути: дефолт определяется БАЛАНСОМ (а не доходом),")
    print("   и линейная модель, и правило независимо указывают на balance > ~порог. AUC высок,")
    print("   но accuracy обманчива из-за дисбаланса — честная картина в AUC/recall. Честно:")
    print("   это статистическая зависимость из данных (как у логрегрессии), не понимание бизнеса —")
    print("   система не знает, ЧТО такое дефолт; она находит, КАКОЙ признак его предсказывает.")


if __name__ == "__main__":
    main()
