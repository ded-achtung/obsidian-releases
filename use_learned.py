#!/usr/bin/env python3
"""Система ИСПОЛЬЗУЕТ выученное: находит аномалии, классифицирует, дополняет.

Одна и та же выученная предиктивная модель применяется для трёх задач:
  1) обнаружение аномалий/ошибок — по ошибке предсказания (surprise, бит/символ)
     система находит, ГДЕ в новом тексте что-то не так (опечатка, чужой язык);
  2) классификация по правдоподобию — какой моделью вход «дешевле» закодировать
     (меньше бит) → к тому классу он и относится (код / русский);
  3) автодополнение — самые вероятные продолжения по контексту.

Запуск: python use_learned.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from thinking_system.predictors.symbolic import SymbolicPredictor
from thinking_system.text.dataset import eval_bpc, make_pairs, train_test_split, unigram_bpc
from thinking_system.text.ingest import load_book
from thinking_system.text.vocab import ByteVocab

VOCAB = ByteVocab()


def train(text, *, context_len=16, hidden=192, updates=30000, eval_every=3000, patience=5, seed=0):
    ids = VOCAB.encode(text)
    tr, te = train_test_split(ids, train_frac=0.9)
    ctx_tr, tgt_tr = make_pairs(tr, context_len)
    ctx_te, tgt_te = make_pairs(te, context_len)
    pred = SymbolicPredictor(VOCAB.size, context_len, emb_dim=32, hidden_dim=hidden, lr=1e-3, weight_decay=1e-4, seed=seed)
    rng = np.random.default_rng(seed)
    snap = lambda: {k: getattr(pred, k).copy() for k in ("E", "W1", "b1", "W2", "b2")}
    best, best_p, stale = 1e9, snap(), 0
    for step in range(updates):
        b = rng.integers(0, len(ctx_tr), 64)
        pred.update(ctx_tr[b], tgt_tr[b])
        if (step + 1) % eval_every == 0:
            bpc = eval_bpc(pred, ctx_te, tgt_te)
            if bpc < best - 1e-3:
                best, best_p, stale = bpc, snap(), 0
            else:
                stale += 1
                if stale >= patience:
                    break
    for k, v in best_p.items():
        setattr(pred, k, v)
    return pred


def per_byte_bits(pred, text):
    """Surprise (бит) на каждый байт текста при данной модели."""
    L = pred.context_len
    ids = np.concatenate([np.full(L, 32, dtype=np.int64), VOCAB.encode(text)])  # left-pad пробелами
    bits = []
    for i in range(L, len(ids)):
        logits = pred.logits(ids[i - L : i])
        p = np.exp(logits - logits.max())
        p /= p.sum()
        bits.append(float(-np.log2(p[int(ids[i])] + 1e-12)))
    return np.array(bits)


def avg_bits(pred, text):
    return float(per_byte_bits(pred, text).mean())


def _surprise_bar(bits, text):
    """Печатает текст и под ним столбик surprise (выше = неожиданнее)."""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = bits.min(), max(bits.max(), bits.min() + 1e-6)
    line = "".join(blocks[min(7, int((b - lo) / (hi - lo) * 7))] for b in bits)
    safe = "".join(c if 32 <= ord(c) < 127 else "·" for c in text)
    print(f"   текст:    {safe}")
    print(f"   surprise: {line}")


def demo_anomaly(code_model):
    print("=" * 72)
    print("1) ОБНАРУЖЕНИЕ АНОМАЛИЙ/ОШИБОК — модель кода находит, ГДЕ не так")
    print("=" * 72)

    print("\nОпечатка в коде (return → retunr): surprise подскакивает на ошибке")
    good = "        return out"
    bad = "        retunr out"
    _surprise_bar(per_byte_bits(code_model, bad), bad)
    print(f"   суммарно бит: правильно {per_byte_bits(code_model, good).sum():.1f}  vs  с опечаткой {per_byte_bits(code_model, bad).sum():.1f}")

    print("\nЧужой язык во входе кода — средний surprise резко выше:")
    samples = {
        "код Python   ": "for i in range(n): self.update(i)",
        "русский текст": "Система учится на реальных данных",
    }
    for name, s in samples.items():
        print(f"   {name}: {avg_bits(code_model, s):.2f} бит/байт")


def demo_classify(code_model, ru_model):
    print("\n" + "=" * 72)
    print("2) КЛАССИФИКАЦИЯ ПО ПРАВДОПОДОБИЮ — система решает «код или русский»")
    print("=" * 72)
    print("   (меньше бит = модель лучше объясняет вход = это её класс)\n")
    tests = [
        ("def add(a, b): return a + b", "код"),
        ("for x in items: print(x)", "код"),
        ("Память нужна для накопления опыта", "русский"),
        ("Система использует выученное", "русский"),
    ]
    correct = 0
    print(f"   {'фрагмент':<36}{'код,бит':>9}{'рус,бит':>9}   вывод")
    for text, label in tests:
        bc, br = avg_bits(code_model, text), avg_bits(ru_model, text)
        guess = "код" if bc < br else "русский"
        correct += guess == label
        mark = "✓" if guess == label else "✗"
        short = (text[:33] + "…") if len(text) > 34 else text
        print(f"   {short:<36}{bc:>9.2f}{br:>9.2f}   {guess} {mark}")
    print(f"\n   верно: {correct}/{len(tests)}")


def demo_autocomplete(code_model):
    print("\n" + "=" * 72)
    print("3) АВТОДОПОЛНЕНИЕ — система предлагает вероятное продолжение")
    print("=" * 72 + "\n")
    L = code_model.context_len
    for prefix in ["    return ", "for i in ra", "def ", "self.", "print("]:
        ids = list(VOCAB.encode(prefix))
        if len(ids) < L:
            ids = [32] * (L - len(ids)) + ids
        logits = code_model.logits(np.array(ids[-L:]))
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.argsort(p)[::-1][:3]
        def show(b):
            b = int(b)
            return "␣" if b == 32 else (r"\n" if b == 10 else (chr(b) if 33 <= b < 127 else f"0x{b:02X}"))
        sug = "  ".join(f"{show(b)!s}({p[int(b)]:.2f})" for b in top)
        print(f"   {prefix!r:16} → {sug}")


def main():
    print("обучаю две модели (код / русский)…\n")
    code = "\n\n".join(Path(p).read_text(encoding="utf-8") for p in sorted(glob.glob("thinking_system/**/*.py", recursive=True)))
    code_model = train(code, updates=30000, seed=0)
    ru_text = load_book("research/brain-inspired-thinking-system.md")
    ru_model = train(ru_text, updates=18000, seed=0)

    demo_anomaly(code_model)
    demo_classify(code_model, ru_model)
    demo_autocomplete(code_model)


if __name__ == "__main__":
    main()
