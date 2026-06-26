#!/usr/bin/env python3
"""Что выучила модель НА КОДЕ: def→имя, отступы, скобки, ключевые слова.

Часть A — книга «Python для непрограммистов» (русский + код), byte-level.
Часть B — чистый Python-код с отступами (наши .py), byte-level — чтобы показать
          обучение ОТСТУПАМ (в книге PDF-извлечение их местами потеряло).

Запуск: python show_code.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from thinking_system.predictors.symbolic import SymbolicPredictor
from thinking_system.text.dataset import eval_bpc, make_pairs, train_test_split, unigram_bpc
from thinking_system.text.ingest import load_book
from thinking_system.text.vocab import ByteVocab


def train(text, vocab, *, context_len, hidden, updates, eval_every=3000, patience=5, lr=1e-3, wd=1e-4, seed=0):
    ids = vocab.encode(text)
    tr, te = train_test_split(ids, train_frac=0.9)
    ctx_tr, tgt_tr = make_pairs(tr, context_len)
    ctx_te, tgt_te = make_pairs(te, context_len)
    pred = SymbolicPredictor(vocab.size, context_len, emb_dim=32, hidden_dim=hidden, lr=lr, weight_decay=wd, seed=seed)
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
    return pred, best, unigram_bpc(tr, tgt_te, vocab.size)


def _ctx(vocab, prefix, L):
    ids = list(vocab.encode(prefix))
    if len(ids) < L:
        ids = [32] * (L - len(ids)) + ids  # left-pad пробелами
    return np.array(ids[-L:])


def _b(b):
    b = int(b)
    if b == 10:
        return r"\n"
    if b == 32:
        return "␣"
    if 33 <= b < 127:
        return chr(b)
    return f"0x{b:02X}"


def topk(pred, vocab, prefix, k=5):
    logits = pred.logits(_ctx(vocab, prefix, pred.context_len))
    p = np.exp(logits - logits.max())
    p /= p.sum()
    top = np.argsort(p)[::-1][:k]
    return "  ".join(f"{_b(x)}:{p[int(x)]:.2f}" for x in top)


def complete(pred, vocab, seed, n, *, temp=0.5, top_k=8, rng=None):
    rng = rng or np.random.default_rng(0)
    L = pred.context_len
    ctx = list(_ctx(vocab, seed, L))
    out = []
    for _ in range(n):
        logits = pred.logits(np.array(ctx[-L:])) / temp
        if top_k < len(logits):
            cut = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits >= cut, logits, -np.inf)
        p = np.exp(logits - np.nanmax(logits))
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        out.append(nxt)
        ctx.append(nxt)
    return seed + vocab.decode(out)


def main():
    vocab = ByteVocab()

    print("=" * 72)
    print("ЧАСТЬ A — книга «Python для непрограммистов» (рус.+код, byte-level)")
    print("=" * 72)
    book = load_book("books/python_dlya_neprogrammistov.pdf")
    predA, bpcA, uniA = train(book, vocab, context_len=16, hidden=224, updates=45000)
    print(f"held-out bits/byte: {bpcA:.2f}  (unigram {uniA:.2f})\n")
    print("Код — предсказание следующего байта (top-5):")
    for pfx in ["retur", "impor", "prin", "def ", "print(", "for i in rang", "rang"]:
        print(f"   {pfx!r:18} → {topk(predA, vocab, pfx)}")
    print("\nДописывание по затравке 'def ':")
    print("   " + repr(complete(predA, vocab, "def ", 90, rng=np.random.default_rng(3))))

    print("\n" + "=" * 72)
    print("ЧАСТЬ B — чистый Python-код с отступами (наши .py, byte-level)")
    print("=" * 72)
    code = "\n\n".join(
        Path(p).read_text(encoding="utf-8") for p in sorted(glob.glob("thinking_system/**/*.py", recursive=True))
    )
    predB, bpcB, uniB = train(code, vocab, context_len=16, hidden=224, updates=30000)
    print(f"held-out bits/byte: {bpcB:.2f}  (unigram {uniB:.2f})  (код, {len(code)} символов)\n")

    print("ОТСТУПЫ — после строки-блока, кончающейся ':\\n', модель ждёт пробелы:")
    for pfx in ["    def update(self):\n", "for i in range(n):\n", "        if value > 0:\n"]:
        print(f"   …{pfx!r} → {topk(predB, vocab, pfx)}")

    print("\nКод-структура (top-5):")
    for pfx in ["def ", "    retur", "self.", "import ", "np.", "        "]:
        print(f"   {pfx!r:18} → {topk(predB, vocab, pfx)}")

    print("\nДописывание кода по затравке 'def ':")
    print("   " + repr(complete(predB, vocab, "def ", 120, rng=np.random.default_rng(4))))


if __name__ == "__main__":
    main()
