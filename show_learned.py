#!/usr/bin/env python3
"""Показать КОНКРЕТНО, что выучила модель: предсказания, дописывание, структура UTF-8.

Обучает небольшие модели и печатает интерпретируемые примеры выученного:
  1) предсказание следующего символа (top-k) для осмысленных контекстов;
  2) дописывание текста по затравке;
  3) на байтовом уровне — что модель сама вывела структуру кодировки UTF-8
     (после ведущего байта кириллицы предсказывает байт-продолжение).

Запуск: python show_learned.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.predictors.symbolic import SymbolicPredictor
from thinking_system.text.dataset import make_pairs, train_test_split, eval_bpc, unigram_bpc
from thinking_system.text.ingest import load_book
from thinking_system.text.vocab import ByteVocab, CharVocab


def train(text, vocab, *, context_len, hidden, updates, lr=1e-3, wd=1e-4, seed=0):
    ids = vocab.encode(text)
    tr, te = train_test_split(ids, train_frac=0.9)
    ctx_tr, tgt_tr = make_pairs(tr, context_len)
    ctx_te, tgt_te = make_pairs(te, context_len)
    pred = SymbolicPredictor(vocab.size, context_len, emb_dim=24, hidden_dim=hidden, lr=lr, weight_decay=wd, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_p, stale = 1e9, None, 0
    snap = lambda: {k: getattr(pred, k).copy() for k in ("E", "W1", "b1", "W2", "b2")}
    for step in range(updates):
        b = rng.integers(0, len(ctx_tr), 64)
        pred.update(ctx_tr[b], tgt_tr[b])
        if (step + 1) % 1000 == 0:
            bpc = eval_bpc(pred, ctx_te, tgt_te)
            if bpc < best - 1e-3:
                best, best_p, stale = bpc, snap(), 0
            else:
                stale += 1
                if stale >= 5:
                    break
    for k, v in (best_p or snap()).items():
        setattr(pred, k, v)
    return pred, best, unigram_bpc(tr, tgt_te, vocab.size)


def _ctx_ids(vocab, prefix, L):
    ids = list(vocab.encode(prefix))
    pad = int(vocab.encode(" ")[0])
    if len(ids) < L:
        ids = [pad] * (L - len(ids)) + ids
    return np.array(ids[-L:])


def topk_next(pred, vocab, prefix, k=5):
    logits = pred.logits(_ctx_ids(vocab, prefix, pred.context_len))
    p = np.exp(logits - logits.max())
    p /= p.sum()
    top = np.argsort(p)[::-1][:k]
    return [(vocab.decode([int(i)]), float(p[int(i)])) for i in top]


def complete(pred, vocab, seed_str, n, *, temp=0.5, top_k=10, rng=None):
    rng = rng or np.random.default_rng(0)
    L = pred.context_len
    ctx = list(_ctx_ids(vocab, seed_str, L))
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
    return seed_str + vocab.decode(out)


def show_char_english():
    print("=" * 70)
    print("ЧТО ВЫУЧИЛА МОДЕЛЬ — английский мат. текст (char-level)")
    print("=" * 70)
    text = load_book("books/sample_mathematics.md")
    vocab = CharVocab(text)
    pred, bpc, uni = train(text, vocab, context_len=10, hidden=128, updates=14000)
    print(f"held-out bits/char: {bpc:.2f}  (unigram-бейзлайн {uni:.2f})\n")

    print("1) Предсказание СЛЕДУЮЩЕГО символа (top-5) для осмысленных контекстов:")
    for prefix in [
        "The square root of 2 is ",
        "mathematical inducti",
        "is a prime numbe",
        "the sum of the firs",
        "Theore",
        "Proof. ",
    ]:
        preds = topk_next(pred, vocab, prefix, k=5)
        shown = "  ".join(f"{c!r}:{p:.2f}" for c, p in preds)
        print(f"   …{prefix!r}\n      → {shown}")

    print("\n2) Дописывание по затравке (модель продолжает сама):")
    rng = np.random.default_rng(1)
    for seed in ["The square root of 2 is ", "A prime number is ", "Proof. Suppose "]:
        print(f"   {seed!r}\n      → {complete(pred, vocab, seed, 80, rng=rng)!r}")


def show_byte_russian():
    print("\n" + "=" * 70)
    print("ЧТО ВЫУЧИЛА МОДЕЛЬ — русский текст (byte-level, словарь 256)")
    print("=" * 70)
    text = load_book("research/brain-inspired-thinking-system.md")
    vocab = ByteVocab()
    pred, bpc, uni = train(text, vocab, context_len=12, hidden=160, updates=16000)
    print(f"held-out bits/byte: {bpc:.2f}  (unigram-бейзлайн {uni:.2f})\n")

    print("3) Модель САМА вывела структуру UTF-8 (её никто не учил кодировке):")
    print("   кириллица в UTF-8 = ведущий байт 0xD0/0xD1 + байт-продолжение (0x80–0xBF).")
    for prefix in ["Это па", "память", "систе"]:
        ids = vocab.encode(prefix)
        # контекст, заканчивающийся ведущим байтом кириллицы
        lead_pos = max(i for i, b in enumerate(ids) if b in (0xD0, 0xD1))
        ctx = ids[: lead_pos + 1]
        ctxL = ctx[-pred.context_len :]
        if len(ctxL) < pred.context_len:
            ctxL = np.concatenate([np.full(pred.context_len - len(ctxL), 32), ctxL])
        logits = pred.logits(ctxL)
        p = np.exp(logits - logits.max()); p /= p.sum()
        top = np.argsort(p)[::-1][:5]
        lead = int(ids[lead_pos])
        cont = all(0x80 <= int(b) <= 0xBF for b in top[:3])
        bytes_str = "  ".join(f"0x{int(b):02X}:{p[int(b)]:.2f}" for b in top)
        print(f"   после ведущего байта 0x{lead:02X} (из {prefix!r}) → топ-байты: {bytes_str}")
        print(f"      все из диапазона продолжений 0x80–0xBF: {'ДА ✓' if cont else 'нет'}")

    print("\n4) Дописывание русского текста по затравке:")
    rng = np.random.default_rng(2)
    for seed in ["Система ", "Память нужна для "]:
        print(f"   {seed!r}\n      → {complete(pred, vocab, seed, 90, top_k=8, rng=rng)!r}")


if __name__ == "__main__":
    show_char_english()
    show_byte_russian()
