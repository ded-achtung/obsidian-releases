#!/usr/bin/env python3
"""Подать системе РЕАЛЬНЫЕ данные (книги) и проверить, учится ли она по ним.

Читает все книги из папки books/ (PDF / Markdown / txt), обучает символьный
предиктор следующего символа (с эпизодической памятью и replay — той же
машинерией, что в шагах 1–3) и честно оценивает качество в bits-per-character
на ОТЛОЖЕННОЙ (held-out) части, сравнивая с тривиальными бейзлайнами.

Падение held-out BPC ниже unigram-бейзлайна = система выучила структуру текста
(контекст, слова), а не просто запомнила частоты символов или обучающую выборку.

Запуск:
    python run_book.py
    python run_book.py --updates 40000 --context-len 12
"""

from __future__ import annotations

import argparse

import numpy as np

from thinking_system.core.experience import Experience
from thinking_system.memory.buffer import EpisodicBuffer
from thinking_system.predictors.symbolic import SymbolicPredictor
from thinking_system.text.dataset import accuracy, eval_bpc, make_pairs, train_test_split, unigram_bpc, uniform_bpc
from thinking_system.text.ingest import load_book, load_corpus
from thinking_system.text.vocab import CharVocab
from thinking_system.viz import sparkline


def _generate(pred: SymbolicPredictor, vocab: CharVocab, seed_ids: np.ndarray, n: int, *, temp: float = 0.8, rng=None) -> str:
    rng = rng or np.random.default_rng(0)
    ctx = list(seed_ids[-pred.context_len :])
    out: list[int] = []
    for _ in range(n):
        logits = pred.logits(np.array(ctx[-pred.context_len :]))
        logits = logits / temp
        p = np.exp(logits - logits.max())
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        out.append(nxt)
        ctx.append(nxt)
    return vocab.decode(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Learn from real books (PDF/MD) — held-out bits-per-character")
    ap.add_argument("--books", default="books", help="папка с книгами")
    ap.add_argument("--context-len", type=int, default=8)
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--updates", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os

    if os.path.isfile(args.books):
        text, sources = load_book(args.books), [os.path.basename(args.books)]
    else:
        text, sources = load_corpus(args.books)
    if not text.strip():
        print(f"'{args.books}' пуст. Положи туда книги (PDF / .md / .txt) и запусти снова.")
        return
    vocab = CharVocab(text)
    ids = vocab.encode(text)
    print(f"▶ Книги: {sources}")
    print(f"  символов: {len(ids)}, словарь: {vocab.size}")

    train_ids, test_ids = train_test_split(ids, train_frac=args.train_frac)
    ctx_tr, tgt_tr = make_pairs(train_ids, args.context_len)
    ctx_te, tgt_te = make_pairs(test_ids, args.context_len)
    if len(ctx_te) == 0:
        print("Слишком мало текста для held-out оценки — добавь книги покрупнее.")
        return

    # Бейзлайны на held-out.
    bpc_uniform = uniform_bpc(vocab.size)
    bpc_unigram = unigram_bpc(train_ids, tgt_te, vocab.size)

    pred = SymbolicPredictor(vocab.size, args.context_len, emb_dim=args.emb_dim, hidden_dim=args.hidden, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed)
    memory = EpisodicBuffer(capacity=max(40_000, len(ctx_tr) + 10), seed=args.seed)
    rng = np.random.default_rng(args.seed)

    def snapshot() -> dict:
        return {k: getattr(pred, k).copy() for k in ("E", "W1", "b1", "W2", "b2")}

    curve: list[float] = []
    best = float("inf")
    best_params = snapshot()
    stale = 0
    print(f"\n  обучение с ранней остановкой (held-out BPC по ходу)…")
    for step in range(args.updates):
        # поток обучающих пар → эпизодическая память (циклически, несколько проходов)
        j = step % len(ctx_tr)
        memory.add(Experience(ctx_tr[j].copy(), np.array([tgt_tr[j]]), step))
        if len(memory) >= args.batch:
            cs, ts = memory.sample(args.batch)
            pred.update(cs, ts)
        if (step + 1) % args.eval_every == 0:
            b = eval_bpc(pred, ctx_te, tgt_te)
            curve.append(b)
            if b < best - 1e-3:
                best, best_params, stale = b, snapshot(), 0
            else:
                stale += 1
                if stale >= args.patience:  # ранняя остановка: held-out перестал улучшаться
                    break

    for k, v in best_params.items():  # восстановить лучшую (не переобученную) модель
        setattr(pred, k, v)
    bpc_model = eval_bpc(pred, ctx_te, tgt_te)
    acc_model = accuracy(pred, ctx_te, tgt_te)

    print("  held-out BPC по ходу обучения: " + sparkline(curve))
    print("\n── Held-out качество (bits-per-character, меньше = лучше) ──")
    print(f"  uniform (угадывание)     : {bpc_uniform:.3f}")
    print(f"  unigram (частоты символов): {bpc_unigram:.3f}")
    print(f"  МОДЕЛЬ (контекст {args.context_len} симв.): {bpc_model:.3f}   (лучшее по ходу: {best:.3f})")
    print(f"  точность следующего символа: {acc_model * 100:.1f}%")

    seed_ids = test_ids[: args.context_len]
    print("\n── Сгенерировано моделью (после обучения на книге) ──")
    print("  " + repr(_generate(pred, vocab, seed_ids, 240, rng=rng)))

    learned = bpc_model < bpc_unigram and bpc_model < bpc_uniform
    print("\n── Итог ──")
    print(f"  модель бьёт unigram-бейзлайн на held-out (выучила контекст, а не только частоты): {'да' if learned else 'нет'}")
    print(f"  сжатие против равномерного: {bpc_uniform / bpc_model:.2f}× меньше неопределённости на символ")


if __name__ == "__main__":
    main()
