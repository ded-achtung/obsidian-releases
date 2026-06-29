"""Тесты текстового домена: ингест (PDF/MD), словарь, и обучение по тексту."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thinking_system.predictors.symbolic import SymbolicPredictor
from thinking_system.text.dataset import eval_bpc, make_pairs, train_test_split, unigram_bpc, uniform_bpc
from thinking_system.text.ingest import load_book
from thinking_system.text.vocab import ByteVocab, CharVocab


def test_vocab_roundtrip() -> None:
    vocab = CharVocab("abc abc xyz\n")
    ids = vocab.encode("cab")
    assert vocab.decode(ids) == "cab"
    assert vocab.size == len("abcxyz \n")


def test_make_pairs_and_split() -> None:
    ids = np.arange(20)
    tr, te = train_test_split(ids, train_frac=0.8)
    assert len(tr) == 16 and len(te) == 4
    ctx, tgt = make_pairs(ids, 3)
    assert ctx.shape == (17, 3) and tgt.shape == (17,)
    assert ctx[0].tolist() == [0, 1, 2] and tgt[0] == 3


def test_load_md(tmp_path: Path) -> None:
    p = tmp_path / "book.md"
    p.write_text("# Title\n\nTheorem: 1 + 1 = 2.\n", encoding="utf-8")
    text = load_book(p)
    assert "Theorem" in text


def test_load_pdf_roundtrip(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")  # опц. зависимость pymupdf → skip, а не fail
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Lemma one plus one equals two")
    pdf = tmp_path / "b.pdf"
    doc.save(pdf)
    doc.close()
    text = load_book(pdf)
    assert "plus one equals two" in text


def test_byte_vocab_universal_roundtrip() -> None:
    # Один словарь 256 на ВСЁ: разные языки, эмодзи и код — точный round-trip.
    v = ByteVocab()
    s = "Hello мир 世界 🌍\ndef f(x): return x + 1  # комментарий\n"
    ids = v.encode(s)
    assert v.size == 256
    assert int(ids.min()) >= 0 and int(ids.max()) < 256
    assert v.decode(ids) == s


def test_byte_model_learns_multilingual_and_code() -> None:
    # Байтовый уровень учит смесь языков + кода и бьёт unigram на held-out.
    text = (
        "def add(a, b): return a + b  # сложение двух чисел\n"
        "The result is positive. Результат положительный.\n"
    ) * 150
    v = ByteVocab()
    ids = v.encode(text)
    train, test = train_test_split(ids, train_frac=0.85)
    ctx_tr, tgt_tr = make_pairs(train, 8)
    ctx_te, tgt_te = make_pairs(test, 8)

    pred = SymbolicPredictor(v.size, 8, emb_dim=16, hidden_dim=64, lr=3e-3, weight_decay=1e-3, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        b = rng.integers(0, len(ctx_tr), 64)
        pred.update(ctx_tr[b], tgt_tr[b])

    assert eval_bpc(pred, ctx_te, tgt_te) < unigram_bpc(train, tgt_te, v.size)


def test_model_beats_unigram_on_structured_text() -> None:
    # На тексте с реальной структурой модель должна обойти unigram-бейзлайн на
    # held-out — т.е. использовать КОНТЕКСТ, а не только частоты символов.
    text = "the quick brown fox jumps over the lazy dog. " * 200
    vocab = CharVocab(text)
    ids = vocab.encode(text)
    train, test = train_test_split(ids, train_frac=0.85)
    ctx_tr, tgt_tr = make_pairs(train, 6)
    ctx_te, tgt_te = make_pairs(test, 6)

    pred = SymbolicPredictor(vocab.size, 6, emb_dim=12, hidden_dim=48, lr=3e-3, weight_decay=1e-3, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        b = rng.integers(0, len(ctx_tr), 64)
        pred.update(ctx_tr[b], tgt_tr[b])

    bpc_model = eval_bpc(pred, ctx_te, tgt_te)
    bpc_unigram = unigram_bpc(train, tgt_te, vocab.size)
    assert bpc_model < bpc_unigram                 # выучила контекст
    assert bpc_model < uniform_bpc(vocab.size)     # и тем более лучше угадывания
