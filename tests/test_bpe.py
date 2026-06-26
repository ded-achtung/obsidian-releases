"""Тесты byte-BPE: точный round-trip, сжатие, универсальность."""

from __future__ import annotations

from thinking_system.text.bpe import BytePairTokenizer


def test_bpe_roundtrip_and_compresses() -> None:
    text = "the cat sat on the mat. " * 50 + "def f(x): return x. " * 50
    tok = BytePairTokenizer().train(text, vocab_size=300)
    assert tok.decode(tok.encode(text)) == text                       # точный round-trip
    assert len(tok.encode(text)) < len(text.encode("utf-8"))          # сжатие (меньше токенов)
    assert 256 < tok.vocab_size <= 300


def test_bpe_universal_any_language() -> None:
    tok = BytePairTokenizer().train("Hello мир 世界 def f(): pass " * 30, vocab_size=290)
    for s in ["Привет world", "世界 код", "def g(x): return x"]:
        assert tok.decode(tok.encode(s)) == s                         # любой текст через байты
