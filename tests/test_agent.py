"""Тесты единого когнитивного контура: учится, выбирает, помнит, использует."""

from __future__ import annotations

import numpy as np

from thinking_system.agent.cognitive import CognitiveAgent
from thinking_system.text.vocab import ByteVocab


def test_agent_runs_and_learns() -> None:
    v = ByteVocab()
    sources = [
        ("prose", v.encode("the quick brown fox jumps over the lazy dog. " * 200)),
        ("code", v.encode("def f(x): return x + 1\n" * 200)),
    ]
    agent = CognitiveAgent(sources, context_len=8, hidden=64, warmup=128, lp_min=400, lp_window=800, seed=0).run(6000)
    bits = np.array(agent.bits)
    assert agent.visits.sum() == 6000
    assert bits[:800].mean() > bits[-800:].mean()  # surprise падает = учится


def test_agent_scan_flags_anomaly() -> None:
    v = ByteVocab()
    agent = CognitiveAgent(
        [("pat", v.encode("abcabcabcabc " * 300))],
        context_len=6, hidden=48, warmup=100, lp_min=300, lp_window=600, seed=0,
    ).run(5000)
    text = "abcabcZabcabc"  # 'Z' — аномалия в выученном паттерне
    bits = agent.scan(text)
    assert bits[text.index("Z")] > np.median(bits)  # модель удивлена на аномалии


def test_agent_source_bpc_beats_uniform() -> None:
    v = ByteVocab()
    agent = CognitiveAgent(
        [("s", v.encode("hello world, this is a small but structured sample. " * 120))],
        context_len=8, hidden=64, warmup=128, lp_min=300, lp_window=600, seed=0,
    ).run(6000)
    assert agent.source_bpc(0) < 8.0  # лучше равномерного угадывания (log2 256)
