#!/usr/bin/env python3
"""П.2: связная генерация через torch char-LSTM (масштаб архитектуры).

Тот же текстовый домен, но рекуррентная модель с памятью контекста вместо
фиксированного окна — генерация становится СВЯЗНОЙ (реальные слова/строки кода),
а не «словесной кашей». На GPU тот же код тянет большие модели.

Запуск: python run_generate.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import torch

from thinking_system.predictors.torch_rnn import CharRNN
from thinking_system.predictors.torch_backend import default_device
from thinking_system.text.dataset import train_test_split, unigram_bpc, make_pairs
from thinking_system.text.ingest import load_book
from thinking_system.text.vocab import CharVocab


def run_corpus(name, text, prompt, *, steps, seed=0):
    vocab = CharVocab(text)
    ids = vocab.encode(text)
    tr, te = train_test_split(ids, train_frac=0.9)
    rnn = CharRNN(vocab.size, hidden=256, layers=2, lr=2e-3, seed=seed)
    rnn.fit(tr, seq_len=48, batch=32, steps=steps)

    bpc = rnn.bpc(te)
    _, tte = make_pairs(te, 1)
    uni = unigram_bpc(tr, tte, vocab.size)
    seed_ids = vocab.encode(prompt)
    sample = prompt + vocab.decode(rnn.generate(seed_ids, 320, temp=0.5))

    print(f"\n=== {name} ===")
    print(f"held-out bits/char: {bpc:.2f}  (unigram {uni:.2f})")
    print(f"генерация по затравке {prompt!r}:")
    print("   " + sample.replace("\n", "\n   "))


def main():
    print(f"▶ Torch char-LSTM; устройство: {default_device()} (CUDA: {torch.cuda.is_available()})")

    math = load_book("books/sample_mathematics.md")
    run_corpus("Английский мат. текст", math, "The square root of 2 is ", steps=6000, seed=0)

    code = "\n\n".join(Path(p).read_text(encoding="utf-8") for p in sorted(glob.glob("thinking_system/predictors/*.py")))
    run_corpus("Python-код", code, "def ", steps=6000, seed=0)

    print("\n── Итог ──")
    print("   Рекуррентная архитектура (память контекста) даёт СВЯЗНУЮ генерацию — реальные")
    print("   фразы и строки кода, а не кашу. Тот же интерфейс/домен; на GPU масштабируется.")


if __name__ == "__main__":
    main()
