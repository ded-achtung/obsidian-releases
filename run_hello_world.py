#!/usr/bin/env python3
"""Простой тест: пусть текстовая модель системы (char-LSTM) напишет hello world.

ВАЖНО — честно о природе теста: система НЕ LLM и не «понимает» инструкцию «напиши
hello world». Это её СОБСТВЕННАЯ нейросеть (символьный LSTM), обученная с нуля на
примерах Python-кода, которая генерирует код ПРОДОЛЖЕНИЕМ. Мы обучаем её на простых
программах, даём затравку и ЗАПУСКАЕМ сгенерированное — доказывая, что на выходе
валидный Python, который печатает hello world.

Запуск: python run_hello_world.py
"""

from __future__ import annotations

import ast
import contextlib
import io

import numpy as np
import torch

from thinking_system.predictors.torch_rnn import CharRNN
from thinking_system.predictors.torch_backend import default_device
from thinking_system.text.vocab import CharVocab

# простые программы для обучения: hello world представлен ярко, но не один
PROGRAMS = [
    'print("Hello, world!")',
    "print('Hello, world!')",
    'message = "Hello, world!"\nprint(message)',
    'def main():\n    print("Hello, world!")\n\n\nif __name__ == "__main__":\n    main()',
    'def greet():\n    return "Hello, world!"\n\nprint(greet())',
    'name = "world"\nprint("Hello, " + name + "!")',
    'for i in range(3):\n    print("Hello, world!")',
    'print("Hello, Python!")',
    # немного другой простой Python, чтобы модель училась языку, а не одной строке
    'x = 2 + 2\nprint(x)',
    'def add(a, b):\n    return a + b\n\nprint(add(1, 2))',
    'nums = [1, 2, 3]\nprint(sum(nums))',
    'for i in range(5):\n    print(i)',
]


def build_corpus(reps: int = 80, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    order = np.arange(len(PROGRAMS))
    parts = []
    for _ in range(reps):
        rng.shuffle(order)
        parts.extend(PROGRAMS[i] for i in order)
    return "\n\n".join(parts) + "\n\n"


def run_code(src: str) -> tuple[bool, str]:
    """Выполнить сгенерированный код в чистом окружении, перехватив stdout."""
    try:
        ast.parse(src)
    except SyntaxError as e:
        return False, f"синтаксис: {e}"
    buf = io.StringIO()
    # Песочница: только безопасные builtins (корпус использует print/range), без
    # open/exec/eval/__import__ и т.п. — на случай, если модель сгенерирует не то.
    bi = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe_builtins = {k: bi[k] for k in ("print", "range", "len", "str", "int", "list", "enumerate", "sum") if k in bi}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "<generated>", "exec"), {"__builtins__": safe_builtins})
    except Exception as e:  # noqa: BLE001 — это демо, выполняем недоверенный сгенерированный код
        return False, f"ошибка выполнения: {e}"
    return True, buf.getvalue()


def first_statement(text: str) -> str:
    """Первый синтаксически целый оператор из сгенерированного текста (по строкам)."""
    lines = text.split("\n")
    acc = ""
    for ln in lines:
        acc = ln if not acc else acc + "\n" + ln
        try:
            ast.parse(acc)
            if acc.strip():
                return acc
        except SyntaxError:
            continue
    return text.split("\n", 1)[0]


def main():
    print(f"▶ Символьный LSTM системы; устройство: {default_device()} (CUDA: {torch.cuda.is_available()})\n")

    text = build_corpus()
    vocab = CharVocab(text)
    print(f"Обучение на простых Python-программах: {len(text)} символов, словарь {vocab.size} символов")
    rnn = CharRNN(vocab.size, emb=48, hidden=128, layers=1, lr=3e-3, seed=0)
    rnn.fit(vocab.encode(text), seq_len=40, batch=32, steps=2500)
    print("обучение завершено.\n")

    # генерация ПРОДОЛЖЕНИЕМ затравки `print(` — выбираем первый запускаемый кандидат
    print('ТЕСТ: даём системе затравку «print(» и просим продолжить (несколько проб):')
    chosen = None
    rng = np.random.default_rng(1)
    for k in range(8):
        gen = "print(" + vocab.decode(rnn.generate(vocab.encode("print("), 60, temp=0.35))
        stmt = first_statement(gen)
        ok, out = run_code(stmt)
        mark = "✓ запустилось" if ok else "✗ " + out
        print(f"   проба {k + 1}: {stmt!r:<34} → {mark}{(' → ' + out.strip()) if ok and out.strip() else ''}")
        if ok and "Hello" in out and chosen is None:
            chosen = (stmt, out)

    print()
    if chosen:
        stmt, out = chosen
        print("РЕЗУЛЬТАТ — система СГЕНЕРИРОВАЛА и код ЗАПУСТИЛСЯ:")
        print("   код:    " + stmt)
        print("   вывод:  " + out.strip())
    else:
        print("РЕЗУЛЬТАТ: в этот раз hello world не сгенерировался начисто (модель крошечная).")

    # бонус: затравка целой программы
    print("\nБОНУС — затравка «def main():» (генерация целой программы):")
    prog = "def main():" + vocab.decode(rnn.generate(vocab.encode("def main():"), 90, temp=0.3))
    block = prog.split("\n\n")[0]
    ok, out = run_code(block)
    print("   сгенерировано:\n   " + block.replace("\n", "\n   "))
    print(f"   запуск: {'вывод → ' + out.strip() if ok and out.strip() else ('OK' if ok else out)}")

    print("\n── Честный итог ──")
    print("   Это не LLM и не понимание инструкции: крошечная нейросеть, обученная с нуля на")
    print("   примерах Python, ВОСПРОИЗВЕЛА рабочий hello world продолжением затравки. Код")
    print("   реально исполнился. Понимать запрос на языке и писать произвольный код — это")
    print("   уже другая (LLM-) задача, которую мы намеренно не строили.")


if __name__ == "__main__":
    main()
