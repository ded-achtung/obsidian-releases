"""Тесты ЕДИНОЙ системы: общая память и ПЕРЕНОСЫ между факультетами (не фасад)."""

from __future__ import annotations

from thinking_system.system import ThinkingSystem

BOOK = """
Пример: разверни [1, 2, 3] = [3, 2, 1]
Пример: разверни [4, 5] = [5, 4]
Пример: удвой [1, 2, 3] = [2, 4, 6]
Пример: удвой [5, 1] = [10, 2]
Пример: сумма [1, 2, 3] = 6
Пример: сумма [4, 5] = 9
Сократ это человек.
человек это млекопитающее.
млекопитающее это животное.
каждый человек смертный.
дождь вызывает мокрый.
мокрый вызывает скользкий.
просто проза
"""


def test_one_shared_library_between_language_and_reasoning() -> None:
    # Словарь языка и библиотека рассуждения — ОДИН объект (общая процедурная память).
    ts = ThinkingSystem()
    assert ts.lexicon.lib is ts.library


def test_reading_feeds_deduction() -> None:
    # Прочитанные факты/причины отвечают на вопросы ДЕДУКЦИЕЙ (семантическая память).
    ts = ThinkingSystem()
    ts.read(BOOK)
    assert ts.ask("Сократ смертный?") == "да"             # силлогизм, не сказан прямо
    assert ts.ask("Сократ это животное?") == "да"          # транзитивность
    assert ts.ask("если включить дождь, будет скользкий?") == "да"  # причинная цепочка
    assert ts.ask("Сократ летает?") == "нет"               # не следует — честно


def test_reading_grows_the_reasoning_library() -> None:
    # ПЕРЕНОС чтение→рассуждение: выученное чтением слово становится примитивом поиска.
    ts = ThinkingSystem()
    out = ts.read(BOOK)
    assert {"разверни", "удвой", "сумма"} <= set(out["learned_words"])
    assert {"разверни", "удвой", "сумма"} <= set(ts.library.abstractions)


def test_language_composes_read_words() -> None:
    # Язык СОСТАВЛЯЕТ прочитанные слова и исполняет (язык пользуется чтением).
    ts = ThinkingSystem()
    ts.read(BOOK)
    assert ts.understand("удвой и сумма [1, 2, 3]")["answer"] == 12


def test_naming_a_found_skill_transfers_reasoning_to_language() -> None:
    # ПЕРЕНОС рассуждение→язык: найденный поиском навык становится словом И сокращает поиск.
    ts = ThinkingSystem()
    # invent=False: проверяем именно КОМПОЗИЦИЮ (без изобретения аффинного мимика из 2 точек)
    assert ts.solve([(2, 5), (3, 10)], max_depth=1, invent=False) is None   # x²+1 не берётся на глубине 1
    assert ts.name_skill("квадрик", [(2, 5), (3, 10)])
    prog = ts.solve([(2, 5), (3, 10)], max_depth=1, invent=False)
    assert prog is not None and str(prog) == "квадрик"           # теперь берётся одним шагом
    assert ts.understand("квадрик 4")["answer"] == 17            # и язык им владеет (на новом входе)


def test_perceive_reduces_surprise_in_shared_cortex() -> None:
    # Предиктивный кортекс учится (bits/byte падает) и копит общую эпизодическую память.
    ts = ThinkingSystem(seed=0)
    text = ("the quick brown fox jumps over the lazy dog. ") * 40
    res = ts.perceive([("текст", text)], steps=1500)
    assert res["bpc_after"][0] < res["bpc_before"][0]
    assert len(ts.cortex.memory) > 0


def test_world_faculty_reaches_goal() -> None:
    # Тот же агент ДЕЙСТВУЕТ: осваивает мир и доходит до цели.
    ts = ThinkingSystem(seed=0)
    ts.learn_world(episodes=40)
    a = ts.act()
    assert a["reached"]


def test_mind_aggregates_all_faculties_in_one_object() -> None:
    # Всё нажитое — в ОДНОМ объекте: слова, абстракции, факты, причины.
    ts = ThinkingSystem()
    ts.read(BOOK)
    ts.name_skill("квадрик", [(2, 5), (3, 10)])
    m = ts.mind()
    assert "квадрик" in m["слова (язык)"]
    assert "квадрик" in m["абстракции (библиотека)"]
    assert m["фактов (семантика)"] >= 4
    assert m["причинных связей"]                                 # непусто
