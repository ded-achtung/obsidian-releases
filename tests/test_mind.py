"""Тесты единого агента: маршрутизация, чтение→библиотека, повестка, память между запусками."""

from __future__ import annotations

from pathlib import Path

from thinking_system.mind import Mind
from thinking_system.mind.lexicon import (GridLexicon, extract_grids,
                                          parse_grid_definition, parse_grid_demo)
from thinking_system.reasoning.grid_seed import full_grid_seed
from thinking_system.reasoning.grids import to_grid

TEXTBOOK = """
# мини-урок
Отражение переставляет клетки слева направо.
отражение [[1, 2], [3, 4]] → [[2, 1], [4, 3]]
отражение [[5, 0, 6]] → [[6, 0, 5]]
переворот [[1, 2], [3, 4]] → [[3, 4], [1, 2]]
переворот [[7], [8]] → [[8], [7]]
поворот это сначала отражение потом переворот
"""

# задача «поворот на 180°»: на глубине 1 сырым seed не решается (rot180 не примитив)
ROT_TRAIN = [([[1, 2, 0], [0, 3, 4]], [[4, 3, 0], [0, 2, 1]]),
             ([[5, 0], [6, 7]], [[7, 6], [0, 5]])]
ROT_TEST = (to_grid([[1, 0], [2, 3]]), to_grid([[3, 2], [0, 1]]))


def test_extract_and_parse_demo() -> None:
    assert extract_grids("до [[1, 2], [3, 4]] после") == [((1, 2), (3, 4))]
    word, gin, gout = parse_grid_demo("отражение [[1, 2]] → [[2, 1]]")
    assert word == "отражение" and gin == ((1, 2),) and gout == ((2, 1),)
    assert parse_grid_demo("просто текст без сеток") is None


def test_parse_definition_needs_known_words() -> None:
    known = {"отражение", "переворот"}
    assert parse_grid_definition("поворот это сначала отражение потом переворот", known) \
        == ("поворот", ["отражение", "переворот"])
    assert parse_grid_definition("поворот это сначала тайна потом переворот", known) is None


def test_lexicon_grounds_word_by_induction() -> None:
    lex = GridLexicon(full_grid_seed())
    ok = lex.learn("отражение", [(to_grid([[1, 2], [3, 4]]), to_grid([[2, 1], [4, 3]]))])
    assert ok and lex.words["отражение"] == ["flip_h"]
    assert not lex.learn("чудо", [(to_grid([[1]]), to_grid([[9]]))])   # вне языка — честный отказ


def test_mind_routes_and_full_loop(tmp_path: Path) -> None:
    state = str(tmp_path / "mind.json")
    mind = Mind(state)
    assert mind.perceive("текст") == "text"
    assert mind.perceive({"id": "t", "train": ROT_TRAIN}) == "grid_task"

    # 1) задача не берётся на глубине 1 → честно в нерешённое
    res = mind.experience({"id": "rot", "train": ROT_TRAIN}, effort=1)
    assert res["routed"] == "grid_task" and not res["solved"] and "rot" in mind.unsolved

    # 2) чтение растит словарь и библиотеку
    res = mind.experience(TEXTBOOK)
    assert res["routed"] == "text"
    assert set(res["выучено_слов"]) == {"отражение", "переворот"}
    assert res["определено"] == ["поворот"] and mind.abstractions == ["flip_h∘flip_v"]

    # 3) повестка: сам возвращается к нерешённому и решает на глубине 1
    assert any("нерешённому" in a for a in mind.agenda())
    work = mind.idle_work(effort=1)
    assert [t for t, _, _ in work["resolved"]] == ["rot"]
    prog = mind.program_for("rot")
    assert prog(ROT_TEST[0]) == ROT_TEST[1]                  # верно и на скрытом примере

    # 4) память переживает процесс: новый агент решает сразу
    mind.save()
    mind2 = Mind(state)
    assert mind2.abstractions == ["flip_h∘flip_v"] and "поворот" in mind2.lexicon.words
    res2 = mind2.attempt("rot2", ROT_TRAIN, effort=1)
    assert res2["solved"] and res2["depth"] == 1


def test_mind_consolidates_repeated_combos(tmp_path: Path) -> None:
    mind = Mind(str(tmp_path / "m.json"))
    mind.solutions = {"a": ["bbox", "fractal"], "b": ["bbox", "fractal"]}
    work = mind.idle_work()
    assert work["consolidated"] == ["bbox∘fractal"]
    assert "bbox∘fractal" in mind.abstractions


def test_mind_uses_parametric_and_template_variables(tmp_path: Path) -> None:
    mind = Mind(str(tmp_path / "m.json"))
    # переменная-цвет из данных: «перекрась всё в 4» решается paint[4] на глубине 1
    res = mind.attempt("paint", [([[3, 0], [0, 2]], [[4, 0], [0, 4]])], effort=1)
    assert res["solved"] and mind.solutions["paint"] == ["paint[4]"]
    # переменная-шаг из опыта: два трёхшаговых решения → шаблон, берущий глубину 3 дёшево
    mind.solutions.update({"a": ["flip_h", "flip_v", "bbox"],
                           "b": ["flip_h", "flip_v", "gravity"]})
    g = to_grid([[3, 0, 2], [0, 3, 0]])
    from thinking_system.reasoning.grids import transpose, flip_h, flip_v
    want = transpose(flip_v(flip_h(g)))
    res = mind.attempt("deep", [(g, want), (to_grid([[1, 2], [0, 4]]),
                                            transpose(flip_v(flip_h(to_grid([[1, 2], [0, 4]])))))],
                       effort=1)                            # глубины 1 не хватает — шаблон берёт
    assert res["solved"] and res.get("via") == "template"
    assert mind.program_for("deep")(g) == want
