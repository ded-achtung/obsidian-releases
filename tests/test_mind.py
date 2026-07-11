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
    from thinking_system.language.morphology import stem

    known = {stem("отражение"), stem("переворот")}           # парсер сравнивает по основам
    assert parse_grid_definition("поворот это сначала отражение потом переворот", known) \
        == ("поворот", ["отражение", "переворот"])
    # падежные формы известных слов тоже узнаются (частично открытый словарь)
    assert parse_grid_definition("поворот это сначала отражения потом переворота", known) \
        == ("поворот", ["отражения", "переворота"])
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


LIB_BASE = """
отражение [[1, 2], [3, 4]] → [[2, 1], [4, 3]]
отражение [[5, 0, 6]] → [[6, 0, 5]]
переворот [[1, 2], [3, 4]] → [[3, 4], [1, 2]]
переворот [[7], [8]] → [[8], [7]]
"""
LIB_ADV = """
поворот это сначала отражение потом переворот
зеркалирование значит отражение
инверсия это сначала обращение потом отражение
"""
LIB_NOISE = """
Головоломки на квадратных полях известны с древности.
Никаких новых операций в заметках не вводится.
"""


def test_peek_value_changes_with_knowledge(tmp_path: Path) -> None:
    mind = Mind(str(tmp_path / "m.json"))
    # до базового учебника продвинутый ничего не даёт (его слова не заземлить)
    assert mind.peek(LIB_ADV)["value"] == 0
    assert mind.peek(LIB_NOISE)["value"] == 0
    assert mind.peek(LIB_BASE)["value"] == 2                 # два новых показа
    mind.read(LIB_BASE)
    peek = mind.peek(LIB_ADV)                                # теперь определения собираются
    assert peek["value"] == 2 and set(peek["groundable"]) == {"поворот", "зеркалирование"}
    assert peek["gaps"] == ["обращение"]                     # незаземлимое слово — вопрос


def test_study_library_orders_by_curiosity_and_skips(tmp_path: Path) -> None:
    mind = Mind(str(tmp_path / "m.json"))
    log = mind.study_library({"adv": LIB_ADV, "noise": LIB_NOISE, "base": LIB_BASE})
    chosen = [e.get("выбрано") for e in log if "выбрано" in e]
    assert chosen == ["base", "adv"]                         # куррикулум возник сам
    assert log[-1].get("пропущено") == ["noise"]             # бесполезное честно пропущено
    assert "поворот" in mind.lexicon.words and mind.abstractions == ["flip_h∘flip_v"]
    assert mind.lexicon.words["зеркалирование"] == ["flip_h"]  # синоним через «значит»
    assert mind.questions == ["обращение"]                   # открытый вопрос в состоянии
    assert any("обращение" in a for a in mind.agenda())      # …и в повестке

    # вопросы переживают процесс и снимаются, когда слово наконец объяснено
    mind.save()
    mind2 = Mind(str(tmp_path / "m.json"))
    assert mind2.questions == ["обращение"]
    mind2.read("обращение [[1, 2]] → [[2, 1]]\nобращение [[3], [4]] → [[3], [4]]")
    assert mind2.questions == []                             # вопрос закрыт показом


def test_open_vocab_forms_and_prose(tmp_path: Path) -> None:
    from thinking_system.reasoning.grids import flip_v, flip_h

    mind = Mind(str(tmp_path / "m.json"))
    # показ ВНУТРИ прозы, без стрелки: две сетки в строке = вход → выход
    res = mind.read("Например отражение превращает [[1, 2], [3, 4]] в [[2, 1], [4, 3]] всегда")
    assert res["выучено_слов"] == ["отражение"]
    assert mind.lexicon.program("отражениями") is not None   # форма слова узнаётся по основе
    # определение падежными формами поверх выученных слов
    mind.read("переворот [[1, 2], [3, 4]] → [[3, 4], [1, 2]]\nпереворот [[7], [8]] → [[8], [7]]")
    res = mind.read("разворот это сначала отражения потом переворота")
    assert res["определено"] == ["разворот"]
    g = to_grid([[1, 2, 0], [0, 3, 4]])
    assert mind.lexicon.program("разворотом")(g) == flip_v(flip_h(g))


def test_guided_smoothing_bounds_adversarial_prior() -> None:
    import math

    from thinking_system.reasoning.deep_search import make_step_cost

    seed = full_grid_seed()
    # враждебный приор: миллионные счётчики на ВСЕХ прочих переходах
    bad = {("^", p.name): 1_000_000 for p in seed if p.name != "flip_h"}
    cost0 = make_step_cost(bad, seed, smooth=0.0)
    cost3 = make_step_cost(bad, seed, smooth=0.3)
    # без сглаживания стоимость непопулярного шага растёт со счётчиками без предела
    assert cost0("^", "flip_h") > 15                         # ≈ log(5e6·|P|)
    # со сглаживанием — ограничена −log(smooth/V), какие бы счётчики ни накопились
    bound = -math.log(0.3 / len(seed))
    assert cost3("^", "flip_h") <= bound + 1e-9 < 6
    # и порядок предпочтений приора сохраняется (сглаживание ≠ стирание опыта)
    assert cost3("^", "gravity") < cost3("^", "flip_h")


def _maze_item(seed: int) -> dict:
    from thinking_system.world.maze_dist import random_maze

    env = random_maze(seed)
    return {"id": f"maze-{seed}",
            "world": {"size": env.size, "walls": sorted(env.walls),
                      "start": list(env.start), "goal": list(env.goal)}}


def test_world_routed_and_solved_by_execution() -> None:
    mind = Mind()
    item = _maze_item(306)
    assert mind.perceive(item) == "world"                    # третий тип опыта
    res = mind.experience(item, effort=2)
    assert res["routed"] == "world" and res["solved"]
    assert res["steps"] >= res["optimal"]                    # реальный проход не короче оптимума


def test_world_ladder_escalates_via_agenda() -> None:
    mind = Mind()
    res = mind.explore("m300", _maze_item(300)["world"], effort=1)
    assert not res["solved"]                                 # беглого взгляда честно не хватило
    assert any("мирам" in a for a in mind.agenda())          # мир попал в повестку
    work = mind.idle_work()                                  # думать дольше = исследовать дольше
    assert [w[0] for w in work["worlds_resolved"]] == ["m300"]
    assert not mind.unsolved_worlds


def test_world_map_survives_process(tmp_path: Path) -> None:
    spec = _maze_item(301)["world"]
    mind = Mind(str(tmp_path / "m.json"))
    assert mind.explore("m301", spec, effort=2)["solved"]
    mind.save()
    mind2 = Mind(str(tmp_path / "m.json"))                   # новый процесс, та же память
    res = mind2.explore("m301", spec, effort=2)
    assert res["solved"] and res["explored"] == 0            # решил сразу по карте из памяти


def test_world_skill_transfers_to_fresh_worlds() -> None:
    mind = Mind()
    for sd in (200, 201, 202):                               # жизнь: решает и практикует навык
        assert mind.explore(f"w{sd}", _maze_item(sd)["world"], effort=2)["solved"]
    assert mind.worlds_practiced == 3
    assert mind.world_skill_sub == [6, 6]                    # подцель навыка выучена из жизни
    fresh = _maze_item(215)["world"]                         # свежий мир того же распределения
    with_skill = mind.explore("t215", fresh, effort=1)
    scratch = Mind().explore("t215", fresh, effort=1)
    assert with_skill["solved"] and not scratch["solved"]    # беглого взгляда хватает ТОЛЬКО с навыком


def test_questions_drive_reading_and_pending_definitions() -> None:
    mind = Mind()
    mind.read(TEXTBOOK)                                      # отражение/переворот заземлены
    res = mind.read("инверсия это сначала обращение потом отражение")
    assert res["вопросы"] == ["обращение"]                   # определение не собрать — вопрос
    assert mind.pending_defs                                 # …и оно ЖДЁТ, а не выброшено
    lib = {
        "ценный": ("падение [[5, 0], [0, 0]] → [[0, 0], [5, 0]]\n"
                   "падение [[3, 3], [0, 0]] → [[0, 0], [3, 3]]\n"
                   "остов [[0, 0], [0, 7]] → [[7]]"),
        "отвечающий": ("обращение [[1, 2], [3, 4]] → [[1, 3], [2, 4]]\n"
                       "обращение [[5, 0, 6]] → [[5], [0], [6]]"),
    }
    log = mind.study_library(lib)
    assert log[0]["выбрано"] == "отвечающий"                 # ЦЕЛЬ важнее ценности (1 < 2)
    assert log[0]["цель"] == ["обращение"]
    assert "обращение" not in mind.questions                 # вопрос снят показом
    assert not mind.pending_defs                             # определение достроено…
    prog = mind.lexicon.program("инверсия")
    g = to_grid([[1, 2], [3, 4]])
    assert prog is not None and prog(g) == to_grid([[3, 1], [4, 2]])  # …и это поворот на 90°


def test_text_to_worlds_actions_and_advice() -> None:
    mind = Mind()
    res = mind.read("вниз (0, 0) → (1, 0)\nвниз (2, 1) → (3, 1)\nнаправо (1, 2) → (1, 3)\n"
                    "если не знаешь куда, иди вниз или направо")
    assert res["выучено_действий"] == ["вниз", "направо"]    # заземлено индукцией из переходов
    assert mind.action_words == {"вниз": 1, "направо": 3}
    assert mind.world_advice == [1, 3]                       # совет составлен из выученных слов
    # неоднозначный показ (диагональ) — честный отказ
    assert not Mind().read("наискосок (0, 0) → (1, 1)")["выучено_действий"]
    fresh = _maze_item(215)["world"]                         # свежий мир, агент миров не решал
    advised = mind.explore("t215", fresh, effort=1)
    scratch = Mind().explore("t215", fresh, effort=1)
    assert advised["solved"] and not scratch["solved"]       # знания из ТЕКСТА хватает бегло
