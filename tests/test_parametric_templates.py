"""Тесты переменных в языке: параметрические примитивы и шаблоны с дыркой."""

from __future__ import annotations

from thinking_system.reasoning import parametric
from thinking_system.reasoning.grid_seed import full_grid_seed
from thinking_system.reasoning.grids import to_grid
from thinking_system.reasoning.search_prior import best_first_induce
from thinking_system.reasoning.templates import HOLE, anti_unify, template_search

G = to_grid([[3, 0, 2], [0, 3, 0]])


def test_parametric_semantics() -> None:
    assert parametric.make("keep", 3).fn(G) == ((3, 0, 0), (0, 3, 0))
    assert parametric.make("drop", 3).fn(G) == ((0, 0, 2), (0, 0, 0))
    assert parametric.make("paint", 7).fn(G) == ((7, 0, 7), (0, 7, 0))


def test_palette_and_instantiate_from_task_data() -> None:
    pairs = [(G, parametric.make("keep", 3).fn(G))]
    assert parametric.task_palette(pairs) == [2, 3]
    names = {p.name for p in parametric.instantiate(pairs)}
    enumerated = {"keep[2]", "keep[3]", "drop[2]", "drop[3]", "paint[2]", "paint[3]"}
    computed = {f"{fam}[{w}]" for fam in ("keep", "drop", "paint") for w in ("top", "rare")}
    assert names == enumerated | computed                    # палитра + вычисляемые


def test_by_name_roundtrip_and_reject() -> None:
    p = parametric.by_name("paint[5]")
    assert p is not None and p.fn(G) == ((5, 0, 5), (0, 5, 0))
    assert parametric.by_name("flip_h") is None


def test_parametric_solves_color_task_in_search() -> None:
    # «оставь только цвет 3 и обрежь» — вне бесцветного seed, решается с переменной-цветом
    target = lambda g: __import__("thinking_system.reasoning.perception", fromlist=["bounding_box"]).bounding_box(parametric.make("keep", 3).fn(g))
    pairs = [(G, target(G))]
    seed = full_grid_seed()
    assert best_first_induce(pairs, seed, max_depth=2, budget=4000)[0] is None
    prog, _ = best_first_induce(pairs, seed + parametric.instantiate(pairs), max_depth=2, budget=8000)
    assert prog is not None and str(prog) == "keep[3] ▸ bbox"


def test_anti_unify_binds_parameter_into_hole() -> None:
    sols = [["keep[3]", "bbox"], ["keep[2]", "bbox"], ["flip_h", "flip_v"]]
    tpls = anti_unify(sols)
    assert (HOLE, "bbox") in tpls                            # разные привязки цвета → переменная
    assert all(len(t) >= 2 and t.count(HOLE) == 1 for t in tpls)


def test_template_search_reaches_depth3_cheaply() -> None:
    seed = full_grid_seed()
    by = {p.name: p for p in seed}
    # опыт: два трёхшаговых решения, различающихся последним шагом → шаблон с дыркой в хвосте
    tpls = anti_unify([["flip_h", "flip_v", "bbox"], ["flip_h", "flip_v", "gravity"]])
    assert tpls == [("flip_h", "flip_v", HOLE)]
    want = lambda g: by["transpose"].fn(by["flip_v"].fn(by["flip_h"].fn(g)))
    pairs = [(G, want(G)), (to_grid([[1, 2], [0, 4]]), want(to_grid([[1, 2], [0, 4]])))]
    prog, checked = template_search(pairs, tpls, seed)
    assert prog is not None and str(prog) == "flip_h ▸ flip_v ▸ transpose"
    assert checked <= len(seed)                              # цена — один слот, не глубина 3


def test_each_applies_to_every_object_in_place() -> None:
    from thinking_system.reasoning import object_param as op

    # два объекта: Г-образный слева-сверху и палочка справа-снизу
    g = to_grid([[1, 1, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 2, 0],
                 [0, 0, 2, 0]])
    out = op.make_each("flip_h").fn(g)
    assert out == to_grid([[1, 1, 0, 0],                     # каждый отражён В СВОЁМ боксе,
                           [0, 1, 0, 0],                     # а не сетка целиком
                           [0, 0, 2, 0],
                           [0, 0, 2, 0]])
    glob = to_grid([[0, 0, 1, 1], [0, 0, 0, 1], [0, 2, 0, 0], [0, 2, 0, 0]])
    from thinking_system.reasoning.grids import flip_h
    assert flip_h(g) == glob and out != glob                 # пообъектно ≠ глобально


def test_each_rejects_size_changing_inner() -> None:
    import pytest
    from thinking_system.reasoning import object_param as op
    from thinking_system.reasoning.structural import replicate_by_self

    with pytest.raises(ValueError):
        op.each_apply(to_grid([[1, 1], [0, 1]]), replicate_by_self)


def test_pick_selects_kth_largest() -> None:
    from thinking_system.reasoning import object_param as op

    g = to_grid([[3, 3, 3, 0, 0],
                 [0, 0, 0, 2, 2],
                 [7, 0, 0, 0, 0]])
    assert op.make_pick(2).fn(g) == to_grid([[0, 0, 0, 0, 0],
                                             [0, 0, 0, 2, 2],
                                             [0, 0, 0, 0, 0]])
    import pytest
    with pytest.raises(ValueError):
        op.pick_apply(g, 4)                                  # объектов меньше k — честный отказ


def test_object_by_name_roundtrip() -> None:
    from thinking_system.reasoning import object_param as op

    g = to_grid([[1, 1, 0], [1, 0, 0], [0, 0, 2]])
    assert op.by_name("each[flip_v]").fn(g) == op.make_each("flip_v").fn(g)
    assert op.by_name("pick[2]").fn(g) == op.make_pick(2).fn(g)
    assert op.by_name("each[чудо]") is None and op.by_name("flip_h") is None


def test_mind_solves_per_object_task() -> None:
    from thinking_system.mind import Mind
    from thinking_system.reasoning import object_param as op

    g1 = to_grid([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 2, 2], [0, 0, 0, 2]])
    g2 = to_grid([[0, 5, 5], [0, 5, 0], [0, 0, 0]])
    want = op.make_each("flip_h").fn
    mind = Mind()
    res = mind.attempt("perobj", [(g1, want(g1)), (g2, want(g2))], effort=1)
    assert res["solved"] and mind.solutions["perobj"] == ["each[flip_h]"]
    assert mind.program_for("perobj")(g1) == want(g1)        # восстановление из имени


def test_where_predicates_semantics() -> None:
    from thinking_system.reasoning import object_param as op

    g = to_grid([[3, 3, 3, 0, 7],
                 [0, 0, 0, 0, 0],
                 [2, 2, 0, 0, 5]])
    # перекрась САМЫЙ БОЛЬШОЙ объект в 8; остальные не тронуты
    big = op.by_name("big[paint[8]]").fn(g)
    assert big == to_grid([[8, 8, 8, 0, 7], [0, 0, 0, 0, 0], [2, 2, 0, 0, 5]])
    # перекрась самый маленький (первый из одиночек) — только он
    small = op.by_name("small[paint[9]]").fn(g)
    assert small == to_grid([[3, 3, 3, 0, 9], [0, 0, 0, 0, 0], [2, 2, 0, 0, 5]])
    # сотри ВСЕ объекты размера 1
    ones = op.by_name("one[paint[0]]").fn(g)
    assert ones == to_grid([[3, 3, 3, 0, 0], [0, 0, 0, 0, 0], [2, 2, 0, 0, 0]])
    assert op.by_name("big[чудо]") is None


def test_predicates_instantiated_from_palette() -> None:
    from thinking_system.reasoning import object_param as op

    pairs = [(to_grid([[3, 0], [0, 7]]), to_grid([[8, 0], [0, 7]]))]
    names = {p.name for p in op.instantiate_predicates(pairs)}
    # палитра {3,7,8} ∪ {0} × предикаты {big,small,one} × две связности
    assert "big[paint[8]]" in names and "one[paint[0]]" in names
    assert "bigc[paint[8]]" in names and "onec[paint[0]]" in names
    assert len(names) == 3 * 4 * 2


def test_mind_solves_predicate_task() -> None:
    from thinking_system.mind import Mind

    g1 = to_grid([[3, 3, 0, 7], [3, 0, 0, 0]])
    g2 = to_grid([[0, 5, 0], [5, 5, 0], [0, 0, 2]])          # объекты разнесены: связность
    want1 = to_grid([[8, 8, 0, 7], [8, 0, 0, 0]])            # без учёта цвета (как keep_largest)
    want2 = to_grid([[0, 8, 0], [8, 8, 0], [0, 0, 2]])       # большой → 8
    mind = Mind()
    res = mind.attempt("recolor_big", [(g1, want1), (g2, want2)], effort=1)
    assert res["solved"] and mind.solutions["recolor_big"] == ["big[paint[8]]"]
    assert mind.program_for("recolor_big")(g1) == want1      # восстановление из имени


def test_same_color_connectivity_is_a_variable() -> None:
    from thinking_system.reasoning import object_param as op

    # связная фигура из 3 с одиночной чужой клеткой 7 ВНУТРИ
    g = to_grid([[3, 3, 3],
                 [3, 7, 3],
                 [3, 3, 0]])
    # цветослепая связность видит ОДИН объект — one[...] не находит одиночек
    assert op.by_name("one[paint[0]]").fn(g) == g
    # одноцветная связность видит 7 как отдельный объект размера 1 — и стирает её
    assert op.by_name("onec[paint[0]]").fn(g) == to_grid([[3, 3, 3],
                                                          [3, 0, 3],
                                                          [3, 3, 0]])
    # bigc красит наибольшую ОДНОЦВЕТНУЮ область (тройки), не трогая 7
    assert op.by_name("bigc[paint[5]]").fn(g) == to_grid([[5, 5, 5],
                                                          [5, 7, 5],
                                                          [5, 5, 0]])


def test_same_color_each_and_pick_roundtrip() -> None:
    from thinking_system.reasoning import object_param as op

    g = to_grid([[1, 2, 0],
                 [1, 2, 0],
                 [0, 0, 5]])
    # одноцветная: три объекта (1-столбик, 2-столбик, 5); цветослепая: два
    assert len(op._components_same_color(g)) == 3
    from thinking_system.reasoning.perception import _components
    assert len(_components(g)) == 2
    assert op.by_name("pickc[2]").fn(g) == to_grid([[0, 2, 0], [0, 2, 0], [0, 0, 0]])
    assert op.by_name("eachc[flip_v]").fn(g) == g            # столбики симметричны


def test_bigram_prior_counts_pairs() -> None:
    from thinking_system.reasoning.deep_search import bigram_prior, _START

    b = bigram_prior([["flip_h", "flip_v", "bbox"], ["flip_h", "flip_v"]])
    assert b[("flip_h", "flip_v")] == 2 and b[("flip_v", "bbox")] == 1
    assert b[(_START, "flip_h")] == 2


def test_guided_induce_finds_depth3_and_bigram_helps() -> None:
    from thinking_system.reasoning.deep_search import bigram_prior, guided_induce

    seed = full_grid_seed()
    by = {p.name: p for p in seed}
    want = lambda g: by["bbox"].fn(by["flip_v"].fn(by["flip_h"].fn(g)))
    g1 = to_grid([[0, 0, 0], [3, 5, 0], [7, 0, 0]])
    g2 = to_grid([[0, 2, 4], [0, 0, 6], [0, 0, 0]])
    pairs = [(g1, want(g1)), (g2, want(g2))]

    blind, n_blind = guided_induce(pairs, seed, None, max_depth=3, budget=30000)
    assert blind is not None and blind(g1) == want(g1)
    # биграммы прежних решений (flip_h→flip_v, flip_v→bbox) ведут поиск короче
    bigr = bigram_prior([["flip_h", "flip_v", "gravity"], ["flip_v", "bbox"]])
    guided, n_guided = guided_induce(pairs, seed, bigr, max_depth=3, budget=30000)
    assert guided is not None and guided(g1) == want(g1)
    assert n_guided < n_blind


def test_guided_heuristic_prefers_target_shape() -> None:
    from thinking_system.reasoning.deep_search import guided_induce

    seed = full_grid_seed()
    by = {p.name: p for p in seed}
    # цель — транспонированная НЕквадратная сетка: форма цели сразу отсекает
    # ветки, сохраняющие исходную форму, в конец очереди
    g = to_grid([[1, 2, 3], [4, 5, 6]])
    want = by["transpose"].fn(g)
    prog, n = guided_induce([(g, want)], seed, None, max_depth=2, budget=5000)
    assert prog is not None and prog(g) == want
    assert n <= len(seed) * 3                               # нашли в первых волнах


def test_mind_effort3_uses_guided_depth3(tmp_path: Path) -> None:
    from thinking_system.mind import Mind

    mind = Mind(str(tmp_path / "m.json"))
    g1 = to_grid([[0, 0, 0], [3, 5, 0], [7, 0, 0]])
    g2 = to_grid([[0, 2, 4], [0, 0, 6], [0, 0, 0]])
    from thinking_system.reasoning.grids import flip_h, flip_v
    from thinking_system.reasoning.perception import bounding_box
    want = lambda g: bounding_box(flip_v(flip_h(g)))
    res = mind.attempt("deep3", [(g1, want(g1)), (g2, want(g2))], effort=3)
    assert res["solved"] and res["depth"] == 3
    assert mind.program_for("deep3")(g1) == want(g1)


def test_expansion_layer_semantics() -> None:
    from thinking_system.reasoning import expansion as ex

    g = to_grid([[1, 2], [0, 3]])
    assert ex.upscale2(g) == to_grid([[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 3, 3], [0, 0, 3, 3]])
    assert ex.tile_h(g) == to_grid([[1, 2, 1, 2], [0, 3, 0, 3]])
    assert ex.tile_v(g) == to_grid([[1, 2], [0, 3], [1, 2], [0, 3]])
    assert ex.mirror_h(g) == to_grid([[1, 2, 2, 1], [0, 3, 3, 0]])
    assert ex.mirror_v(g) == to_grid([[1, 2], [0, 3], [0, 3], [1, 2]])


def test_expansion_composes_in_search() -> None:
    from thinking_system.reasoning import expansion as ex
    from thinking_system.reasoning.perception import bounding_box
    from thinking_system.reasoning.search_prior import best_first_induce

    seed = full_grid_seed()
    # «обрежь до занятого и увеличь ×2» — мотив масштабирующих задач ARC
    want = lambda g: ex.upscale2(bounding_box(g))
    g1 = to_grid([[0, 0, 0], [0, 5, 1], [0, 0, 0]])
    g2 = to_grid([[7, 0], [0, 0]])
    prog, _ = best_first_induce([(g1, want(g1)), (g2, want(g2))], seed, max_depth=2, budget=4000)
    assert prog is not None and str(prog) == "bbox ▸ upscale2"
    # взрывной рост в глубокой композиции отсекается guard-ом, а не виснет
    from thinking_system.reasoning.grid_seed import guard
    import pytest
    big = to_grid([[1] * 30] * 30)
    up2 = next(p for p in seed if p.name == "upscale2")
    with pytest.raises(ValueError):
        guard(up2).fn(ex.upscale3(big))                      # upscale2 после ×3 на 30×30


def test_pairwise_layer_semantics() -> None:
    from thinking_system.reasoning import pairwise as pw

    # лево|разделитель|право (нечётная ширина — средний столбец отброшен)
    g = to_grid([[1, 0, 9, 2, 2],
                 [0, 1, 9, 0, 2]])
    assert pw.make("and", "h").fn(g) == to_grid([[1, 0], [0, 1]])   # занято в обеих
    assert pw.make("or", "h").fn(g) == to_grid([[1, 2], [0, 1]])    # приоритет первой
    assert pw.make("xor", "h").fn(g) == to_grid([[0, 2], [0, 0]])   # ровно в одной
    assert pw.make("diff", "h").fn(g) == to_grid([[0, 0], [0, 0]])  # первая минус вторая
    v = to_grid([[5, 0], [0, 5], [5, 5]])                           # верх/низ, средний ряд прочь
    assert pw.make("and", "v").fn(v) == to_grid([[5, 0]])
    import pytest
    with pytest.raises(ValueError):
        pw.make("and", "h").fn(to_grid([[7], [7]]))                 # не из чего взять половины


def test_pairwise_composes_with_paint_in_search() -> None:
    from thinking_system.reasoning import pairwise as pw, parametric
    from thinking_system.reasoning.search_prior import best_first_induce

    seed = full_grid_seed()
    xor_h = pw.make("xor", "h").fn
    want = lambda g: parametric.make("paint", 3).fn(xor_h(g))       # мотив «совмести половины»
    g1 = to_grid([[1, 0, 9, 2, 2], [0, 1, 9, 0, 2]])
    g2 = to_grid([[5, 5, 9, 5, 0], [0, 0, 9, 5, 5]])
    pairs = [(g1, want(g1)), (g2, want(g2))]
    prims = seed + parametric.instantiate(pairs)
    prog, _ = best_first_induce(pairs, prims, max_depth=2, budget=20000)
    assert prog is not None and str(prog) == "xor_h ▸ paint[3]"


def test_computed_arguments_semantics_and_ties() -> None:
    from thinking_system.reasoning import parametric as p

    g = to_grid([[3, 3, 7], [0, 3, 7]])                      # 3×3, 7×2
    assert p.by_name("paint[top]").fn(g) == to_grid([[3, 3, 3], [0, 3, 3]])
    assert p.by_name("drop[rare]").fn(g) == to_grid([[3, 3, 0], [0, 3, 0]])
    tie = to_grid([[5, 2], [2, 5]])                          # ничья 2×2 — меньший цвет
    assert p.by_name("keep[top]").fn(tie) == to_grid([[0, 2], [2, 0]])
    import pytest
    with pytest.raises(ValueError):
        p.by_name("paint[top]").fn(to_grid([[0, 0]]))        # пустая сетка — честный отказ


def test_computed_argument_generalizes_across_palettes() -> None:
    from thinking_system.mind import Mind

    # роль «шум = самый редкий цвет» играет РАЗНЫЙ цвет в разных парах:
    # перечисляемый drop[c] не согласуется с обеими, вычисляемый drop[rare] — да
    g1 = to_grid([[3, 3, 3], [3, 7, 3]])                     # редкий 7
    g2 = to_grid([[5, 5, 1], [5, 5, 5]])                     # редкий 1
    want1 = to_grid([[3, 3, 3], [3, 0, 3]])
    want2 = to_grid([[5, 5, 0], [5, 5, 5]])
    mind = Mind()
    res = mind.attempt("denoise_rare", [(g1, want1), (g2, want2)], effort=1)
    # на двух цветах keep[top] ≡ drop[rare] — принимаем любой ВЫЧИСЛЯЕМЫЙ вариант
    assert res["solved"] and mind.solutions["denoise_rare"] in (["drop[rare]"], ["keep[top]"])
    assert mind.program_for("denoise_rare")(g1) == want1     # восстановление из имени
