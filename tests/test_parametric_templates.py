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
    assert names == {"keep[2]", "keep[3]", "drop[2]", "drop[3]", "paint[2]", "paint[3]"}


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
