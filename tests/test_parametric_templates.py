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
