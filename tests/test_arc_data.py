"""Тесты загрузчика ARC и честности протокола run_arc (на фикстурах, без сети)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thinking_system.reasoning.arc_data import load_arc, load_arc_dir


def _write_task(path: Path, name: str, train, test) -> None:
    (path / f"{name}.json").write_text(json.dumps({"train": train, "test": test}))


def _fixture_dir(tmp_path: Path) -> Path:
    d = tmp_path / "training"
    d.mkdir()
    _write_task(d, "aaaa0001",
                train=[{"input": [[1, 2]], "output": [[2, 1]]},
                       {"input": [[3, 4]], "output": [[4, 3]]}],
                test=[{"input": [[5, 6]], "output": [[6, 5]]}])
    _write_task(d, "bbbb0002",
                train=[{"input": [[0, 1], [2, 3]], "output": [[2, 3], [0, 1]]}],
                test=[{"input": [[7, 8], [9, 0]], "output": [[9, 0], [7, 8]]}])
    return d


def test_load_arc_dir_parses_pairs_and_sorts(tmp_path: Path) -> None:
    tasks = load_arc_dir(str(_fixture_dir(tmp_path)))
    assert [t.task_id for t in tasks] == ["aaaa0001", "bbbb0002"]
    t = tasks[0]
    assert len(t.train) == 2 and len(t.test) == 1
    i, o = t.train[0]
    assert i == ((1, 2),) and o == ((2, 1),)                 # Grid = кортеж кортежей int
    assert all(isinstance(v, int) for row in i for v in row)


def test_load_arc_prefers_local_dir(tmp_path: Path) -> None:
    _fixture_dir(tmp_path)
    tasks = load_arc("training", data_dir=str(tmp_path))
    assert len(tasks) == 2


def test_load_arc_unknown_split_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_arc("test", data_dir=str(tmp_path))


def test_protocol_search_never_sees_hidden_test(tmp_path: Path) -> None:
    """Программа ищется по train-парам; скрытый test только проверяет её."""
    from thinking_system.reasoning.grids import grid_primitives
    from thinking_system.reasoning.search_prior import best_first_induce

    task = load_arc_dir(str(_fixture_dir(tmp_path)))[0]      # отражение по горизонтали
    prog, _ = best_first_induce(list(task.train), grid_primitives(), max_depth=1)
    assert prog is not None and str(prog) == "flip_h"
    assert all(prog(i) == o for i, o in task.test)           # верна и на скрытом test


def test_load_arc_via_arckit_if_installed() -> None:
    pytest.importorskip("arckit")
    tasks = load_arc("training", data_dir="/nonexistent")
    assert len(tasks) == 400                                 # ARC-AGI-1 training
    assert all(t.train and t.test for t in tasks[:5])
