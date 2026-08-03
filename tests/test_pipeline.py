"""Experiment-log bookkeeping tests — writing a run back into experiments.json must be idempotent"""

from __future__ import annotations

import json

import pytest

from movie_retrieval.config import Paths
from movie_retrieval.pipeline import _record_run


@pytest.fixture
def paths(tmp_path) -> Paths:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "experiments.json").write_text(
        json.dumps({"experiments": [{"run": "R1-dim32", "val_recall@10": 0.05}]})
    )
    return Paths(root=tmp_path)


def _runs(paths: Paths) -> list[dict]:
    return json.loads((paths.artifacts_dir / "experiments.json").read_text())["experiments"]


class TestRecordRun:
    def test_appends_new_runs_in_matrix_order(self, paths):
        # R4 is recorded before R0 — the file must still read in experiment-matrix order
        _record_run(paths, {"run": "R4-serving", "reload_consistent": True})
        _record_run(paths, {"run": "R0-popularity", "val_recall@10": 0.06})
        assert [r["run"] for r in _runs(paths)] == ["R0-popularity", "R1-dim32", "R4-serving"]

    def test_rerun_replaces_instead_of_duplicating(self, paths):
        _record_run(paths, {"run": "R4-serving", "query_latency_ms_p95": 0.9})
        _record_run(paths, {"run": "R4-serving", "query_latency_ms_p95": 0.2})

        runs = _runs(paths)
        assert [r["run"] for r in runs] == ["R1-dim32", "R4-serving"]
        assert runs[-1]["query_latency_ms_p95"] == 0.2  # latest value wins, no two accumulated rows

    def test_missing_log_reports_actionable_error(self, paths):
        (paths.artifacts_dir / "experiments.json").unlink()
        with pytest.raises(FileNotFoundError, match="movie-retrieval train"):
            _record_run(paths, {"run": "R4-serving"})
