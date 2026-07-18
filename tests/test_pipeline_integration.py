"""Integration test บน MovieLens 100K จริง — skip อัตโนมัติถ้ายังไม่ได้ดาวน์โหลด

ใช้ tmp_path เป็น project root แยกขาดจาก artifacts จริง (copy zip ที่มีอยู่ไปใช้
เพื่อไม่ต้องดาวน์โหลดซ้ำ และ checksum ยังถูก verify เหมือนเดิม)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from movie_retrieval.config import ExperimentConfig, Paths
from movie_retrieval.data import EXPECTED_N_USERS
from movie_retrieval.pipeline import load_splits, prepare
from movie_retrieval.splits import audit_no_leakage

REAL_ZIP = Path(__file__).resolve().parents[1] / "data" / "raw" / "ml-100k.zip"

pytestmark = pytest.mark.skipif(
    not REAL_ZIP.exists(),
    reason="ml-100k.zip not downloaded yet (run `movie-retrieval prepare` first)",
)


@pytest.fixture
def tmp_paths(tmp_path) -> Paths:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    shutil.copy(REAL_ZIP, raw_dir / "ml-100k.zip")
    return Paths(root=tmp_path)


class TestPrepareOnRealData:
    def test_prepare_full_contract_and_split(self, tmp_paths):
        cfg = ExperimentConfig()
        summary = prepare(tmp_paths, cfg)

        # ml-100k: ทุก user มี >= 20 ratings → ไม่มีใครถูก drop ที่ min_history=5
        assert summary["n_users_dropped"] == 0
        assert summary["n_test_users"] == EXPECTED_N_USERS
        assert summary["n_test"] == EXPECTED_N_USERS  # k_test=1 ต่อ user
        assert summary["n_train"] + summary["n_val"] + summary["n_test"] == 100_000

        # โหลดกลับจาก CSV แล้ว audit ต้องผ่านเหมือนเดิม (round-trip integrity)
        split = load_splits(tmp_paths)
        report = audit_no_leakage(split, cfg.split)
        assert report["no_test_train_overlap"] is True

        # artifacts ที่ต้องเกิดขึ้น
        assert (tmp_paths.artifacts_dir / "split_config.json").exists()
        assert (tmp_paths.artifacts_dir / "data_card.json").exists()
