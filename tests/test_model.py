"""Two-tower model tests — synthetic data ขนาดเล็ก (TF ต้องถูกติดตั้ง)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movie_retrieval.config import ModelConfig
from movie_retrieval.model import TwoTowerModel, build_vocabs

CFG = ModelConfig(embedding_dim=8, epochs=3, batch_size=32, seed=7)


@pytest.fixture(scope="module")
def tiny_train() -> pd.DataFrame:
    """20 users แบ่ง 2 กลุ่ม: กลุ่ม A ดูแต่ m0-m2, กลุ่ม B ดูแต่ m3-m5

    โครงสร้างชัดเจนพอที่ two-tower ควรเรียนรู้ separation ได้ในไม่กี่ epoch
    """
    rng = np.random.default_rng(0)
    rows = []
    for u in range(20):
        group_movies = ["m0", "m1", "m2"] if u < 10 else ["m3", "m4", "m5"]
        for t, movie in enumerate(rng.permutation(group_movies * 3)):
            rows.append(
                {"user_id": f"u{u}", "movie_id": movie, "rating": 4.0, "timestamp": t}
            )
    return pd.DataFrame(rows).drop_duplicates(["user_id", "movie_id"])


class TestBuildVocabs:
    def test_sorted_numeric_ids(self):
        df = pd.DataFrame({"user_id": ["10", "2"], "movie_id": ["100", "20"]})
        users, movies = build_vocabs(df)
        assert users == ["2", "10"]
        assert movies == ["20", "100"]


class TestTwoTowerModel:
    def test_training_reduces_loss(self, tiny_train):
        users, movies = build_vocabs(tiny_train)
        cfg = ModelConfig(embedding_dim=8, epochs=10, batch_size=32, seed=7)
        model = TwoTowerModel(users, movies, cfg)
        history = model.fit(tiny_train)
        assert len(history) == cfg.epochs
        assert all(np.isfinite(loss) for loss in history)
        # dataset เล็กมาก epoch แรกๆ อาจแกว่ง — เทียบค่าเฉลี่ยท้ายกับ epoch แรก
        assert np.mean(history[-3:]) < history[0]

    def test_learns_group_structure(self, tiny_train):
        """user กลุ่ม A ต้อง score หนังกลุ่ม A สูงกว่าหนังกลุ่ม B"""
        users, movies = build_vocabs(tiny_train)
        model = TwoTowerModel(users, movies, ModelConfig(embedding_dim=8, epochs=10, seed=7))
        model.fit(tiny_train)
        scores = model.score_users(["u0"])[0]
        col = {m: j for j, m in enumerate(model.movie_vocab)}
        group_a = np.mean([scores[col[m]] for m in ["m0", "m1", "m2"]])
        group_b = np.mean([scores[col[m]] for m in ["m3", "m4", "m5"]])
        assert group_a > group_b

    def test_save_load_round_trip(self, tiny_train, tmp_path):
        users, movies = build_vocabs(tiny_train)
        model = TwoTowerModel(users, movies, CFG)
        model.fit(tiny_train)
        model.save(tmp_path / "model")

        restored = TwoTowerModel.load(tmp_path / "model", users, movies)
        np.testing.assert_allclose(
            model.score_users(users[:5]), restored.score_users(users[:5]), rtol=1e-6
        )

    def test_score_matrix_shape(self, tiny_train):
        users, movies = build_vocabs(tiny_train)
        model = TwoTowerModel(users, movies, CFG)
        assert model.score_users(users[:3]).shape == (3, len(movies))
