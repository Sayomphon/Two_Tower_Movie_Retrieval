"""Popularity baseline tests"""

from __future__ import annotations

import pandas as pd
import pytest

from movie_retrieval.baseline import PopularityRecommender


@pytest.fixture
def train_df() -> pd.DataFrame:
    # ความนิยม: m1 x3, m2 x2, m3 x1
    rows = [
        ("u1", "m1"), ("u2", "m1"), ("u3", "m1"),
        ("u1", "m2"), ("u2", "m2"),
        ("u1", "m3"),
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id"])


class TestPopularityRecommender:
    def test_ranks_by_train_count(self, train_df):
        pop = PopularityRecommender().fit(train_df)
        assert pop.top_items(3) == ["m1", "m2", "m3"]

    def test_excludes_seen_items(self, train_df):
        pop = PopularityRecommender().fit(train_df)
        seen = {"u1": {"m1", "m2", "m3"}}
        assert pop.recommend("u1", k=2, seen=seen) == []
        seen = {"u2": {"m1"}}
        assert pop.recommend("u2", k=2, seen=seen) == ["m2", "m3"]

    def test_include_seen_flag(self, train_df):
        pop = PopularityRecommender().fit(train_df)
        seen = {"u1": {"m1"}}
        assert pop.recommend("u1", k=1, seen=seen, exclude_seen=False) == ["m1"]

    def test_deterministic_tie_break(self):
        # m2 และ m1 count เท่ากัน → เรียงตาม movie_id
        df = pd.DataFrame([("u1", "m2"), ("u2", "m1")], columns=["user_id", "movie_id"])
        pop = PopularityRecommender().fit(df)
        assert pop.top_items(2) == ["m1", "m2"]

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _ = PopularityRecommender().item_counts
