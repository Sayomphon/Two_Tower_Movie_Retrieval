"""Metric correctness tests — ตรวจกับค่าที่คำนวณมือ"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from movie_retrieval.evaluate import (
    catalogue_coverage,
    item_popularity_slices,
    popularity_bias,
    ranking_metrics,
    sliced_metrics,
    topk_recommendations,
    user_activity_slices,
)


class TestRankingMetrics:
    def test_hit_at_position_two(self):
        recs = {"u1": ["a", "b", "c"]}
        truth = {"u1": ["b"]}
        metrics = ranking_metrics(recs, truth, ks=(1, 2))
        assert metrics["recall@1"] == 0.0
        assert metrics["recall@2"] == 1.0
        # b อยู่ตำแหน่ง index 1 → DCG = 1/log2(3), IDCG = 1
        assert metrics["ndcg@2"] == pytest.approx(1 / math.log2(3))
        assert metrics["hit_rate@2"] == 1.0

    def test_perfect_first_position(self):
        metrics = ranking_metrics({"u1": ["x"]}, {"u1": ["x"]}, ks=(1,))
        assert metrics["recall@1"] == 1.0
        assert metrics["ndcg@1"] == 1.0

    def test_average_over_users(self):
        recs = {"u1": ["a"], "u2": ["b"]}
        truth = {"u1": ["a"], "u2": ["z"]}
        metrics = ranking_metrics(recs, truth, ks=(1,))
        assert metrics["recall@1"] == 0.5
        assert metrics["n_users_evaluated"] == 2

    def test_no_overlap_raises(self):
        with pytest.raises(ValueError):
            ranking_metrics({"u1": ["a"]}, {"u9": ["a"]}, ks=(1,))


class TestTopKRecommendations:
    def test_orders_by_score_and_masks_seen(self):
        scores = np.array([[1.0, 3.0, 2.0]])
        recs = topk_recommendations(
            scores, ["u1"], ["m1", "m2", "m3"], k=2, seen={"u1": {"m2"}}
        )
        assert recs["u1"] == ["m3", "m1"]  # m2 ถูก mask ออก

    def test_no_seen_filtering_when_disabled(self):
        scores = np.array([[1.0, 3.0, 2.0]])
        recs = topk_recommendations(
            scores, ["u1"], ["m1", "m2", "m3"], k=2, seen={"u1": {"m2"}}, exclude_seen=False
        )
        assert recs["u1"] == ["m2", "m3"]

    def test_no_duplicate_items(self):
        # score เท่ากันคือเคสที่ tie-break พลาดแล้วคืนหนังซ้ำได้ง่ายที่สุด
        scores = np.array([[2.0, 2.0, 2.0, 1.0], [3.0, 1.0, 3.0, 3.0]])
        recs = topk_recommendations(scores, ["u1", "u2"], ["m1", "m2", "m3", "m4"], k=4)
        for user_id, items in recs.items():
            assert len(items) == len(set(items)), f"duplicate movie for {user_id}"


class TestCoverageAndBias:
    def test_coverage(self):
        recs = {"u1": ["a", "b"], "u2": ["b", "c"]}
        assert catalogue_coverage(recs, catalogue_size=10) == pytest.approx(0.3)

    def test_bias_all_most_popular(self):
        counts = pd.Series([5, 4, 3, 2, 1], index=["a", "b", "c", "d", "e"])
        recs = {f"u{i}": ["a"] for i in range(5)}
        bias = popularity_bias(recs, counts)
        assert bias["mean_popularity_percentile"] == 1.0
        assert bias["top10pct_share"] == 1.0
        # exposure = [0,0,0,0,5] → gini = (n+1-2*sum(cum)/cum[-1])/n = 0.8
        assert bias["gini_exposure"] == pytest.approx(0.8)


class TestSlices:
    def test_user_activity_terciles(self):
        rows = (
            [("low", f"m{i}") for i in range(2)]
            + [("mid", f"m{i}") for i in range(5)]
            + [("high", f"m{i}") for i in range(9)]
        )
        df = pd.DataFrame(rows, columns=["user_id", "movie_id"])
        slices = user_activity_slices(df)
        assert slices["low"] == "low_activity"
        assert slices["mid"] == "medium_activity"
        assert slices["high"] == "high_activity"

    def test_item_popularity_slices(self):
        counts = pd.Series(range(1, 21), index=[f"m{i}" for i in range(20)])
        groups = item_popularity_slices(counts)
        assert groups["m19"] == "head"  # count สูงสุด
        assert groups["m0"] == "tail"

    def test_sliced_metrics_partitions_users(self):
        recs = {"u1": ["a"], "u2": ["b"]}
        truth = {"u1": ["a"], "u2": ["b"]}
        groups = {"u1": "g1", "u2": "g2"}
        result = sliced_metrics(recs, truth, groups, ks=(1,))
        assert result["g1"]["recall@1"] == 1.0
        assert result["g1"]["n_users_evaluated"] == 1
        assert result["g2"]["n_users_evaluated"] == 1
