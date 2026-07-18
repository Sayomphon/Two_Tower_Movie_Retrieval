"""Temporal split + leakage audit tests — invariant สำคัญที่สุดของโปรเจค"""

from __future__ import annotations

import pandas as pd
import pytest

from movie_retrieval.config import SplitConfig
from movie_retrieval.splits import (
    LeakageError,
    SplitResult,
    audit_no_leakage,
    seen_items_map,
    temporal_leave_last_k,
)

CFG = SplitConfig(k_test=1, k_val=1, min_train=3)


class TestTemporalSplit:
    def test_latest_interaction_goes_to_test(self, synthetic_ratings):
        split = temporal_leave_last_k(synthetic_ratings, CFG)
        # interaction ล่าสุดของทุก user คือ m6 (timestamp สูงสุด)
        assert (split.test["movie_id"] == "m6").all()
        assert (split.val["movie_id"] == "m5").all()

    def test_temporal_ordering_per_user(self, synthetic_ratings):
        split = temporal_leave_last_k(synthetic_ratings, CFG)
        for user_id in split.test["user_id"].unique():
            train_max = split.train[split.train["user_id"] == user_id]["timestamp"].max()
            val_ts = split.val[split.val["user_id"] == user_id]["timestamp"].min()
            test_ts = split.test[split.test["user_id"] == user_id]["timestamp"].min()
            assert train_max <= val_ts <= test_ts

    def test_short_history_user_is_dropped_everywhere(self, synthetic_ratings):
        split = temporal_leave_last_k(synthetic_ratings, CFG)
        for frame in (split.train, split.val, split.test):
            assert "u6" not in set(frame["user_id"])
        assert split.n_users_dropped == 1

    def test_split_sizes(self, synthetic_ratings):
        split = temporal_leave_last_k(synthetic_ratings, CFG)
        assert split.summary()["n_test"] == 5  # 5 users ที่เหลือ x k_test=1
        assert split.summary()["n_val"] == 5
        assert split.summary()["n_train"] == 20  # 5 users x 4

    def test_deterministic_with_timestamp_ties(self):
        """interaction ที่ timestamp ชนกันต้องแบ่งเหมือนเดิมทุกครั้ง (tie-break ด้วย movie_id)"""
        rows = [
            {"user_id": "u1", "movie_id": m, "rating": 3.0, "timestamp": 100}
            for m in ["m3", "m1", "m2", "m5", "m4", "m6"]
        ]
        df = pd.DataFrame(rows)
        first = temporal_leave_last_k(df, CFG)
        second = temporal_leave_last_k(df.sample(frac=1, random_state=7), CFG)
        assert first.test["movie_id"].tolist() == second.test["movie_id"].tolist()
        assert first.val["movie_id"].tolist() == second.val["movie_id"].tolist()


class TestLeakageAudit:
    def test_future_leakage_raises(self, synthetic_ratings):
        split = temporal_leave_last_k(synthetic_ratings, CFG)
        # จำลอง bug: ย้าย test row (อนาคต) เข้า train
        poisoned = SplitResult(
            train=pd.concat([split.train, split.test]),
            val=split.val,
            test=split.test,
            n_users_dropped=0,
        )
        with pytest.raises(LeakageError):
            audit_no_leakage(poisoned, CFG)

    def test_test_user_without_history_raises(self, synthetic_ratings):
        split = temporal_leave_last_k(synthetic_ratings, CFG)
        poisoned = SplitResult(
            train=split.train[split.train["user_id"] != "u1"],
            val=split.val,
            test=split.test,
            n_users_dropped=0,
        )
        with pytest.raises(LeakageError, match="no train history"):
            audit_no_leakage(poisoned, CFG)


class TestSeenItemsMap:
    def test_merges_multiple_frames(self, synthetic_ratings):
        split = temporal_leave_last_k(synthetic_ratings, CFG)
        seen = seen_items_map(split.train, split.val)
        assert seen["u1"] == {"m1", "m2", "m3", "m4", "m5"}
        assert "m6" not in seen["u1"]  # test item ต้องไม่ถูกนับว่า seen
