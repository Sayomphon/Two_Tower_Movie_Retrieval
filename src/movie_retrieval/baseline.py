"""Global popularity baseline (R0 ใน experiment matrix).

กติกาสำคัญ:
- popularity counts มาจาก train เท่านั้น (ห้ามเห็น val/test)
- exclude seen items ของแต่ละ user ก่อนคืน Top-K
- deterministic: ties ใน count ตัดสินด้วย movie_id

Recommender ใดๆ ในโปรเจคนี้ต้องคืน dict[user_id, list[movie_id]] (เรียงจากดีสุด)
เพื่อให้ evaluate ด้วย evaluator ตัวเดียวกันได้ทั้ง baseline และ two-tower
"""

from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """แนะนำหนังตามจำนวน interaction ใน train (non-personalized)"""

    def __init__(self) -> None:
        self._ranked_items: list[str] = []
        self._counts: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> PopularityRecommender:
        counts = (
            train_df.groupby("movie_id")
            .size()
            .rename("count")
            .reset_index()
            # count มาก่อน (desc), movie_id ตัดสิน ties (asc) — deterministic
            .sort_values(["count", "movie_id"], ascending=[False, True], kind="mergesort")
        )
        self._ranked_items = counts["movie_id"].tolist()
        self._counts = counts.set_index("movie_id")["count"]
        return self

    @property
    def item_counts(self) -> pd.Series:
        if self._counts is None:
            raise RuntimeError("PopularityRecommender is not fitted")
        return self._counts

    def top_items(self, k: int) -> list[str]:
        """Global Top-K (ไม่ filter seen) — ใช้เป็น fallback สำหรับ unknown user"""
        return self._ranked_items[:k]

    def recommend(
        self,
        user_id: str,
        k: int,
        seen: dict[str, set[str]] | None = None,
        exclude_seen: bool = True,
    ) -> list[str]:
        user_seen = (seen or {}).get(user_id, set()) if exclude_seen else set()
        recs: list[str] = []
        for movie_id in self._ranked_items:
            if movie_id in user_seen:
                continue
            recs.append(movie_id)
            if len(recs) == k:
                break
        return recs

    def recommend_batch(
        self,
        user_ids: list[str],
        k: int,
        seen: dict[str, set[str]] | None = None,
        exclude_seen: bool = True,
    ) -> dict[str, list[str]]:
        return {
            user_id: self.recommend(user_id, k, seen=seen, exclude_seen=exclude_seen)
            for user_id in user_ids
        }
