"""Temporal leave-last-k-out split ต่อ user + leakage audit.

หลักการ (blueprint บทที่ 3):
- ห้ามใช้ random split เพราะ interaction อนาคตของ user จะรั่วเข้า train
- ต่อ user: เรียงตาม timestamp แล้วเก็บรายการล่าสุดเป็น test, ก่อนหน้าเป็น val
- popularity statistics และ vocabulary สร้างจาก train เท่านั้น
- ties ใน timestamp ตัดสินด้วย movie_id เพื่อให้ deterministic
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SplitConfig


class LeakageError(AssertionError):
    """Raised เมื่อ audit พบ future leakage — ผลการทดลองใช้ไม่ได้"""


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users_dropped: int

    def summary(self) -> dict:
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "n_train_users": self.train["user_id"].nunique(),
            "n_test_users": self.test["user_id"].nunique(),
            "n_users_dropped": self.n_users_dropped,
        }


def temporal_leave_last_k(df: pd.DataFrame, cfg: SplitConfig) -> SplitResult:
    """แบ่ง train/val/test ต่อ user ตาม timestamp (ล่าสุด → test)"""
    n_users_before = df["user_id"].nunique()

    # ตัด users ที่ history ไม่พอทั้งหมด (declared policy ใน SplitConfig)
    counts = df.groupby("user_id")["movie_id"].transform("count")
    df = df[counts >= cfg.min_history].copy()
    n_users_dropped = n_users_before - df["user_id"].nunique()

    # deterministic ordering: timestamp หลัก, movie_id รอง (ties พบได้จริงใน ml-100k)
    df = df.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort")
    df["rank_desc"] = df.groupby("user_id").cumcount(ascending=False)

    test = df[df["rank_desc"] < cfg.k_test].drop(columns="rank_desc")
    val = df[(df["rank_desc"] >= cfg.k_test) & (df["rank_desc"] < cfg.k_test + cfg.k_val)].drop(
        columns="rank_desc"
    )
    train = df[df["rank_desc"] >= cfg.k_test + cfg.k_val].drop(columns="rank_desc")

    result = SplitResult(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        n_users_dropped=n_users_dropped,
    )
    audit_no_leakage(result, cfg)
    return result


def audit_no_leakage(split: SplitResult, cfg: SplitConfig) -> dict:
    """ตรวจ invariant ทุกข้อของ temporal split — raise LeakageError ถ้าพบปัญหา

    Invariants:
    1. max(train ts) <= min(val ts) <= min(test ts) ต่อ user
    2. ทุก test user มี train history >= min_train
    3. (user, movie) ใน test ต้องไม่โผล่ใน train (test positive ไม่ถูก observe ตอน train)
    """
    train, val, test = split.train, split.val, split.test

    train_max_ts = train.groupby("user_id")["timestamp"].max()
    val_min_ts = val.groupby("user_id")["timestamp"].min()
    test_min_ts = test.groupby("user_id")["timestamp"].min()

    # 1. temporal ordering ต่อ user (เทียบเฉพาะ users ที่มีทั้งสองฝั่ง)
    common_tv = train_max_ts.index.intersection(val_min_ts.index)
    if not (train_max_ts.loc[common_tv] <= val_min_ts.loc[common_tv]).all():
        raise LeakageError("train timestamp is newer than validation timestamp for some user")
    common_vt = val_min_ts.index.intersection(test_min_ts.index)
    if not (val_min_ts.loc[common_vt] <= test_min_ts.loc[common_vt]).all():
        raise LeakageError("validation timestamp is newer than test timestamp for some user")

    # 2. ทุก test user ต้องมี train history เพียงพอ
    train_counts = train.groupby("user_id").size()
    test_users = test["user_id"].unique()
    missing = set(test_users) - set(train_counts.index)
    if missing:
        raise LeakageError(f"{len(missing)} test users have no train history")
    if not (train_counts.loc[list(test_users)] >= cfg.min_train).all():
        raise LeakageError("some test user has fewer than min_train interactions in train")

    # 3. test positives ต้องไม่อยู่ใน train
    train_pairs = set(zip(train["user_id"], train["movie_id"], strict=True))
    test_pairs = set(zip(test["user_id"], test["movie_id"], strict=True))
    overlap = train_pairs & test_pairs
    if overlap:
        raise LeakageError(f"{len(overlap)} (user, movie) pairs leak from test into train")

    return {
        "temporal_ordering_ok": True,
        "min_train_history_ok": True,
        "no_test_train_overlap": True,
        **split.summary(),
    }


def seen_items_map(*frames: pd.DataFrame) -> dict[str, set[str]]:
    """รวม (user -> set of movie_ids) จากหลาย splits สำหรับ seen-item filtering"""
    seen: dict[str, set[str]] = {}
    for frame in frames:
        for user_id, movie_id in zip(frame["user_id"], frame["movie_id"], strict=True):
            seen.setdefault(user_id, set()).add(movie_id)
    return seen
