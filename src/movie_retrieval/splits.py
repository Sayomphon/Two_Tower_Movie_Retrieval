"""Per-user temporal leave-last-k-out split + leakage audit.

Principles (blueprint chapter 3):
- never use a random split: a user's future interactions would leak into train
- per user: sort by timestamp, keep the latest items as test and the ones before as val
- popularity statistics and vocabulary are built from train only
- ties in timestamp are broken by movie_id to stay deterministic
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SplitConfig


class LeakageError(AssertionError):
    """Raised when the audit finds future leakage — the experiment results are void"""


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
    """Split train/val/test per user by timestamp (most recent → test)"""
    n_users_before = df["user_id"].nunique()

    # drop users whose history is too short (declared policy in SplitConfig)
    counts = df.groupby("user_id")["movie_id"].transform("count")
    df = df[counts >= cfg.min_history].copy()
    n_users_dropped = n_users_before - df["user_id"].nunique()

    # deterministic ordering: timestamp first, movie_id second (ties do occur in ml-100k)
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
    """Check every invariant of the temporal split — raise LeakageError on any problem

    Invariants:
    1. max(train ts) <= min(val ts) <= min(test ts) per user
    2. every test user has train history >= min_train
    3. (user, movie) in test must not appear in train (test positives are unobserved at train time)
    """
    train, val, test = split.train, split.val, split.test

    train_max_ts = train.groupby("user_id")["timestamp"].max()
    val_min_ts = val.groupby("user_id")["timestamp"].min()
    test_min_ts = test.groupby("user_id")["timestamp"].min()

    # 1. temporal ordering per user (compare only users present on both sides)
    common_tv = train_max_ts.index.intersection(val_min_ts.index)
    if not (train_max_ts.loc[common_tv] <= val_min_ts.loc[common_tv]).all():
        raise LeakageError("train timestamp is newer than validation timestamp for some user")
    common_vt = val_min_ts.index.intersection(test_min_ts.index)
    if not (val_min_ts.loc[common_vt] <= test_min_ts.loc[common_vt]).all():
        raise LeakageError("validation timestamp is newer than test timestamp for some user")

    # 2. every test user must have enough train history
    train_counts = train.groupby("user_id").size()
    test_users = test["user_id"].unique()
    missing = set(test_users) - set(train_counts.index)
    if missing:
        raise LeakageError(f"{len(missing)} test users have no train history")
    if not (train_counts.loc[list(test_users)] >= cfg.min_train).all():
        raise LeakageError("some test user has fewer than min_train interactions in train")

    # 3. test positives must not be in train
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
    """Merge (user -> set of movie_ids) across splits for seen-item filtering"""
    seen: dict[str, set[str]] = {}
    for frame in frames:
        for user_id, movie_id in zip(frame["user_id"], frame["movie_id"], strict=True):
            seen.setdefault(user_id, set()).add(movie_id)
    return seen
