"""Retrieval evaluation — operates on Top-K recommendation lists

Principles (blueprint chapter 7):
- every recommender is scored through the same interface: dict[user_id, ranked list],
  which makes the baseline-vs-two-tower comparison fair
- full-catalogue evaluation (1,682 items) — no sampling bias
- metrics are not accuracy only: coverage/popularity bias act as guardrails
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Recommendations = dict[str, list[str]]
Truth = dict[str, list[str]]


# ----------------------------------------------------------------- top-k

def topk_recommendations(
    scores: np.ndarray,
    user_ids: list[str],
    movie_vocab: list[str],
    k: int,
    seen: dict[str, set[str]] | None = None,
    exclude_seen: bool = True,
) -> Recommendations:
    """Turn a score matrix [n_users, n_movies] → Top-K per user (seen items masked with -inf)"""
    scores = scores.copy().astype(np.float64)
    col_of = {movie_id: j for j, movie_id in enumerate(movie_vocab)}

    if exclude_seen and seen:
        for i, user_id in enumerate(user_ids):
            for movie_id in seen.get(user_id, ()):  # noqa: B909
                j = col_of.get(movie_id)
                if j is not None:
                    scores[i, j] = -np.inf

    recs: Recommendations = {}
    movie_arr = np.asarray(movie_vocab, dtype=object)
    k = min(k, scores.shape[1])
    for i, user_id in enumerate(user_ids):
        top_idx = np.argpartition(-scores[i], k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[i][top_idx], kind="stable")]
        recs[user_id] = movie_arr[top_idx].tolist()
    return recs


# ---------------------------------------------------------- ranking metrics

def _dcg(hits: np.ndarray) -> float:
    """hits: binary relevance ordered by position 0..K-1"""
    positions = np.arange(len(hits))
    return float(np.sum(hits / np.log2(positions + 2)))


def ranking_metrics(recs: Recommendations, truth: Truth, ks: tuple[int, ...] = (10, 50)) -> dict:
    """Recall@K, NDCG@K, HitRate@K averaged per user (only users that have truth)"""
    results: dict[str, float] = {}
    users = [u for u in truth if u in recs]
    if not users:
        raise ValueError("no overlapping users between recommendations and truth")

    for k in ks:
        recalls, ndcgs, hit_rates = [], [], []
        for user_id in users:
            relevant = set(truth[user_id])
            top_k = recs[user_id][:k]
            hits = np.array([1.0 if m in relevant else 0.0 for m in top_k])
            n_hits = float(hits.sum())

            recalls.append(n_hits / len(relevant))
            hit_rates.append(1.0 if n_hits > 0 else 0.0)

            ideal_hits = np.ones(min(k, len(relevant)))
            idcg = _dcg(ideal_hits)
            ndcgs.append(_dcg(hits) / idcg if idcg > 0 else 0.0)

        results[f"recall@{k}"] = float(np.mean(recalls))
        results[f"ndcg@{k}"] = float(np.mean(ndcgs))
        results[f"hit_rate@{k}"] = float(np.mean(hit_rates))
    results["n_users_evaluated"] = len(users)
    return results


# ------------------------------------------------------- coverage and bias

def catalogue_coverage(recs: Recommendations, catalogue_size: int) -> float:
    """Share of the catalogue recommended to at least one user"""
    recommended = {m for items in recs.values() for m in items}
    return len(recommended) / catalogue_size


def popularity_bias(recs: Recommendations, item_counts: pd.Series) -> dict:
    """Measure how much the recommendations concentrate on popular movies

    - mean_popularity_percentile: 1.0 = only the most popular movies are recommended
    - top10pct_share: share of slots taken by the top-10%-popular movies
    - gini_exposure: 0 = every movie gets equal exposure, 1 = maximally concentrated
    """
    counts = item_counts.astype(float)
    percentile = counts.rank(pct=True)  # 1.0 = most popular
    top10_cut = counts.quantile(0.9)
    top10_items = set(counts[counts >= top10_cut].index)

    all_slots = [m for items in recs.values() for m in items]
    slot_percentiles = [percentile.get(m, 0.0) for m in all_slots]
    top10_share = sum(1 for m in all_slots if m in top10_items) / len(all_slots)

    exposure = pd.Series(all_slots).value_counts()
    exposure = exposure.reindex(counts.index, fill_value=0).sort_values().to_numpy(dtype=float)
    n = len(exposure)
    if exposure.sum() == 0:
        gini = 0.0
    else:
        cum = np.cumsum(exposure)
        gini = float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

    return {
        "mean_popularity_percentile": float(np.mean(slot_percentiles)),
        "top10pct_share": float(top10_share),
        "gini_exposure": gini,
    }


# ----------------------------------------------------------------- slices

def user_activity_slices(train_df: pd.DataFrame) -> dict[str, str]:
    """Bucket users into low/medium/high by train interaction count (terciles)"""
    counts = train_df.groupby("user_id").size()
    low_cut, high_cut = counts.quantile([1 / 3, 2 / 3])
    slices: dict[str, str] = {}
    for user_id, count in counts.items():
        if count <= low_cut:
            slices[user_id] = "low_activity"
        elif count <= high_cut:
            slices[user_id] = "medium_activity"
        else:
            slices[user_id] = "high_activity"
    return slices


def item_popularity_slices(item_counts: pd.Series, head_quantile: float = 0.9) -> dict[str, str]:
    """head = top-10% popular items, tail = everything else (unseen in train counts as tail)"""
    cut = item_counts.quantile(head_quantile)
    return {
        str(movie_id): ("head" if count >= cut else "tail")
        for movie_id, count in item_counts.items()
    }


def sliced_metrics(
    recs: Recommendations,
    truth: Truth,
    groups: dict[str, str],
    ks: tuple[int, ...] = (10,),
    default_group: str | None = None,
) -> dict[str, dict]:
    """Compute ranking metrics separately per user group"""
    by_group: dict[str, Truth] = {}
    for user_id, items in truth.items():
        group = groups.get(user_id, default_group)
        if group is None:
            continue
        by_group.setdefault(group, {})[user_id] = items

    return {
        group: ranking_metrics(recs, group_truth, ks=ks)
        for group, group_truth in sorted(by_group.items())
    }


def truth_from_frame(test_df: pd.DataFrame) -> Truth:
    """Turn the test DataFrame → dict[user_id, list of relevant movie_ids]"""
    return test_df.groupby("user_id")["movie_id"].apply(list).to_dict()
