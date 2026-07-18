"""End-to-end pipeline orchestration: prepare → train → evaluate → export

แต่ละขั้นเขียน artifact ลงดิสก์เพื่อให้รันแยกขั้นได้ (rerun-friendly ตาม blueprint บทที่ 4)
และ final test evaluation ทำครั้งเดียวหลังเลือก config จาก validation เท่านั้น
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, replace
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from . import data as data_mod
from .baseline import PopularityRecommender
from .config import ExperimentConfig, ModelConfig, Paths
from .evaluate import (
    catalogue_coverage,
    item_popularity_slices,
    popularity_bias,
    ranking_metrics,
    sliced_metrics,
    topk_recommendations,
    truth_from_frame,
    user_activity_slices,
)
from .index import BruteForceIndex, RetrievalService, load_index, save_index
from .model import TwoTowerModel, build_vocabs
from .splits import SplitResult, seen_items_map, temporal_leave_last_k

logger = logging.getLogger(__name__)

EVAL_KS = (10, 50)
SELECT_K = 10  # model selection ใช้ val recall@10


# ---------------------------------------------------------------- prepare

def prepare(paths: Paths, cfg: ExperimentConfig) -> dict:
    """Download + validate + positive rule + temporal split → data/processed/"""
    ratings = data_mod.load_ratings(paths.raw_dir)
    card = data_mod.validate_ratings(ratings)
    logger.info("data contract OK: %s", card)

    positives = data_mod.apply_positive_rule(ratings, cfg.split.positive_rating_threshold)
    split = temporal_leave_last_k(positives, cfg.split)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(paths.processed_dir / "train.csv", index=False)
    split.val.to_csv(paths.processed_dir / "val.csv", index=False)
    split.test.to_csv(paths.processed_dir / "test.csv", index=False)

    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(paths.artifacts_dir / "split_config.json")
    (paths.artifacts_dir / "data_card.json").write_text(
        json.dumps(asdict(card), indent=2, ensure_ascii=False)
    )

    summary = split.summary()
    logger.info("split summary: %s", summary)
    return summary


def load_splits(paths: Paths) -> SplitResult:
    dtypes = {"user_id": str, "movie_id": str, "rating": float, "timestamp": int}

    def _read(name: str) -> pd.DataFrame:
        return pd.read_csv(paths.processed_dir / f"{name}.csv", dtype=dtypes)

    return SplitResult(
        train=_read("train"), val=_read("val"), test=_read("test"), n_users_dropped=0
    )


# ------------------------------------------------------------------ train

def val_recall(model: TwoTowerModel, split: SplitResult, seen_train: dict) -> float:
    """Val recall@SELECT_K ด้วย full-catalogue scoring (mask เฉพาะ train-seen)"""
    val_truth = truth_from_frame(split.val)
    users = sorted(val_truth, key=int)
    scores = model.score_users(users)
    recs = topk_recommendations(scores, users, model.movie_vocab, SELECT_K, seen=seen_train)
    return ranking_metrics(recs, val_truth, ks=(SELECT_K,))[f"recall@{SELECT_K}"]


def train(paths: Paths, cfg: ExperimentConfig) -> dict:
    """รัน experiment matrix (R1 dim32, R2 dim64) เลือก winner จาก val recall@10"""
    split = load_splits(paths)
    seen_train = seen_items_map(split.train)
    user_vocab, movie_vocab = build_vocabs(split.train)

    candidates = [
        ("R1-dim32", replace(cfg.model, embedding_dim=32)),
        ("R2-dim64", replace(cfg.model, embedding_dim=64, l2_regularization=1e-5)),
    ]

    experiments: list[dict] = []
    best: tuple[str, TwoTowerModel, float] | None = None
    for run_name, model_cfg in candidates:
        logger.info("=== training %s: %s ===", run_name, model_cfg)
        model = TwoTowerModel(user_vocab, movie_vocab, model_cfg)
        started = time.perf_counter()
        history = model.fit(split.train)
        train_seconds = time.perf_counter() - started

        recall = val_recall(model, split, seen_train)
        experiments.append(
            {
                "run": run_name,
                "config": asdict(model_cfg),
                "final_train_loss": history[-1],
                f"val_recall@{SELECT_K}": recall,
                "train_seconds": round(train_seconds, 2),
            }
        )
        logger.info("%s val recall@%d = %.4f", run_name, SELECT_K, recall)
        if best is None or recall > best[2]:
            best = (run_name, model, recall)

    assert best is not None
    winner_name, winner_model, winner_recall = best

    # baseline บน val เพื่อเทียบใน selection report
    pop = PopularityRecommender().fit(split.train)
    val_truth = truth_from_frame(split.val)
    val_users = sorted(val_truth, key=int)
    pop_recs = pop.recommend_batch(val_users, SELECT_K, seen=seen_train)
    pop_recall = ranking_metrics(pop_recs, val_truth, ks=(SELECT_K,))[f"recall@{SELECT_K}"]

    winner_model.save(paths.model_dir)
    report = {
        "experiments": experiments,
        "selected_run": winner_name,
        f"selected_val_recall@{SELECT_K}": winner_recall,
        f"baseline_val_recall@{SELECT_K}": pop_recall,
        "beats_baseline": winner_recall > pop_recall,
    }
    (paths.artifacts_dir / "experiments.json").write_text(json.dumps(report, indent=2))
    logger.info("selected %s (val recall@%d %.4f vs baseline %.4f)",
                winner_name, SELECT_K, winner_recall, pop_recall)
    return report


# --------------------------------------------------------------- evaluate

def _evaluate_recs(recs, truth, item_counts, catalogue_size) -> dict:
    metrics = ranking_metrics(recs, truth, ks=EVAL_KS)
    metrics["catalogue_coverage@10"] = catalogue_coverage(
        {u: items[:10] for u, items in recs.items()}, catalogue_size
    )
    metrics["popularity_bias@10"] = popularity_bias(
        {u: items[:10] for u, items in recs.items()}, item_counts
    )
    return metrics


def evaluate(paths: Paths, cfg: ExperimentConfig, with_sensitivity: bool = False) -> dict:
    """Final test evaluation ครั้งเดียว + slices + export serving artifacts"""
    split = load_splits(paths)
    user_vocab, movie_vocab = build_vocabs(split.train)
    model = TwoTowerModel.load(paths.model_dir, user_vocab, movie_vocab)
    pop = PopularityRecommender().fit(split.train)
    item_counts = pop.item_counts
    catalogue_size = len(movie_vocab)

    # test evaluation: mask ทุกอย่างที่ user เคยเห็นก่อน test (train + val)
    seen_before_test = seen_items_map(split.train, split.val)
    test_truth = truth_from_frame(split.test)
    test_users = sorted(test_truth, key=int)
    max_k = max(EVAL_KS)

    scores = model.score_users(test_users)
    model_recs = topk_recommendations(
        scores, test_users, movie_vocab, max_k, seen=seen_before_test
    )
    pop_recs = pop.recommend_batch(test_users, max_k, seen=seen_before_test)

    # OOV test items: หนังที่ไม่อยู่ใน train vocab — model ไม่มีทาง retrieve ได้
    vocab_set = set(movie_vocab)
    oov_rate = float(np.mean([m not in vocab_set for items in test_truth.values() for m in items]))

    results = {
        "baseline": _evaluate_recs(pop_recs, test_truth, item_counts, catalogue_size),
        "two_tower": _evaluate_recs(model_recs, test_truth, item_counts, catalogue_size),
        "oov_test_item_rate": oov_rate,
    }

    # ---- slices (blueprint บทที่ 7) ----
    activity = user_activity_slices(split.train)
    results["two_tower"]["slices_by_user_activity"] = sliced_metrics(
        model_recs, test_truth, activity, ks=(10,)
    )
    pop_slices = item_popularity_slices(item_counts)
    test_item_group = {
        user: pop_slices.get(items[0], "tail") for user, items in test_truth.items()
    }
    results["two_tower"]["slices_by_test_item_popularity"] = sliced_metrics(
        model_recs, test_truth, test_item_group, ks=(10,)
    )

    # ---- export serving artifacts + consistency/latency checks ----
    serving = export_serving_artifacts(paths, cfg, model, pop, split)
    results["serving"] = serving

    if with_sensitivity:
        results["sensitivity_rating_ge_4"] = sensitivity_threshold(paths, cfg)

    results["generated_at"] = datetime.now(UTC).isoformat()
    (paths.artifacts_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    logger.info("final metrics written to %s", paths.artifacts_dir / "metrics.json")
    return results


def export_serving_artifacts(
    paths: Paths,
    cfg: ExperimentConfig,
    model: TwoTowerModel,
    pop: PopularityRecommender,
    split: SplitResult,
) -> dict:
    """สร้าง index SavedModel + vocab/seen/titles/versions และตรวจ reload consistency"""
    movie_vocab = model.movie_vocab
    popularity_scores = (
        pop.item_counts.reindex(movie_vocab).fillna(0).to_numpy(dtype=np.float32)
    )
    index = BruteForceIndex(
        user_vocab=model.user_vocab,
        user_embeddings=model.user_embedding.embeddings.numpy(),
        movie_vocab=movie_vocab,
        movie_embeddings=model.movie_embedding_matrix(),
        popularity_scores=popularity_scores,
    )

    sample_users = model.user_vocab[:20]
    before = index.recommend(np.array(sample_users, dtype=object), 10)["movie_ids"].numpy()

    save_index(index, paths.index_dir)
    reloaded = load_index(paths.index_dir)
    after = reloaded.recommend(
        np.array(sample_users, dtype=object), np.int32(10)
    )["movie_ids"].numpy()
    reload_consistent = bool((before == after).all())
    if not reload_consistent:
        raise RuntimeError("index reload produced different Top-K — artifact is not trustworthy")

    # latency benchmark (single-user query แบบ production shape)
    latencies = []
    for user_id in model.user_vocab[:100]:
        started = time.perf_counter()
        reloaded.recommend(np.array([user_id], dtype=object), np.int32(10))
        latencies.append((time.perf_counter() - started) * 1000)

    # serving metadata artifacts
    seen = seen_items_map(split.train, split.val)
    (paths.artifacts_dir / "vocab.json").write_text(
        json.dumps(
            {
                "user_vocab": model.user_vocab,
                "movie_vocab": movie_vocab,
                "oov_policy": {
                    "unknown_user": "popularity fallback + fallback_used flag",
                    "unknown_movie": "excluded from candidate catalogue",
                },
            }
        )
    )
    (paths.artifacts_dir / "seen_items.json").write_text(
        json.dumps({user: sorted(items) for user, items in seen.items()})
    )
    titles = data_mod.load_movie_titles(paths.raw_dir)
    (paths.artifacts_dir / "titles.json").write_text(json.dumps(titles, ensure_ascii=False))
    (paths.artifacts_dir / "versions.json").write_text(
        json.dumps(
            {
                "model_version": cfg.model_version,
                "index_version": f"catalog-{date.today().isoformat()}",
            },
            indent=2,
        )
    )

    return {
        "reload_consistent": reload_consistent,
        "query_latency_ms_p50": round(float(np.percentile(latencies, 50)), 3),
        "query_latency_ms_p95": round(float(np.percentile(latencies, 95)), 3),
        "index_type": "brute_force_saved_model",
    }


def sensitivity_threshold(paths: Paths, cfg: ExperimentConfig) -> dict:
    """R3: rating >= 4 เป็น positive แทนการใช้ทุก rating — วัดผลต่อ recall"""
    ratings = data_mod.load_ratings(paths.raw_dir)
    positives = data_mod.apply_positive_rule(ratings, 4.0)
    strict_cfg = replace(cfg.split, positive_rating_threshold=4.0)
    split = temporal_leave_last_k(positives, strict_cfg)

    user_vocab, movie_vocab = build_vocabs(split.train)
    model = TwoTowerModel(user_vocab, movie_vocab, ModelConfig(seed=cfg.model.seed))
    model.fit(split.train)

    seen_before_test = seen_items_map(split.train, split.val)
    test_truth = truth_from_frame(split.test)
    users = sorted(test_truth, key=int)
    recs = topk_recommendations(
        model.score_users(users), users, movie_vocab, max(EVAL_KS), seen=seen_before_test
    )
    metrics = ranking_metrics(recs, test_truth, ks=EVAL_KS)
    metrics["split_summary"] = split.summary()
    return metrics


# ---------------------------------------------------------------- serving

def recommend_cli(paths: Paths, user_id: str, k: int, exclude_seen: bool) -> dict:
    service = RetrievalService.from_artifacts(paths.artifacts_dir)
    return service.recommend(user_id, k=k, exclude_seen=exclude_seen)
