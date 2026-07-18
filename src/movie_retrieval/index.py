"""Serving index — brute-force Top-K retrieval เป็น SavedModel

Design decisions:
- ใช้ pure TF ops (StaticHashTable + matmul + top_k) — artifact ไม่ผูกกับ Keras
  version ใดๆ โหลดได้ทุก environment ที่มี TF
- unknown user (OOV) → fallback เป็น popularity scores พร้อม flag `fallback_used`
- seen-item filtering เป็น post-retrieval business filter อยู่ฝั่ง RetrievalService
  (production จะเป็น filtering service แยกชั้น ตาม blueprint บทที่ 9)
- BruteForce เหมาะกับ catalogue 1,682 เรื่อง — scale ใหญ่ต้องเปลี่ยนเป็น ANN
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf

MAX_K = 100
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class BruteForceIndex(tf.Module):
    """Queryable Top-K index: user_id (string) → movie ids + scores + fallback flag"""

    def __init__(
        self,
        user_vocab: list[str],
        user_embeddings: np.ndarray,  # [n_users + 1, D] แถว 0 = OOV
        movie_vocab: list[str],
        movie_embeddings: np.ndarray,  # [n_movies, D] เรียงตาม movie_vocab
        popularity_scores: np.ndarray,  # [n_movies] fallback สำหรับ unknown user
    ) -> None:
        super().__init__()
        if user_embeddings.shape[0] != len(user_vocab) + 1:
            raise ValueError("user_embeddings must include OOV row at index 0")
        if movie_embeddings.shape[0] != len(movie_vocab):
            raise ValueError("movie_embeddings must align with movie_vocab")

        self.user_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(user_vocab),
                tf.range(1, len(user_vocab) + 1, dtype=tf.int64),
            ),
            default_value=tf.constant(0, dtype=tf.int64),  # 0 = unknown user
        )
        self.user_embeddings = tf.Variable(
            user_embeddings.astype(np.float32), trainable=False, name="user_embeddings"
        )
        self.movie_ids = tf.constant(movie_vocab)
        self.movie_embeddings = tf.Variable(
            movie_embeddings.astype(np.float32), trainable=False, name="movie_embeddings"
        )
        self.popularity = tf.Variable(
            popularity_scores.astype(np.float32), trainable=False, name="popularity"
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name="user_ids"),
            tf.TensorSpec(shape=[], dtype=tf.int32, name="k"),
        ]
    )
    def recommend(self, user_ids: tf.Tensor, k: tf.Tensor) -> dict[str, tf.Tensor]:
        idx = self.user_table.lookup(user_ids)  # [B]
        user_emb = tf.gather(self.user_embeddings, idx)  # [B, D]
        scores = tf.matmul(user_emb, self.movie_embeddings, transpose_b=True)  # [B, N]

        fallback = tf.equal(idx, 0)
        scores = tf.where(fallback[:, None], self.popularity[None, :], scores)

        k = tf.minimum(k, tf.shape(self.movie_ids)[0])
        top_scores, top_idx = tf.math.top_k(scores, k=k)
        return {
            "movie_ids": tf.gather(self.movie_ids, top_idx),
            "scores": top_scores,
            "fallback_used": fallback,
        }


def save_index(index: BruteForceIndex, index_dir: Path) -> None:
    tf.saved_model.save(
        index,
        str(index_dir),
        signatures={"recommend": index.recommend.get_concrete_function()},
    )


def load_index(index_dir: Path) -> tf.Module:
    return tf.saved_model.load(str(index_dir))


class RetrievalService:
    """Inference wrapper ตาม contract ใน blueprint บทที่ 9

    - validate input (user_id charset/length, 1 <= k <= MAX_K)
    - seen-item filtering หลัง retrieval
    - แนบ model/index version ทุก response
    """

    def __init__(
        self,
        index: tf.Module,
        seen: dict[str, set[str]],
        titles: dict[str, str],
        model_version: str,
        index_version: str,
        catalogue_size: int,
    ) -> None:
        self._index = index
        self._seen = seen
        self._titles = titles
        self._model_version = model_version
        self._index_version = index_version
        self._catalogue_size = catalogue_size

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path) -> RetrievalService:
        index = load_index(artifacts_dir / "bruteforce_index")
        vocab = json.loads((artifacts_dir / "vocab.json").read_text())
        seen_raw = json.loads((artifacts_dir / "seen_items.json").read_text())
        titles = json.loads((artifacts_dir / "titles.json").read_text())
        versions = json.loads((artifacts_dir / "versions.json").read_text())
        return cls(
            index=index,
            seen={user: set(items) for user, items in seen_raw.items()},
            titles=titles,
            model_version=versions["model_version"],
            index_version=versions["index_version"],
            catalogue_size=len(vocab["movie_vocab"]),
        )

    def recommend(self, user_id: str, k: int = 10, exclude_seen: bool = True) -> dict:
        # ---- input validation (ป้องกัน abuse/typo ก่อนแตะ index) ----
        if not isinstance(user_id, str) or not _USER_ID_PATTERN.match(user_id):
            raise ValueError("user_id must be 1-64 chars of [A-Za-z0-9_-]")
        if not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be an integer in [1, {MAX_K}]")

        user_seen = self._seen.get(user_id, set()) if exclude_seen else set()
        # ขอเผื่อจาก index เท่าจำนวน seen เพื่อให้เหลือครบ k หลัง filter
        fetch_k = min(k + len(user_seen), self._catalogue_size)

        result = self._index.recommend(tf.constant([user_id]), tf.constant(fetch_k, tf.int32))
        movie_ids = [m.decode() for m in result["movie_ids"].numpy()[0]]
        scores = result["scores"].numpy()[0].tolist()
        fallback_used = bool(result["fallback_used"].numpy()[0])

        recommendations = []
        for movie_id, score in zip(movie_ids, scores, strict=True):
            if movie_id in user_seen:
                continue
            recommendations.append(
                {
                    "movie_id": movie_id,
                    "title": self._titles.get(movie_id, "(unknown title)"),
                    "score": round(float(score), 4),
                }
            )
            if len(recommendations) == k:
                break

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "fallback_used": fallback_used,
            "strategy": "popularity_fallback" if fallback_used else "two_tower_retrieval",
            "model_version": self._model_version,
            "index_version": self._index_version,
        }
