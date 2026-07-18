"""Two-tower retrieval model (pure TensorFlow/Keras 3).

Design decision: implement retrieval loss เองแทนการใช้ TensorFlow Recommenders
เพราะ TFRS อยู่ใน maintenance mode และไม่ compatible กับ Keras 3 โดยตรง
(ต้องใช้ TF_USE_LEGACY_KERAS env hack) — การ own โค้ด ~60 บรรทัดนี้
โปร่งใสกว่าและเป็น interview material ที่ดีกว่า

Architecture (blueprint บทที่ 6):
    user_id  → StringLookup → Embedding(dim) ┐
                                             ├→ dot product → Top-K
    movie_id → StringLookup → Embedding(dim) ┘

Loss: in-batch sampled softmax — ใช้ movie อื่นใน batch เดียวกันเป็น negatives
พร้อม accidental-hit masking (movie ซ้ำใน batch ต้องไม่ถูกนับเป็น negative)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import ModelConfig

logger = logging.getLogger(__name__)

keras = tf.keras


class TwoTowerModel:
    """User/movie embedding towers + dot-product retrieval

    Index 0 ของทุก embedding table สงวนให้ OOV (unknown user/movie)
    """

    def __init__(self, user_vocab: list[str], movie_vocab: list[str], cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.user_vocab = list(user_vocab)
        self.movie_vocab = list(movie_vocab)

        keras.utils.set_random_seed(cfg.seed)
        initializer = keras.initializers.TruncatedNormal(stddev=0.1, seed=cfg.seed)

        self.user_lookup = keras.layers.StringLookup(
            vocabulary=self.user_vocab, num_oov_indices=1, mask_token=None
        )
        self.movie_lookup = keras.layers.StringLookup(
            vocabulary=self.movie_vocab, num_oov_indices=1, mask_token=None
        )
        self.user_embedding = keras.layers.Embedding(
            len(self.user_vocab) + 1, cfg.embedding_dim, embeddings_initializer=initializer
        )
        self.movie_embedding = keras.layers.Embedding(
            len(self.movie_vocab) + 1, cfg.embedding_dim, embeddings_initializer=initializer
        )
        # build weights ทันทีเพื่อให้ save/load ได้ก่อน train
        self.user_embedding.build((None,))
        self.movie_embedding.build((None,))

    # ---------------------------------------------------------------- towers

    def user_tower(self, user_ids: tf.Tensor) -> tf.Tensor:
        return self.user_embedding(self.user_lookup(user_ids))

    def movie_tower(self, movie_ids: tf.Tensor) -> tf.Tensor:
        return self.movie_embedding(self.movie_lookup(movie_ids))

    # ------------------------------------------------------------- training

    def _loss_step(self, user_ids: tf.Tensor, movie_ids: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """คืน (sum_loss สำหรับ gradients, mean_loss สำหรับ logging)

        ใช้ SUM reduction ตาม semantics ของ TFRS Retrieval task — gradient
        scale ใหญ่พอให้ Adagrad เดินได้จริง (mean reduction ทำให้ step เล็ก
        จน loss แบนบน dataset เล็ก)
        """
        user_emb = self.user_tower(user_ids)  # [B, D]
        movie_idx = self.movie_lookup(movie_ids)  # [B]
        movie_emb = self.movie_embedding(movie_idx)  # [B, D]

        logits = tf.matmul(user_emb, movie_emb, transpose_b=True)  # [B, B]
        batch = tf.shape(logits)[0]
        labels = tf.range(batch)

        # accidental-hit masking: movie เดียวกันที่ตำแหน่งอื่นใน batch
        # เป็น positive ปลอมๆ ห้ามนับเป็น negative
        same_movie = tf.equal(movie_idx[None, :], movie_idx[:, None])  # [B, B]
        not_diagonal = ~tf.eye(batch, dtype=tf.bool)
        logits = tf.where(same_movie & not_diagonal, tf.float32.min, logits)

        per_example = keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        sum_loss = tf.reduce_sum(per_example)
        mean_loss = tf.reduce_mean(per_example)

        if self.cfg.l2_regularization > 0:
            sum_loss += self.cfg.l2_regularization * (
                tf.reduce_sum(tf.square(user_emb)) + tf.reduce_sum(tf.square(movie_emb))
            )
        return sum_loss, mean_loss

    def fit(
        self,
        train_df: pd.DataFrame,
        epoch_callback: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        """Train ด้วย custom loop (GradientTape) — คืน loss history ต่อ epoch"""
        cfg = self.cfg
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "user_id": train_df["user_id"].to_numpy(dtype=object),
                "movie_id": train_df["movie_id"].to_numpy(dtype=object),
            }
        )
        dataset = (
            dataset.shuffle(len(train_df), seed=cfg.seed, reshuffle_each_iteration=True)
            .batch(cfg.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        optimizer = keras.optimizers.Adagrad(learning_rate=cfg.learning_rate)
        variables = (
            self.user_embedding.trainable_variables + self.movie_embedding.trainable_variables
        )

        @tf.function
        def train_step(batch: dict[str, tf.Tensor]) -> tf.Tensor:
            with tf.GradientTape() as tape:
                sum_loss, mean_loss = self._loss_step(batch["user_id"], batch["movie_id"])
            gradients = tape.gradient(sum_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables, strict=True))
            return mean_loss

        history: list[float] = []
        for epoch in range(cfg.epochs):
            epoch_losses = []
            for batch in dataset:
                epoch_losses.append(float(train_step(batch)))
            mean_loss = float(np.mean(epoch_losses))
            history.append(mean_loss)
            logger.info("epoch %d/%d loss=%.4f", epoch + 1, cfg.epochs, mean_loss)
            if epoch_callback is not None:
                epoch_callback(epoch, mean_loss)
        return history

    # -------------------------------------------------------------- scoring

    def movie_embedding_matrix(self) -> np.ndarray:
        """Embedding ของทุกหนังใน vocab (เรียงตาม self.movie_vocab)"""
        return self.movie_tower(tf.constant(self.movie_vocab)).numpy()

    def score_users(self, user_ids: list[str]) -> np.ndarray:
        """คืน score matrix [len(user_ids), len(movie_vocab)] — full-catalogue retrieval

        catalogue 1,682 เรื่อง → matmul ตรงๆ ถูกต้องแม่นยำกว่า sampled evaluation
        """
        user_emb = self.user_tower(tf.constant(user_ids)).numpy()
        return user_emb @ self.movie_embedding_matrix().T

    # ------------------------------------------------------------ artifacts

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            model_dir / "weights.npz",
            user_embeddings=self.user_embedding.embeddings.numpy(),
            movie_embeddings=self.movie_embedding.embeddings.numpy(),
        )
        (model_dir / "model_config.json").write_text(
            json.dumps(
                {
                    "embedding_dim": self.cfg.embedding_dim,
                    "learning_rate": self.cfg.learning_rate,
                    "epochs": self.cfg.epochs,
                    "batch_size": self.cfg.batch_size,
                    "l2_regularization": self.cfg.l2_regularization,
                    "seed": self.cfg.seed,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, model_dir: Path, user_vocab: list[str], movie_vocab: list[str]) -> TwoTowerModel:
        payload = json.loads((model_dir / "model_config.json").read_text())
        model = cls(user_vocab, movie_vocab, ModelConfig(**payload))
        weights = np.load(model_dir / "weights.npz")
        model.user_embedding.embeddings.assign(weights["user_embeddings"])
        model.movie_embedding.embeddings.assign(weights["movie_embeddings"])
        return model


def build_vocabs(train_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Vocabulary จาก train เท่านั้น (entities ที่โผล่ครั้งแรกใน val/test = OOV)

    เรียงด้วย (length, lexicographic) — deterministic เสมอ และเทียบเท่า numeric
    order สำหรับ id ตัวเลขแบบ ml-100k โดยไม่ assume ว่า id ต้องเป็นตัวเลข
    """
    key = lambda s: (len(s), s)  # noqa: E731
    users = sorted(train_df["user_id"].unique(), key=key)
    movies = sorted(train_df["movie_id"].unique(), key=key)
    return users, movies
