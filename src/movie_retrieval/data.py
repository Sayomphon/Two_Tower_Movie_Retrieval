"""Dataset ingestion + data contract.

Security/compliance design:
- download over HTTPS from GroupLens only, and verify SHA-256 before extracting
  (guards against a tampered/corrupted archive)
- zip-slip safe extraction: check that every member path stays inside the target dir
- never commit the raw dataset (MovieLens research-use terms forbid redistribution)
- the data contract checks schema/range/duplicates before any training
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
# SHA-256 of ml-100k.zip (the dataset has been frozen since 1998 — safe to pin)
ML100K_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"

# Known dataset statistics from the GroupLens README — used as the data contract
EXPECTED_N_RATINGS = 100_000
EXPECTED_N_USERS = 943
EXPECTED_N_MOVIES = 1_682

RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


class DataContractError(ValueError):
    """Raised when data fails the contract — training must not continue"""


@dataclass(frozen=True)
class DataCard:
    """Dataset lineage metadata, recorded into artifacts"""

    source_url: str
    sha256: str
    n_ratings: int
    n_users: int
    n_movies: int


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_ml100k(raw_dir: Path, url: str = ML100K_URL, timeout: int = 60) -> Path:
    """Download ml-100k.zip if missing, and always verify its SHA-256"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-100k.zip"

    if not zip_path.exists():
        logger.info("Downloading MovieLens 100K from %s", url)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    actual = sha256_of(zip_path)
    if actual != ML100K_SHA256:
        # delete the mismatched file immediately — it must not be used
        zip_path.unlink(missing_ok=True)
        raise DataContractError(
            f"SHA-256 mismatch for {zip_path.name}: expected {ML100K_SHA256}, got {actual}. "
            "File deleted; re-run to download again."
        )
    return zip_path


def _safe_extract(zip_path: Path, dest: Path, members: list[str]) -> None:
    """Extract only the files we need, guarding against zip-slip path traversal"""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in members:
            target = (dest / member).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise DataContractError(f"Unsafe path in archive: {member}")
            zf.extract(member, dest)


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    """Load u.data → DataFrame[user_id, movie_id, rating, timestamp]

    user_id/movie_id are always cast to string (categorical identity, not numbers)
    """
    zip_path = download_ml100k(raw_dir)
    data_file = raw_dir / "ml-100k" / "u.data"
    if not data_file.exists():
        _safe_extract(zip_path, raw_dir, ["ml-100k/u.data", "ml-100k/u.item", "ml-100k/README"])

    df = pd.read_csv(
        data_file,
        sep="\t",
        names=RATINGS_COLUMNS,
        dtype={"user_id": str, "movie_id": str, "rating": float, "timestamp": int},
    )
    return df


def load_movie_titles(raw_dir: Path) -> dict[str, str]:
    """Load u.item → mapping movie_id -> title (used for interpretability/demo)"""
    item_file = raw_dir / "ml-100k" / "u.item"
    if not item_file.exists():
        _safe_extract(download_ml100k(raw_dir), raw_dir, ["ml-100k/u.item"])
    titles: dict[str, str] = {}
    # u.item is latin-1 encoded, pipe-separated: id|title|release_date|...
    with item_file.open(encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 2:
                titles[parts[0]] = parts[1]
    return titles


def validate_ratings(df: pd.DataFrame, strict_ml100k: bool = True) -> DataCard:
    """Data contract — raise DataContractError if any check fails"""
    problems: list[str] = []

    missing_cols = set(RATINGS_COLUMNS) - set(df.columns)
    if missing_cols:
        problems.append(f"missing columns: {sorted(missing_cols)}")
    else:
        if df[RATINGS_COLUMNS].isna().any().any():
            problems.append("null values found")
        if not df["rating"].between(1, 5).all():
            problems.append("rating out of range [1, 5]")
        if not (df["timestamp"] > 0).all():
            problems.append("non-positive timestamps")
        if df.duplicated(subset=["user_id", "movie_id"]).any():
            problems.append("duplicate (user_id, movie_id) interactions")
        if not (df["user_id"].map(type).eq(str).all() and df["movie_id"].map(type).eq(str).all()):
            problems.append("user_id/movie_id must be str")

    if strict_ml100k and not problems:
        if len(df) != EXPECTED_N_RATINGS:
            problems.append(f"expected {EXPECTED_N_RATINGS} ratings, got {len(df)}")
        if df["user_id"].nunique() != EXPECTED_N_USERS:
            problems.append(f"expected {EXPECTED_N_USERS} users, got {df['user_id'].nunique()}")
        if df["movie_id"].nunique() > EXPECTED_N_MOVIES:
            problems.append(f"more than {EXPECTED_N_MOVIES} movies: {df['movie_id'].nunique()}")

    if problems:
        raise DataContractError("; ".join(problems))

    return DataCard(
        source_url=ML100K_URL,
        sha256=ML100K_SHA256,
        n_ratings=len(df),
        n_users=df["user_id"].nunique(),
        n_movies=df["movie_id"].nunique(),
    )


def apply_positive_rule(df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    """Turn explicit ratings into implicit positives per the declared rule

    threshold=None  → every rating counts as a positive interaction
    threshold=4.0   → only rating >= 4 is positive (sensitivity analysis R3)
    """
    if threshold is None:
        return df.copy()
    return df[df["rating"] >= threshold].copy()
