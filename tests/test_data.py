"""Data contract + security tests (no network access)"""

from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from movie_retrieval.data import (
    DataContractError,
    _safe_extract,
    apply_positive_rule,
    validate_ratings,
)


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["1", "1", "2"],
            "movie_id": ["10", "20", "10"],
            "rating": [3.0, 5.0, 1.0],
            "timestamp": [100, 200, 150],
        }
    )


class TestValidateRatings:
    def test_valid_frame_passes(self):
        card = validate_ratings(_valid_df(), strict_ml100k=False)
        assert card.n_ratings == 3
        assert card.n_users == 2

    def test_rating_out_of_range_fails(self):
        df = _valid_df()
        df.loc[0, "rating"] = 6.0
        with pytest.raises(DataContractError, match="rating out of range"):
            validate_ratings(df, strict_ml100k=False)

    def test_duplicate_interaction_fails(self):
        df = pd.concat([_valid_df(), _valid_df().iloc[[0]]], ignore_index=True)
        with pytest.raises(DataContractError, match="duplicate"):
            validate_ratings(df, strict_ml100k=False)

    def test_null_fails(self):
        df = _valid_df()
        df.loc[0, "timestamp"] = None
        with pytest.raises(DataContractError):
            validate_ratings(df, strict_ml100k=False)

    def test_strict_mode_rejects_wrong_row_count(self):
        with pytest.raises(DataContractError, match="expected 100000 ratings"):
            validate_ratings(_valid_df(), strict_ml100k=True)


class TestPositiveRule:
    def test_none_keeps_all(self):
        assert len(apply_positive_rule(_valid_df(), None)) == 3

    def test_threshold_filters(self):
        filtered = apply_positive_rule(_valid_df(), 4.0)
        assert len(filtered) == 1
        assert filtered["rating"].min() >= 4.0


class TestSafeExtract:
    def test_zip_slip_is_blocked(self, tmp_path):
        """An archive containing path traversal (../) must be rejected — security guard"""
        malicious = tmp_path / "evil.zip"
        with zipfile.ZipFile(malicious, "w") as zf:
            zf.writestr("../evil.txt", "pwned")
        with pytest.raises(DataContractError, match="Unsafe path"):
            _safe_extract(malicious, tmp_path / "out", ["../evil.txt"])
        assert not (tmp_path / "evil.txt").exists()
