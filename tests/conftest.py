"""Shared fixtures — fully controlled synthetic data for unit tests

Unit tests never touch the network or the real dataset (the integration test lives in its
own file and skips automatically when the dataset is not on the machine yet)
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_ratings() -> pd.DataFrame:
    """6 users: u1-u5 have 6 interactions (enough for min_history=5), u6 has only 2 (gets dropped)

    Timestamps increase strictly per user — the most recent interaction is m6
    """
    rows = []
    for u in range(1, 6):
        for t in range(6):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "movie_id": f"m{t + 1}",
                    "rating": float((t % 5) + 1),
                    "timestamp": 1000 * u + t,
                }
            )
    rows.append({"user_id": "u6", "movie_id": "m1", "rating": 5.0, "timestamp": 100})
    rows.append({"user_id": "u6", "movie_id": "m2", "rating": 4.0, "timestamp": 101})
    return pd.DataFrame(rows)
