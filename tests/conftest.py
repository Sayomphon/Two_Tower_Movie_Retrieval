"""Shared fixtures — synthetic data ที่ควบคุมได้ 100% สำหรับ unit tests

Unit tests ไม่แตะ network/dataset จริง (integration test แยกไฟล์และ skip
อัตโนมัติถ้ายังไม่มีไฟล์ dataset ในเครื่อง)
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_ratings() -> pd.DataFrame:
    """6 users: u1-u5 มี 6 interactions (พอสำหรับ min_history=5), u6 มีแค่ 2 (ต้องถูก drop)

    timestamp เรียงเพิ่มขึ้นต่อ user อย่างชัดเจน — interaction ล่าสุดคือ m6
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
