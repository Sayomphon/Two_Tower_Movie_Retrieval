"""Serving index tests — reload consistency, fallback, input validation"""

from __future__ import annotations

import numpy as np
import pytest

from movie_retrieval.index import BruteForceIndex, RetrievalService, load_index, save_index

USER_VOCAB = ["u1", "u2"]
MOVIE_VOCAB = ["m1", "m2", "m3"]
# u1 ชอบ m3, u2 ชอบ m1 (ออกแบบ embedding ให้ dot product ชัดเจน)
USER_EMB = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # แถว 0 = OOV
MOVIE_EMB = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
POPULARITY = np.array([30.0, 20.0, 10.0], dtype=np.float32)  # m1 popular สุด


@pytest.fixture
def index() -> BruteForceIndex:
    return BruteForceIndex(USER_VOCAB, USER_EMB, MOVIE_VOCAB, MOVIE_EMB, POPULARITY)


class TestBruteForceIndex:
    def test_known_user_personalized(self, index):
        result = index.recommend(np.array(["u1", "u2"], dtype=object), 1)
        movie_ids = result["movie_ids"].numpy()
        assert movie_ids[0][0].decode() == "m3"  # u1 → m3
        assert movie_ids[1][0].decode() == "m1"  # u2 → m1
        assert not result["fallback_used"].numpy().any()

    def test_unknown_user_falls_back_to_popularity(self, index):
        result = index.recommend(np.array(["stranger"], dtype=object), 3)
        assert result["fallback_used"].numpy()[0]
        top = [m.decode() for m in result["movie_ids"].numpy()[0]]
        assert top == ["m1", "m2", "m3"]  # เรียงตาม popularity

    def test_k_is_capped_at_catalogue_size(self, index):
        result = index.recommend(np.array(["u1"], dtype=object), 999)
        assert result["movie_ids"].shape[1] == len(MOVIE_VOCAB)

    def test_save_load_consistency(self, index, tmp_path):
        users = np.array(["u1", "u2", "stranger"], dtype=object)
        before = index.recommend(users, 3)["movie_ids"].numpy()
        save_index(index, tmp_path / "idx")
        reloaded = load_index(tmp_path / "idx")
        after = reloaded.recommend(users, np.int32(3))["movie_ids"].numpy()
        assert (before == after).all()

    def test_misaligned_embeddings_rejected(self):
        with pytest.raises(ValueError, match="OOV row"):
            BruteForceIndex(USER_VOCAB, USER_EMB[:2], MOVIE_VOCAB, MOVIE_EMB, POPULARITY)


class TestRetrievalService:
    @pytest.fixture
    def service(self, index) -> RetrievalService:
        return RetrievalService(
            index=index,
            seen={"u1": {"m3"}},
            titles={"m1": "Movie One", "m2": "Movie Two", "m3": "Movie Three"},
            model_version="test-v1",
            index_version="catalog-test",
            catalogue_size=len(MOVIE_VOCAB),
        )

    def test_response_contract(self, service):
        response = service.recommend("u2", k=2)
        assert response["model_version"] == "test-v1"
        assert response["index_version"] == "catalog-test"
        assert response["fallback_used"] is False
        assert len(response["recommendations"]) == 2
        first = response["recommendations"][0]
        assert set(first) == {"movie_id", "title", "score"}
        assert first["movie_id"] == "m1"
        assert first["title"] == "Movie One"

    def test_seen_filtering(self, service):
        response = service.recommend("u1", k=3)
        rec_ids = [r["movie_id"] for r in response["recommendations"]]
        assert "m3" not in rec_ids  # u1 เคยดู m3 แล้ว

    def test_response_has_no_duplicate_movies(self, service):
        # u1 เห็น m3 แล้ว → fetch_k = k + 1 แล้วค่อย filter ทีหลัง
        # ถ้าตรรกะ over-fetch เพี้ยน หนังซ้ำจะโผล่ที่ชั้น serving นี้ก่อนที่อื่น
        response = service.recommend("u1", k=2)
        movie_ids = [r["movie_id"] for r in response["recommendations"]]
        assert len(movie_ids) == len(set(movie_ids))

    def test_include_seen(self, service):
        response = service.recommend("u1", k=3, exclude_seen=False)
        rec_ids = [r["movie_id"] for r in response["recommendations"]]
        assert rec_ids[0] == "m3"

    def test_unknown_user_fallback_strategy(self, service):
        response = service.recommend("newcomer", k=2)
        assert response["fallback_used"] is True
        assert response["strategy"] == "popularity_fallback"
        assert response["recommendations"][0]["movie_id"] == "m1"

    @pytest.mark.parametrize("bad_user", ["", "a" * 65, "user id", "u1; DROP TABLE", "x/../y"])
    def test_invalid_user_id_rejected(self, service, bad_user):
        with pytest.raises(ValueError, match="user_id"):
            service.recommend(bad_user, k=5)

    @pytest.mark.parametrize("bad_k", [0, -1, 101, "10"])
    def test_invalid_k_rejected(self, service, bad_k):
        with pytest.raises(ValueError, match="k must be"):
            service.recommend("u1", k=bad_k)
