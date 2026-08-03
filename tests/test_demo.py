"""Demo UI tests — สิ่งที่ผู้ใช้เห็นบนหน้าจอ ทดสอบได้โดยไม่ต้องติดตั้ง gradio

(`build_demo()` import gradio ข้างในตัวเอง จึงไม่ถูกแตะจากไฟล์นี้)
หน้าจอเป็น HTML ที่เราประกอบเอง — test ชุดนี้จึงคุมสองอย่างเป็นหลัก:
ค่าที่มาจากข้อมูลต้องถูก escape เสมอ และ input แปลก ๆ ต้องกลายเป็นข้อความ ไม่ใช่ exception
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from movie_retrieval.demo import (
    MIN_BAR_WIDTH,
    bar_widths,
    header_stats,
    history_chips,
    message_badge,
    recommend_for_ui,
    recommendation_rows,
    render_hero,
    render_results,
    status_badge,
)
from movie_retrieval.index import BruteForceIndex, RetrievalService

USER_VOCAB = ["u1", "u2"]
MOVIE_VOCAB = ["m1", "m2", "m3"]
USER_EMB = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # แถว 0 = OOV
MOVIE_EMB = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
POPULARITY = np.array([30.0, 20.0, 10.0], dtype=np.float32)
TITLES = {"m1": "Alpha", "m2": "Beta", "m3": "Gamma"}

XSS_TITLE = "<script>alert('x')</script>"


def make_service(titles: dict[str, str] | None = None) -> RetrievalService:
    index = BruteForceIndex(USER_VOCAB, USER_EMB, MOVIE_VOCAB, MOVIE_EMB, POPULARITY)
    return RetrievalService(
        index=index,
        seen={"u1": {"m3", "m2"}},
        titles=titles or TITLES,
        model_version="test-v1",
        index_version="catalog-test",
        catalogue_size=len(MOVIE_VOCAB),
    )


@pytest.fixture
def service() -> RetrievalService:
    return make_service()


class TestRows:
    def test_rows_are_ranked_from_one(self, service):
        rows = recommendation_rows(service.recommend("u2", k=3))
        assert [rank for rank, _, _ in rows] == [1, 2, 3]
        assert rows[0][1] == "Alpha"  # u2 ชอบ m1
        assert isinstance(rows[0][2], float)

    def test_empty_recommendations_render_as_no_rows(self):
        assert recommendation_rows({"recommendations": []}) == []


class TestBarWidths:
    def test_widths_span_from_floor_to_full(self):
        assert bar_widths([1.0, 2.0, 3.0]) == [MIN_BAR_WIDTH, (MIN_BAR_WIDTH + 100) / 2, 100.0]

    def test_identical_scores_do_not_divide_by_zero(self):
        assert bar_widths([2.0, 2.0]) == [100.0, 100.0]

    def test_negative_scores_stay_within_bounds(self):
        widths = bar_widths([-4.0, -1.0, 0.5])
        assert widths[0] == MIN_BAR_WIDTH
        assert all(MIN_BAR_WIDTH <= w <= 100 for w in widths)

    def test_no_scores(self):
        assert bar_widths([]) == []


class TestRenderResults:
    def test_every_row_shows_rank_title_and_score(self, service):
        html = render_results(recommendation_rows(service.recommend("u2", k=2)))
        assert html.count('class="tt-row"') == 2
        assert ">Alpha<" in html and "1</div>" in html

    def test_titles_are_escaped(self):
        html = render_results([(1, XSS_TITLE, 1.0)])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_bar_width_is_a_plain_number(self):
        html = render_results([(1, "Alpha", 1.0), (2, "Beta", 0.0)])
        assert 'style="width:100.0%"' in html
        assert f'style="width:{MIN_BAR_WIDTH}.0%"' in html

    def test_no_rows_renders_nothing(self):
        assert render_results([]) == ""


class TestBadges:
    def test_known_user_gets_the_ok_tone_and_versions(self, service):
        badge = status_badge(service.recommend("u2", k=1))
        assert "tt-status--ok" in badge
        assert "Two-tower retrieval" in badge
        assert "model test-v1 · index catalog-test" in badge

    def test_unknown_user_gets_the_warn_tone(self, service):
        badge = status_badge(service.recommend("stranger", k=1))
        assert "tt-status--warn" in badge
        assert "not in the training vocabulary" in badge

    def test_message_text_is_escaped(self):
        assert "&lt;b&gt;" in message_badge("<b>hi</b>")
        assert "<b>" not in message_badge("<b>hi</b>")


class TestHistoryChips:
    def test_titles_are_sorted_and_capped(self, service):
        assert history_chips(service, "u1").count('class="tt-chip"') == 2
        capped = history_chips(service, "u1", limit=1)
        assert "tt-chip--more" in capped and "+1 more" in capped

    def test_user_without_history_says_so(self, service):
        assert "No interaction history" in history_chips(service, "stranger")

    def test_titles_are_escaped(self):
        chips = history_chips(make_service({"m2": XSS_TITLE, "m3": "Gamma"}), "u1")
        assert "<script>" not in chips and "&lt;script&gt;" in chips


class TestHeaderStats:
    def test_stats_come_from_the_metrics_file(self, tmp_path):
        (tmp_path / "metrics.json").write_text(
            json.dumps(
                {
                    "two_tower": {"recall@50": 0.23117, "catalogue_coverage@10": 0.89279},
                    "serving": {"query_latency_ms_p95": 1.097},
                }
            )
        )
        assert header_stats(tmp_path) == [
            ("0.231", "Recall@50"),
            ("89.3%", "Coverage@10"),
            ("1.10 ms", "p95 latency"),
        ]

    @pytest.mark.parametrize("content", [None, "not json", '{"two_tower": {}}'])
    def test_unusable_metrics_hide_the_stats_instead_of_crashing(self, tmp_path, content):
        if content is not None:
            (tmp_path / "metrics.json").write_text(content)
        assert header_stats(tmp_path) == []

    def test_hero_escapes_stat_values(self):
        assert "&lt;i&gt;" in render_hero([("<i>", "label")])


class TestRecommendForUi:
    def test_happy_path_fills_all_three_panels(self, service):
        results, status, history = recommend_for_ui(service, "u1", k=1)
        assert results.count('class="tt-row"') == 1
        assert ">Alpha<" in results  # m2/m3 ถูกกรองเพราะ u1 เคยดูแล้ว
        assert "tt-status--ok" in status
        assert "Beta" in history

    def test_whitespace_is_trimmed(self, service):
        results, _, _ = recommend_for_ui(service, "  u2  ", k=2)
        assert results.count('class="tt-row"') == 2

    def test_blank_input_prompts_instead_of_erroring(self, service):
        results, status, history = recommend_for_ui(service, "   ", k=5)
        assert (results, history) == ("", "")
        assert "Enter a user id" in status and "tt-status--muted" in status

    @pytest.mark.parametrize("bad_user", ["u1; DROP TABLE", "x/../y", "a" * 65, "user id"])
    def test_rejected_input_becomes_a_message_not_an_exception(self, service, bad_user):
        results, status, history = recommend_for_ui(service, bad_user, k=5)
        assert (results, history) == ("", "")
        assert "user_id" in status and "tt-status--warn" in status

    def test_unknown_user_still_returns_candidates(self, service):
        results, status, history = recommend_for_ui(service, "stranger", k=2)
        assert results.index("Alpha") < results.index("Beta")  # เรียงตาม popularity
        assert "Popularity fallback" in status
        assert "No interaction history" in history
