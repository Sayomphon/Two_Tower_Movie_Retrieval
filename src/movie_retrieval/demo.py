"""Gradio demo — type a user id and see Top-K candidates from the real serving index

Why it exists: the README and the notebook can explain the *results*, but they don't let
a reader **try it**. This demo calls the same `RetrievalService` the CLI calls, on the same
artifacts — so input validation, seen filtering and cold-start fallback are all real, not mocked.

Gradio is an optional dependency (`pip install -e ".[demo]"`), so it is `import`ed inside
`build_demo()`/`main()` only — the HTML-assembling parts stay testable without a UI stack.

**The screen is HTML we write ourselves, which means nobody sanitizes it for us.** Every
value coming from data (movie titles, versions, error messages) must go through `escape()`,
and numbers must be formatted as numbers before being interpolated into a style/attribute.
"""

from __future__ import annotations

import json
from html import escape
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Paths
from .index import RetrievalService

if TYPE_CHECKING:  # pragma: no cover — used during type checking only
    import gradio as gr

# deliberately below the service's MAX_K: this is as much as one screen stays readable with
MAX_DEMO_K = 20
DEFAULT_K = 10
HISTORY_PREVIEW = 12

# minimum score-bar width (%) — the last rank must still read as a bar, not a vanishing line
MIN_BAR_WIDTH = 8

DEFAULT_USER_ID = "42"
EXAMPLE_USER_IDS = [DEFAULT_USER_ID, "1", "196", "brand-new-user"]

TAGLINE = (
    "Type a user id — the exported serving index returns the Top-K candidates it would hand "
    "to a ranking stage. Same <code>RetrievalService</code> the CLI calls, on the same "
    "artifacts: real input validation, real seen-item filtering, real cold-start fallback."
)

FOOTER = """Scores are **not** predicted ratings: for a known user they are dot-product
affinity between the user and movie embeddings; for an unknown user they are train
interaction counts from the popularity fallback.

Data: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) (GroupLens) —
research use only, downloaded at runtime, never redistributed ·
[source code](https://github.com/Sayomphon/Two_Tower_Movie_Retrieval)
"""

# every color references a Gradio CSS variable → light/dark switch themselves, no second set
CSS = """
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }

.tt-hero h1 {
  font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15;
  margin: 0 0 8px; color: var(--body-text-color);
}
.tt-hero p { margin: 0 0 16px; line-height: 1.6; color: var(--body-text-color-subdued); }
.tt-hero code {
  padding: 1px 5px; border-radius: 5px; font-size: 0.88em;
  background: var(--background-fill-secondary);
}
.tt-stats { display: flex; flex-wrap: wrap; gap: 8px; }
.tt-stat {
  flex: 1 1 120px; padding: 10px 14px; border-radius: 12px;
  border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary);
}
.tt-stat b {
  display: block; font-size: 1.15rem; font-weight: 650; line-height: 1.3;
  color: var(--body-text-color);
}
.tt-stat span {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--body-text-color-subdued);
}

.tt-list { display: flex; flex-direction: column; gap: 2px; }
.tt-row {
  display: grid; grid-template-columns: 30px 1fr auto; align-items: center; gap: 14px;
  padding: 9px 10px; border-radius: 10px; transition: background 120ms ease;
}
/* in the dark theme --background-fill-secondary is nearly the page background, so hover
   disappears — an overlay derived from the text color shows in both themes
   (the first line is the fallback for older browsers) */
.tt-row:hover {
  background: var(--background-fill-secondary);
  background: color-mix(in srgb, var(--body-text-color) 7%, transparent);
}
.tt-rank {
  font-variant-numeric: tabular-nums; font-size: 0.85rem; text-align: right;
  color: var(--body-text-color-subdued);
}
.tt-title { font-size: 0.97rem; line-height: 1.35; color: var(--body-text-color); }
.tt-track {
  height: 5px; margin-top: 6px; border-radius: 999px;
  background: var(--border-color-primary);
}
.tt-bar {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--primary-400), var(--primary-600));
}
.tt-score {
  font-variant-numeric: tabular-nums; font-size: 0.88rem; min-width: 58px; text-align: right;
  color: var(--body-text-color);
}

.tt-status {
  display: flex; gap: 10px; align-items: flex-start; margin-top: 10px;
  padding: 11px 14px; border-radius: 12px; line-height: 1.5; font-size: 0.9rem;
  border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary);
  color: var(--body-text-color);
}
.tt-dot { width: 8px; height: 8px; margin-top: 7px; border-radius: 999px; flex: none; }
.tt-status--ok .tt-dot { background: #16a34a; }
.tt-status--warn .tt-dot { background: #ea8c3a; }
.tt-status--muted .tt-dot { background: var(--body-text-color-subdued); }
.tt-meta {
  margin-top: 4px; font-size: 0.76rem; font-variant-numeric: tabular-nums;
  color: var(--body-text-color-subdued);
}

.tt-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.tt-chip {
  padding: 4px 10px; border-radius: 999px; font-size: 0.82rem;
  border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued);
}
.tt-chip--more { border-style: dashed; }
.tt-empty { color: var(--body-text-color-subdued); font-size: 0.9rem; margin: 0; }
"""


def header_stats(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Real numbers from `metrics.json` as chips on the hero — evidence, not decoration

    If the file is missing/unreadable (e.g. evaluate has never run), the hero simply gets
    no chips — better than letting the whole page break over an ornament.
    """
    try:
        metrics = json.loads((artifacts_dir / "metrics.json").read_text())
        two_tower, serving = metrics["two_tower"], metrics["serving"]
        return [
            (f"{two_tower['recall@50']:.3f}", "Recall@50"),
            (f"{two_tower['catalogue_coverage@10']:.1%}", "Coverage@10"),
            (f"{serving['query_latency_ms_p95']:.2f} ms", "p95 latency"),
        ]
    except (OSError, JSONDecodeError, KeyError, TypeError):
        return []


def render_hero(stats: list[tuple[str, str]]) -> str:
    """Top of the screen: title + one explanatory paragraph + the numbers that prove it's real"""
    chips = "".join(
        f'<div class="tt-stat"><b>{escape(value)}</b><span>{escape(label)}</span></div>'
        for value, label in stats
    )
    return (
        '<div class="tt-hero">'
        "<h1>Two-Tower Movie Retrieval</h1>"
        f"<p>{TAGLINE}</p>"
        f'<div class="tt-stats">{chips}</div>'
        "</div>"
    )


def recommendation_rows(response: dict) -> list[tuple[int, str, float]]:
    """response → result rows (rank, movie title, score)"""
    return [
        (rank, item["title"], item["score"])
        for rank, item in enumerate(response["recommendations"], start=1)
    ]


def bar_widths(scores: list[float]) -> list[float]:
    """Scores → bar widths (%), relative to each other within a single response

    Scores from different paths are not comparable (a dot product can be negative, while
    popularity is a count in the hundreds), so they are normalized by that set's min/max —
    a bar says "how far apart within this list", not an absolute value.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:  # all scores equal → fill every bar, better than dividing by zero
        return [100.0] * len(scores)
    return [round(MIN_BAR_WIDTH + (100 - MIN_BAR_WIDTH) * (s - lo) / (hi - lo), 1) for s in scores]


def render_results(rows: list[tuple[int, str, float]]) -> str:
    """Result rows → HTML (titles always escaped, bar widths are plain numbers)"""
    if not rows:
        return ""
    widths = bar_widths([score for _, _, score in rows])
    cards = [
        f'<div class="tt-row">'
        f'<div class="tt-rank">{rank}</div>'
        f"<div>"
        f'<div class="tt-title">{escape(title)}</div>'
        f'<div class="tt-track"><div class="tt-bar" style="width:{width:.1f}%"></div></div>'
        f"</div>"
        f'<div class="tt-score">{score:g}</div>'
        f"</div>"
        for (rank, title, score), width in zip(rows, widths, strict=True)
    ]
    return f'<div class="tt-list">{"".join(cards)}</div>'


def status_badge(response: dict) -> str:
    """Status bar under the results — which path produced them, and from which artifact version"""
    if response["fallback_used"]:
        tone, headline = "warn", "Popularity fallback"
        detail = (
            "this id is not in the training vocabulary, so the index returns the globally "
            "most-rated movies instead of a personalized list."
        )
    else:
        tone, headline = "ok", "Two-tower retrieval"
        detail = "personalized candidates; movies this user already rated are filtered out."

    meta = f"model {response['model_version']} · index {response['index_version']}"
    return (
        f'<div class="tt-status tt-status--{tone}"><span class="tt-dot"></span>'
        f"<div><b>{headline}</b> — {detail}"
        f'<div class="tt-meta">{escape(meta)}</div></div></div>'
    )


def message_badge(text: str, tone: str = "muted") -> str:
    """Message bar for states with no results (nothing entered / input failed validation)

    `tone` only ever comes from our own code, never from user input — `text` is always escaped
    """
    return (
        f'<div class="tt-status tt-status--{tone}"><span class="tt-dot"></span>'
        f"<div>{escape(text)}</div></div>"
    )


def history_chips(service: RetrievalService, user_id: str, limit: int = HISTORY_PREVIEW) -> str:
    """Movies this user already rated — context for judging whether the recs make sense"""
    titles = service.seen_titles(user_id)
    if not titles:
        return '<p class="tt-empty">No interaction history for this id — nothing was filtered.</p>'

    chips = "".join(f'<span class="tt-chip">{escape(title)}</span>' for title in titles[:limit])
    if len(titles) > limit:
        chips += f'<span class="tt-chip tt-chip--more">+{len(titles) - limit} more</span>'
    return f'<div class="tt-chips">{chips}</div>'


def recommend_for_ui(service: RetrievalService, user_id: str, k: int) -> tuple[str, str, str]:
    """Bridge between the UI and the service — returns what the three panels on screen need

    Input that fails the service's validation is turned into a message for the user
    (rather than letting the exception bubble up for Gradio to show as a raw error)
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return "", message_badge("Enter a user id to see candidates."), ""

    try:
        response = service.recommend(user_id, k=int(k))
    except ValueError as exc:
        return "", message_badge(str(exc), tone="warn"), ""

    return (
        render_results(recommendation_rows(response)),
        status_badge(response),
        history_chips(service, user_id),
    )


def build_demo(service: RetrievalService, stats: list[tuple[str, str]] | None = None) -> gr.Blocks:
    """Assemble the screen: hero → input → results → status → the filtered-out history"""
    import gradio as gr

    def on_submit(raw_user_id: str, raw_k: float) -> tuple[str, str, str]:
        return recommend_for_ui(service, raw_user_id, int(raw_k))

    # analytics_enabled=False: no telemetry leaves the machine
    with gr.Blocks(title="Two-Tower Movie Retrieval", analytics_enabled=False) as demo:
        # create the output slots first so the Try button can reference them, then .render()
        # them in on-screen order
        results = gr.HTML(render=False)
        status = gr.HTML(render=False)
        history = gr.HTML(render=False)
        outputs = [results, status, history]

        gr.HTML(render_hero(stats or []))

        with gr.Row():
            user_id = gr.Textbox(
                label="User ID",
                value=DEFAULT_USER_ID,
                placeholder="1–943 for a trained user, or anything else to see the fallback",
                max_lines=1,
                scale=3,
            )
            k = gr.Slider(1, MAX_DEMO_K, value=DEFAULT_K, step=1, label="Candidates", scale=2)
            submit = gr.Button("Recommend", variant="primary", scale=1)

        gr.Examples(
            examples=[[example_id, DEFAULT_K] for example_id in EXAMPLE_USER_IDS],
            example_labels=EXAMPLE_USER_IDS,
            inputs=[user_id, k],
            outputs=outputs,
            fn=on_submit,
            run_on_click=True,  # one click shows results — not just filling the fields
            cache_examples=False,
            label="Try",
        )

        results.render()
        status.render()

        with gr.Accordion("Already rated by this user (excluded above)", open=False):
            history.render()

        gr.Markdown(FOOTER)

        submit.click(on_submit, inputs=[user_id, k], outputs=outputs)
        user_id.submit(on_submit, inputs=[user_id, k], outputs=outputs)
        # real results on page load, without having to guess what to type
        demo.load(on_submit, inputs=[user_id, k], outputs=outputs)

    return demo


def main() -> None:
    """Run the demo on the exported artifacts (`movie-retrieval all` must have run first)"""
    import gradio as gr

    paths = Paths.default()
    if not paths.index_dir.exists():
        raise SystemExit(
            f"serving artifacts not found under {paths.artifacts_dir} — "
            "run `movie-retrieval all` first"
        )

    service = RetrievalService.from_artifacts(paths.artifacts_dir)
    # read metrics once at startup — the hero should not touch disk on every request
    stats = header_stats(paths.artifacts_dir)
    stats.append((f"{service.catalogue_size:,}", "Movies indexed"))

    # max_size caps the queue against load spikes · api_open=False closes the API endpoint
    # the demo does not use
    build_demo(service, stats).queue(max_size=32, api_open=False).launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
