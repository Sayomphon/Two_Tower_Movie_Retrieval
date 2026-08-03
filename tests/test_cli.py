"""CLI contract tests — argv goes in, and we check what the pipeline gets called with

No network/dataset/artifacts involved: every pipeline function is monkeypatched. What is
tested is the contract the user sees (subcommand, flag, default) — the layer module-level
tests cannot see, especially `--include-seen`, which is inverted into `exclude_seen`
before being passed on.
"""

from __future__ import annotations

import json

import pytest

from movie_retrieval import cli


@pytest.fixture
def calls(monkeypatch) -> list[tuple[str, dict]]:
    """Record which pipeline stage the CLI calls, with which arguments, in order"""
    recorded: list[tuple[str, dict]] = []

    def spy(name: str):
        def fake(paths, *positional, **kwargs):
            recorded.append((name, kwargs))
            return {"stage": name, **kwargs}

        return fake

    for stage in ("prepare", "train", "evaluate", "recommend_cli"):
        monkeypatch.setattr(cli, stage, spy(stage))
    return recorded


class TestParser:
    def test_command_is_required(self):
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args([])
        assert exc.value.code == 2

    def test_unknown_command_is_rejected(self):
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args(["deploy"])
        assert exc.value.code == 2

    def test_recommend_requires_user_id(self):
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args(["recommend"])
        assert exc.value.code == 2

    def test_recommend_defaults(self):
        args = cli.build_parser().parse_args(["recommend", "--user-id", "42"])
        assert (args.user_id, args.k, args.include_seen) == ("42", 10, False)

    def test_sensitivity_is_opt_in(self):
        parser = cli.build_parser()
        assert parser.parse_args(["evaluate"]).sensitivity is False
        assert parser.parse_args(["evaluate", "--sensitivity"]).sensitivity is True


class TestMain:
    def test_prepare_prints_json_and_exits_zero(self, calls, capsys):
        assert cli.main(["prepare"]) == 0
        assert [name for name, _ in calls] == ["prepare"]
        assert json.loads(capsys.readouterr().out)["stage"] == "prepare"

    def test_all_runs_three_stages_in_order(self, calls, capsys):
        assert cli.main(["all", "--sensitivity"]) == 0
        assert [name for name, _ in calls] == ["prepare", "train", "evaluate"]
        assert calls[-1][1] == {"with_sensitivity": True}

    def test_evaluate_forwards_sensitivity_flag(self, calls, capsys):
        cli.main(["evaluate"])
        assert calls[0][1] == {"with_sensitivity": False}

    def test_recommend_inverts_include_seen(self, calls, capsys):
        cli.main(["recommend", "--user-id", "42", "--k", "5"])
        assert calls[0][1] == {"user_id": "42", "k": 5, "exclude_seen": True}

        calls.clear()
        cli.main(["recommend", "--user-id", "42", "--include-seen"])
        assert calls[0][1] == {"user_id": "42", "k": 10, "exclude_seen": False}
