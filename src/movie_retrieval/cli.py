"""Command line interface.

Usage:
    movie-retrieval prepare                 # download + validate + temporal split
    movie-retrieval train                   # experiment matrix + select winner
    movie-retrieval evaluate [--sensitivity]  # final test eval + export serving artifacts
    movie-retrieval all [--sensitivity]     # ทุกขั้นตอนต่อเนื่อง
    movie-retrieval recommend --user-id 42 --k 10 [--include-seen]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import ExperimentConfig, Paths
from .pipeline import evaluate, prepare, recommend_cli, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="movie-retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="download dataset, validate contract, create temporal splits")
    sub.add_parser("train", help="train two-tower candidates and select by val recall@10")

    eval_parser = sub.add_parser("evaluate", help="final test evaluation + export artifacts")
    eval_parser.add_argument(
        "--sensitivity", action="store_true",
        help="also run rating>=4 positive-rule sensitivity (R3)",
    )

    all_parser = sub.add_parser("all", help="prepare + train + evaluate")
    all_parser.add_argument("--sensitivity", action="store_true")

    rec_parser = sub.add_parser("recommend", help="query serving index for one user")
    rec_parser.add_argument("--user-id", required=True)
    rec_parser.add_argument("--k", type=int, default=10)
    rec_parser.add_argument(
        "--include-seen", action="store_true",
        help="do not filter movies the user already rated",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    paths = Paths.default()
    cfg = ExperimentConfig()

    if args.command == "prepare":
        result = prepare(paths, cfg)
    elif args.command == "train":
        result = train(paths, cfg)
    elif args.command == "evaluate":
        result = evaluate(paths, cfg, with_sensitivity=args.sensitivity)
    elif args.command == "all":
        prepare(paths, cfg)
        train(paths, cfg)
        result = evaluate(paths, cfg, with_sensitivity=args.sensitivity)
    elif args.command == "recommend":
        result = recommend_cli(
            paths, user_id=args.user_id, k=args.k, exclude_seen=not args.include_seen
        )
    else:  # pragma: no cover — argparse บังคับ command อยู่แล้ว
        raise SystemExit(2)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
