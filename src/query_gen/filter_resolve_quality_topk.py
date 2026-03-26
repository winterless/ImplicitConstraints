#!/usr/bin/env python3
"""
Filter and rank rows from toucan_resolve_quality.csv.

Rules:
- Keep rows whose content length is fewer than 150 characters.
- Keep only rows whose content matches at least one regex from the pattern file.
- Rank rows by making question_quality_avg high and response_quality_avg low:
  rank_score = q_weight * question_quality_avg - r_weight * response_quality_avg

Example:
    python src/query_gen/filter_resolve_quality_topk.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

DEFAULT_INPUT = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/toucan_resolve_quality.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/toucan_resolve_quality_top500.csv"
)
DEFAULT_PATTERN_FILE = Path(__file__).with_name("filter_resolve_quality_patterns.txt")


def passes_length_filter(text: str, max_chars: int) -> tuple[bool, int]:
    char_count = len(text)
    return char_count < max_chars, char_count


def load_patterns(pattern_file: Path) -> list[str]:
    if not pattern_file.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_file}")

    patterns: list[str] = []
    with pattern_file.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or line in {'"', "“", "”"}:
                continue
            patterns.append(line)
    return patterns


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def matches_keyword_patterns(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Toucan quality CSV and export top-K ranked rows."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Number of top rows to keep after ranking (default: 500).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=150,
        help="Keep rows with content length below this limit (default: 150).",
    )
    parser.add_argument(
        "--q-weight",
        type=float,
        default=1.0,
        help="Weight for question_quality_avg in ranking (default: 1.0).",
    )
    parser.add_argument(
        "--r-weight",
        type=float,
        default=1.0,
        help="Weight for response_quality_avg in ranking (default: 1.0).",
    )
    parser.add_argument(
        "--pattern-file",
        type=Path,
        default=DEFAULT_PATTERN_FILE,
        help="Text file with one regex pattern per line.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    compiled_patterns = compile_patterns(load_patterns(args.pattern_file))
    ranked_rows: list[dict[str, str | float]] = []
    with args.input.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            content = row.get("content") or row.get("first_user_content", "") or ""
            if not matches_keyword_patterns(content, compiled_patterns):
                continue
            keep, char_count = passes_length_filter(content, max_chars=args.max_chars)
            if not keep:
                continue

            question_quality = parse_float(row.get("question_quality_avg", ""))
            response_quality = parse_float(row.get("response_quality_avg", ""))
            if question_quality == 0.0 or response_quality == 0.0:
                continue

            rank_score = args.q_weight * question_quality - args.r_weight * response_quality
            enriched: dict[str, str | float] = dict(row)
            enriched["content"] = content
            enriched["char_count"] = str(char_count)
            enriched["rank_score"] = rank_score
            enriched["_question_quality_value"] = question_quality
            enriched["_response_quality_value"] = response_quality
            ranked_rows.append(enriched)

    ranked_rows.sort(
        key=lambda row: (
            float(row["rank_score"]),
            float(row["_question_quality_value"]),
            -float(row["_response_quality_value"]),
            str(row["first_user_content"]),
        ),
        reverse=True,
    )
    top_rows = ranked_rows[: args.top_k]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "source_file",
            "line_number",
            "uuid",
            "content",
            "first_user_content",
            "question_quality_avg",
            "response_quality_avg",
            "char_count",
            "rank_score",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in top_rows:
            output_row = dict(row)
            output_row["rank_score"] = f"{float(output_row['rank_score']):.6f}"
            output_row.pop("_question_quality_value", None)
            output_row.pop("_response_quality_value", None)
            writer.writerow(output_row)

    print(
        f"Kept {len(ranked_rows)} rows after regex and length filters, "
        f"wrote top {len(top_rows)} rows to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
