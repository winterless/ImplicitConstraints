#!/usr/bin/env python3
"""
Export per-record quality stats from Toucan resolved jsonl files into a CSV.

Example:
    python src/query_gen/export_resolve_quality_csv.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/seed_datasets/Toucan-1.5M-resolve"
)
DEFAULT_OUTPUT = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/toucan_resolve_quality.csv"
)


def iter_jsonl_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.jsonl"))


def json_maybe_load(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def extract_first_user_content(messages_raw: Any) -> str:
    messages = json_maybe_load(messages_raw)
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if content is None:
            return ""
        return str(content)
    return ""


def extract_question_content(record: dict[str, Any]) -> str:
    question = record.get("question")
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        return json.dumps(question, ensure_ascii=False)
    if question is not None:
        return str(question)
    return extract_first_user_content(record.get("messages"))


def _extract_nested_score(assessment_raw: Any, key: str) -> float | None:
    assessment = json_maybe_load(assessment_raw)
    if not isinstance(assessment, dict):
        return None
    value = assessment.get(key)
    if not isinstance(value, dict):
        return None
    score = value.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def extract_question_quality_score(assessment_raw: Any) -> float | None:
    scenario_realism = _extract_nested_score(assessment_raw, "scenario_realism")
    question_quality = _extract_nested_score(assessment_raw, "question_quality")
    if scenario_realism is None or question_quality is None:
        return None
    return (scenario_realism + question_quality) / 2.0


def extract_response_quality_score(assessment_raw: Any) -> float | None:
    return _extract_nested_score(assessment_raw, "completeness")


def format_score(score: float | None) -> str:
    if score is None:
        return ""
    return f"{score:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Toucan quality scores from resolved jsonl files."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input jsonl file or directory (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input path not found: {args.input}")

    jsonl_files = iter_jsonl_files(args.input)
    if not jsonl_files:
        raise SystemExit(f"No jsonl files found under: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with args.output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source_file",
                "line_number",
                "uuid",
                "content",
                "first_user_content",
                "question_quality_avg",
                "response_quality_avg",
            ],
        )
        writer.writeheader()

        for jsonl_file in jsonl_files:
            with jsonl_file.open("r", encoding="utf-8") as fh:
                for line_number, line in enumerate(fh, 1):
                    text = line.strip()
                    if not text:
                        continue
                    record = json.loads(text)
                    question_quality = extract_question_quality_score(
                        record.get("question_quality_assessment")
                    )
                    response_quality = extract_response_quality_score(
                        record.get("response_quality_assessment")
                    )
                    if (
                        question_quality is None
                        or response_quality is None
                        or question_quality == 0
                        or response_quality == 0
                    ):
                        continue
                    writer.writerow(
                        {
                            "source_file": str(jsonl_file),
                            "line_number": line_number,
                            "uuid": record.get("uuid", ""),
                            "content": extract_question_content(record),
                            "first_user_content": extract_first_user_content(record.get("messages")),
                            "question_quality_avg": format_score(question_quality),
                            "response_quality_avg": format_score(response_quality),
                        }
                    )
                    row_count += 1

    print(f"Wrote {row_count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
