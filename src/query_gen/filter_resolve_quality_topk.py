#!/usr/bin/env python3
"""
Select top-K short-query rows from seed_dataset_query_stats.csv and export
their available tools plus full context as JSONL.

Example:
    python src/query_gen/filter_resolve_quality_topk.py
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from export_resolve_quality_csv import (
    extract_available_tools,
    has_nonempty_value,
    iter_records,
    json_maybe_load,
)

DEFAULT_INPUT = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/seed_dataset_query_stats.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/seed_dataset_query_top1000.jsonl"
)
DEFAULT_SEED_ROOT = Path("/home/unlimitediw/workspace/ImplicitConstraints/seed_datasets")
DEFAULT_TURN_COUNT_INPUT = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/seed_dataset_query_stats_turn_count.csv"
)
DEFAULT_PATTERN_FILE = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/filter_resolve_quality_patterns.txt"
)


def maximize_csv_field_limit() -> None:
    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_int(value: str | None) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter seed dataset query stats and export top-K full records."
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
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--seed-root",
        type=Path,
        default=DEFAULT_SEED_ROOT,
        help=f"Root directory used by source_pointer paths (default: {DEFAULT_SEED_ROOT}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of top rows to keep after sorting (default: 1000).",
    )
    parser.add_argument(
        "--max-query-chars",
        type=int,
        default=60,
        help="Keep rows whose query length is below this threshold (default: 60).",
    )
    parser.add_argument(
        "--max-user-turns",
        type=int,
        default=1,
        help="Keep rows with at most this many user turns (default: 1).",
    )
    parser.add_argument(
        "--sort-field",
        choices=("conversation_length", "turn_count", "selected_metric_value"),
        default="conversation_length",
        help="Numeric field used for top-k ranking (default: conversation_length).",
    )
    parser.add_argument(
        "--pattern-file",
        type=Path,
        default=DEFAULT_PATTERN_FILE,
        help=(
            "Whitelist regex file. Only queries matching at least one pattern are "
            f"eligible for ranking (default: {DEFAULT_PATTERN_FILE})."
        ),
    )
    return parser.parse_args()


def parse_source_pointer(pointer: str, seed_root: Path) -> tuple[Path, int]:
    file_part, _, remainder = pointer.partition("#")
    if not file_part or not remainder:
        raise ValueError(f"Invalid source pointer: {pointer}")

    location, _, _record_id = remainder.partition(":")
    if location.startswith("L"):
        record_number = int(location[1:])
    else:
        record_number = int(location)

    path = Path(file_part)
    if not path.is_absolute():
        path = seed_root / path
    return path, record_number


def parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def read_csv_fieldnames(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader.fieldnames or [])


def resolve_input_csv(input_path: Path, sort_field: str) -> Path:
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    fieldnames = set(read_csv_fieldnames(input_path))
    required_fields = {"has_available_tools", "turn_count", "user_turn_count", sort_field}
    if required_fields.issubset(fieldnames):
        return input_path

    if input_path == DEFAULT_INPUT and sort_field == "turn_count":
        fallback_path = DEFAULT_TURN_COUNT_INPUT
        if fallback_path.exists():
            fallback_fields = set(read_csv_fieldnames(fallback_path))
            if required_fields.issubset(fallback_fields):
                print(
                    "Default input is missing 'turn_count'; "
                    f"using fallback CSV: {fallback_path}"
                )
                return fallback_path

    missing_fields = sorted(required_fields - fieldnames)
    raise SystemExit(
        f"Input CSV is missing required fields {missing_fields}: {input_path}. "
        "Please regenerate it with export_resolve_quality_csv.py first."
    )


def load_whitelist_patterns(pattern_file: Path) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    with pattern_file.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, 1):
            pattern = raw_line.strip()
            if not pattern or pattern.startswith("#"):
                continue
            try:
                patterns.append(re.compile(boundary_safe_pattern(pattern), re.IGNORECASE))
            except re.error as exc:
                raise SystemExit(
                    f"Invalid regex in {pattern_file} line {line_number}: {pattern!r} ({exc})"
                ) from exc

    if not patterns:
        raise SystemExit(f"No usable whitelist patterns found in: {pattern_file}")
    return patterns


def matches_whitelist(query: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(query) for pattern in patterns)


def boundary_safe_pattern(pattern: str) -> str:
    # English patterns should not match inside larger English words, e.g.
    # "on time" should not match "executiON TIMEd out".
    if not re.search(r"[A-Za-z]", pattern):
        return pattern
    return rf"(?<![A-Za-z0-9])(?:{pattern})(?![A-Za-z0-9])"


def dedupe_query_key(query: str) -> str:
    return " ".join(query.split()).casefold()


def row_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(row["sort_value"]),
        int(row["turn_count"]),
        int(row["conversation_length"]),
        int(row["query_length"]),
        str(row["source_pointer"]),
    )


def select_top_rows(
    csv_path: Path,
    max_query_chars: int,
    max_user_turns: int,
    top_k: int,
    sort_field: str,
    whitelist_patterns: list[re.Pattern[str]],
) -> list[dict[str, Any]]:
    kept_rows = 0
    excluded_invalid_rows = 0
    excluded_user_turn_rows = 0
    excluded_nonmatching_rows = 0
    excluded_duplicate_queries = 0
    best_row_by_query: dict[str, dict[str, Any]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "has_available_tools" not in reader.fieldnames:
            raise ValueError(
                "Input CSV is missing 'has_available_tools'. "
                "Please regenerate it with export_resolve_quality_csv.py first."
            )
        if sort_field not in reader.fieldnames:
            raise ValueError(
                f"Input CSV is missing '{sort_field}'. "
                "Please regenerate it with export_resolve_quality_csv.py first."
            )
        for row in reader:
            original_query = row.get("original_query", "")
            query_length = parse_int(row.get("query_length"))
            conversation_length = parse_int(row.get("conversation_length"))
            turn_count = parse_int(row.get("turn_count"))
            user_turn_count = parse_int(row.get("user_turn_count"))
            sort_value = parse_int(row.get(sort_field))
            source_pointer = row.get("source_pointer", "")
            has_available_tools = parse_bool(row.get("has_available_tools"))
            if (
                not original_query
                or query_length is None
                or conversation_length is None
                or turn_count is None
                or user_turn_count is None
                or sort_value is None
                or not source_pointer
                or not has_available_tools
                or query_length <= 0
                or conversation_length <= 0
                or turn_count <= 0
                or user_turn_count <= 0
                or sort_value <= 0
                or query_length >= max_query_chars
            ):
                excluded_invalid_rows += 1
                continue

            if user_turn_count > max_user_turns:
                excluded_user_turn_rows += 1
                continue

            if not matches_whitelist(original_query, whitelist_patterns):
                excluded_nonmatching_rows += 1
                continue

            kept_rows += 1
            enriched = {
                "original_query": original_query,
                "query_length": query_length,
                "conversation_length": conversation_length,
                "turn_count": turn_count,
                "user_turn_count": user_turn_count,
                "source_pointer": source_pointer,
                "sort_field": sort_field,
                "sort_value": sort_value,
            }
            query_key = dedupe_query_key(original_query)
            previous = best_row_by_query.get(query_key)
            if previous is None or row_sort_key(enriched) > row_sort_key(previous):
                if previous is not None:
                    excluded_duplicate_queries += 1
                best_row_by_query[query_key] = enriched
            else:
                excluded_duplicate_queries += 1

    top_rows = sorted(best_row_by_query.values(), key=row_sort_key, reverse=True)[:top_k]
    print(
        f"Kept {kept_rows} rows with query_length < {max_query_chars} "
        f"and user_turn_count <= {max_user_turns}, "
        f"excluded {excluded_invalid_rows} invalid/no-tools rows, "
        f"excluded {excluded_user_turn_rows} user-turn-limit rows, "
        f"excluded {excluded_nonmatching_rows} whitelist-miss rows, "
        f"excluded {excluded_duplicate_queries} duplicate-query rows, "
        f"selected top {len(top_rows)} by {sort_field}"
    )
    return top_rows


def extract_full_context(record: dict[str, Any]) -> Any:
    for key in ("messages", "conversations", "content"):
        value = json_maybe_load(record.get(key))
        if value is not None:
            return value
    if "text" in record:
        return record.get("text")
    return record


def index_rows_by_source(
    rows: list[dict[str, Any]], seed_root: Path
) -> dict[Path, dict[int, dict[str, Any]]]:
    grouped: dict[Path, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        path, record_number = parse_source_pointer(str(row["source_pointer"]), seed_root)
        grouped[path][record_number] = row
    return grouped


def enrich_rows(
    selected_rows: list[dict[str, Any]], seed_root: Path
) -> list[dict[str, Any]]:
    grouped_rows = index_rows_by_source(selected_rows, seed_root)
    enriched_by_pointer: dict[str, dict[str, Any]] = {}

    for path, record_map in grouped_rows.items():
        target_numbers = set(record_map)
        found_count = 0
        for record_number, record in iter_records(path):
            if record_number not in target_numbers:
                continue
            base_row = dict(record_map[record_number])
            base_row["available_tools"] = extract_available_tools(record)
            base_row["full_context"] = extract_full_context(record)
            enriched_by_pointer[str(base_row["source_pointer"])] = base_row
            found_count += 1
            if found_count == len(target_numbers):
                break

        missing = sorted(target_numbers - {parse_source_pointer(pointer, seed_root)[1] for pointer in enriched_by_pointer if parse_source_pointer(pointer, seed_root)[0] == path})
        if missing:
            raise ValueError(f"Missing records in {path}: {missing[:10]}")

    return [
        enriched_by_pointer[str(row["source_pointer"])]
        for row in selected_rows
        if str(row["source_pointer"]) in enriched_by_pointer
    ]


def filter_rows_with_available_tools(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if has_nonempty_value(row.get("available_tools"))
        and has_nonempty_value(row.get("full_context"))
        and has_nonempty_value(row.get("original_query"))
        and has_nonempty_value(row.get("source_pointer"))
    ]
    excluded = len(rows) - len(filtered)
    print(f"Excluded {excluded} rows with empty available_tools/full_context")
    return filtered


def sanitize_string(text: str) -> str:
    # Replace invalid surrogate code points before UTF-8 JSONL serialization.
    return text.encode("utf-8", errors="replace").decode("utf-8").replace("\x00", "")


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            sanitize_string(str(key)): sanitize_json_value(item)
            for key, item in value.items()
        }
    return value


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rank, row in enumerate(rows, 1):
            payload = {
                "rank": rank,
                "original_query": row["original_query"],
                "query_length": row["query_length"],
                "conversation_length": row["conversation_length"],
                "turn_count": row["turn_count"],
                "user_turn_count": row["user_turn_count"],
                "sort_field": row["sort_field"],
                "sort_value": row["sort_value"],
                "source_pointer": row["source_pointer"],
                "available_tools": row.get("available_tools"),
                "full_context": row.get("full_context"),
            }
            fh.write(
                json.dumps(
                    sanitize_json_value(payload),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            fh.write("\n")


def main() -> int:
    args = parse_args()
    maximize_csv_field_limit()
    if not args.pattern_file.exists():
        raise SystemExit(f"Pattern file not found: {args.pattern_file}")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.max_query_chars <= 0:
        raise SystemExit("--max-query-chars must be positive")
    if args.max_user_turns <= 0:
        raise SystemExit("--max-user-turns must be positive")
    input_path = resolve_input_csv(args.input, args.sort_field)
    whitelist_patterns = load_whitelist_patterns(args.pattern_file)

    top_rows = select_top_rows(
        input_path,
        args.max_query_chars,
        args.max_user_turns,
        args.top_k,
        args.sort_field,
        whitelist_patterns,
    )
    enriched_rows = enrich_rows(top_rows, args.seed_root)
    final_rows = filter_rows_with_available_tools(enriched_rows)
    write_jsonl(final_rows, args.output)

    print(f"Wrote {len(final_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
