#!/usr/bin/env python3
"""
Export lightweight query/conversation stats for all datasets under seed_datasets.

Collected fields:
1. original_query
2. query_length
3. conversation_length
4. turn_count
5. user_turn_count
6. source_pointer

Example:
    python src/query_gen/export_resolve_quality_csv.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INPUT = Path("/home/unlimitediw/workspace/ImplicitConstraints/seed_datasets")
DEFAULT_OUTPUT = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/seed_dataset_query_stats.csv"
)
FIELDNAMES = [
    "original_query",
    "query_length",
    "conversation_length",
    "turn_count",
    "user_turn_count",
    "source_pointer",
    "has_available_tools",
    "selected_metric_name",
    "selected_metric_value",
]
SKIP_DIR_NAMES = {".cache", ".git", "__pycache__"}
USER_ROLES = {"user", "human"}
SPEAKER_PREFIX_RE = re.compile(r"^(User|Human|Assistant|System|GPT|Tool):\s*(.*)$")
SETUP_HINTS = (
    "i will ask you",
    "for each of your turn",
    "your operation should be",
    "your goal is",
    "you should help me",
    "you can execute",
    "the following functions are defined",
    "the toolkit",
    "contains the following functions",
    "available actions",
    "you have access to",
    "in the execute_ipython_cell",
    "act like a person",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export query and conversation length stats from seed datasets."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input dataset file or directory (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of files to process in parallel.",
    )
    parser.add_argument(
        "--metric",
        choices=("conversation_length", "turn_count"),
        default="conversation_length",
        help="Primary metric to expose in selected_metric_value (default: conversation_length).",
    )
    return parser.parse_args()


def iter_dataset_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in SKIP_DIR_NAMES and not name.startswith(".")
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in {".jsonl", ".json"}:
                files.append(path)
    return sorted(files)


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


def has_nonempty_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [stringify(item).strip() for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                nested = stringify(value[key]).strip()
                if nested:
                    return nested
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def looks_like_setup(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    if any(hint in normalized for hint in SETUP_HINTS):
        return True
    if normalized.startswith("you are an assistant"):
        return True
    if len(text) > 1200 and (
        "example:" in normalized
        or "args:" in normalized
        or "function" in normalized
        or "tool" in normalized
    ):
        return True
    return False


def is_available_tools_message(role: str, text: str) -> bool:
    if role != "system":
        return False
    normalized = text.lower()
    return (
        "<|im_system|>tool_declare" in normalized
        or "available tools" in normalized
        or "the following functions are defined" in normalized
        or "toolkit" in normalized
    )


def normalize_role(item: dict[str, Any], default_role: str = "") -> str:
    role = item.get("role") or item.get("from") or item.get("speaker") or default_role
    return str(role).strip().lower()


def extract_turns_from_messages(messages_raw: Any) -> list[tuple[str, str]]:
    messages = json_maybe_load(messages_raw)
    if not isinstance(messages, list):
        return []

    turns: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message)
        text = stringify(message.get("content")).strip()
        if text:
            turns.append((role, text))
    return turns


def extract_turns_from_conversations(conversations_raw: Any) -> list[tuple[str, str]]:
    conversations = json_maybe_load(conversations_raw)
    if not isinstance(conversations, list):
        return []

    turns: list[tuple[str, str]] = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message)
        text = stringify(message.get("content") or message.get("value")).strip()
        if text:
            turns.append((role, text))
    return turns


def extract_turns_from_std_content(content_raw: Any) -> list[tuple[str, str]]:
    content = json_maybe_load(content_raw)
    if not isinstance(content, list):
        return []

    turns: list[tuple[str, str]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        class_name = str(item.get("class_", "")).strip().lower()
        role = "user" if class_name == "text_observation" else "assistant"
        parts = [stringify(item.get("description")).strip(), stringify(item.get("content")).strip()]
        text = "\n".join(part for part in parts if part).strip()
        if text:
            turns.append((role, text))
    return turns


def parse_transcript_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_role = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        match = SPEAKER_PREFIX_RE.match(line)
        if match:
            if current_role and current_lines:
                blocks.append((current_role, "\n".join(current_lines).strip()))
            current_role = match.group(1).lower()
            current_lines = [match.group(2)]
        elif current_role:
            current_lines.append(line)

    if current_role and current_lines:
        blocks.append((current_role, "\n".join(current_lines).strip()))
    return blocks


def extract_turns(record: dict[str, Any]) -> list[tuple[str, str]]:
    if "messages" in record:
        turns = extract_turns_from_messages(record.get("messages"))
        if turns:
            return turns
    if "conversations" in record:
        turns = extract_turns_from_conversations(record.get("conversations"))
        if turns:
            return turns
    if "content" in record:
        turns = extract_turns_from_std_content(record.get("content"))
        if turns:
            return turns
    if isinstance(record.get("text"), str):
        return parse_transcript_blocks(record["text"])
    return []


def extract_original_query(record: dict[str, Any], turns: list[tuple[str, str]]) -> str:
    for field_name in ("question", "query", "prompt", "instruction", "task"):
        value = stringify(record.get(field_name)).strip()
        if value:
            return value

    user_messages = [text.strip() for role, text in turns if role in USER_ROLES and text.strip()]
    if user_messages:
        for candidate in user_messages:
            if not looks_like_setup(candidate):
                return candidate
        return user_messages[0]

    text = stringify(record.get("text")).strip()
    if not text:
        return ""

    if turns:
        for role, candidate in turns:
            if role in USER_ROLES and candidate and not looks_like_setup(candidate):
                return candidate.strip()

    first_block = text.split("\n\n", 1)[0].strip()
    if first_block.lower().startswith(("thought:", "action:", "observation:")):
        return ""
    return re.sub(r"^(User|Human):\s*", "", first_block).strip()


def extract_conversation_length(record: dict[str, Any], turns: list[tuple[str, str]]) -> int:
    if turns:
        included_parts = [
            text
            for role, text in turns
            if text and not is_available_tools_message(role, text)
        ]
        return sum(len(part) for part in included_parts)

    text = stringify(record.get("text")).strip()
    return len(text)


def extract_turn_count(record: dict[str, Any], turns: list[tuple[str, str]]) -> int:
    if turns:
        return sum(
            1
            for role, text in turns
            if text and not is_available_tools_message(role, text)
        )

    text = stringify(record.get("text")).strip()
    return 1 if text else 0


def extract_user_turn_count(record: dict[str, Any], turns: list[tuple[str, str]]) -> int:
    if turns:
        return sum(
            1
            for role, text in turns
            if role in USER_ROLES
            and text
            and not is_available_tools_message(role, text)
            and not looks_like_setup(text)
        )

    text = stringify(record.get("text")).strip()
    return 1 if text else 0


def extract_available_tools(record: dict[str, Any], turns: list[tuple[str, str]] | None = None) -> Any:
    for key in ("tools", "available_tools"):
        value = json_maybe_load(record.get(key))
        if has_nonempty_value(value):
            return value

    target_tools = json_maybe_load(record.get("target_tools"))
    if has_nonempty_value(target_tools):
        return {"target_tools": target_tools}

    system_value = stringify(record.get("system")).strip()
    if system_value and looks_like_setup(system_value):
        return system_value

    resolved_turns = turns if turns is not None else extract_turns(record)
    tool_messages: list[dict[str, str]] = []
    for role, text in resolved_turns:
        if is_available_tools_message(role, text):
            tool_messages.append({"role": role, "content": text})
            continue
        if role in USER_ROLES and looks_like_setup(text) and (
            "function" in text.lower()
            or "toolkit" in text.lower()
            or "execute_ipython_cell" in text.lower()
        ):
            tool_messages.append({"role": role, "content": text})

    if tool_messages:
        return tool_messages
    return None


def make_source_pointer(
    path: Path, input_root: Path, record_number: int, record: dict[str, Any]
) -> str:
    relative_base = input_root.parent if input_root.is_file() else input_root
    try:
        relative_path = path.relative_to(relative_base)
    except ValueError:
        relative_path = path

    pointer = (
        f"{relative_path}#L{record_number}"
        if path.suffix.lower() == ".jsonl"
        else f"{relative_path}#{record_number}"
    )

    record_id = record.get("uuid", record.get("id", ""))
    if record_id not in ("", None):
        pointer = f"{pointer}:{record_id}"
    return str(pointer)


def iter_jsonl_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if isinstance(record, dict):
                yield line_number, record


def iter_json_array_records(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[tuple[int, dict[str, Any]]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
        buffer = ""
        index = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, eof
            chunk = fh.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def ensure_data() -> None:
            while index >= len(buffer) and not eof:
                refill()

        def skip_whitespace() -> None:
            nonlocal index
            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index < len(buffer) or eof:
                    return
                refill()

        ensure_data()
        skip_whitespace()
        if index >= len(buffer) or buffer[index] != "[":
            raise ValueError(f"Expected top-level JSON array in {path}")
        index += 1

        record_number = 0
        while True:
            skip_whitespace()
            if index >= len(buffer):
                if eof:
                    break
                refill()
                continue
            if buffer[index] == "]":
                break

            while True:
                try:
                    record, next_index = decoder.raw_decode(buffer, index)
                    index = next_index
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()

            if isinstance(record, dict):
                record_number += 1
                yield record_number, record

            skip_whitespace()
            if index < len(buffer) and buffer[index] == ",":
                index += 1
            elif index < len(buffer) and buffer[index] == "]":
                break
            elif eof:
                break

            if index > chunk_size * 4:
                buffer = buffer[index:]
                index = 0


def iter_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl_records(path)
        return
    if suffix == ".json":
        yield from iter_json_array_records(path)
        return
    raise ValueError(f"Unsupported file suffix: {path}")


def build_row(
    path: Path,
    input_root: Path,
    record_number: int,
    record: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    turns = extract_turns(record)
    query = extract_original_query(record, turns)
    conversation_length = extract_conversation_length(record, turns)
    turn_count = extract_turn_count(record, turns)
    user_turn_count = extract_user_turn_count(record, turns)
    available_tools = extract_available_tools(record, turns)
    metric_value = (
        conversation_length if metric == "conversation_length" else turn_count
    )
    return {
        "original_query": query,
        "query_length": len(query),
        "conversation_length": conversation_length,
        "turn_count": turn_count,
        "user_turn_count": user_turn_count,
        "source_pointer": make_source_pointer(path, input_root, record_number, record),
        "has_available_tools": int(has_nonempty_value(available_tools)),
        "selected_metric_name": metric,
        "selected_metric_value": metric_value,
    }


def process_file_to_temp_csv(
    path: Path, input_root: Path, temp_dir: Path, metric: str
) -> tuple[Path, int]:
    temp_fd, temp_name = tempfile.mkstemp(
        prefix="query_stats_", suffix=".csv", dir=temp_dir
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)

    row_count = 0
    with temp_path.open("w", encoding="utf-8", newline="") as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=FIELDNAMES)
        for record_number, record in iter_records(path):
            writer.writerow(build_row(path, input_root, record_number, record, metric))
            row_count += 1

    return temp_path, row_count


def append_temp_csv(temp_path: Path, output_handle: Any) -> None:
    with temp_path.open("r", encoding="utf-8", newline="") as temp_file:
        shutil.copyfileobj(temp_file, output_handle)


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input path not found: {args.input}")

    dataset_files = iter_dataset_files(args.input)
    if not dataset_files:
        raise SystemExit(f"No .jsonl/.json files found under: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        row_count = 0
        with tempfile.TemporaryDirectory(prefix="query_stats_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(
                        process_file_to_temp_csv, path, args.input, temp_dir, args.metric
                    ): path
                    for path in dataset_files
                }
                for future in as_completed(futures):
                    path = futures[future]
                    temp_path, file_rows = future.result()
                    append_temp_csv(temp_path, csv_file)
                    safe_unlink(temp_path)
                    row_count += file_rows
                    print(f"Processed {path} ({file_rows} rows)")

    print(
        f"Wrote {row_count} rows from {len(dataset_files)} files to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
