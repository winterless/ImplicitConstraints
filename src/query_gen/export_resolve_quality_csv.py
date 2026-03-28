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
    "available functions",
    "available actions",
    "you have access to",
    "in the execute_ipython_cell",
    "act like a person",
)
TOOL_FIELD_NAMES = ("tools", "available_tools", "functions", "apis", "toolkit")
ACTION_FIELD_NAMES = ("action_space", "available_actions")
TOOL_HEADER_PATTERNS = (
    re.compile(r"available tools?", re.IGNORECASE),
    re.compile(r"available functions?", re.IGNORECASE),
    re.compile(r"the following functions are defined", re.IGNORECASE),
    re.compile(r"contains the following functions", re.IGNORECASE),
    re.compile(r"you have access to", re.IGNORECASE),
    re.compile(r"\btoolkit\b", re.IGNORECASE),
)
ACTION_HEADER_PATTERNS = (
    re.compile(r"available actions?:", re.IGNORECASE),
)
FUNCTION_SIGNATURE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()\n]{0,200})\)\s*->\s*[^:\n]+"
)
XML_FUNCTION_RE = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)>", re.IGNORECASE)
EXECUTE_TAG_RE = re.compile(r"<(execute_[A-Za-z_][A-Za-z0-9_]*)>", re.IGNORECASE)
API_CALL_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*[A-Z_][A-Za-z0-9_]*)\s*\(")
TRAJECTORY_ACTION_RE = re.compile(
    r"\b(click|type|select|hover|goto|scroll|stop|fill|press|drag)\b(?:\s*\[|\s*\()",
    re.IGNORECASE,
)
ACTION_NAME_RE = re.compile(r"\b([a-z][a-z0-9_]*)\b")
AVAILABLE_TOOLS_TAG_RE = re.compile(
    r"<available_tools>\s*(.*?)\s*</available_tools>",
    re.IGNORECASE | re.DOTALL,
)
TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
GENERIC_FUNCTION_NAMES = {
    "api",
    "args",
    "dict",
    "end",
    "example",
    "false",
    "float",
    "function",
    "int",
    "json",
    "list",
    "none",
    "null",
    "object",
    "optional",
    "required",
    "response",
    "return",
    "str",
    "string",
    "tool",
    "true",
    "value",
}


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


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


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
        or "<available_tools>" in normalized
        or "available tools" in normalized
        or "the following functions are defined" in normalized
        or "toolkit" in normalized
    )


def normalize_role(item: dict[str, Any], default_role: str = "") -> str:
    role = item.get("role") or item.get("from") or item.get("speaker") or default_role
    return str(role).strip().lower()


def normalize_tool_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.strip().strip(":,.;"))
    return normalized


def filter_candidate_names(names: Iterable[str]) -> list[str]:
    filtered: list[str] = []
    for name in names:
        normalized = normalize_tool_name(name)
        lowered = normalized.lower()
        if not normalized or lowered in GENERIC_FUNCTION_NAMES:
            continue
        if len(normalized) == 1:
            continue
        filtered.append(normalized)
    return unique_preserve_order(filtered)


def compact_evidence_text(text: str, limit: int = 2000) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def infer_execution_protocol_tools(text: str) -> list[str]:
    normalized = " ".join(text.lower().split())
    inferred: list[str] = []

    if "computer shell" in normalized or "bash code block" in normalized:
        inferred.append("execute_bash")
    if "python code block" in normalized or "python interpreter" in normalized:
        inferred.append("execute_python")
    if "sql" in normalized and ("database" in normalized or "query" in normalized):
        inferred.append("execute_sql")
    if "execute_ipython_cell" in normalized:
        inferred.append("execute_ipython_cell")

    return filter_candidate_names(inferred)


def build_available_tools_payload(
    kind: str,
    source: str,
    tool_names: Iterable[str] | None = None,
    raw_text: str | None = None,
    data: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "source": source,
    }
    names = filter_candidate_names(tool_names or [])
    if names:
        payload["tool_names"] = names
    if raw_text:
        payload["evidence"] = compact_evidence_text(raw_text)
    if data is not None:
        payload["data"] = data
    return payload


def extract_function_names_from_tool_specs(specs: Any) -> list[str]:
    names: list[str] = []

    if isinstance(specs, dict):
        function_obj = specs.get("function")
        if isinstance(function_obj, dict):
            function_name = stringify(function_obj.get("name")).strip()
            if function_name:
                names.append(function_name)
        for key in ("name", "tool_name"):
            value = stringify(specs.get(key)).strip()
            if value:
                names.append(value)
    elif isinstance(specs, list):
        for item in specs:
            names.extend(extract_function_names_from_tool_specs(item))

    return filter_candidate_names(names)


def extract_tagged_json_block(text: str, pattern: re.Pattern[str]) -> Any:
    if not text:
        return None
    match = pattern.search(text)
    if not match:
        return None
    raw_block = match.group(1).strip()
    if not raw_block:
        return None
    try:
        return json.loads(raw_block)
    except json.JSONDecodeError:
        return None


def extract_available_tools_from_tagged_text(
    text: str, source: str
) -> dict[str, Any] | None:
    specs = extract_tagged_json_block(text, AVAILABLE_TOOLS_TAG_RE)
    if not has_nonempty_value(specs):
        return None

    return build_available_tools_payload(
        kind="explicit_tool_definition",
        source=source,
        tool_names=extract_function_names_from_tool_specs(specs),
        raw_text=text,
        data=specs,
    )


def extract_tool_calls_from_tagged_text(text: str, source: str) -> dict[str, Any] | None:
    calls = extract_tagged_json_block(text, TOOL_CALL_TAG_RE)
    if not has_nonempty_value(calls):
        return None

    return build_available_tools_payload(
        kind="observed_tool_usage",
        source=source,
        tool_names=extract_function_names_from_tool_specs(calls),
        raw_text=text,
        data=calls,
    )


def extract_action_names_from_text(text: str) -> list[str]:
    if not text:
        return []

    action_names: list[str] = []
    for match in re.finditer(r"->\s*([A-Z_]{2,})\b", text):
        action_names.append(match.group(1).lower())
    for match in TRAJECTORY_ACTION_RE.finditer(text):
        action_names.append(match.group(1).lower())
    for match in re.finditer(r"\bAction:\s*([a-z][a-z0-9_ ]{1,80})", text, re.IGNORECASE):
        action_text = match.group(1).strip()
        first_token = action_text.split()[0]
        action_names.append(first_token.lower())
    return filter_candidate_names(action_names)


def extract_function_names_from_text(text: str) -> list[str]:
    if not text:
        return []

    names: list[str] = []
    names.extend(match.group(1) for match in XML_FUNCTION_RE.finditer(text))
    names.extend(match.group(1) for match in EXECUTE_TAG_RE.finditer(text))
    names.extend(match.group(1) for match in API_CALL_RE.finditer(text))
    names.extend(match.group(1) for match in FUNCTION_SIGNATURE_RE.finditer(text))
    return filter_candidate_names(names)


def extract_action_space_from_text(text: str, source: str) -> dict[str, Any] | None:
    if not text:
        return None
    if not any(pattern.search(text) for pattern in ACTION_HEADER_PATTERNS):
        return None

    match = re.search(
        r"available actions?:\s*(.{0,1500})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    segment = match.group(1) if match else text[:1500]
    segment = re.split(r"\b(?:assistant|observation|thought|action):", segment, maxsplit=1, flags=re.IGNORECASE)[0]

    candidates: list[str] = []
    for piece in re.split(r"[\n,;]+", segment):
        cleaned = piece.strip()
        if not cleaned:
            continue
        token_match = ACTION_NAME_RE.search(cleaned.lower())
        if token_match:
            candidates.append(token_match.group(1))

    return build_available_tools_payload(
        kind="action_space",
        source=source,
        tool_names=candidates or extract_action_names_from_text(segment),
        raw_text=segment,
    )


def extract_explicit_tool_definition_from_text(
    text: str, source: str
) -> dict[str, Any] | None:
    if not text:
        return None
    header_match = next((pattern.search(text) for pattern in TOOL_HEADER_PATTERNS if pattern.search(text)), None)
    if not header_match:
        return None

    segment = text[header_match.start() : header_match.start() + 2500]

    tool_names = extract_function_names_from_text(segment)
    if not tool_names:
        tool_names = infer_execution_protocol_tools(segment)

    return build_available_tools_payload(
        kind="explicit_tool_definition",
        source=source,
        tool_names=tool_names,
        raw_text=segment,
    )


def action_name_from_raw_value(value: Any) -> str | None:
    text = stringify(value).strip()
    if not text:
        return None

    match = re.search(r"->\s*([A-Z_]{2,})\b", text)
    if match:
        return match.group(1).lower()

    match = TRAJECTORY_ACTION_RE.search(text)
    if match:
        return match.group(1).lower()

    first_token = text.split()[0].strip("[]():,.;").lower()
    if re.fullmatch(r"[a-z][a-z0-9_]{1,40}", first_token):
        return first_token
    return None


def extract_observed_tool_usage_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    tool_names: list[str] = []

    content = json_maybe_load(record.get("content"))
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            class_name = str(item.get("class_", "")).strip().lower()
            function_name = stringify(item.get("function")).strip()
            if class_name == "api_action" and function_name:
                tool_names.append(function_name)
            if class_name == "code_action":
                language = stringify(item.get("language")).strip().lower()
                if language:
                    tool_names.append(f"execute_{language}")

    step_data = json_maybe_load(record.get("step_data"))
    if isinstance(step_data, dict):
        for key in ("action", "parsed_action"):
            name = action_name_from_raw_value(step_data.get(key))
            if name:
                tool_names.append(name)

    for key in ("action_reprs", "available_actions"):
        values = json_maybe_load(record.get(key))
        if isinstance(values, list):
            for item in values[:100]:
                name = action_name_from_raw_value(item)
                if name:
                    tool_names.append(name)

    actions = json_maybe_load(record.get("actions"))
    if isinstance(actions, list):
        for item in actions[:100]:
            if not isinstance(item, dict):
                name = action_name_from_raw_value(item)
                if name:
                    tool_names.append(name)
                continue
            for key in ("action", "action_type", "operation", "name"):
                name = action_name_from_raw_value(item.get(key))
                if name:
                    tool_names.append(name)

    tool_names = filter_candidate_names(tool_names)
    if not tool_names:
        return None

    return build_available_tools_payload(
        kind="observed_tool_usage",
        source="record_structure",
        tool_names=tool_names,
    )


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
    for key in TOOL_FIELD_NAMES:
        value = json_maybe_load(record.get(key))
        if has_nonempty_value(value):
            return build_available_tools_payload(
                kind="explicit_tool_definition",
                source=f"record.{key}",
                data=value,
            )

    for key in ACTION_FIELD_NAMES:
        value = json_maybe_load(record.get(key))
        if has_nonempty_value(value):
            return build_available_tools_payload(
                kind="action_space",
                source=f"record.{key}",
                data=value,
            )

    target_tools = json_maybe_load(record.get("target_tools"))
    if has_nonempty_value(target_tools):
        return build_available_tools_payload(
            kind="explicit_tool_definition",
            source="record.target_tools",
            data={"target_tools": target_tools},
        )

    resolved_turns = turns if turns is not None else extract_turns(record)
    for index, (role, text) in enumerate(resolved_turns):
        source = f"turns[{index}].{role or 'unknown'}"
        tagged_tools = extract_available_tools_from_tagged_text(text, source)
        if tagged_tools:
            return tagged_tools

        action_space = extract_action_space_from_text(text, source)
        if action_space:
            return action_space

        explicit_tools = extract_explicit_tool_definition_from_text(text, source)
        if explicit_tools:
            return explicit_tools

        tagged_calls = extract_tool_calls_from_tagged_text(text, source)
        if tagged_calls:
            return tagged_calls

        if role == "system":
            inferred_tools = infer_execution_protocol_tools(text)
            if inferred_tools:
                return build_available_tools_payload(
                    kind="explicit_tool_definition",
                    source=source,
                    tool_names=inferred_tools,
                    raw_text=text,
                )

    system_value = stringify(record.get("system")).strip()
    tagged_tools = extract_available_tools_from_tagged_text(
        system_value,
        "record.system",
    )
    if tagged_tools:
        return tagged_tools

    explicit_from_system = extract_explicit_tool_definition_from_text(
        system_value,
        "record.system",
    )
    if explicit_from_system:
        return explicit_from_system

    inferred_tools = infer_execution_protocol_tools(system_value)
    if inferred_tools:
        return build_available_tools_payload(
            kind="explicit_tool_definition",
            source="record.system",
            tool_names=inferred_tools,
            raw_text=system_value,
        )

    observed_from_record = extract_observed_tool_usage_from_record(record)
    if observed_from_record:
        return observed_from_record

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
        joined_text = "\n\n".join(item["content"] for item in tool_messages)
        explicit_tools = extract_explicit_tool_definition_from_text(
            joined_text,
            "turn_messages",
        )
        if explicit_tools:
            return explicit_tools
        action_space = extract_action_space_from_text(joined_text, "turn_messages")
        if action_space:
            return action_space

    text_value = stringify(record.get("text")).strip()
    tagged_tools = extract_available_tools_from_tagged_text(text_value, "record.text")
    if tagged_tools:
        return tagged_tools

    tagged_calls = extract_tool_calls_from_tagged_text(text_value, "record.text")
    if tagged_calls:
        return tagged_calls

    action_space = extract_action_space_from_text(text_value, "record.text")
    if action_space:
        return action_space

    explicit_tools = extract_explicit_tool_definition_from_text(
        text_value,
        "record.text",
    )
    if explicit_tools:
        return explicit_tools

    function_names = extract_function_names_from_text(text_value)
    action_names = extract_action_names_from_text(text_value)
    recovered_names = filter_candidate_names(function_names + action_names)
    if recovered_names:
        return build_available_tools_payload(
            kind="observed_tool_usage",
            source="record.text",
            tool_names=recovered_names,
            raw_text=text_value,
        )

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
