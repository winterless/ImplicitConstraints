#!/usr/bin/env python3
"""
Batch-filter implicit queries with Qwen and export a compressed top-K set.

This script:
1. Reads query candidates from JSONL files under `data_prepare/`
2. Calls a Qwen-compatible chat completion API with `filter_prompt.txt`
3. Stores per-row structured judgments incrementally for resume
4. Selects a final compressed query for each source row
5. Deduplicates and exports the top-K implicit-eval queries

The API settings intentionally follow the same DashScope-compatible conventions
documented in `/home/unlimitediw/workspace/DataBot/docs/qwen_api_call_guide.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_INPUT_DIR = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/data_prepare"
)
DEFAULT_PROMPT_PATH = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/filter_prompt.txt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/out/implicit_query_filter"
)
DEFAULT_CONFIG_PATH = Path(
    "/home/unlimitediw/workspace/DataBot/configs/datasearcher_api.json"
)
DEFAULT_API_KEY_FILE = Path("/home/unlimitediw/workspace/DataBot/.secrets/alicloud_api_key.txt")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_PROVIDER = "aliyun"
DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 2
DEFAULT_TOP_K = 200

QUERY_FIELDS = ("query", "original_query", "content", "user_prompt")
VALID_RULE_FAMILIES = {
    "deadline_buffer",
    "conflict_and_exclusivity",
    "safety_and_irreversibility",
    "latent_goal_vs_literal_request",
    "multi_hop_evidence",
    "side_effect_optimization",
    "explanation_obligation",
    "none",
}


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [ImplicitQueryFilter] {msg}", file=sys.stderr, flush=True)


def _api_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}{path}" if base.endswith("/v1") else f"{base}/v1{path}"


def _post_json(
    url: str,
    payload: Dict[str, Any],
    timeout_s: int,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_completion(
    base_url: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    timeout_s: int,
    retries: int,
    headers: Optional[Dict[str, str]] = None,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    last_err = None
    for attempt in range(1, retries + 2):
        try:
            resp = _post_json(
                _api_url(base_url, "/chat/completions"),
                payload,
                timeout_s=timeout_s,
                headers=headers,
            )
            choice = (resp.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content", "")
            if isinstance(content, list):
                parts: List[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    else:
                        parts.append(str(block))
                content_text = "\n".join(parts).strip()
            else:
                content_text = str(content).strip()
            usage = resp.get("usage") or {}
            return {
                "content": content_text,
                "usage": {
                    "prompt_tokens": parse_int(usage.get("prompt_tokens"), 0),
                    "completion_tokens": parse_int(usage.get("completion_tokens"), 0),
                    "total_tokens": parse_int(usage.get("total_tokens"), 0),
                },
                "raw_response": resp,
            }
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            last_err = f"HTTP {e.code}: {body or str(e)}"
            if e.code in (429, 500, 502, 503, 504) and attempt <= retries:
                time.sleep(attempt)
                continue
            raise RuntimeError(last_err) from e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = str(e)
            if attempt <= retries:
                time.sleep(attempt)
                continue
            raise RuntimeError(f"Chat API failed: {last_err}") from e
    raise RuntimeError(last_err or "Chat API failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and compress implicit queries with Qwen."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Small delay between successful requests.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse existing scored output.",
    )
    parser.add_argument(
        "--keep-non-candidate",
        action="store_true",
        help="Allow non-candidate rows into final top-k if scores are high enough.",
    )
    parser.add_argument(
        "--allow-non-daily",
        action="store_true",
        help="Allow non-daily rows into final top-k.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_api_settings(
    config_path: Path,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[str, str, Dict[str, str]]:
    cfg = load_config(config_path)
    providers_cfg = cfg.get("providers") or {}
    provider_cfg = providers_cfg.get(provider) or {}
    resolved_base_url = (
        base_url
        or os.getenv("BASE_URL")
        or os.getenv("ALIYUN_BASE_URL")
        or str(provider_cfg.get("base_url", DEFAULT_BASE_URL))
    )
    resolved_model = (
        model
        or os.getenv("MODEL")
        or os.getenv("ALIYUN_MODEL")
        or str(provider_cfg.get("model", DEFAULT_MODEL))
    )

    if provider != "aliyun":
        return resolved_base_url, resolved_model, {}

    api_key_env = str(provider_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    resolved_key = (
        api_key
        or os.getenv("API_KEY")
        or os.getenv(api_key_env)
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("ALIYUN_API_KEY")
        or (
            DEFAULT_API_KEY_FILE.read_text(encoding="utf-8").strip()
            if DEFAULT_API_KEY_FILE.exists()
            else ""
        )
    )
    if not resolved_key:
        raise ValueError(
            "Aliyun provider requires API key via --api-key or DASHSCOPE_API_KEY/ALIYUN_API_KEY."
        )
    return resolved_base_url, resolved_model, {"Authorization": f"Bearer {resolved_key}"}


def iter_query_rows(input_dir: Path) -> Iterable[Dict[str, Any]]:
    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line_number, raw_line in enumerate(fh, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    _log(f"Skip invalid JSONL row: {jsonl_path.name}#{line_number}: {exc}")
                    continue
                query_value = ""
                query_field = ""
                for field in QUERY_FIELDS:
                    value = obj.get(field)
                    if isinstance(value, str) and value.strip():
                        query_value = value.strip()
                        query_field = field
                        break
                if not query_value:
                    continue
                yield {
                    "source_id": f"{jsonl_path.name}#{line_number}",
                    "source_file": str(jsonl_path),
                    "source_filename": jsonl_path.name,
                    "line_number": line_number,
                    "query_field": query_field,
                    "original_query": query_value,
                    "source_record": obj,
                }


def render_prompt(prompt_template: str, query: str) -> str:
    if "{{query}}" in prompt_template:
        return prompt_template.replace("{{query}}", query)
    return f"{prompt_template.rstrip()}\n\n{query}"


def extract_first_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("No JSON object found in model response")


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def normalize_result(
    raw_obj: Dict[str, Any],
    original_query: str,
    parse_error: Optional[str] = None,
) -> Dict[str, Any]:
    compressed_versions = clean_string_list(raw_obj.get("compressed_versions"))
    compressed_version = str(raw_obj.get("compressed_version", "")).strip()
    if compressed_version:
        compressed_versions = [compressed_version]
    implicit_points = clean_string_list(raw_obj.get("implicit_points"))

    base_implicit_count = max(0, parse_int(raw_obj.get("base_implicit_count"), 0))
    compressible_count = max(0, parse_int(raw_obj.get("compressible_count"), 0))
    final_implicit_count = parse_int(
        raw_obj.get("final_implicit_count"),
        base_implicit_count + compressible_count,
    )
    if final_implicit_count < base_implicit_count + compressible_count:
        final_implicit_count = base_implicit_count + compressible_count

    primary_rule_family = str(raw_obj.get("primary_rule_family", "none")).strip() or "none"
    if primary_rule_family not in VALID_RULE_FAMILIES:
        primary_rule_family = "none"

    secondary_rule_families = [
        item for item in clean_string_list(raw_obj.get("secondary_rule_families"))
        if item in VALID_RULE_FAMILIES and item != primary_rule_family
    ][:2]

    confidence = parse_float(raw_obj.get("confidence"), 0.0)
    confidence = max(0.0, min(1.0, confidence))

    normalized = {
        "is_candidate": parse_bool(raw_obj.get("is_candidate")),
        "is_daily": parse_bool(raw_obj.get("is_daily")),
        "primary_rule_family": primary_rule_family,
        "secondary_rule_families": secondary_rule_families,
        "base_implicit_count": base_implicit_count,
        "compressible_count": compressible_count,
        "final_implicit_count": max(0, final_implicit_count),
        "compressed_version": compressed_versions[0] if compressed_versions else "",
        "implicit_points": implicit_points[:8],
        "reason": str(raw_obj.get("reason", "")).strip(),
        "daily_reason": str(raw_obj.get("daily_reason", "")).strip(),
        "compression_reason": str(raw_obj.get("compression_reason", "")).strip(),
        "confidence": confidence,
        "parse_error": parse_error,
        "raw_result": raw_obj,
    }

    # If the prompt is the older version without explicit final count fields,
    # keep behavior deterministic.
    if not normalized["reason"] and parse_error:
        normalized["reason"] = parse_error

    if not normalized["compressed_version"]:
        normalized["compressible_count"] = 0
        normalized["final_implicit_count"] = normalized["base_implicit_count"]

    # Drop "compressed" variants that are identical to the original query.
    original_key = " ".join(original_query.split()).casefold()
    if normalized["compressed_version"]:
        if " ".join(normalized["compressed_version"].split()).casefold() == original_key:
            normalized["compressed_version"] = ""
    normalized["compressible_count"] = min(
        normalized["compressible_count"],
        1 if normalized["compressed_version"] else 0,
    )
    normalized["final_implicit_count"] = max(
        normalized["final_implicit_count"],
        normalized["base_implicit_count"] + normalized["compressible_count"],
    )
    return normalized


def choose_selected_query(original_query: str, compressed_version: str) -> str:
    if not compressed_version:
        return original_query
    return compressed_version


def select_query_for_export(row: Dict[str, Any]) -> Dict[str, Any]:
    compressed_version = str(row.get("compressed_version", "")).strip()
    if not compressed_version:
        compressed_versions = clean_string_list(row.get("compressed_versions"))
        compressed_version = compressed_versions[0] if compressed_versions else ""
    final_version = choose_selected_query(row["original_query"], compressed_version)
    out = dict(row)
    out["compressed_version"] = compressed_version
    out["final_version"] = final_version
    out["final_version_length"] = len(final_version)
    out["used_compressed_version"] = final_version != row["original_query"]
    # Keep backward-compatible aliases for existing downstream files.
    out["selected_query"] = final_version
    out["selected_query_length"] = len(final_version)
    out["used_compressed_query"] = final_version != row["original_query"]
    return out


def normalize_query_key(query: str) -> str:
    return " ".join(query.split()).casefold()


def ranking_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    rank = parse_int(row.get("rank"), 10**9)
    return (
        1 if parse_bool(row.get("is_candidate")) else 0,
        1 if parse_bool(row.get("is_daily")) else 0,
        parse_int(row.get("final_implicit_count"), 0),
        parse_int(row.get("compressible_count"), 0),
        parse_float(row.get("confidence"), 0.0),
        len(clean_string_list(row.get("implicit_points"))),
        -parse_int(row.get("final_version_length"), len(str(row.get("final_version", "")))),
        -rank,
    )


def load_existing_scored(scored_path: Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not scored_path.exists():
        return existing
    with scored_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_id = str(obj.get("source_id", "")).strip()
            if source_id:
                existing[source_id] = obj
    return existing


def export_top_queries(
    scored_rows: Iterable[Dict[str, Any]],
    output_dir: Path,
    top_k: int,
    keep_non_candidate: bool,
    allow_non_daily: bool,
) -> Dict[str, Any]:
    scored_rows_list = list(scored_rows)
    prepared: List[Dict[str, Any]] = []
    excluded_non_candidate = 0
    excluded_non_daily = 0

    for raw_row in scored_rows_list:
        row = select_query_for_export(raw_row)
        if not keep_non_candidate and not parse_bool(row.get("is_candidate")):
            excluded_non_candidate += 1
            continue
        if not allow_non_daily and not parse_bool(row.get("is_daily")):
            excluded_non_daily += 1
            continue
        prepared.append(row)

    best_by_query: Dict[str, Dict[str, Any]] = {}
    duplicate_count = 0
    for row in prepared:
        key = normalize_query_key(row["final_version"])
        previous = best_by_query.get(key)
        if previous is None or ranking_key(row) > ranking_key(previous):
            if previous is not None:
                duplicate_count += 1
            best_by_query[key] = row
        else:
            duplicate_count += 1

    top_rows = sorted(best_by_query.values(), key=ranking_key, reverse=True)[:top_k]

    top_jsonl_path = output_dir / f"implicit_query_top{top_k}.jsonl"
    top_txt_path = output_dir / f"implicit_query_top{top_k}.txt"
    summary_path = output_dir / f"implicit_query_top{top_k}_summary.json"

    with top_jsonl_path.open("w", encoding="utf-8") as fh:
        for rank, row in enumerate(top_rows, 1):
            payload = dict(row)
            payload["rank"] = rank
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with top_txt_path.open("w", encoding="utf-8") as fh:
        for row in top_rows:
            fh.write(row["final_version"])
            fh.write("\n")

    summary = {
        "input_scored_rows": len(scored_rows_list),
        "eligible_rows": len(prepared),
        "excluded_non_candidate": excluded_non_candidate,
        "excluded_non_daily": excluded_non_daily,
        "duplicate_selected_query_rows": duplicate_count,
        "final_top_k": len(top_rows),
        "top_jsonl_path": str(top_jsonl_path),
        "top_txt_path": str(top_txt_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_usage_summary(scored_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_total = 0
    completion_total = 0
    total_total = 0
    with_usage = 0
    max_row: Optional[Dict[str, Any]] = None

    for row in scored_rows:
        usage = row.get("llm_usage") or {}
        pt = parse_int(usage.get("prompt_tokens"), 0)
        ct = parse_int(usage.get("completion_tokens"), 0)
        tt = parse_int(usage.get("total_tokens"), 0)
        if pt or ct or tt:
            with_usage += 1
        prompt_total += pt
        completion_total += ct
        total_total += tt
        row_total = tt or (pt + ct)
        if max_row is None or row_total > parse_int((max_row.get("llm_usage") or {}).get("total_tokens"), 0):
            max_row = row

    avg_denominator = with_usage or 1
    return {
        "rows_with_usage": with_usage,
        "prompt_tokens_sum": prompt_total,
        "completion_tokens_sum": completion_total,
        "total_tokens_sum": total_total,
        "avg_prompt_tokens": prompt_total / avg_denominator,
        "avg_completion_tokens": completion_total / avg_denominator,
        "avg_total_tokens": total_total / avg_denominator,
        "max_total_tokens_row": (
            {
                "source_id": max_row.get("source_id"),
                "original_query": max_row.get("original_query"),
                "final_version": max_row.get("final_version"),
                "llm_usage": max_row.get("llm_usage"),
            }
            if max_row is not None
            else None
        ),
    }


def run_filter(
    input_dir: Path,
    prompt_path: Path,
    output_dir: Path,
    config_path: Path,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout_s: int,
    retries: int,
    top_k: int,
    max_items: int,
    sleep_seconds: float,
    resume: bool,
    keep_non_candidate: bool,
    allow_non_daily: bool,
) -> Dict[str, Any]:
    if not input_dir.exists():
        raise ValueError(f"Input dir not found: {input_dir}")
    if not prompt_path.exists():
        raise ValueError(f"Prompt file not found: {prompt_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "implicit_query_scored.jsonl"
    error_path = output_dir / "implicit_query_errors.jsonl"

    prompt_template = prompt_path.read_text(encoding="utf-8")
    resolved_base_url, resolved_model, headers = resolve_api_settings(
        config_path=config_path,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    rows = list(iter_query_rows(input_dir))
    if max_items > 0:
        rows = rows[:max_items]

    existing_rows = load_existing_scored(scored_path) if resume else {}
    if existing_rows:
        _log(f"Resume enabled: skip {len(existing_rows)} already scored rows.")

    total = len(rows)
    ok_count = 0
    fail_count = 0
    processed = 0

    scored_mode = "a" if (resume and scored_path.exists()) else "w"
    error_mode = "a" if (resume and error_path.exists()) else "w"

    with scored_path.open(scored_mode, encoding="utf-8") as scored_fh, error_path.open(
        error_mode, encoding="utf-8"
    ) as error_fh:
        for index, row in enumerate(rows, 1):
            source_id = row["source_id"]
            if resume and source_id in existing_rows:
                processed += 1
                continue

            prompt = render_prompt(prompt_template, row["original_query"])
            messages = [{"role": "user", "content": prompt}]

            try:
                chat_result = chat_completion(
                    resolved_base_url,
                    resolved_model,
                    messages=messages,
                    timeout_s=timeout_s,
                    retries=retries,
                    headers=headers,
                )
                content = str(chat_result.get("content", ""))
                raw_obj = extract_first_json_object(content)
                normalized = normalize_result(raw_obj, row["original_query"])
                payload = {
                    "source_id": source_id,
                    "source_file": row["source_file"],
                    "source_filename": row["source_filename"],
                    "line_number": row["line_number"],
                    "query_field": row["query_field"],
                    "original_query": row["original_query"],
                    "original_query_length": len(row["original_query"]),
                    "rank": row["source_record"].get("rank"),
                    "query_length": row["source_record"].get("query_length"),
                    "turn_count": row["source_record"].get("turn_count"),
                    "conversation_length": row["source_record"].get("conversation_length"),
                    "source_pointer": row["source_record"].get("source_pointer"),
                    **normalized,
                    "llm_usage": chat_result.get("usage", {}),
                    "raw_model_output": content,
                }
                payload = select_query_for_export(payload)
                scored_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                scored_fh.flush()
                existing_rows[source_id] = payload
                ok_count += 1
                processed += 1
                _log(
                    f"[{index}/{total}] OK {source_id} | "
                    f"candidate={payload['is_candidate']} daily={payload['is_daily']} "
                    f"implicit={payload['final_implicit_count']} compressed={payload['compressible_count']} "
                    f"tokens={payload['llm_usage'].get('total_tokens', 0)}"
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            except Exception as exc:
                fail_count += 1
                processed += 1
                err_payload = {
                    "source_id": source_id,
                    "source_file": row["source_file"],
                    "line_number": row["line_number"],
                    "original_query": row["original_query"],
                    "error": str(exc),
                }
                error_fh.write(json.dumps(err_payload, ensure_ascii=False) + "\n")
                error_fh.flush()
                _log(f"[{index}/{total}] FAIL {source_id}: {exc}")

    final_rows = list(existing_rows.values())
    export_summary = export_top_queries(
        scored_rows=final_rows,
        output_dir=output_dir,
        top_k=top_k,
        keep_non_candidate=keep_non_candidate,
        allow_non_daily=allow_non_daily,
    )
    usage_summary = build_usage_summary(final_rows)
    (output_dir / "usage_summary.json").write_text(
        json.dumps(usage_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_summary = {
        "provider": provider,
        "base_url": resolved_base_url,
        "model": resolved_model,
        "prompt_path": str(prompt_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_input_rows": total,
        "processed_this_run": processed,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "resume": resume,
        "scored_path": str(scored_path),
        "error_path": str(error_path),
        "top_k": top_k,
        "export_summary": export_summary,
        "usage_summary": usage_summary,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_summary


def main() -> int:
    args = parse_args()
    try:
        summary = run_filter(
            input_dir=args.input_dir,
            prompt_path=args.prompt,
            output_dir=args.output_dir,
            config_path=args.config,
            provider=args.provider.strip().lower(),
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout,
            retries=args.retries,
            top_k=args.top_k,
            max_items=args.max_items,
            sleep_seconds=args.sleep_seconds,
            resume=not args.no_resume,
            keep_non_candidate=args.keep_non_candidate,
            allow_non_daily=args.allow_non_daily,
        )
    except Exception as exc:
        _log(f"FAILED: {exc}")
        return 1

    _log(
        "DONE: "
        f"ok={summary['ok_count']} fail={summary['fail_count']} "
        f"top={summary['export_summary']['final_top_k']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
