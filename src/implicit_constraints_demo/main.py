from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

from .agent import BaseAgent, HeuristicPlanningAgent, QwenPlanningAgent
from .evaluator import BaseScenarioEvaluator, build_evaluator
from .llm_client import ChatCompletionClient
from .orchestrator import ScenarioOrchestrator, ScenarioRunError
from .runtime_config import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    RoleRuntimeConfig,
    RuntimeConfig,
    load_runtime_config,
    override_runtime_config,
)
from .schemas import load_scenario
from .tool_registry import ToolRegistry
from .world import build_world

DEFAULT_SCENARIO_MANIFEST = "scenario_manifest.yaml"
RULE_FAMILY_LABELS = [
    "时间/截止/缓冲/时区/ETA",
    "先查真实状态再行动",
    "风险防呆/确认后执行",
    "排序/筛选/最优选择",
    "跨对象依赖/整体损失或整体可行性",
    "个性化/关系/偏好识别",
    "自动化/提醒/真正落地执行",
]


def _binary_scenario_score(passed_count: int, total_count: int) -> float:
    """整题评分：全部小项通过为 1，否则 0。"""
    if total_count <= 0:
        return 0.0
    return 1.0 if passed_count == total_count else 0.0


def _extract_rule_family_label(text: str) -> str | None:
    match = re.match(r"\s*【([^】]+)】", text or "")
    if not match:
        return None
    label = match.group(1).strip()
    return label if label in RULE_FAMILY_LABELS else None


def _build_batch_aggregate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    completed_results = [item for item in results if item.get("status") == "completed"]
    passed_all_count = sum(
        1 for item in completed_results if item.get("passed_all") is True
    )
    sum_passed_subitems = sum(
        int(item.get("passed_count", 0)) for item in completed_results
    )
    sum_total_subitems = sum(
        int(item.get("total_count", 0)) for item in completed_results
    )
    completed_count = len(completed_results)

    dimension_totals: dict[str, dict[str, Any]] = {
        label: {
            "earned_points": 0,
            "total_points": 0,
        }
        for label in RULE_FAMILY_LABELS
    }
    for item in completed_results:
        for point_result in item.get("point_results", []):
            if not isinstance(point_result, dict):
                continue
            rule_label = point_result.get("rule_label")
            if rule_label not in RULE_FAMILY_LABELS:
                continue
            bucket = dimension_totals[rule_label]
            bucket["total_points"] += 1
            if point_result.get("passed") is True:
                bucket["earned_points"] += 1

    rule_family_scores: dict[str, Any] = {}
    for rule_label, bucket in dimension_totals.items():
        total_points = int(bucket["total_points"])
        if total_points <= 0:
            continue
        rule_family_scores[rule_label] = {
            "earned": int(bucket["earned_points"]),
            "total": total_points,
            "score": round(bucket["earned_points"] / total_points, 6),
        }

    batch_all_correct_rate = (
        round(passed_all_count / completed_count, 6) if completed_count else None
    )
    batch_subitem_rate = (
        round(sum_passed_subitems / sum_total_subitems, 6)
        if sum_total_subitems > 0
        else None
    )
    return {
        "batch_all_correct": {
            "earned": passed_all_count,
            "total": completed_count,
            "rate": batch_all_correct_rate,
        },
        "batch_subitems": {
            "earned": sum_passed_subitems,
            "total": sum_total_subitems,
            "rate": batch_subitem_rate,
        },
        "rule_family_scores": rule_family_scores,
    }


def _load_existing_batch_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        raise SystemExit(f"Batch summary not found: {summary_path}")
    try:
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid batch summary JSON: {summary_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"Batch summary must be a JSON object: {summary_path}")
    if raw.get("mode") != "batch":
        raise SystemExit(f"Expected a batch summary file: {summary_path}")
    results = raw.get("results", [])
    if not isinstance(results, list):
        raise SystemExit(f"Batch summary results must be a list: {summary_path}")
    return raw


def _summary_entry_for_result(
    *,
    result: dict[str, Any],
    scenario_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if result["status"] == "completed":
        scenario = load_scenario(scenario_path)
        eval_summary = result["evaluation"]["summary"]
        evaluation_results = result["evaluation"].get("evaluation_results", [])
        passed_count = int(eval_summary.get("passed_count", 0))
        total_points = int(eval_summary.get("total_count", 0))
        binary = float(
            eval_summary.get(
                "binary_scenario_score",
                _binary_scenario_score(passed_count, total_points),
            )
        )
        raw_points = scenario.raw.get("implicit_eval_points", [])
        point_results: list[dict[str, Any]] = []
        if isinstance(raw_points, list):
            for idx, point_text in enumerate(raw_points):
                passed = False
                if idx < len(evaluation_results) and isinstance(evaluation_results[idx], dict):
                    passed = bool(evaluation_results[idx].get("passed", False))
                point_results.append(
                    {
                        "rule_label": _extract_rule_family_label(str(point_text)),
                        "passed": passed,
                    }
                )
        return {
            "scenario_path": str(scenario_path),
            "scenario_id": result["scenario_id"],
            "status": "completed",
            "output_path": str(output_path),
            "normalized_scenario_score": eval_summary["normalized_scenario_score"],
            "binary_scenario_score": binary,
            "passed_all": eval_summary["passed_all"],
            "passed_count": passed_count,
            "total_count": total_points,
            "point_results": point_results,
        }
    return {
        "scenario_path": str(scenario_path),
        "scenario_id": result["scenario_id"],
        "status": "failed",
        "output_path": str(output_path),
        "error": result.get("error", "Unknown error"),
    }


def _build_batch_summary(
    *,
    results: list[dict[str, Any]],
    scenario_dir: str,
    scenario_manifest: str | None,
    batch_output_dir: Path,
) -> dict[str, Any]:
    completed_count = sum(1 for item in results if item.get("status") == "completed")
    failed_count = sum(1 for item in results if item.get("status") == "failed")
    summary: dict[str, Any] = {
        "mode": "batch",
        "scenario_dir": scenario_dir,
        "scenario_manifest": scenario_manifest,
        "output_dir": str(batch_output_dir),
        "total_scenarios": len(results),
        "completed_scenarios": completed_count,
        "failed_scenarios": failed_count,
        "results": results,
    }
    summary.update(_build_batch_aggregate_metrics(results))
    return summary


def _failed_scenarios_from_summary(
    summary: dict[str, Any],
    *,
    error_filters: list[str],
) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()
    for item in summary.get("results", []):
        if not isinstance(item, dict) or item.get("status") != "failed":
            continue
        scenario_id = str(item.get("scenario_id", "")).strip()
        if not scenario_id or scenario_id in seen_ids:
            continue
        error = str(item.get("error", ""))
        if error_filters and not any(fragment in error for fragment in error_filters):
            continue
        raw_path = str(item.get("scenario_path", "")).strip()
        if not raw_path:
            raise SystemExit(
                f"Failed summary entry is missing scenario_path: {scenario_id}"
            )
        scenario_path = Path(raw_path)
        if not scenario_path.is_absolute():
            scenario_path = (Path.cwd() / scenario_path).resolve()
        if not scenario_path.exists():
            raise SystemExit(f"Scenario from summary does not exist: {scenario_path}")
        selected.append((scenario_id, scenario_path))
        seen_ids.add(scenario_id)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one scenario or all executable scenarios in a directory."
    )
    parser.add_argument(
        "--scenario",
        help="Optional path to a single scenario YAML file. If omitted, run all executable scenarios in --scenario-dir.",
    )
    parser.add_argument(
        "--scenario-dir",
        default="data/scenarios",
        help="Directory to scan when --scenario is omitted.",
    )
    parser.add_argument(
        "--scenario-manifest",
        help=(
            "Optional scenario manifest YAML. If omitted, the runner will use "
            "<scenario-dir>/scenario_manifest.yaml when present, otherwise it will scan *.yaml directly."
        ),
    )
    parser.add_argument(
        "--tool-dir",
        default="data/tool_schemas",
        help="Directory containing tool descriptor YAML files.",
    )
    parser.add_argument(
        "--output",
        help="Optional directory to store the run artifact as <scenario_id>.json.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_RUNTIME_CONFIG_PATH,
        help="Runtime config YAML file for agent/world/evaluator roles.",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["llm", "heuristic"],
        help="Optional override for the agent role.",
    )
    parser.add_argument(
        "--world-mode",
        choices=["llm", "mock"],
        help="Optional override for the world role.",
    )
    parser.add_argument(
        "--evaluator-mode",
        choices=["llm", "deterministic"],
        help="Optional override for the evaluator role.",
    )
    parser.add_argument(
        "--base-url",
        help="Optional global override for all LLM role base URLs.",
    )
    parser.add_argument(
        "--model",
        help="Optional global override for all LLM role models.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional global API key override for all LLM roles.",
    )
    parser.add_argument(
        "--api-key-env",
        help="Optional global API key env var override for all LLM roles.",
    )
    parser.add_argument(
        "--api-key-file",
        help="Optional global API key file override for all LLM roles.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Optional global timeout override for all LLM roles.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        help="Optional global retry override for all LLM roles.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional global temperature override for all LLM roles.",
    )
    parser.add_argument(
        "--allow-missing-api-key",
        action="store_true",
        help="Allow all LLM roles to run without an API key, useful for local compatible endpoints.",
    )
    parser.add_argument(
        "--rerun-from-summary",
        help=(
            "Optional existing batch _summary.json path. When set, rerun failed scenarios "
            "from that summary, overwrite their artifacts, and rebuild the summary."
        ),
    )
    parser.add_argument(
        "--rerun-error-contains",
        action="append",
        default=[],
        help=(
            "Optional substring filter for --rerun-from-summary. Repeatable. "
            "Only failed scenarios whose error contains one of these substrings will be rerun."
        ),
    )
    parser.add_argument(
        "--no-retry-on-error",
        action="store_true",
        help="Disable the default one retry when a scenario run fails with an error.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.scenario and args.rerun_from_summary:
        raise SystemExit("--scenario cannot be combined with --rerun-from-summary")

    registry = ToolRegistry.from_directory(args.tool_dir)
    runtime_config = override_runtime_config(
        load_runtime_config(args.config),
        agent_mode=args.agent_mode,
        world_mode=args.world_mode,
        evaluator_mode=args.evaluator_mode,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        api_key_file=args.api_key_file,
        require_api_key=False if args.allow_missing_api_key else None,
        timeout_s=args.timeout,
        retries=args.retries,
        temperature=args.temperature,
    )
    agent = _build_agent(runtime_config.agent, api_key=args.api_key)
    world_client = None
    if runtime_config.world.mode == "llm":
        world_client = _require_role_client(runtime_config.world, role_name="world", api_key=args.api_key)
    evaluator = _build_evaluator(runtime_config.evaluator, api_key=args.api_key)

    if args.scenario:
        scenario_path = Path(args.scenario)
        result = _execute_scenario(
            scenario_path=scenario_path,
            registry=registry,
            agent=agent,
            world_mode=runtime_config.world.mode,
            world_client=world_client,
            evaluator=evaluator,
            runtime_config=runtime_config,
            retry_once_on_error=not args.no_retry_on_error,
        )
        output_path = _single_output_path(result["scenario_id"], args.output)
        _write_json(output_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] == "failed":
            raise SystemExit(1)
        return

    scenario_dir = Path(args.scenario_dir)
    manifest_path = _resolve_scenario_manifest_path(scenario_dir, args.scenario_manifest)
    if args.rerun_from_summary:
        source_summary_path = Path(args.rerun_from_summary)
        source_summary = _load_existing_batch_summary(source_summary_path)
        rerun_targets = _failed_scenarios_from_summary(
            source_summary,
            error_filters=args.rerun_error_contains,
        )
        if not rerun_targets:
            raise SystemExit(
                f"No failed scenarios matched in summary: {source_summary_path}"
            )

        batch_output_dir = (
            _batch_output_dir(args.output)
            if args.output
            else source_summary_path.parent
        )
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        existing_results = source_summary.get("results", [])
        results_by_id: dict[str, dict[str, Any]] = {}
        ordered_scenario_ids: list[str] = []
        for item in existing_results:
            if not isinstance(item, dict):
                continue
            scenario_id = str(item.get("scenario_id", "")).strip()
            if not scenario_id or scenario_id in results_by_id:
                continue
            results_by_id[scenario_id] = item
            ordered_scenario_ids.append(scenario_id)

        print(
            f"[resume] rerunning {len(rerun_targets)} failed scenarios from {source_summary_path}",
            file=sys.stderr,
        )
        for _, scenario_path in rerun_targets:
            result = _execute_scenario(
                scenario_path=scenario_path,
                registry=registry,
                agent=agent,
                world_mode=runtime_config.world.mode,
                world_client=world_client,
                evaluator=evaluator,
                runtime_config=runtime_config,
                retry_once_on_error=not args.no_retry_on_error,
            )
            output_path = batch_output_dir / f"{result['scenario_id']}.json"
            _write_json(output_path, result)
            results_by_id[result["scenario_id"]] = _summary_entry_for_result(
                result=result,
                scenario_path=scenario_path,
                output_path=output_path,
            )
            if result["status"] == "completed":
                print(
                    f"[completed] {result['scenario_id']} -> {output_path}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[failed] {result['scenario_id']} -> {output_path}: {result.get('error', 'Unknown error')}",
                    file=sys.stderr,
                )

        merged_results = [
            results_by_id[scenario_id]
            for scenario_id in ordered_scenario_ids
            if scenario_id in results_by_id
        ]
        summary = _build_batch_summary(
            results=merged_results,
            scenario_dir=str(source_summary.get("scenario_dir", args.scenario_dir)),
            scenario_manifest=(
                str(source_summary["scenario_manifest"])
                if source_summary.get("scenario_manifest") is not None
                else None
            ),
            batch_output_dir=batch_output_dir,
        )
        summary_path = batch_output_dir / "_summary.json"
        _write_json(summary_path, summary)
        _print_batch_score_line(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    scenario_paths = _discover_executable_scenarios(scenario_dir, manifest_path=manifest_path)
    if not scenario_paths:
        raise SystemExit(f"No executable scenarios found in: {args.scenario_dir}")

    batch_output_dir = _batch_output_dir(args.output)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    summary_results: list[dict[str, Any]] = []

    for scenario_path in scenario_paths:
        result = _execute_scenario(
            scenario_path=scenario_path,
            registry=registry,
            agent=agent,
            world_mode=runtime_config.world.mode,
            world_client=world_client,
            evaluator=evaluator,
            runtime_config=runtime_config,
            retry_once_on_error=not args.no_retry_on_error,
        )
        output_path = batch_output_dir / f"{result['scenario_id']}.json"
        _write_json(output_path, result)
        summary_results.append(
            _summary_entry_for_result(
                result=result,
                scenario_path=scenario_path,
                output_path=output_path,
            )
        )
        if result["status"] == "completed":
            print(
                f"[completed] {result['scenario_id']} -> {output_path}",
                file=sys.stderr,
            )
        else:
            print(
                f"[failed] {result['scenario_id']} -> {output_path}: {result.get('error', 'Unknown error')}",
                file=sys.stderr,
            )

    summary = _build_batch_summary(
        results=summary_results,
        scenario_dir=str(scenario_dir),
        scenario_manifest=str(manifest_path) if manifest_path is not None else None,
        batch_output_dir=batch_output_dir,
    )
    summary_path = batch_output_dir / "_summary.json"
    _write_json(summary_path, summary)
    _print_batch_score_line(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _print_batch_score_line(summary: dict[str, Any]) -> None:
    """在 stderr 打印简化后的批量得分摘要。"""
    bac = summary.get("batch_all_correct")
    bsub = summary.get("batch_subitems")
    if not isinstance(bac, dict) or not isinstance(bsub, dict):
        return
    earned_ac, total_ac, rate_ac = (
        bac.get("earned"),
        bac.get("total"),
        bac.get("rate"),
    )
    earned_si, total_si, rate_si = (
        bsub.get("earned"),
        bsub.get("total"),
        bsub.get("rate"),
    )
    if total_ac is None or int(total_ac) <= 0:
        return
    sub_part = (
        f"小项累计 {earned_si}/{total_si} (= {rate_si})"
        if total_si is not None and int(total_si) > 0
        else "小项累计 N/A"
    )
    print(
        "[batch scores] " f"整题全对 {earned_ac}/{total_ac} (= {rate_ac}); " + sub_part,
        file=sys.stderr,
    )
    rule_scores = summary.get("rule_family_scores")
    if not isinstance(rule_scores, dict):
        return
    parts: list[str] = []
    for label in RULE_FAMILY_LABELS:
        bucket = rule_scores.get(label)
        if not isinstance(bucket, dict):
            continue
        parts.append(
            f"{label} {bucket.get('earned')}/{bucket.get('total')} (= {bucket.get('score')})"
        )
    if parts:
        print("[rule scores] " + "; ".join(parts), file=sys.stderr)


def _run_single_scenario(
    scenario_path: Path,
    registry: ToolRegistry,
    agent: BaseAgent,
    world_mode: str,
    world_client: ChatCompletionClient | None,
    evaluator: BaseScenarioEvaluator,
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    world = build_world(
        scenario=scenario,
        registry=registry,
        mode=world_mode,
        client=world_client,
    )
    orchestrator = ScenarioOrchestrator(
        scenario=scenario,
        registry=registry,
        agent=agent,
        world=world,
        evaluator=evaluator,
    )
    result = orchestrator.run()
    result["runtime_config"] = _runtime_config_snapshot(runtime_config)
    return result


def _execute_scenario(
    *,
    scenario_path: Path,
    registry: ToolRegistry,
    agent: BaseAgent,
    world_mode: str,
    world_client: ChatCompletionClient | None,
    evaluator: BaseScenarioEvaluator,
    runtime_config: RuntimeConfig,
    retry_once_on_error: bool = True,
) -> dict[str, Any]:
    max_attempts = 2 if retry_once_on_error else 1
    first_error_summary: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = _run_single_scenario(
                scenario_path=scenario_path,
                registry=registry,
                agent=agent,
                world_mode=world_mode,
                world_client=world_client,
                evaluator=evaluator,
                runtime_config=runtime_config,
            )
            result["attempts_used"] = attempt
            if attempt > 1 and first_error_summary is not None:
                result["recovered_after_retry"] = True
                result["first_attempt_error"] = first_error_summary
            return result
        except ScenarioRunError as exc:
            artifact = dict(exc.artifact)
            err_msg = str(artifact.get("error") or exc.cause or exc)
            artifact["runtime_config"] = _runtime_config_snapshot(runtime_config)
            artifact["attempts_used"] = attempt
            if attempt < max_attempts:
                first_error_summary = err_msg
                sid = artifact.get("scenario_id", scenario_path.stem)
                print(
                    f"[retry] {sid} attempt {attempt}/{max_attempts} failed: {err_msg}; retrying once...",
                    file=sys.stderr,
                )
                continue
            return artifact
        except Exception as exc:
            err_msg = str(exc)
            if attempt < max_attempts:
                first_error_summary = err_msg
                print(
                    f"[retry] {scenario_path.name} attempt {attempt}/{max_attempts} failed: {err_msg}; retrying once...",
                    file=sys.stderr,
                )
                continue
            scenario = load_scenario(scenario_path)
            return {
                "status": "failed",
                "scenario_id": scenario.scenario_id,
                "category": scenario.category,
                "user_prompt": scenario.user_prompt,
                "input_snapshot": {
                    "context": scenario.context,
                    "allowed_tools": registry.export_for_keys(scenario.allowed_tools),
                    "agent_mode": agent.mode,
                    "world_mode": world_mode,
                    "evaluator_mode": evaluator.mode,
                },
                "messages": [],
                "model_logs": [],
                "world_logs": [],
                "final_world_state": scenario.state,
                "evaluation": None,
                "error": err_msg,
                "runtime_config": _runtime_config_snapshot(runtime_config),
                "attempts_used": attempt,
            }


def _resolve_scenario_manifest_path(directory: Path, manifest_arg: str | None) -> Path | None:
    if manifest_arg:
        manifest_path = Path(manifest_arg)
        if not manifest_path.exists():
            raise SystemExit(f"Scenario manifest not found: {manifest_path}")
        return manifest_path
    candidate = directory.parent / DEFAULT_SCENARIO_MANIFEST
    if candidate.exists():
        return candidate
    return None


def _discover_executable_scenarios(directory: Path, *, manifest_path: Path | None = None) -> list[Path]:
    if manifest_path is not None:
        return _discover_scenarios_from_manifest(directory, manifest_path)

    scenario_paths: list[Path] = []
    for path in sorted(directory.glob("*.yaml")):
        raw = load_yaml(path)
        if _is_executable_scenario(raw):
            scenario_paths.append(path)
        else:
            print(f"[skip] non-executable scenario file: {path}", file=sys.stderr)
    return scenario_paths


def _discover_scenarios_from_manifest(directory: Path, manifest_path: Path) -> list[Path]:
    raw = load_yaml(manifest_path)
    if not isinstance(raw, dict):
        raise SystemExit(f"Scenario manifest must be a mapping: {manifest_path}")

    entries = raw.get("scenarios", [])
    if not isinstance(entries, list):
        raise SystemExit(f"Scenario manifest must contain a 'scenarios' list: {manifest_path}")

    selected: list[tuple[int, str, Path]] = []
    seen_paths: set[Path] = set()
    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise SystemExit(f"Invalid manifest entry at index {idx}: expected mapping")
        if not bool(entry.get("enabled", True)):
            continue

        raw_path = entry.get("path")
        if not raw_path:
            raise SystemExit(f"Manifest entry at index {idx} is missing 'path'")

        scenario_path = Path(str(raw_path))
        if not scenario_path.is_absolute():
            scenario_path = manifest_path.parent / scenario_path
        scenario_path = scenario_path.resolve()
        if scenario_path in seen_paths:
            raise SystemExit(f"Duplicate scenario path in manifest: {scenario_path}")
        if not scenario_path.exists():
            raise SystemExit(f"Scenario listed in manifest does not exist: {scenario_path}")
        if directory.resolve() not in scenario_path.parents:
            raise SystemExit(
                f"Scenario listed in manifest must be inside scenario directory {directory}: {scenario_path}"
            )

        scenario_raw = load_yaml(scenario_path)
        if not _is_executable_scenario(scenario_raw):
            raise SystemExit(f"Scenario listed in manifest is not executable: {scenario_path}")

        order = int(entry.get("order", idx))
        selected.append((order, str(raw_path), scenario_path))
        seen_paths.add(scenario_path)

    selected.sort(key=lambda item: (item[0], item[1]))
    return [path for _, _, path in selected]


def _is_executable_scenario(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    required_keys = {"id", "category", "user_prompt", "world"}
    return required_keys.issubset(raw.keys())


def _single_output_path(scenario_id: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir) / f"{scenario_id}.json"
    return Path("runs") / f"{scenario_id}.json"


def _batch_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path("runs") / "batch"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_agent(config: RoleRuntimeConfig, api_key: str) -> BaseAgent:
    if config.mode == "heuristic":
        return HeuristicPlanningAgent()
    client = _require_role_client(config, role_name="agent", api_key=api_key)
    return QwenPlanningAgent(client=client)


def _build_evaluator(config: RoleRuntimeConfig, api_key: str) -> BaseScenarioEvaluator:
    client = None
    if config.mode == "llm":
        client = _require_role_client(config, role_name="evaluator", api_key=api_key)
    return build_evaluator(config.mode, client=client)


def _build_role_client(
    config: RoleRuntimeConfig,
    api_key: str,
    *,
    role_name: str,
) -> ChatCompletionClient | None:
    if config.mode not in {"llm"}:
        return None
    api_key_envs = tuple(
        part.strip()
        for part in config.api_key_env.split(",")
        if part.strip()
    )
    return ChatCompletionClient(
        provider=config.provider,
        base_url=config.base_url,
        model=config.model,
        model_agent=config.model_agent,
        sub_account_name=config.sub_account_name,
        api_key=api_key,
        api_key_file=config.api_key_file,
        api_key_envs=api_key_envs,
        require_api_key=config.require_api_key,
        timeout_s=config.timeout_s,
        retries=config.retries,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        thinking_budget=_default_thinking_budget(config, role_name=role_name),
    )


def _require_role_client(
    config: RoleRuntimeConfig,
    *,
    role_name: str,
    api_key: str,
) -> ChatCompletionClient:
    try:
        client = _build_role_client(config, api_key=api_key, role_name=role_name)
    except ValueError as exc:
        raise SystemExit(f"{role_name} role requires a usable LLM client: {exc}") from exc
    if client is None:
        raise SystemExit(f"{role_name} role is not configured to use an LLM client.")
    return client


def _default_thinking_budget(
    config: RoleRuntimeConfig,
    *,
    role_name: str,
) -> int | None:
    provider = config.provider.strip().lower()
    model = config.model.strip().lower()
    if provider != "gemini":
        return None
    if role_name != "agent":
        if _gemini_model_requires_thinking(model):
            return None
        return 0
    if _gemini_model_requires_thinking(model):
        return 256
    if not _gemini_model_supports_thinking(model):
        return None
    return 256


def _gemini_model_supports_thinking(model_name: str) -> bool:
    return model_name.startswith("gemini-2.5") or model_name.startswith("gemini-3")


def _gemini_model_requires_thinking(model_name: str) -> bool:
    return model_name.startswith("gemini-3.1")


def _runtime_config_snapshot(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "agent": _role_config_snapshot(config.agent),
        "world": _role_config_snapshot(config.world),
        "evaluator": _role_config_snapshot(config.evaluator),
    }


def _role_config_snapshot(config: RoleRuntimeConfig) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "model_agent": config.model_agent,
        "sub_account_name": config.sub_account_name,
        "api_key_env": config.api_key_env,
        "api_key_file": config.api_key_file,
        "require_api_key": config.require_api_key,
        "timeout_s": config.timeout_s,
        "retries": config.retries,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }


if __name__ == "__main__":
    main()
