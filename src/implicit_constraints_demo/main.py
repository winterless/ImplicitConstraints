from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .agent import HeuristicPlanningAgent, QwenPlanningAgent
from .llm_client import (
    DEFAULT_ALIYUN_API_KEY_FILE,
    DEFAULT_ALIYUN_BASE_URL,
    DEFAULT_ALIYUN_MODEL,
    ChatCompletionClient,
    resolve_aliyun_api_key,
)
from .orchestrator import ScenarioOrchestrator
from .schemas import load_scenario, load_yaml
from .tool_registry import ToolRegistry
from .world import build_world


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
        "--tool-dir",
        default="data/tool_schemas",
        help="Directory containing tool descriptor YAML files.",
    )
    parser.add_argument(
        "--output",
        help="Optional directory to store the run artifact as <scenario_id>.json.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_ALIYUN_BASE_URL,
        help="Chat completion base URL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ALIYUN_MODEL,
        help="Model ID. DataBot uses qwen-plus by default on DashScope compatible mode.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key. If empty, read DASHSCOPE_API_KEY / ALIYUN_API_KEY / .secrets file.",
    )
    parser.add_argument(
        "--api-key-file",
        default=DEFAULT_ALIYUN_API_KEY_FILE,
        help="Optional API key file path.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Chat request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count for chat requests.",
    )
    parser.add_argument(
        "--world-mode",
        required=True,
        choices=["mock", "llm"],
        help="Execution mode for the world model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    registry = ToolRegistry.from_directory(args.tool_dir)
    resolved_api_key = resolve_aliyun_api_key(
        api_key=args.api_key,
        api_key_file=args.api_key_file,
    )
    if args.world_mode == "llm" and not resolved_api_key:
        raise SystemExit(
            "world-mode=llm requires an API key. Set DASHSCOPE_API_KEY / ALIYUN_API_KEY or pass --api-key."
        )
    llm_client: ChatCompletionClient | None = None
    if resolved_api_key:
        llm_client = ChatCompletionClient(
            base_url=args.base_url,
            model=args.model,
            api_key=resolved_api_key,
            timeout_s=args.timeout,
            retries=args.retries,
        )
        agent = QwenPlanningAgent(client=llm_client)
    else:
        print(
            "No API key found; falling back to local heuristic agent.",
            file=sys.stderr,
        )
        agent = HeuristicPlanningAgent()

    if args.scenario:
        scenario_path = Path(args.scenario)
        result = _run_single_scenario(
            scenario_path,
            registry,
            agent,
            world_mode=args.world_mode,
            llm_client=llm_client,
        )
        output_path = _single_output_path(result["scenario_id"], args.output)
        _write_json(output_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    scenario_paths = _discover_executable_scenarios(Path(args.scenario_dir))
    if not scenario_paths:
        raise SystemExit(f"No executable scenarios found in: {args.scenario_dir}")

    batch_output_dir = _batch_output_dir(args.output)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "mode": "batch",
        "scenario_dir": str(Path(args.scenario_dir)),
        "output_dir": str(batch_output_dir),
        "total_scenarios": len(scenario_paths),
        "completed_scenarios": 0,
        "failed_scenarios": 0,
        "results": [],
    }

    for scenario_path in scenario_paths:
        try:
            result = _run_single_scenario(
                scenario_path,
                registry,
                agent,
                world_mode=args.world_mode,
                llm_client=llm_client,
            )
            output_path = batch_output_dir / f"{result['scenario_id']}.json"
            _write_json(output_path, result)
            summary["completed_scenarios"] += 1
            summary["results"].append(
                {
                    "scenario_path": str(scenario_path),
                    "scenario_id": result["scenario_id"],
                    "status": "completed",
                    "output_path": str(output_path),
                    "normalized_scenario_score": result["evaluation"]["summary"][
                        "normalized_scenario_score"
                    ],
                    "passed_all": result["evaluation"]["summary"]["passed_all"],
                }
            )
            print(
                f"[completed] {result['scenario_id']} -> {output_path}",
                file=sys.stderr,
            )
        except Exception as exc:
            summary["failed_scenarios"] += 1
            summary["results"].append(
                {
                    "scenario_path": str(scenario_path),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[failed] {scenario_path}: {exc}", file=sys.stderr)

    summary_path = batch_output_dir / "_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_single_scenario(
    scenario_path: Path,
    registry: ToolRegistry,
    agent: HeuristicPlanningAgent | QwenPlanningAgent,
    world_mode: str,
    llm_client: ChatCompletionClient | None,
) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    world = build_world(
        scenario=scenario,
        registry=registry,
        mode=world_mode,
        client=llm_client,
    )
    orchestrator = ScenarioOrchestrator(
        scenario=scenario,
        registry=registry,
        agent=agent,
        world=world,
    )
    return orchestrator.run()


def _discover_executable_scenarios(directory: Path) -> list[Path]:
    scenario_paths: list[Path] = []
    for path in sorted(directory.glob("*.yaml")):
        raw = load_yaml(path)
        if _is_executable_scenario(raw):
            scenario_paths.append(path)
        else:
            print(f"[skip] non-executable scenario file: {path}", file=sys.stderr)
    return scenario_paths


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


if __name__ == "__main__":
    main()
