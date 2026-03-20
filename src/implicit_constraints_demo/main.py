from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
        )
        output_path = _single_output_path(result["scenario_id"], args.output)
        _write_json(output_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] == "failed":
            raise SystemExit(1)
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
        result = _execute_scenario(
            scenario_path=scenario_path,
            registry=registry,
            agent=agent,
            world_mode=runtime_config.world.mode,
            world_client=world_client,
            evaluator=evaluator,
            runtime_config=runtime_config,
        )
        output_path = batch_output_dir / f"{result['scenario_id']}.json"
        _write_json(output_path, result)
        if result["status"] == "completed":
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
        else:
            summary["failed_scenarios"] += 1
            summary["results"].append(
                {
                    "scenario_path": str(scenario_path),
                    "scenario_id": result["scenario_id"],
                    "status": "failed",
                    "output_path": str(output_path),
                    "error": result.get("error", "Unknown error"),
                }
            )
            print(
                f"[failed] {result['scenario_id']} -> {output_path}: {result.get('error', 'Unknown error')}",
                file=sys.stderr,
            )

    summary_path = batch_output_dir / "_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
) -> dict[str, Any]:
    try:
        return _run_single_scenario(
            scenario_path=scenario_path,
            registry=registry,
            agent=agent,
            world_mode=world_mode,
            world_client=world_client,
            evaluator=evaluator,
            runtime_config=runtime_config,
        )
    except ScenarioRunError as exc:
        artifact = dict(exc.artifact)
        artifact["runtime_config"] = _runtime_config_snapshot(runtime_config)
        return artifact
    except Exception as exc:
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
            "error": str(exc),
            "runtime_config": _runtime_config_snapshot(runtime_config),
        }


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


def _build_role_client(config: RoleRuntimeConfig, api_key: str) -> ChatCompletionClient | None:
    if config.mode not in {"llm"}:
        return None
    api_key_envs = tuple(
        part.strip()
        for part in config.api_key_env.split(",")
        if part.strip()
    )
    return ChatCompletionClient(
        base_url=config.base_url,
        model=config.model,
        api_key=api_key,
        api_key_file=config.api_key_file,
        api_key_envs=api_key_envs,
        require_api_key=config.require_api_key,
        timeout_s=config.timeout_s,
        retries=config.retries,
        temperature=config.temperature,
    )


def _require_role_client(
    config: RoleRuntimeConfig,
    *,
    role_name: str,
    api_key: str,
) -> ChatCompletionClient:
    try:
        client = _build_role_client(config, api_key=api_key)
    except ValueError as exc:
        raise SystemExit(f"{role_name} role requires a usable LLM client: {exc}") from exc
    if client is None:
        raise SystemExit(f"{role_name} role is not configured to use an LLM client.")
    return client


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
        "api_key_env": config.api_key_env,
        "api_key_file": config.api_key_file,
        "require_api_key": config.require_api_key,
        "timeout_s": config.timeout_s,
        "retries": config.retries,
        "temperature": config.temperature,
    }


if __name__ == "__main__":
    main()
