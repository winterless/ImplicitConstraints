from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agent import HeuristicPlanningAgent, QwenPlanningAgent
from .llm_client import (
    DEFAULT_ALIYUN_API_KEY_FILE,
    DEFAULT_ALIYUN_BASE_URL,
    DEFAULT_ALIYUN_MODEL,
    ChatCompletionClient,
    resolve_aliyun_api_key,
)
from .orchestrator import ScenarioOrchestrator
from .schemas import load_scenario
from .tool_registry import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the implicit constraints demo scenario.")
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to the scenario YAML file.",
    )
    parser.add_argument(
        "--tool-dir",
        default="data/tool_schemas/device",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

    scenario = load_scenario(args.scenario)
    registry = ToolRegistry.from_directory(args.tool_dir)
    resolved_api_key = resolve_aliyun_api_key(
        api_key=args.api_key,
        api_key_file=args.api_key_file,
    )
    if resolved_api_key:
        agent = QwenPlanningAgent(
            client=ChatCompletionClient(
                base_url=args.base_url,
                model=args.model,
                api_key=resolved_api_key,
                timeout_s=args.timeout,
                retries=args.retries,
            )
        )
    else:
        print(
            "No API key found; falling back to local heuristic agent.",
            file=sys.stderr,
        )
        agent = HeuristicPlanningAgent()

    orchestrator = ScenarioOrchestrator(
        scenario=scenario,
        registry=registry,
        agent=agent,
    )
    result = orchestrator.run()

    if args.output:
        output_path = Path(args.output) / f"{scenario.scenario_id}.json"
    else:
        output_path = Path("runs") / f"{scenario.scenario_id}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
