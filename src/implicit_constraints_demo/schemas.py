from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ToolDescriptor:
    server: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    success_response_schema: dict[str, Any]
    returns: dict[str, Any] | str | None = None
    state_changes: list[str] = field(default_factory=list)
    failure_conditions: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.server}:{self.tool_name}"


@dataclass(slots=True)
class ToolCall:
    server: str
    tool_name: str
    arguments: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.server}:{self.tool_name}"


@dataclass(slots=True)
class ToolResult:
    success: bool
    message: str
    data: dict[str, Any]
    state_changes: dict[str, Any]
    model_log: dict[str, Any] | None = None


@dataclass(slots=True)
class AgentDecision:
    thought: str
    tool_call: ToolCall | None = None
    task_complete: bool = False
    model_log: dict[str, Any] | None = None


@dataclass(slots=True)
class Scenario:
    raw: dict[str, Any]

    @property
    def scenario_id(self) -> str:
        return self.raw["id"]

    @property
    def category(self) -> str:
        return self.raw["category"]

    @property
    def user_prompt(self) -> str:
        return self.raw["user_prompt"]

    @property
    def max_steps(self) -> int:
        return int(self.raw.get("max_steps", 8))

    @property
    def world(self) -> dict[str, Any]:
        return self.raw["world"]

    @property
    def context(self) -> dict[str, Any]:
        return self.world["context"]

    @property
    def state(self) -> dict[str, Any]:
        return self.world["state"]

    @property
    def allowed_tools(self) -> list[str]:
        return list(self.raw.get("allowed_tools", []))

    @property
    def rubric(self) -> list[dict[str, Any]]:
        return list(self.raw.get("evaluation_rubric", []))

    @property
    def execution_rules(self) -> list[str]:
        return list(self.raw.get("execution_rules", []))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_scenario(path: str | Path) -> Scenario:
    return Scenario(raw=load_yaml(path))
