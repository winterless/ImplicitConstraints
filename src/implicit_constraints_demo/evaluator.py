from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

from .llm_client import ChatCompletionClient, extract_first_json_object
from .schemas import Scenario


class BaseScenarioEvaluator(ABC):
    @property
    @abstractmethod
    def mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        scenario: Scenario,
        messages: list[dict[str, Any]],
        final_state: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class DeterministicScenarioEvaluator(BaseScenarioEvaluator):
    @property
    def mode(self) -> str:
        return "deterministic"

    def evaluate(
        self,
        scenario: Scenario,
        messages: list[dict[str, Any]],
        final_state: dict[str, Any],
    ) -> dict[str, Any]:
        execution_messages = _execution_messages(messages)
        results = []
        passed_count = 0
        for item in scenario.rubric:
            passed, reasoning = self._evaluate_item(item["check"], execution_messages, final_state)
            if passed:
                passed_count += 1
            results.append(
                {
                    "criterion": item["criterion"],
                    "passed": passed,
                    "reasoning": reasoning,
                }
            )

        total = len(results)
        return {
            "evaluation_results": results,
            "summary": {
                "passed_all": passed_count == total,
                "passed_count": passed_count,
                "total_count": total,
                "scenario_pass_rate": 1.0 if passed_count == total else 0.0,
                "normalized_scenario_score": round(passed_count / total, 3) if total else 0.0,
            },
        }

    def _evaluate_item(
        self,
        check: dict[str, Any],
        execution_messages: list[dict[str, Any]],
        final_state: dict[str, Any],
    ) -> tuple[bool, str]:
        kind = check["type"]
        if kind == "action_called":
            tool = check["tool"]
            passed = any(entry["tool_key"] == tool for entry in execution_messages)
            return passed, f"Tool {tool} {'was' if passed else 'was not'} called."

        if kind == "action_before":
            first_index = _first_index(execution_messages, check["first"])
            second_index = _first_index(execution_messages, check["second"])
            passed = first_index is not None and second_index is not None and first_index < second_index
            return passed, (
                f"First tool index={first_index}, second tool index={second_index}; "
                f"ordering {'ok' if passed else 'failed'}."
            )

        if kind == "state_eq":
            actual = _resolve_path(final_state, check["path"])
            expected = check["value"]
            passed = actual == expected
            return passed, f"Expected {check['path']}={expected!r}, got {actual!r}."

        if kind == "arrival_before_event_buffer":
            arrival = _resolve_path(final_state, check["arrival_path"])
            event_start = _resolve_path(final_state, check["event_path"])
            buffer_minutes = int(_resolve_path(final_state, check["buffer_path"]))
            required_arrival = _subtract_minutes(event_start, buffer_minutes)
            passed = arrival is not None and arrival <= required_arrival
            return passed, (
                f"Expected arrival <= {required_arrival}, got {arrival!r}."
            )

        raise ValueError(f"Unsupported rubric check type: {kind}")


class LLMScenarioEvaluator(BaseScenarioEvaluator):
    def __init__(self, client: ChatCompletionClient) -> None:
        self.client = client

    @property
    def mode(self) -> str:
        return "llm"

    def evaluate(
        self,
        scenario: Scenario,
        messages: list[dict[str, Any]],
        final_state: dict[str, Any],
    ) -> dict[str, Any]:
        system_prompt = _build_evaluator_system_prompt()
        user_prompt = _build_evaluator_user_prompt(
            scenario=scenario,
            messages=messages,
            final_state=final_state,
        )
        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw_response = self.client.chat_completion(request_messages)
        parsed = extract_first_json_object(raw_response)
        normalized = _normalize_llm_evaluation(parsed)
        normalized["model_log"] = {
            "evaluator_mode": self.mode,
            "request_messages": request_messages,
            "raw_response": raw_response,
            "parsed_response": parsed,
        }
        return normalized


ScenarioEvaluator = DeterministicScenarioEvaluator


def build_evaluator(
    mode: str,
    client: ChatCompletionClient | None = None,
) -> BaseScenarioEvaluator:
    if mode == "deterministic":
        return DeterministicScenarioEvaluator()
    if mode == "llm":
        if client is None:
            raise ValueError("Evaluator mode 'llm' requires an API-backed chat client")
        return LLMScenarioEvaluator(client=client)
    raise ValueError(f"Unsupported evaluator mode: {mode}")


def _resolve_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        current = current[part]
    return current


def _execution_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == "world" and "tool_key" in message]


def _first_index(execution_messages: list[dict[str, Any]], tool_key: str) -> int | None:
    for idx, entry in enumerate(execution_messages):
        if entry["tool_key"] == tool_key:
            return idx
    return None


def _subtract_minutes(local_time: str, minutes: int) -> str:
    base = datetime.strptime(local_time, "%H:%M")
    shifted = base - timedelta(minutes=minutes)
    return shifted.strftime("%H:%M")


def _build_evaluator_system_prompt() -> str:
    return (
        "You are the evaluator for an implicit-intelligence benchmark.\n"
        "Judge whether the agent successfully solved the user's request.\n"
        "Use the scenario metadata, hidden constraints, execution rules, tool trace, and final world state.\n"
        "Be strict, mechanical, and evidence-based.\n"
        "If an explicit evaluation rubric is present, use it directly.\n"
        "If the rubric is empty, derive 3 to 6 concrete criteria from the hidden constraints, implicit reason, "
        "suggested plan, and execution rules.\n"
        "A criterion should only pass when the trace or final state clearly supports it.\n"
        "Return exactly one JSON object with keys criteria, overall_passed, overall_reasoning.\n"
        "criteria must be an array of objects with keys criterion, passed, reasoning.\n"
        "Do not use markdown fences."
    )


def _build_evaluator_user_prompt(
    scenario: Scenario,
    messages: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> str:
    payload = {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "user_prompt": scenario.user_prompt,
        "implicit_reason": scenario.raw.get("implicit_reason"),
        "hidden_constraints": scenario.raw.get("hidden_constraints", []),
        "suggested_plan": scenario.raw.get("suggested_plan", []),
        "execution_rules": scenario.execution_rules,
        "evaluation_rubric": scenario.rubric,
        "messages": messages,
        "final_world_state": final_state,
    }
    return (
        "Evaluate this scenario run.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _normalize_llm_evaluation(parsed: dict[str, Any]) -> dict[str, Any]:
    raw_criteria = parsed.get("criteria", [])
    criteria: list[dict[str, Any]] = []
    if isinstance(raw_criteria, list):
        for item in raw_criteria:
            if not isinstance(item, dict):
                continue
            criterion = str(item.get("criterion", "")).strip() or "Unnamed criterion"
            passed = bool(item.get("passed"))
            reasoning = str(item.get("reasoning", "")).strip() or "No reasoning provided."
            criteria.append(
                {
                    "criterion": criterion,
                    "passed": passed,
                    "reasoning": reasoning,
                }
            )
    if not criteria:
        criteria = [
            {
                "criterion": "Overall task success",
                "passed": bool(parsed.get("overall_passed")),
                "reasoning": str(parsed.get("overall_reasoning", "")).strip() or "No reasoning provided.",
            }
        ]

    passed_count = sum(1 for item in criteria if item["passed"])
    total = len(criteria)
    overall_passed = bool(parsed.get("overall_passed"))
    if "overall_passed" not in parsed:
        overall_passed = passed_count == total
    overall_reasoning = str(parsed.get("overall_reasoning", "")).strip()

    return {
        "evaluation_results": criteria,
        "summary": {
            "passed_all": overall_passed,
            "passed_count": passed_count,
            "total_count": total,
            "scenario_pass_rate": 1.0 if overall_passed else 0.0,
            "normalized_scenario_score": round(passed_count / total, 3) if total else 0.0,
            "overall_reasoning": overall_reasoning,
        },
    }
