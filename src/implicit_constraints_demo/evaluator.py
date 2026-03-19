from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .schemas import Scenario


class ScenarioEvaluator:
    def evaluate(
        self,
        scenario: Scenario,
        messages: list[dict[str, Any]],
        final_state: dict[str, Any],
    ) -> dict[str, Any]:
        tool_messages = _tool_messages(messages)
        results = []
        passed_count = 0
        for item in scenario.rubric:
            passed, reasoning = self._evaluate_item(item["check"], tool_messages, final_state)
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
        tool_messages: list[dict[str, Any]],
        final_state: dict[str, Any],
    ) -> tuple[bool, str]:
        kind = check["type"]
        if kind == "action_called":
            tool = check["tool"]
            passed = any(entry["tool_key"] == tool for entry in tool_messages)
            return passed, f"Tool {tool} {'was' if passed else 'was not'} called."

        if kind == "action_before":
            first_index = _first_index(tool_messages, check["first"])
            second_index = _first_index(tool_messages, check["second"])
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


def _resolve_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        current = current[part]
    return current


def _tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == "tool"]


def _first_index(tool_messages: list[dict[str, Any]], tool_key: str) -> int | None:
    for idx, entry in enumerate(tool_messages):
        if entry["tool_key"] == tool_key:
            return idx
    return None


def _subtract_minutes(local_time: str, minutes: int) -> str:
    base = datetime.strptime(local_time, "%H:%M")
    shifted = base - timedelta(minutes=minutes)
    return shifted.strftime("%H:%M")
