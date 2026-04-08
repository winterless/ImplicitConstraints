from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

from .llm_client import ChatCompletionClient
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

        if kind == "action_argument_present":
            tool = check["tool"]
            arg_name = check["argument"]
            expected_value = check.get("value")
            for entry in execution_messages:
                if entry.get("tool_key") != tool:
                    continue
                args = entry.get("arguments")
                if not isinstance(args, dict) or arg_name not in args:
                    continue
                actual_value = args.get(arg_name)
                if expected_value is not None and actual_value != expected_value:
                    continue
                return True, (
                    f"Tool {tool} called with {arg_name}={actual_value!r}"
                    + (f" (required value {expected_value!r})" if expected_value is not None else "")
                    + "."
                )
            detail = f"Tool {tool} was not called with argument {arg_name!r}"
            if expected_value is not None:
                detail += f"={expected_value!r}"
            return False, detail + "."

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
        completion = self.client.chat_completion_json(request_messages, repair_retries=1)
        raw_response = completion.content
        parsed = completion.parsed
        normalized = _normalize_llm_evaluation(parsed)
        normalized["model_log"] = {
            "evaluator_mode": self.mode,
            "request_messages": request_messages,
            "raw_response": raw_response,
            "parsed_response": parsed,
            "repair_attempts": completion.repair_attempts,
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
        "你是一个隐式约束基准的评测器。\n"
        "你的任务是判断智能体是否成功解决了用户请求。\n"
        "【重要】评分标准**仅**来自用户消息里 payload 中的 implicit_eval_points（字符串列表）："
        "你必须逐条对照 implicit_eval_points 进行评判，每条对应一条 criterion，不得自行引入与列表无关的考点。\n"
        "若 implicit_eval_points 为空，则仅根据 user_prompt、messages 与 final_world_state 做整体判断，并在 reasoning 中说明「场景未提供 implicit_eval_points」。\n"
        "证据必须来自 messages（含 assistant 与 world 工具轨迹：tool_key、arguments、返回 data）以及 final_world_state；不得臆测未出现的工具结果。\n"
        "评测风格要严格、基于证据；只有当轨迹或最终状态能清楚支持该点时，该条 implicit_eval_points 才算通过。\n"
        "criterion 文案可与对应条目的语义一致或简要概括，passed 与 reasoning 必须使用中文。\n"
        "必须只返回一个包含 criteria、overall_passed、overall_reasoning 的 JSON 对象。\n"
        "其中 criteria 必须是一个数组，数组元素是包含 criterion、passed、reasoning 的对象；criteria 条数应与 implicit_eval_points 条数一致（除非 implicit_eval_points 为空时可为一条总评）。\n"
        "不要使用 markdown 代码块围栏。"
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
        "implicit_eval_points": scenario.raw.get("implicit_eval_points", []),
        "messages": messages,
        "final_world_state": final_state,
    }
    return (
        "请评估这次场景运行结果，并用中文填写所有解释字段。\n"
        "评分仅依据 payload 中的 implicit_eval_points 与下列轨迹、终态。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _normalize_llm_evaluation(parsed: dict[str, Any]) -> dict[str, Any]:
    raw_criteria = parsed.get("criteria", [])
    criteria: list[dict[str, Any]] = []
    if isinstance(raw_criteria, list):
        for item in raw_criteria:
            if not isinstance(item, dict):
                continue
            criterion = str(item.get("criterion", "")).strip() or "未命名标准"
            passed = bool(item.get("passed"))
            reasoning = str(item.get("reasoning", "")).strip() or "未提供理由。"
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
