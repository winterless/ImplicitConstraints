from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from typing import Any

from .llm_client import ChatCompletionClient
from .schemas import AgentDecision, Scenario, ToolCall, ToolResult


@dataclass
class AgentMemory:
    observations: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    @property
    @abstractmethod
    def mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def decide(
        self,
        scenario: Scenario,
        memory: AgentMemory,
        available_tools: list[dict[str, object]],
        messages: list[dict[str, Any]],
    ) -> AgentDecision:
        raise NotImplementedError

    def observe(self, memory: AgentMemory, call: ToolCall, result: ToolResult) -> None:
        if not result.success:
            return
        if call.tool_name == "calendar.get_next_event":
            memory.observations["next_event"] = result.data["event"]
        elif call.tool_name == "maps.get_routes":
            memory.observations["routes"] = result.data["routes"]
            memory.observations["routes_destination"] = result.data["destination"]
        elif call.tool_name == "maps.get_navigation_status":
            memory.observations["navigation"] = result.data["navigation"]
        elif call.tool_name == "maps.start_navigation":
            memory.observations["navigation_started"] = True
            memory.observations["expected_arrival_time"] = result.data["expected_arrival_time"]


class QwenPlanningAgent(BaseAgent):
    def __init__(self, client: ChatCompletionClient) -> None:
        self.client = client

    @property
    def mode(self) -> str:
        return "llm"

    def decide(
        self,
        scenario: Scenario,
        memory: AgentMemory,
        available_tools: list[dict[str, object]],
        messages: list[dict[str, Any]],
    ) -> AgentDecision:
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(
            scenario=scenario,
            memory=memory,
            available_tools=available_tools,
            messages=messages,
        )
        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        completion = self.client.chat_completion_json(request_messages, repair_retries=1)
        parsed = completion.parsed
        model_log = {
            "agent_type": "llm",
            "request_messages": request_messages,
            "raw_response": completion.content,
            "parsed_response": parsed,
            "repair_attempts": completion.repair_attempts,
        }
        thought = str(parsed.get("thought", "")).strip() or "No thought provided."
        if bool(parsed.get("task_complete")):
            return AgentDecision(
                thought=thought,
                task_complete=True,
                model_log=model_log,
            )

        tool = parsed.get("tool_call")
        if not isinstance(tool, dict):
            raise ValueError("Model response must include 'tool_call' or 'task_complete'.")
        server = str(tool.get("server", "")).strip()
        tool_name = str(tool.get("tool_name", "")).strip()
        arguments = tool.get("arguments", {})
        allowed_tool_keys = {
            (str(tool_def.get("server", "")).strip(), str(tool_def.get("tool_name", "")).strip())
            for tool_def in available_tools
        }
        if not server or not tool_name or not isinstance(arguments, dict):
            raise ValueError("Invalid tool_call returned by model.")
        if (server, tool_name) not in allowed_tool_keys:
            return AgentDecision(
                thought=(
                    f"{thought} The model did not choose a valid allowed tool, "
                    "so the run is being ended instead of crashing."
                ).strip(),
                task_complete=True,
                model_log=model_log,
            )
        return AgentDecision(
            thought=thought,
            tool_call=ToolCall(
                server=server,
                tool_name=tool_name,
                arguments=arguments,
            ),
            model_log=model_log,
        )


class HeuristicPlanningAgent(BaseAgent):
    """Local fallback agent for running the demo without a model API key."""

    @property
    def mode(self) -> str:
        return "heuristic"

    def decide(
        self,
        scenario: Scenario,
        memory: AgentMemory,
        available_tools: list[dict[str, object]],
        messages: list[dict[str, Any]],
    ) -> AgentDecision:
        del messages  # The heuristic agent only uses structured memory and scenario context.

        tool_names = {str(tool["tool_name"]) for tool in available_tools}
        next_event = memory.observations.get("next_event")
        routes = memory.observations.get("routes")
        navigation_started = bool(memory.observations.get("navigation_started"))
        destination = (
            memory.observations.get("routes_destination")
            or (next_event or {}).get("location")
            or scenario.context.get("destination_hint")
        )

        if navigation_started:
            return AgentDecision(
                thought="导航已经启动，当前任务已完成。",
                task_complete=True,
            )

        if "calendar.get_next_event" in tool_names and next_event is None:
            return AgentDecision(
                thought=(
                    "这是一个时间敏感的出行请求。先读取最近的日历事件，"
                    "确认航班时间和应提前到达的缓冲时间，再决定路线。"
                ),
                tool_call=ToolCall(
                    server="device",
                    tool_name="calendar.get_next_event",
                    arguments={},
                ),
            )

        if "maps.get_routes" in tool_names and routes is None and destination:
            return AgentDecision(
                thought=(
                    f"已知目标是 {destination}。接下来查询可用路线，"
                    "比较 ETA 后再决定是否可以准时到达。"
                ),
                tool_call=ToolCall(
                    server="device",
                    tool_name="maps.get_routes",
                    arguments={"destination": destination},
                ),
            )

        if "maps.start_navigation" in tool_names and routes and destination:
            selected_route = _select_best_route(scenario, next_event, routes)
            if selected_route is not None:
                arrival_time = _add_minutes(
                    str(scenario.context["local_time"]),
                    int(selected_route["eta_minutes"]),
                )
                rationale = (
                    f"比较路线后，选择 {selected_route['route_id']}。"
                    f"预计 {arrival_time} 到达。"
                )
                if next_event and next_event.get("start_time") and next_event.get("boarding_buffer_minutes") is not None:
                    latest_arrival = _subtract_minutes(
                        str(next_event["start_time"]),
                        int(next_event["boarding_buffer_minutes"]),
                    )
                    rationale = (
                        f"需要最晚在 {latest_arrival} 前到达。"
                        f"{selected_route['route_id']} 预计 {arrival_time} 到达，"
                        "是当前最合适的路线。"
                    )
                return AgentDecision(
                    thought=rationale,
                    tool_call=ToolCall(
                        server="device",
                        tool_name="maps.start_navigation",
                        arguments={
                            "destination": destination,
                            "route_id": str(selected_route["route_id"]),
                        },
                    ),
                )

        return AgentDecision(
            thought="已经完成可执行的导航决策，当前任务结束。",
            task_complete=True,
        )


def _build_system_prompt() -> str:
    return (
        "你是一个求解该任务的智能体，每一步只能选择一次工具调用来帮助用户。\n"
        "规则：\n"
        "1. 只能使用提供给你的工具。\n"
        "2. 当关键信息可能缺失时，优先先观察、再执行修改性操作。\n"
        "3. 你能看到的上下文可能不完整，重要约束可能隐藏在只读工具的返回里。\n"
        "4. 对涉及时间、截止时间、准时、预约、航班、会议或是否赶得上的请求，只要存在相关工具，就应先查询日程、事件或时刻信息，再决定路线或其他不可逆动作。\n"
        "5. 如果一个时间敏感的出行请求同时存在日历类工具和地图类工具，通常应先检查日历类工具。\n"
        "6. 不要默认最短距离就是最优方案；当路线或时间重要时，应比较可选项。\n"
        "7. 必须使用提供工具列表中的精确 tool_name 和 server 字符串，绝不要臆造工具名，例如 none。\n"
        "8. 不要输出 <think>、思维链、分析过程，也不要在 JSON 对象前后输出任何额外文本。\n"
        "9. 必须只返回一个 JSON 对象。\n"
        "10. `thought` 字段必须使用中文，简洁说明当前判断依据。\n"
        "11. 如果任务已经完成，返回 {\"thought\": \"...\", \"task_complete\": true}。\n"
        "12. 否则返回：\n"
        "{\n"
        '  "thought": "中文说明",\n'
        '  "task_complete": false,\n'
        '  "tool_call": {\n'
        '    "server": "device",\n'
        '    "tool_name": "tool.name",\n'
        '    "arguments": {}\n'
        "  }\n"
        "}\n"
        "不要包含 markdown 代码块围栏。"
    )


def _build_user_prompt(
    scenario: Scenario,
    memory: AgentMemory,
    available_tools: list[dict[str, object]],
    messages: list[dict[str, Any]],
) -> str:
    return (
        f"用户请求：\n{scenario.user_prompt}\n\n"
        f"世界上下文：\n{json.dumps(scenario.context, ensure_ascii=False, indent=2)}\n\n"
        f"可用工具：\n{json.dumps(_compact_tools(available_tools), ensure_ascii=False, indent=2)}\n\n"
        f"已观察到的记忆：\n{json.dumps(memory.observations, ensure_ascii=False, indent=2)}\n\n"
        f"最近执行轨迹：\n{json.dumps(_compact_messages(messages), ensure_ascii=False, indent=2)}\n\n"
        "请选出当前最合适的下一步，只返回一个 JSON 对象。"
    )


def _compact_tools(available_tools: list[dict[str, object]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for tool in available_tools:
        input_schema = tool.get("input_schema", {})
        required_args: list[str] = []
        if isinstance(input_schema, dict):
            required = input_schema.get("required", [])
            if isinstance(required, list):
                required_args = [str(item) for item in required]
        compact.append(
            {
                "server": str(tool.get("server", "")).strip(),
                "tool_name": str(tool.get("tool_name", "")).strip(),
                "description": str(tool.get("description", "")).strip(),
                "read_only": bool(tool.get("read_only", False)),
                "required_arguments": required_args,
            }
        )
    return compact


def _compact_messages(messages: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for message in messages[-limit:]:
        entry = {"role": message.get("role")}
        if message.get("role") == "world":
            entry["tool_key"] = message.get("tool_key")
            entry["success"] = message.get("success")
            entry["message"] = message.get("message")
            if "data" in message:
                entry["data"] = message.get("data")
        elif "content" in message:
            entry["content"] = message.get("content")
        compact.append(entry)
    return compact


def _select_best_route(
    scenario: Scenario,
    next_event: dict[str, Any] | None,
    routes: Any,
) -> dict[str, Any] | None:
    if not isinstance(routes, list) or not routes:
        return None

    deadline: str | None = None
    if next_event and next_event.get("start_time") and next_event.get("boarding_buffer_minutes") is not None:
        deadline = _subtract_minutes(
            str(next_event["start_time"]),
            int(next_event["boarding_buffer_minutes"]),
        )

    viable_routes: list[dict[str, Any]] = []
    for route in routes:
        eta_minutes = route.get("eta_minutes")
        if not isinstance(eta_minutes, int):
            continue
        arrival_time = _add_minutes(str(scenario.context["local_time"]), eta_minutes)
        enriched_route = dict(route)
        enriched_route["_arrival_time"] = arrival_time
        if deadline is None or arrival_time <= deadline:
            viable_routes.append(enriched_route)

    candidates = viable_routes if viable_routes else [dict(route) for route in routes]
    return min(
        candidates,
        key=lambda route: int(route.get("eta_minutes", 10**9)),
    )


def _add_minutes(local_time: str, minutes: int) -> str:
    base = datetime.strptime(local_time, "%H:%M")
    shifted = base + timedelta(minutes=minutes)
    return shifted.strftime("%H:%M")


def _subtract_minutes(local_time: str, minutes: int) -> str:
    base = datetime.strptime(local_time, "%H:%M")
    shifted = base - timedelta(minutes=minutes)
    return shifted.strftime("%H:%M")
