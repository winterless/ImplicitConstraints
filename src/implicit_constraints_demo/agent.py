from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from typing import Any

from .llm_client import ChatCompletionClient, extract_first_json_object
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
        content = self.client.chat_completion(request_messages)
        parsed = extract_first_json_object(content)
        model_log = {
            "agent_type": "llm",
            "request_messages": request_messages,
            "raw_response": content,
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
        if not server or not tool_name or not isinstance(arguments, dict):
            raise ValueError("Invalid tool_call returned by model.")
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
        "You are an agent that must help the user by selecting one tool call at a time.\n"
        "Rules:\n"
        "1. Only use the tools shown to you.\n"
        "2. Prefer observation before mutation when important context may be missing.\n"
        "3. The visible context may be incomplete. Important constraints can be hidden in read-only tools.\n"
        "4. For requests involving time, deadlines, punctuality, appointments, flights, meetings, or arriving on time, "
        "query schedule or event tools before you commit to a route or other irreversible action whenever such tools exist.\n"
        "5. If both a calendar-like tool and a maps-like tool exist for a timing-sensitive travel request, the calendar-like tool should usually be checked first.\n"
        "6. Avoid assuming that the shortest distance is best; compare available options when route or timing matters.\n"
        "7. Respond with exactly one JSON object.\n"
        "8. If the task is complete, return {\"thought\": \"...\", \"task_complete\": true}.\n"
        "9. Otherwise return:\n"
        "{\n"
        '  "thought": "...",\n'
        '  "task_complete": false,\n'
        '  "tool_call": {\n'
        '    "server": "device",\n'
        '    "tool_name": "tool.name",\n'
        '    "arguments": {}\n'
        "  }\n"
        "}\n"
        "Do not include markdown fences."
    )


def _build_user_prompt(
    scenario: Scenario,
    memory: AgentMemory,
    available_tools: list[dict[str, object]],
    messages: list[dict[str, Any]],
) -> str:
    return (
        f"User request:\n{scenario.user_prompt}\n\n"
        f"World context:\n{json.dumps(scenario.context, ensure_ascii=False, indent=2)}\n\n"
        f"Available tools:\n{json.dumps(available_tools, ensure_ascii=False, indent=2)}\n\n"
        f"Observed memory:\n{json.dumps(memory.observations, ensure_ascii=False, indent=2)}\n\n"
        f"Conversation so far:\n{json.dumps(messages, ensure_ascii=False, indent=2)}\n\n"
        "Choose the single best next step."
    )


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
