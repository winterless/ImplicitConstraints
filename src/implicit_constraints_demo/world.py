from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import json
from typing import Any

from .llm_client import ChatCompletionClient, extract_first_json_object
from .schemas import Scenario, ToolCall, ToolDescriptor, ToolResult
from .tool_registry import ToolRegistry


class BaseWorld:
    def __init__(self, scenario: Scenario, registry: ToolRegistry) -> None:
        self.scenario = scenario
        self.registry = registry
        self.state = deepcopy(scenario.state)

    @property
    def mode(self) -> str:
        return "base"

    def execute(self, call: ToolCall) -> ToolResult:
        if call.key not in self.scenario.allowed_tools:
            raise ValueError(f"Tool not allowed by scenario: {call.key}")

        descriptor = self.registry.get(call.key)
        self._validate_arguments(descriptor.input_schema, call.arguments)
        return self._execute_validated(call, descriptor)

    def _execute_validated(self, call: ToolCall, descriptor: ToolDescriptor) -> ToolResult:
        raise NotImplementedError

    def snapshot_state(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def _validate_arguments(self, schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for name in required:
            if name not in arguments:
                raise ValueError(f"Missing required argument: {name}")
        for name, value in arguments.items():
            declared = properties.get(name)
            if declared is None:
                raise ValueError(f"Unexpected argument: {name}")
            expected = declared.get("type")
            if expected == "string" and not isinstance(value, str):
                raise ValueError(f"Argument {name} must be a string")
            if expected == "boolean" and not isinstance(value, bool):
                raise ValueError(f"Argument {name} must be a boolean")
            if expected == "integer" and not isinstance(value, int):
                raise ValueError(f"Argument {name} must be an integer")
            if expected == "number" and not isinstance(value, (int, float)):
                raise ValueError(f"Argument {name} must be a number")


class MockWorld(BaseWorld):
    """Deterministic mock executor for MCP-style tool calls."""

    @property
    def mode(self) -> str:
        return "mock"

    def _execute_validated(self, call: ToolCall, descriptor: ToolDescriptor) -> ToolResult:
        handler_name = call.tool_name.replace(".", "_")
        handler = getattr(self, f"_handle_{handler_name}", None)
        if handler is None:
            return self._handle_generic_tool(call, descriptor)
        return handler(call)

    def _handle_calendar_get_next_event(self, call: ToolCall) -> ToolResult:
        del call
        event = deepcopy(self.state["calendar"]["next_event"])
        return ToolResult(
            success=True,
            message="Fetched next calendar event.",
            data={"event": event},
            state_changes={},
            model_log=None,
        )

    def _handle_maps_get_routes(self, call: ToolCall) -> ToolResult:
        destination = call.arguments["destination"]
        requested = destination.strip().lower()
        routes_by_destination = self.state["maps"]["routes_by_destination"]
        matched_key = None
        for key in routes_by_destination:
            if key.strip().lower() == requested:
                matched_key = key
                break
        if matched_key is None:
            return ToolResult(
                success=False,
                message=f"No routes available for destination: {destination}",
                data={},
                state_changes={},
                model_log=None,
            )
        routes = deepcopy(routes_by_destination[matched_key])
        return ToolResult(
            success=True,
            message=f"Fetched {len(routes)} route options.",
            data={"destination": matched_key, "routes": routes},
            state_changes={},
            model_log=None,
        )

    def _handle_maps_start_navigation(self, call: ToolCall) -> ToolResult:
        destination = call.arguments["destination"]
        route_id = call.arguments["route_id"]
        route = self._find_route(destination, route_id)
        if route is None:
            return ToolResult(
                success=False,
                message=f"Unknown route_id '{route_id}' for destination '{destination}'.",
                data={},
                state_changes={},
                model_log=None,
            )

        expected_arrival = _add_minutes(self.scenario.context["local_time"], route["eta_minutes"])
        self.state["navigation"] = {
            "active_destination": destination,
            "active_route_id": route_id,
            "guidance_started": True,
            "expected_arrival_time": expected_arrival,
            "selected_route_summary": {
                "label": route["label"],
                "distance_km": route["distance_km"],
                "eta_minutes": route["eta_minutes"],
                "traffic_level": route["traffic_level"],
            },
        }
        return ToolResult(
            success=True,
            message=f"Navigation started to {destination}. ETA {expected_arrival}.",
            data={"expected_arrival_time": expected_arrival},
            state_changes={"navigation": deepcopy(self.state["navigation"])},
            model_log=None,
        )

    def _handle_maps_get_navigation_status(self, call: ToolCall) -> ToolResult:
        del call
        return ToolResult(
            success=True,
            message="Fetched navigation status.",
            data={"navigation": deepcopy(self.state["navigation"])},
            state_changes={},
            model_log=None,
        )

    def _handle_generic_tool(self, call: ToolCall, descriptor: ToolDescriptor) -> ToolResult:
        if descriptor.read_only:
            data = self._generic_read_data(call.tool_name)
            return ToolResult(
                success=True,
                message=f"Fetched data for {call.tool_name}.",
                data=data,
                state_changes={},
                model_log=None,
            )

        self.state.setdefault("operation_log", []).append(
            {
                "tool_name": call.tool_name,
                "arguments": deepcopy(call.arguments),
            }
        )
        configured_updates = self.state.get("tool_state_updates", {}).get(call.tool_name, {})
        if isinstance(configured_updates, dict):
            _deep_merge(self.state, deepcopy(configured_updates))
        data = self._generic_write_data(call.tool_name, descriptor.success_response_schema)
        state_changes = {
            "operation_log": deepcopy(self.state["operation_log"][-1]),
        }
        if configured_updates:
            state_changes["configured_updates"] = deepcopy(configured_updates)
        return ToolResult(
            success=True,
            message=f"Executed {call.tool_name}.",
            data=data,
            state_changes=state_changes,
            model_log=None,
        )

    def _generic_read_data(self, tool_name: str) -> dict[str, Any]:
        if tool_name == "schedule.get_next_event" and "calendar" in self.state:
            return {"event": deepcopy(self.state["calendar"]["next_event"])}
        if tool_name == "schedule.list_free_slots" and "schedule" in self.state:
            return {"slots": deepcopy(self.state["schedule"].get("free_slots", []))}

        tool_responses = self.state.get("tool_responses", {})
        response = tool_responses.get(tool_name, {})
        return deepcopy(response) if isinstance(response, dict) else {}

    def _generic_write_data(
        self,
        tool_name: str,
        success_response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        tool_responses = self.state.get("tool_responses", {})
        response = tool_responses.get(tool_name)
        if isinstance(response, dict) and response:
            return deepcopy(response)
        return _default_response_from_schema(tool_name, success_response_schema)

    def _find_route(self, destination: str, route_id: str) -> dict[str, Any] | None:
        routes = self.state["maps"]["routes_by_destination"].get(destination, [])
        for route in routes:
            if route["route_id"] == route_id:
                return route
        return None


class LLMWorldModel(BaseWorld):
    """Paper-aligned world model implemented as a constrained LLM executor."""

    def __init__(self, scenario: Scenario, registry: ToolRegistry, client: ChatCompletionClient) -> None:
        super().__init__(scenario, registry)
        self.client = client

    @property
    def mode(self) -> str:
        return "llm_world_model"

    def _execute_validated(self, call: ToolCall, descriptor: ToolDescriptor) -> ToolResult:
        system_prompt = _build_world_model_system_prompt()
        user_prompt = _build_world_model_user_prompt(
            scenario=self.scenario,
            descriptor=descriptor,
            current_state=self.state,
            tool_call=call,
        )
        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw_response = self.client.chat_completion(request_messages)
        parsed = extract_first_json_object(raw_response)
        result = _normalize_world_model_result(parsed, descriptor)
        if result.success and result.state_changes:
            _deep_merge(self.state, deepcopy(result.state_changes))
        result.model_log = {
            "world_mode": self.mode,
            "request_messages": request_messages,
            "raw_response": raw_response,
            "parsed_response": parsed,
        }
        return result


def build_world(
    scenario: Scenario,
    registry: ToolRegistry,
    mode: str,
    client: ChatCompletionClient | None = None,
) -> BaseWorld:
    if mode == "llm":
        if client is None:
            raise ValueError("world mode 'llm' requires an API-backed chat client")
        return LLMWorldModel(scenario=scenario, registry=registry, client=client)
    if mode == "mock":
        return MockWorld(scenario=scenario, registry=registry)
    raise ValueError(f"Unsupported world mode: {mode}")


def _build_world_model_system_prompt() -> str:
    return (
        "You are the World Model for an implicit-intelligence benchmark.\n"
        "You simulate the environment for exactly one tool call.\n"
        "Your role is constrained and mechanical, not creative.\n"
        "Rules:\n"
        "1. Use only the provided scenario context, execution rules, current world state, tool descriptor, and tool call.\n"
        "2. Do not help the agent, reveal hidden constraints, or give advice.\n"
        "3. If the tool descriptor includes a returns field, treat it as the primary return template. Otherwise use success_response_schema as the return shape for successful calls.\n"
        "4. Read-only operations must return state_changes as {}.\n"
        "5. State-modifying operations may return a partial state_changes object that can be deep-merged into the world state.\n"
        "6. If the tool should fail based on current state or rules, set success=false and explain why neutrally.\n"
        "7. Return exactly one JSON object with keys thought_process, success, message, data, state_changes.\n"
        "Do not use markdown fences."
    )


def _build_world_model_user_prompt(
    scenario: Scenario,
    descriptor: ToolDescriptor,
    current_state: dict[str, Any],
    tool_call: ToolCall,
) -> str:
    return (
        f"Scenario ID: {scenario.scenario_id}\n"
        f"Category: {scenario.category}\n"
        f"User prompt: {scenario.user_prompt}\n\n"
        f"Scenario context:\n{json.dumps(scenario.context, ensure_ascii=False, indent=2)}\n\n"
        f"Execution rules:\n{json.dumps(scenario.execution_rules, ensure_ascii=False, indent=2)}\n\n"
        f"Current world state:\n{json.dumps(current_state, ensure_ascii=False, indent=2)}\n\n"
        f"Tool descriptor:\n{json.dumps(_descriptor_payload(descriptor), ensure_ascii=False, indent=2)}\n\n"
        f"Tool call:\n{json.dumps({'server': tool_call.server, 'tool_name': tool_call.tool_name, 'arguments': tool_call.arguments}, ensure_ascii=False, indent=2)}\n\n"
        "Simulate the environment response for this one action."
    )


def _descriptor_payload(descriptor: ToolDescriptor) -> dict[str, Any]:
    return {
        "server": descriptor.server,
        "tool_name": descriptor.tool_name,
        "description": descriptor.description,
        "input_schema": descriptor.input_schema,
        "read_only": descriptor.read_only,
        "returns": descriptor.returns,
        "success_response_schema": descriptor.success_response_schema,
        "state_changes": descriptor.state_changes,
        "failure_conditions": descriptor.failure_conditions,
    }


def _normalize_world_model_result(
    parsed: dict[str, Any],
    descriptor: ToolDescriptor,
) -> ToolResult:
    success = bool(parsed.get("success"))
    message = str(parsed.get("message", "")).strip() or "No message returned."
    data = parsed.get("data", {})
    state_changes = parsed.get("state_changes", {})

    if not isinstance(data, dict):
        data = {}
    if not isinstance(state_changes, dict):
        state_changes = {}

    if success and not data:
        data = _default_response_from_schema(descriptor.tool_name, descriptor.success_response_schema)
    if descriptor.read_only:
        state_changes = {}
    elif not success:
        state_changes = {}

    return ToolResult(
        success=success,
        message=message,
        data=data,
        state_changes=state_changes,
        model_log=None,
    )


def _add_minutes(local_time: str, minutes: int) -> str:
    base = datetime.strptime(local_time, "%H:%M")
    shifted = base + timedelta(minutes=minutes)
    return shifted.strftime("%H:%M")


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _default_response_from_schema(
    tool_name: str,
    success_response_schema: dict[str, Any],
) -> dict[str, Any]:
    properties = success_response_schema.get("properties", {})
    data: dict[str, Any] = {}
    for key, config in properties.items():
        schema_type = config.get("type")
        if key.endswith("_id"):
            data[key] = f"{tool_name.replace('.', '_')}_001"
        elif schema_type == "boolean":
            data[key] = True
        elif schema_type == "integer":
            data[key] = 1
        elif schema_type == "number":
            data[key] = 1.0
        elif schema_type == "array":
            data[key] = []
        elif schema_type == "object":
            data[key] = {}
        else:
            data[key] = "ok"
    return data
