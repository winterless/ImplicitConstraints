from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

from .schemas import Scenario, ToolCall, ToolResult
from .tool_registry import ToolRegistry


class MockWorld:
    """Deterministic world executor for MCP-style mock tools."""

    def __init__(self, scenario: Scenario, registry: ToolRegistry) -> None:
        self.scenario = scenario
        self.registry = registry
        self.state = deepcopy(scenario.state)

    def execute(self, call: ToolCall) -> ToolResult:
        if call.key not in self.scenario.allowed_tools:
            raise ValueError(f"Tool not allowed by scenario: {call.key}")

        descriptor = self.registry.get(call.key)
        self._validate_arguments(descriptor.input_schema, call.arguments)

        handler_name = call.tool_name.replace(".", "_")
        handler = getattr(self, f"_handle_{handler_name}", None)
        if handler is None:
            raise NotImplementedError(f"No handler implemented for {call.key}")
        return handler(call)

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

    def _handle_calendar_get_next_event(self, call: ToolCall) -> ToolResult:
        event = deepcopy(self.state["calendar"]["next_event"])
        return ToolResult(
            success=True,
            message="Fetched next calendar event.",
            data={"event": event},
            state_changes={},
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
            )
        routes = deepcopy(routes_by_destination[matched_key])
        return ToolResult(
            success=True,
            message=f"Fetched {len(routes)} route options.",
            data={"destination": matched_key, "routes": routes},
            state_changes={},
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
        )

    def _handle_maps_get_navigation_status(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            success=True,
            message="Fetched navigation status.",
            data={"navigation": deepcopy(self.state["navigation"])},
            state_changes={},
        )

    def _find_route(self, destination: str, route_id: str) -> dict[str, Any] | None:
        routes = self.state["maps"]["routes_by_destination"].get(destination, [])
        for route in routes:
            if route["route_id"] == route_id:
                return route
        return None


def _add_minutes(local_time: str, minutes: int) -> str:
    base = datetime.strptime(local_time, "%H:%M")
    shifted = base + timedelta(minutes=minutes)
    return shifted.strftime("%H:%M")
