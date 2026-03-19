from __future__ import annotations

from pathlib import Path

from .schemas import ToolDescriptor, load_yaml


class ToolRegistry:
    def __init__(self, descriptors: dict[str, ToolDescriptor]) -> None:
        self._descriptors = descriptors

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ToolRegistry":
        descriptors: dict[str, ToolDescriptor] = {}
        for path in sorted(Path(directory).glob("*.yaml")):
            raw = load_yaml(path)
            descriptor = ToolDescriptor(
                server=raw["server"],
                tool_name=raw["tool_name"],
                description=raw["description"],
                input_schema=raw.get("input_schema", {}),
                read_only=bool(raw.get("read_only", False)),
                success_response_schema=raw.get("success_response_schema", {}),
                state_changes=list(raw.get("state_changes", [])),
                failure_conditions=list(raw.get("failure_conditions", [])),
            )
            descriptors[descriptor.key] = descriptor
        return cls(descriptors)

    def get(self, key: str) -> ToolDescriptor:
        if key not in self._descriptors:
            raise KeyError(f"Unknown tool descriptor: {key}")
        return self._descriptors[key]

    def export_for_keys(self, keys: list[str]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for key in keys:
            descriptor = self.get(key)
            result.append(
                {
                    "server": descriptor.server,
                    "tool_name": descriptor.tool_name,
                    "description": descriptor.description,
                    "input_schema": descriptor.input_schema,
                    "read_only": descriptor.read_only,
                }
            )
        return result
