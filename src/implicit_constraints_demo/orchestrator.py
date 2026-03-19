from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .agent import AgentMemory, BaseAgent
from .evaluator import ScenarioEvaluator
from .schemas import AgentDecision, Scenario
from .tool_registry import ToolRegistry
from .world import MockWorld


class ScenarioOrchestrator:
    def __init__(
        self,
        scenario: Scenario,
        registry: ToolRegistry,
        agent: BaseAgent,
        evaluator: ScenarioEvaluator | None = None,
    ) -> None:
        self.scenario = scenario
        self.registry = registry
        self.agent = agent
        self.evaluator = evaluator or ScenarioEvaluator()

    def run(self) -> dict[str, Any]:
        world = MockWorld(self.scenario, self.registry)
        memory = AgentMemory()
        messages: list[dict[str, Any]] = [self._user_message()]
        model_logs: list[dict[str, Any]] = []
        available_tools = self.registry.export_for_keys(self.scenario.allowed_tools)
        last_step = 0

        for step in range(1, self.scenario.max_steps + 1):
            last_step = step
            decision = self.agent.decide(
                scenario=self.scenario,
                memory=memory,
                available_tools=available_tools,
                messages=messages,
            )
            if decision.model_log is not None:
                model_logs.append({"step": step, **decision.model_log})
            messages.append(self._assistant_message(step, decision))

            if decision.task_complete:
                break
            if decision.tool_call is None:
                raise ValueError("Agent decision must include a tool call or task_complete=True")

            result = world.execute(decision.tool_call)
            self.agent.observe(memory, decision.tool_call, result)
            messages.append(self._tool_message(step, decision, result))
            messages.append(self._world_message(step, decision, result))

        final_state = world.snapshot_state()
        evaluation = self.evaluator.evaluate(self.scenario, messages, final_state)
        messages.append(self._evaluator_message(last_step + 1, evaluation))
        return {
            "scenario_id": self.scenario.scenario_id,
            "category": self.scenario.category,
            "user_prompt": self.scenario.user_prompt,
            "input_snapshot": {
                "context": self.scenario.context,
                "allowed_tools": available_tools,
            },
            "messages": messages,
            "model_logs": model_logs,
            "final_world_state": final_state,
            "evaluation": evaluation,
        }

    def _user_message(self) -> dict[str, Any]:
        return {
            "step": 0,
            "role": "user",
            "content": self.scenario.user_prompt,
        }

    def _assistant_message(self, step: int, decision: AgentDecision) -> dict[str, Any]:
        payload = {
            "step": step,
            "role": "assistant",
            "content": decision.thought,
            "task_complete": decision.task_complete,
        }
        if decision.tool_call is not None:
            payload["tool_call"] = asdict(decision.tool_call)
        return payload

    def _tool_message(
        self,
        step: int,
        decision: AgentDecision,
        result: Any,
    ) -> dict[str, Any]:
        if decision.tool_call is None:
            raise ValueError("Tool message requires a tool call.")
        return {
            "step": step,
            "role": "tool",
            "tool_key": decision.tool_call.key,
            "tool_name": decision.tool_call.tool_name,
            "success": result.success,
            "message": result.message,
            "data": result.data,
        }

    def _world_message(
        self,
        step: int,
        decision: AgentDecision,
        result: Any,
    ) -> dict[str, Any]:
        if decision.tool_call is None:
            raise ValueError("World message requires a tool call.")
        return {
            "step": step,
            "role": "world",
            "applied_from": decision.tool_call.key,
            "state_changes": result.state_changes,
        }

    def _evaluator_message(self, step: int, evaluation: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": step,
            "role": "evaluator",
            "content": "Scenario evaluation completed.",
            "evaluation_results": evaluation["evaluation_results"],
            "summary": evaluation["summary"],
        }
