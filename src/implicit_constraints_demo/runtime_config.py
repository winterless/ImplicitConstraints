from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .llm_client import (
    DEFAULT_ALIYUN_API_KEY_FILE,
    DEFAULT_ALIYUN_BASE_URL,
    DEFAULT_ALIYUN_MODEL,
)
from .schemas import load_yaml

DEFAULT_RUNTIME_CONFIG_PATH = "llm_config.yaml"

AGENT_MODES = {"llm", "heuristic"}
WORLD_MODES = {"llm", "mock"}
EVALUATOR_MODES = {"llm", "deterministic"}


@dataclass(slots=True)
class RoleRuntimeConfig:
    mode: str
    provider: str = "openai_compatible"
    base_url: str = DEFAULT_ALIYUN_BASE_URL
    model: str = DEFAULT_ALIYUN_MODEL
    api_key_env: str = "DASHSCOPE_API_KEY,ALIYUN_API_KEY"
    api_key_file: str = DEFAULT_ALIYUN_API_KEY_FILE
    require_api_key: bool = True
    timeout_s: int = 120
    retries: int = 2
    temperature: float = 0.0


@dataclass(slots=True)
class RuntimeConfig:
    agent: RoleRuntimeConfig
    world: RoleRuntimeConfig
    evaluator: RoleRuntimeConfig


def default_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        agent=RoleRuntimeConfig(mode="llm"),
        world=RoleRuntimeConfig(mode="llm"),
        evaluator=RoleRuntimeConfig(mode="llm"),
    )


def load_runtime_config(path: str | Path = DEFAULT_RUNTIME_CONFIG_PATH) -> RuntimeConfig:
    config_path = Path(path)
    raw = load_yaml(config_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Runtime config must be a mapping: {config_path}")

    defaults = default_runtime_config()
    return RuntimeConfig(
        agent=_load_role_config(raw, "agent", defaults.agent, AGENT_MODES),
        world=_load_role_config(raw, "world", defaults.world, WORLD_MODES),
        evaluator=_load_role_config(raw, "evaluator", defaults.evaluator, EVALUATOR_MODES),
    )


def override_runtime_config(
    config: RuntimeConfig,
    *,
    agent_mode: str | None = None,
    world_mode: str | None = None,
    evaluator_mode: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    api_key_file: str | None = None,
    require_api_key: bool | None = None,
    timeout_s: int | None = None,
    retries: int | None = None,
    temperature: float | None = None,
) -> RuntimeConfig:
    return RuntimeConfig(
        agent=_override_role_config(
            config.agent,
            mode=agent_mode,
            allowed_modes=AGENT_MODES,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            api_key_file=api_key_file,
            require_api_key=require_api_key,
            timeout_s=timeout_s,
            retries=retries,
            temperature=temperature,
        ),
        world=_override_role_config(
            config.world,
            mode=world_mode,
            allowed_modes=WORLD_MODES,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            api_key_file=api_key_file,
            require_api_key=require_api_key,
            timeout_s=timeout_s,
            retries=retries,
            temperature=temperature,
        ),
        evaluator=_override_role_config(
            config.evaluator,
            mode=evaluator_mode,
            allowed_modes=EVALUATOR_MODES,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            api_key_file=api_key_file,
            require_api_key=require_api_key,
            timeout_s=timeout_s,
            retries=retries,
            temperature=temperature,
        ),
    )


def _load_role_config(
    raw: dict[str, object],
    key: str,
    defaults: RoleRuntimeConfig,
    allowed_modes: set[str],
) -> RoleRuntimeConfig:
    section = raw.get(key, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")

    config = RoleRuntimeConfig(
        mode=str(section.get("mode", defaults.mode)),
        provider=str(section.get("provider", defaults.provider)),
        base_url=str(section.get("base_url", defaults.base_url)),
        model=str(section.get("model", defaults.model)),
        api_key_env=str(section.get("api_key_env", defaults.api_key_env)),
        api_key_file=str(section.get("api_key_file", defaults.api_key_file)),
        require_api_key=bool(section.get("require_api_key", defaults.require_api_key)),
        timeout_s=int(section.get("timeout_s", defaults.timeout_s)),
        retries=int(section.get("retries", defaults.retries)),
        temperature=float(section.get("temperature", defaults.temperature)),
    )
    if config.mode not in allowed_modes:
        raise ValueError(
            f"Unsupported mode '{config.mode}' for section '{key}'. "
            f"Allowed values: {sorted(allowed_modes)}"
        )
    return config


def _override_role_config(
    config: RoleRuntimeConfig,
    *,
    mode: str | None,
    allowed_modes: set[str],
    base_url: str | None,
    model: str | None,
    api_key_env: str | None,
    api_key_file: str | None,
    require_api_key: bool | None,
    timeout_s: int | None,
    retries: int | None,
    temperature: float | None,
) -> RoleRuntimeConfig:
    updated = replace(
        config,
        base_url=base_url or config.base_url,
        model=model or config.model,
        api_key_env=api_key_env or config.api_key_env,
        api_key_file=api_key_file or config.api_key_file,
        require_api_key=config.require_api_key if require_api_key is None else require_api_key,
        timeout_s=config.timeout_s if timeout_s is None else timeout_s,
        retries=config.retries if retries is None else retries,
        temperature=config.temperature if temperature is None else temperature,
    )
    if mode is not None:
        if mode not in allowed_modes:
            raise ValueError(f"Unsupported mode override '{mode}'. Allowed values: {sorted(allowed_modes)}")
        updated = replace(updated, mode=mode)
    return updated
