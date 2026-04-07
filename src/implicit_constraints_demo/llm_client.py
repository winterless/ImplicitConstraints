from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ALIYUN_MODEL = "qwen-plus"
DEFAULT_ALIYUN_API_KEY_FILE = ".secrets/alicloud_api_key.txt"
DEFAULT_API_KEY_ENV_VARS = ("DASHSCOPE_API_KEY", "ALIYUN_API_KEY")
DEFAULT_MAX_TOKENS = 2048
INTERNAL_GATEWAY_PROVIDER = "internal_gateway"


def _api_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/v1{path}" if not base.endswith("/v1") else f"{base}{path}"


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_api_key(
    api_key: str = "",
    api_key_file: str = DEFAULT_ALIYUN_API_KEY_FILE,
    api_key_envs: tuple[str, ...] = DEFAULT_API_KEY_ENV_VARS,
) -> str:
    if api_key:
        return api_key
    for env_name in api_key_envs:
        if not env_name:
            continue
        env_key = os.getenv(env_name) or ""
        if env_key:
            return env_key
    if api_key_file:
        path = Path(api_key_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return ""


def resolve_aliyun_api_key(api_key: str = "", api_key_file: str = DEFAULT_ALIYUN_API_KEY_FILE) -> str:
    return resolve_api_key(api_key=api_key, api_key_file=api_key_file)


@dataclass(slots=True)
class ParsedChatCompletion:
    content: str
    parsed: dict[str, Any]
    repair_attempts: int = 0


class ModelResponseFormatError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        raw_response: str,
        request_messages: list[dict[str, str]],
        repair_attempts: int,
    ) -> None:
        self.raw_response = raw_response
        self.request_messages = request_messages
        self.repair_attempts = repair_attempts
        preview = _preview_text(raw_response, limit=400)
        super().__init__(f"{message} Raw response preview: {preview}")


def xxx(
    *,
    model_agent: str,
    req_data: dict[str, Any],
    sub_account_name: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    user_prompt = ""
    for message in req_data.get("messages", []):
        if str(message.get("role", "")).strip().lower() == "user":
            user_prompt = str(message.get("content", ""))
            break
    return {
        "code": 0,
        "message": "mock success",
        "data": {
            "provider": INTERNAL_GATEWAY_PROVIDER,
            "model_agent": model_agent,
            "sub_account_name": sub_account_name,
            "model": model,
            "timeout": timeout,
            "content": f"[mock_xxx] model={model}, user={user_prompt[:120]}",
            "request_data": req_data,
        },
    }


class ChatCompletionClient:
    def __init__(
        self,
        provider: str = "openai_compatible",
        base_url: str = DEFAULT_ALIYUN_BASE_URL,
        model: str = DEFAULT_ALIYUN_MODEL,
        model_agent: str = "",
        sub_account_name: str = "",
        api_key: str = "",
        api_key_file: str = DEFAULT_ALIYUN_API_KEY_FILE,
        api_key_envs: tuple[str, ...] = DEFAULT_API_KEY_ENV_VARS,
        require_api_key: bool = True,
        timeout_s: int = 120,
        retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking_budget: int | None = None,
    ) -> None:
        resolved_api_key = resolve_api_key(
            api_key=api_key,
            api_key_file=api_key_file,
            api_key_envs=api_key_envs,
        )
        normalized_provider = provider.strip().lower()
        if require_api_key and normalized_provider != INTERNAL_GATEWAY_PROVIDER and not resolved_api_key:
            raise ValueError(
                "Missing API key. Configure api_key, api_key_env, or api_key_file for this role."
            )
        self.provider = normalized_provider
        self.base_url = base_url
        self.model = model
        self.model_agent = model_agent
        self.sub_account_name = sub_account_name
        self.timeout_s = timeout_s
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.headers = {}
        self._gemini_client: genai.Client | None = None
        if resolved_api_key:
            self.headers["Authorization"] = f"Bearer {resolved_api_key}"
        if self.provider == "gemini":
            self._gemini_client = genai.Client(api_key=resolved_api_key)
        elif self.provider not in {
            "openai_compatible",
            "local_openai_compatible",
            INTERNAL_GATEWAY_PROVIDER,
        }:
            raise ValueError(
                f"Unsupported provider '{provider}'. Expected one of "
                f"'openai_compatible', 'local_openai_compatible', '{INTERNAL_GATEWAY_PROVIDER}', or 'gemini'."
            )

    def chat_completion(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> str:
        if self.provider == "gemini":
            return self._chat_completion_gemini(messages, max_tokens=max_tokens)
        if self.provider == INTERNAL_GATEWAY_PROVIDER:
            return self._chat_completion_internal_gateway(messages, max_tokens=max_tokens)
        return self._chat_completion_openai_compatible(messages, max_tokens=max_tokens)

    def _chat_completion_openai_compatible(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": self.temperature,
        }
        last_err: str | None = None
        for attempt in range(1, self.retries + 2):
            try:
                resp = _post_json(
                    _api_url(self.base_url, "/chat/completions"),
                    payload,
                    timeout_s=self.timeout_s,
                    headers=self.headers,
                )
                choice = (resp.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content", "")
                return _normalize_response_content(content)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_err = str(exc)
                if attempt <= self.retries:
                    time.sleep(attempt)
                    continue
                raise RuntimeError(f"Chat API failed: {last_err}") from exc
        raise RuntimeError(last_err or "Chat failed")

    def _chat_completion_gemini(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> str:
        if self._gemini_client is None:
            raise RuntimeError("Gemini client is not initialized.")
        contents, system_instruction = _build_gemini_request(messages)
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": max_tokens if max_tokens is not None else self.max_tokens,
            "systemInstruction": system_instruction or None,
        }
        if self.thinking_budget is not None:
            config_kwargs["thinkingConfig"] = genai_types.ThinkingConfig(
                thinkingBudget=self.thinking_budget
            )
        config = genai_types.GenerateContentConfig(
            **config_kwargs,
        )
        last_err: str | None = None
        for attempt in range(1, self.retries + 2):
            try:
                resp = self._gemini_client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return _normalize_response_content(_extract_gemini_response_text(resp))
            except Exception as exc:  # Gemini SDK raises provider-specific errors.
                last_err = str(exc)
                if attempt <= self.retries:
                    time.sleep(attempt)
                    continue
                raise RuntimeError(f"Gemini API failed: {last_err}") from exc
        raise RuntimeError(last_err or "Gemini chat failed")

    def _chat_completion_internal_gateway(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> str:
        req_data = {
            "messages": messages,
            "tools": None,
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": self.temperature,
        }
        timeout_ms = max(1, int(self.timeout_s * 1000))
        last_err: str | None = None
        for attempt in range(1, self.retries + 2):
            try:
                resp = xxx(
                    model_agent=self.model_agent,
                    req_data=req_data,
                    sub_account_name=self.sub_account_name,
                    model=self.model,
                    timeout=timeout_ms,
                )
                return _normalize_response_content(_extract_internal_gateway_response_text(resp))
            except Exception as exc:
                last_err = str(exc)
                if attempt <= self.retries:
                    time.sleep(attempt)
                    continue
                raise RuntimeError(f"Internal gateway call failed: {last_err}") from exc
        raise RuntimeError(last_err or "Internal gateway call failed")

    def chat_completion_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        repair_retries: int = 1,
    ) -> ParsedChatCompletion:
        raw_response = self.chat_completion(messages, max_tokens=max_tokens)
        try:
            return ParsedChatCompletion(
                content=raw_response,
                parsed=extract_first_json_object(raw_response),
                repair_attempts=0,
            )
        except ValueError as exc:
            last_error: ValueError = exc
            last_raw = raw_response

        for repair_idx in range(1, repair_retries + 1):
            repair_messages = list(messages) + [
                {"role": "assistant", "content": _truncate_for_repair(last_raw)},
                {
                    "role": "user",
                    "content": (
                        "你上一条回复无效，因为它没有包含一个完整的 JSON 对象。"
                        "现在请只返回一个完整的 JSON 对象。"
                        "不要包含 <think>、分析过程、markdown 代码块围栏，也不要在 JSON 前后输出任何额外文本。"
                        "如果 JSON 内有 thought、thought_process、reasoning、overall_reasoning、message 等解释性字段，请统一使用中文。"
                    ),
                },
            ]
            repaired_raw = self.chat_completion(repair_messages, max_tokens=max_tokens)
            try:
                return ParsedChatCompletion(
                    content=repaired_raw,
                    parsed=extract_first_json_object(repaired_raw),
                    repair_attempts=repair_idx,
                )
            except ValueError as exc:
                last_error = exc
                last_raw = repaired_raw

        raise ModelResponseFormatError(
            str(last_error),
            raw_response=last_raw,
            request_messages=messages,
            repair_attempts=repair_retries,
        )


def extract_first_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if not text:
        raise ValueError("No JSON object found in model response.")

    parsed = _parse_full_json_dict(text)
    if parsed is None:
        parsed = _parse_python_dict_literal(text)
    if parsed is None:
        parsed = _scan_for_json_dict(text)
    if parsed is None:
        parsed = _scan_for_python_dict(text)
    if parsed is None:
        raise ValueError("No JSON object found in model response.")
    return _unwrap_response_envelope(parsed)


def _parse_full_json_dict(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _scan_for_python_dict(text: str) -> dict[str, Any] | None:
    for idx, char in enumerate(text):
        if char != "{":
            continue
        parsed = _parse_python_dict_literal(text[idx:])
        if parsed is not None:
            return parsed
    return None


def _scan_for_json_dict(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and not text[idx + end :].strip():
            return obj

    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_python_dict_literal(text: str) -> dict[str, Any] | None:
    if not text.startswith("{") or not text.endswith("}"):
        return None
    try:
        obj = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _unwrap_response_envelope(payload: dict[str, Any], max_depth: int = 4) -> dict[str, Any]:
    current = payload
    for _ in range(max_depth):
        embedded = _extract_embedded_response_content(current)
        if embedded is None:
            return current
        if isinstance(embedded, dict):
            current = embedded
            continue
        embedded_text = _normalize_response_content(embedded)
        if not embedded_text:
            return current
        next_payload = _parse_full_json_dict(embedded_text)
        if next_payload is None:
            next_payload = _parse_python_dict_literal(embedded_text)
        if next_payload is None:
            next_payload = _scan_for_json_dict(embedded_text)
        if next_payload is None:
            next_payload = _scan_for_python_dict(embedded_text)
        if next_payload is None:
            return current
        current = next_payload
    return current


def _extract_embedded_response_content(payload: dict[str, Any]) -> Any | None:
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict) and "content" in message:
                return message["content"]
            delta = choice.get("delta")
            if isinstance(delta, dict) and "content" in delta:
                return delta["content"]

    data = payload.get("data")
    if isinstance(data, dict) and "content" in data:
        return data["content"]

    message = payload.get("message")
    if isinstance(message, dict) and "content" in message:
        return message["content"]

    if "content" in payload and any(key in payload for key in ("id", "model", "object", "role")):
        return payload["content"]

    return None


def _normalize_response_content(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_gemini_request(
    messages: list[dict[str, str]],
) -> tuple[list[genai_types.Content], str]:
    system_messages: list[str] = []
    contents: list[genai_types.Content] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_messages.append(content)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            genai_types.Content(
                role=gemini_role,
                parts=[genai_types.Part.from_text(text=content)],
            )
        )
    if not contents:
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text="")],
            )
        ]
    return contents, "\n\n".join(system_messages).strip()


def _extract_gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        content_parts = getattr(content, "parts", None) or []
        for part in content_parts:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(str(part_text))
    return "\n".join(parts).strip()


def _extract_internal_gateway_response_text(response: Any) -> str:
    parsed_response = _coerce_internal_gateway_payload(response)
    if isinstance(parsed_response, dict):
        choices = parsed_response.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict) and "content" in message:
                    return _normalize_response_content(message["content"])
                delta = choice.get("delta")
                if isinstance(delta, dict) and "content" in delta:
                    return _normalize_response_content(delta["content"])

    if isinstance(parsed_response, dict):
        data = parsed_response.get("data")
        if isinstance(data, dict):
            content = data.get("content")
            if content is not None:
                return str(content)
        content = parsed_response.get("content")
        if content is not None:
            return str(content)
    return _normalize_response_content(parsed_response)


def _coerce_internal_gateway_payload(response: Any) -> Any:
    if isinstance(response, str):
        text = response.strip()
        if not text:
            return ""
        parsed = _parse_full_json_dict(text)
        if parsed is None:
            parsed = _parse_python_dict_literal(text)
        if parsed is not None:
            return parsed
        return text
    return response


def _truncate_for_repair(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit]


def _preview_text(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
