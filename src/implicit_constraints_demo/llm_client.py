from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ALIYUN_MODEL = "qwen-plus"
DEFAULT_ALIYUN_API_KEY_FILE = ".secrets/alicloud_api_key.txt"


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


def resolve_aliyun_api_key(api_key: str = "", api_key_file: str = DEFAULT_ALIYUN_API_KEY_FILE) -> str:
    if api_key:
        return api_key
    env_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY") or ""
    if env_key:
        return env_key
    path = Path(api_key_file)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


class ChatCompletionClient:
    def __init__(
        self,
        base_url: str = DEFAULT_ALIYUN_BASE_URL,
        model: str = DEFAULT_ALIYUN_MODEL,
        api_key: str = "",
        api_key_file: str = DEFAULT_ALIYUN_API_KEY_FILE,
        timeout_s: int = 120,
        retries: int = 2,
    ) -> None:
        resolved_api_key = resolve_aliyun_api_key(api_key=api_key, api_key_file=api_key_file)
        if not resolved_api_key:
            raise ValueError(
                "Missing API key. Set DASHSCOPE_API_KEY / ALIYUN_API_KEY or provide .secrets/alicloud_api_key.txt."
            )
        self.base_url = base_url
        self.model = model
        self.timeout_s = timeout_s
        self.retries = retries
        self.headers = {"Authorization": f"Bearer {resolved_api_key}"}

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.0,
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
                return str(content).strip()
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_err = str(exc)
                if attempt <= self.retries:
                    time.sleep(attempt)
                    continue
                raise RuntimeError(f"Chat API failed: {last_err}") from exc
        raise RuntimeError(last_err or "Chat failed")


def extract_first_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response.")
    depth = 0
    for idx, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError("Incomplete JSON object in model response.")
