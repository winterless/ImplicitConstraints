from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


DEFAULT_MODEL = "gemini-3-pro-preview"


def build_request_data(
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    tools: Any = None,
    max_tokens: int = 10000,
    temperature: float = 0.0,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": tools,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def get_model_response(
    system: str,
    user: str,
    *,
    invoke_fn: Callable[..., Any],
    model_agent: str,
    sub_account_name: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 1500,
    log_path: str = "model_requests.jsonl",
    tools: Any = None,
    max_tokens: int = 10000,
    temperature: float = 0.0,
) -> Any:
    req_data = build_request_data(
        system=system,
        user=user,
        model=model,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = invoke_fn(
        model_agent=model_agent,
        req_data=req_data,
        sub_account_name=sub_account_name,
        model=model,
        timeout=timeout,
    )

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fw:
        fw.write(
            json.dumps(
                {
                    "request_data": req_data,
                    "output": result,
                },
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )

    return result
