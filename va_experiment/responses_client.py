"""OpenAI Responses API client: function tools together with reasoning.

gpt-6-luna rejects function tools with reasoning in Chat Completions (HTTP 400,
"use /v1/responses or set reasoning_effort to 'none'"). This client keeps the
pipeline's stateless contract: every call sends the full history, with
``store: false``. Each assistant message keeps the provider's verbatim output
items (including encrypted reasoning) in ``provider_output`` and replays them,
so later turns and report forks see the same context the model produced, and
the frozen execution alone reproduces every request. Raw HTTP is used because
the pinned SDK predates ``responses``; no SDK is imported here.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
import random
from typing import TYPE_CHECKING, Any

import httpx

from va_experiment.model_client import (ModelCallCancelledError, ModelCallError, ModelCapabilityCallError,
                                        _is_context_limit, _redact)
from va_experiment.models import ModelConfig, ModelReply

if TYPE_CHECKING:
    from utils.functions_runtime import FunctionsRuntime
    from utils.types import ChatMessage

ENDPOINT = "https://api.openai.com/v1/responses"
MAX_ATTEMPTS = 10
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
# Tool-call arguments that are not valid JSON are passed through under this key,
# so the executor reports "Unexpected tool arguments" to the model instead of
# ending the run (the legacy Chat path raised and ended the execution).
INVALID_ARGS_KEY = "__invalid_json_arguments__"


class _ProviderHTTPError(Exception):
    """Carries the provider error body in the shape model_client helpers expect."""

    def __init__(self, status: int, body: Any, request_id: str | None):
        super().__init__(f"HTTP {status}: {json.dumps(body)[:2000] if not isinstance(body, str) else body[:2000]}")
        self.status, self.body, self.request_id = status, body, request_id


def _call_args(call: Any) -> tuple[str, str | None, dict]:
    if isinstance(call, dict):
        return call["function"], call.get("id"), call.get("args") or {}
    return call.function, call.id, dict(call.args)


def messages_to_input(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert pipeline messages to Responses input items (system role kept as system)."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role in {"system", "user"}:
            items.append({"role": role, "content": message["content"]})
        elif role == "assistant":
            if message.get("provider_output"):
                items.extend(deepcopy(message["provider_output"]))
                continue
            # History produced elsewhere (e.g. a synthesized empty reply): rebuild minimally.
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for call in message.get("tool_calls") or []:
                name, call_id, args = _call_args(call)
                items.append({"type": "function_call", "call_id": call_id, "name": name,
                              "arguments": json.dumps(args, ensure_ascii=False)})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": message["content"]})
        else:
            raise ValueError(f"Unsupported message role for Responses API: {role}")
    return items


def tools_payload(runtime: FunctionsRuntime) -> list[dict[str, Any]]:
    return [{"type": "function", "name": f.name, "description": f.description,
             "parameters": f.parameters.model_json_schema(), "strict": False}
            for f in runtime.functions.values()]


def parse_output(response: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return (assistant message, finish_reason) from a Responses payload."""
    from utils.functions_runtime import FunctionCall

    items = response.get("output") or []
    texts, refusals, calls = [], [], []
    for item in items:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "refusal":
                    refusals.append(part.get("refusal", ""))
        elif item.get("type") == "function_call":
            raw = item.get("arguments") or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    raise ValueError("arguments are not a JSON object")
            except ValueError:
                args = {INVALID_ARGS_KEY: raw}
            calls.append(FunctionCall(function=item["name"], args=args, id=item.get("call_id")))
    content = "".join(texts) or ("\n".join(refusals) if refusals else "")
    message = {"role": "assistant", "content": content, "tool_calls": calls or None,
               "provider_output": deepcopy(items)}
    status = response.get("status")
    if status == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        finish = {"max_output_tokens": "length", "content_filter": "content_filter"}.get(reason, f"incomplete:{reason}")
    elif refusals:
        finish = "refusal"
    elif calls:
        finish = "tool_calls"
    elif status == "completed":
        finish = "stop"
    else:
        finish = f"status:{status}"
    return message, finish


class ResponsesModelClient:
    def __init__(self, config: ModelConfig, *, transport: httpx.AsyncBaseTransport | None = None,
                 sleep=asyncio.sleep):
        if config.provider != "openai" or config.reasoning_effort is None:
            raise ValueError("ResponsesModelClient requires provider openai and an explicit reasoning_effort")
        self.config = config.model_copy(deep=True)
        self.calls: list[dict[str, Any]] = []
        self._credential = os.environ.get(config.api_key_env)
        if not self._credential:
            raise ValueError(f"Credential environment variable {config.api_key_env} is empty or missing")
        self._transport = transport
        self._sleep = sleep

    def request_body(self, messages: list[ChatMessage], runtime: FunctionsRuntime) -> dict[str, Any]:
        effort = self.config.reasoning_effort
        body: dict[str, Any] = {
            "model": self.config.model_id, "input": messages_to_input(messages),
            "max_output_tokens": self.config.max_output_tokens, "store": False,
            "reasoning": {"effort": effort},
        }
        if effort != "none":
            body["reasoning"]["summary"] = "auto"
            body["include"] = ["reasoning.encrypted_content"]
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        tools = tools_payload(runtime)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    async def _post(self, client: httpx.AsyncClient, body: dict, calls: list, purpose: str) -> tuple[dict, str]:
        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            record = {"operation": "responses.create", "purpose": purpose, "attempt_index": attempt,
                      "request": _redact(deepcopy(body), self._credential), "response": None, "error": None}
            calls.append(record)
            try:
                http = await client.post(ENDPOINT, json=body)
            except httpx.TransportError as exc:
                record["error"] = _redact({"type": type(exc).__name__, "message": str(exc), "body": None},
                                          self._credential)
                last = exc
            else:
                request_id = http.headers.get("x-request-id")
                try:
                    payload = http.json()
                except ValueError:
                    payload = http.text
                if http.status_code == 200 and isinstance(payload, dict):
                    record["response"] = _redact(payload, self._credential)
                    return payload, request_id or payload.get("id", "")
                error = _ProviderHTTPError(http.status_code, payload, request_id)
                record["error"] = _redact({"type": "HTTPStatusError", "status": http.status_code,
                                           "message": str(error), "body": payload}, self._credential)
                if http.status_code not in RETRY_STATUS:
                    raise error
                last = error
            if attempt < MAX_ATTEMPTS:
                await self._sleep(min(120.0, random.uniform(0, 10 * 2 ** (attempt - 1))))
        raise last  # type: ignore[misc]

    async def complete(self, messages: list[ChatMessage], runtime: FunctionsRuntime, *, purpose: str) -> ModelReply:
        calls: list[dict[str, Any]] = []
        body = self.request_body(deepcopy(messages), runtime)
        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._credential}"}, timeout=600, transport=self._transport,
            ) as client:
                payload, request_id = await self._post(client, body, calls, purpose)
            message, finish = parse_output(payload)
            if finish.startswith("status:"):
                raise ModelCallError(f"Responses API returned {finish}",
                                     _redact({"provider_response": payload, "calls": calls}, self._credential))
            return ModelReply(
                message=message,
                raw_response=_redact({"provider_response": payload, "calls": calls}, self._credential),
                finish_reason=finish, usage=deepcopy(payload.get("usage") or {}), request_id=request_id or "",
            )
        except ModelCallError:
            raise
        except asyncio.CancelledError as exc:
            envelope = _redact({"provider_response": None, "calls": calls}, self._credential)
            raise ModelCallCancelledError(_redact(str(exc), self._credential), envelope) from None
        except Exception as exc:
            envelope = _redact({"provider_response": None, "calls": calls}, self._credential)
            if _is_context_limit(exc):
                return ModelReply(message={"role": "assistant", "content": "", "tool_calls": None},
                                  raw_response=envelope, finish_reason="context_limit", usage={},
                                  request_id=getattr(exc, "request_id", None) or "")
            body = getattr(exc, "body", None)
            error = body.get("error") if isinstance(body, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            param = error.get("param") if isinstance(error, dict) else None
            if code in {"unsupported_value", "unsupported_parameter", "model_not_found"} or (
                    getattr(exc, "status", None) == 400 and param in {"reasoning", "reasoning.effort", "temperature", "tools"}):
                raise ModelCapabilityCallError(_redact(str(exc), self._credential), envelope) from None
            raise ModelCallError(_redact(str(exc), self._credential), envelope) from None
        finally:
            self.calls.extend(deepcopy(calls))
