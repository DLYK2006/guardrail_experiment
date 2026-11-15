"""Offline tests for the Responses API client (reasoning + function tools)."""

import asyncio
import json

import httpx
import pytest

from utils.functions_runtime import FunctionCall, FunctionsRuntime, make_function
from va_experiment.model_client import ModelCallCancelledError, ModelCallError, ModelCapabilityCallError, create_model_client
from va_experiment.models import ModelConfig
from va_experiment.responses_client import INVALID_ARGS_KEY, ResponsesModelClient, messages_to_input

SECRET = "responses-test-secret"


def get_balance(account: str) -> str:
    """Get the balance of an account.

    :param account: Account name.
    """
    return "$1"


def cfg(**overrides):
    values = dict(provider="openai", model_id="gpt-6-luna", api_key_env="VA_TEST_KEY", temperature=None,
                  max_output_tokens=16000, reasoning_effort="medium")
    values.update(overrides)
    return ModelConfig(**values)


def runtime():
    rt = FunctionsRuntime([])
    rt.register_function(make_function(get_balance))
    return rt


REASONING = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}],
             "encrypted_content": "ENC"}
CALL = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_balance",
        "arguments": '{"account": "A"}', "status": "completed"}


def payload(output, status="completed", incomplete=None):
    return {"id": "resp_1", "status": status, "incomplete_details": incomplete, "output": output,
            "usage": {"input_tokens": 5, "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 3}}}


def client_with(responses, captured):
    queue = list(responses)

    def handler(request):
        captured.append(json.loads(request.content))
        assert request.headers["authorization"] == f"Bearer {SECRET}"
        status, body = queue.pop(0)
        return httpx.Response(status, json=body, headers={"x-request-id": "req-1"})

    async def no_sleep(_):
        return None

    return ResponsesModelClient(cfg(), transport=httpx.MockTransport(handler), sleep=no_sleep)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("VA_TEST_KEY", SECRET)


def test_config_routes_reasoning_to_responses_client():
    assert isinstance(create_model_client(cfg()), ResponsesModelClient)


@pytest.mark.parametrize("bad", [dict(temperature=0.0), dict(reasoning_effort=None),
                                 dict(provider="anthropic", api_key_env="VA_TEST_KEY")])
def test_invalid_reasoning_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        cfg(**bad)


def test_reasoning_none_may_keep_temperature():
    assert cfg(reasoning_effort="none", temperature=0.0).temperature == 0.0


@pytest.mark.asyncio
async def test_tool_turn_request_shape_and_lossless_reply():
    captured = []
    client = client_with([(200, payload([REASONING, CALL]))], captured)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    reply = await client.complete(messages, runtime(), purpose="actor_execution")
    body = captured[0]
    assert body["input"] == messages  # system role is kept as system, not developer
    assert body["store"] is False and body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "temperature" not in body and body["max_output_tokens"] == 16000
    assert body["tools"][0]["name"] == "get_balance" and body["tool_choice"] == "auto"
    assert reply.finish_reason == "tool_calls" and reply.request_id == "req-1"
    call = reply.message["tool_calls"][0]
    assert (call.function, call.id, dict(call.args)) == ("get_balance", "call_1", {"account": "A"})
    assert reply.message["provider_output"] == [REASONING, CALL]
    assert reply.usage["output_tokens_details"]["reasoning_tokens"] == 3
    assert SECRET not in json.dumps(reply.raw_response)


@pytest.mark.asyncio
async def test_history_replays_provider_items_and_tool_outputs():
    captured = []
    client = client_with([(200, payload([{"type": "message", "id": "msg_1", "role": "assistant",
                                          "content": [{"type": "output_text", "text": "done"}]}]))], captured)
    assistant = {"role": "assistant", "content": "", "provider_output": [REASONING, CALL],
                 "tool_calls": [FunctionCall(function="get_balance", args={"account": "A"}, id="call_1")]}
    tool = {"role": "tool", "tool_call_id": "call_1", "content": '{"result": "$1"}', "error": None,
            "tool_call": FunctionCall(function="get_balance", args={"account": "A"}, id="call_1")}
    messages = [{"role": "user", "content": "hi"}, assistant, tool, {"role": "user", "content": "report"}]
    reply = await client.complete(messages, FunctionsRuntime([]), purpose="actor_report")
    assert captured[0]["input"] == [
        {"role": "user", "content": "hi"}, REASONING, CALL,
        {"type": "function_call_output", "call_id": "call_1", "output": '{"result": "$1"}'},
        {"role": "user", "content": "report"},
    ]
    assert "tools" not in captured[0]
    assert reply.finish_reason == "stop" and reply.message["content"] == "done"
    assert reply.message["tool_calls"] is None


def test_history_without_provider_items_is_rebuilt():
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        FunctionCall(function="get_balance", args={"account": "B"}, id="call_9")]}]
    assert messages_to_input(messages) == [
        {"type": "function_call", "call_id": "call_9", "name": "get_balance", "arguments": '{"account": "B"}'}]


@pytest.mark.asyncio
@pytest.mark.parametrize("response,finish", [
    (payload([REASONING], status="incomplete", incomplete={"reason": "max_output_tokens"}), "length"),
    (payload([], status="incomplete", incomplete={"reason": "content_filter"}), "content_filter"),
    (payload([{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]), "refusal"),
])
async def test_finish_reasons(response, finish):
    reply = await client_with([(200, response)], []).complete([{"role": "user", "content": "x"}], runtime(),
                                                              purpose="actor_execution")
    assert reply.finish_reason == finish


@pytest.mark.asyncio
async def test_invalid_json_arguments_reach_executor_as_tool_error():
    bad = dict(CALL, arguments="{not json")
    reply = await client_with([(200, payload([bad]))], []).complete([{"role": "user", "content": "x"}], runtime(),
                                                                    purpose="actor_execution")
    assert dict(reply.message["tool_calls"][0].args) == {INVALID_ARGS_KEY: "{not json"}


@pytest.mark.asyncio
async def test_transient_errors_are_retried_and_each_attempt_recorded():
    captured = []
    client = client_with([(429, {"error": {"message": "slow down"}}), (503, {"error": {"message": "busy"}}),
                          (200, payload([CALL]))], captured)
    reply = await client.complete([{"role": "user", "content": "x"}], runtime(), purpose="actor_execution")
    assert [c["attempt_index"] for c in reply.raw_response["calls"]] == [1, 2, 3]
    assert reply.raw_response["calls"][0]["error"]["status"] == 429


@pytest.mark.asyncio
async def test_context_limit_is_not_a_completion():
    body = {"error": {"code": "context_length_exceeded", "message": "too long", "type": "invalid_request_error"}}
    reply = await client_with([(400, body)], []).complete([{"role": "user", "content": "x"}], runtime(),
                                                          purpose="actor_execution")
    assert reply.finish_reason == "context_limit" and reply.message["tool_calls"] is None


@pytest.mark.asyncio
async def test_capability_rejection_is_typed_and_redacted():
    body = {"error": {"code": None, "param": "temperature", "message": f"bad {SECRET}"}}
    with pytest.raises(ModelCapabilityCallError) as info:
        await client_with([(400, body)], []).complete([{"role": "user", "content": "x"}], runtime(),
                                                      purpose="actor_execution")
    assert SECRET not in str(info.value) and SECRET not in json.dumps(info.value.raw_response)


@pytest.mark.asyncio
async def test_other_client_errors_are_not_retried():
    captured = []
    with pytest.raises(ModelCallError):
        await client_with([(401, {"error": {"message": "nope"}})], captured).complete(
            [{"role": "user", "content": "x"}], runtime(), purpose="actor_execution")
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_cancellation_keeps_request_journal():
    def handler(request):
        raise asyncio.CancelledError()

    client = ResponsesModelClient(cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelCallCancelledError) as info:
        await client.complete([{"role": "user", "content": "x"}], runtime(), purpose="actor_execution")
    assert info.value.raw_response["calls"][0]["operation"] == "responses.create"
