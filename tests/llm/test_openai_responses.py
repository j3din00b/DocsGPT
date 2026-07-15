"""Unit tests for the OpenAI Responses API path in application/llm/openai.py.

Covers the api_flavor gating, Chat-Completions -> Responses request
translation, tool/structured-output mapping, reasoning-item carryover, the
streaming-event normalization into the existing handler contract, and the
previous_response_id trimming used for cross-turn chaining.
"""

import types
from unittest.mock import MagicMock

import pytest

from application.core.model_settings import ModelCapabilities


def _make_llm(monkeypatch, capabilities=None, store_responses=False):
    monkeypatch.setattr("application.llm.openai.OpenAI", MagicMock())
    monkeypatch.setattr(
        "application.llm.openai.StorageCreator",
        types.SimpleNamespace(get_storage=lambda: None),
    )
    monkeypatch.setattr(
        "application.llm.openai.settings",
        types.SimpleNamespace(
            OPENAI_API_KEY="k",
            API_KEY="k",
            OPENAI_BASE_URL="",
            AZURE_DEPLOYMENT_NAME="dep",
            OPENAI_RESPONSES_STORE=store_responses,
        ),
    )
    from application.llm.openai import OpenAILLM

    llm = OpenAILLM(api_key="k")
    llm.capabilities = capabilities
    return llm


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _responses_caps(reasoning_effort=None):
    return ModelCapabilities(
        supports_tools=True,
        supports_structured_output=True,
        api_flavor="responses",
        reasoning_effort=reasoning_effort,
    )


def _bare_agent():
    from application.agents.base import BaseAgent

    class _Agent(BaseAgent):
        def _gen_inner(self, query, log_context):
            yield from ()

    return _Agent.__new__(_Agent)


# ── api_flavor gating ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_uses_responses_api_true(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    assert llm._uses_responses_api() is True


@pytest.mark.unit
def test_uses_responses_api_false_for_chat(monkeypatch):
    caps = ModelCapabilities(api_flavor="chat_completions")
    assert _make_llm(monkeypatch, caps)._uses_responses_api() is False


@pytest.mark.unit
def test_uses_responses_api_false_without_caps(monkeypatch):
    assert _make_llm(monkeypatch, None)._uses_responses_api() is False


# ── message translation ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_to_responses_input_tool_roundtrip(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"x"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "assistant", "content": "final"},
    ]
    items = llm._to_responses_input(messages)
    assert items == [
        {"role": "system", "content": [{"type": "input_text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "search",
            "arguments": '{"q":"x"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        # The Responses API requires output_text (not input_text) for the
        # assistant role; input_text 400s. Locked in here so it can't regress.
        {"role": "assistant", "content": [{"type": "output_text", "text": "final"}]},
    ]


@pytest.mark.unit
def test_to_responses_input_reinjects_reasoning(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    reasoning_item = {
        "type": "reasoning", "id": "rs_1",
        "encrypted_content": "enc", "summary": [],
    }
    llm._reasoning_for_calls = {"call_1": [reasoning_item]}
    # The call must be paired with its output — unpaired calls are dropped
    # (the API rejects them with "No tool output found for function call").
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "t", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    items = llm._to_responses_input(messages)
    # Reasoning item is emitted immediately before its function call.
    assert items[0] == reasoning_item
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_1"


@pytest.mark.unit
def test_chat_flavor_drops_responses_only_message_fields(monkeypatch):
    llm = _make_llm(
        monkeypatch,
        ModelCapabilities(api_flavor="chat_completions"),
    )
    cleaned = llm._clean_messages_openai([{
        "role": "assistant",
        "content": "answer",
        "responses_reasoning_items": [{"type": "reasoning", "id": "rs_1"}],
    }])
    assert cleaned == [{"role": "assistant", "content": "answer"}]


@pytest.mark.unit
def test_build_messages_gates_reasoning_and_keeps_tool_chronology(monkeypatch):
    agent = _bare_agent()
    agent.compressed_summary = None
    agent.model_id = "model"
    agent.model_user_id = None
    agent.user = "u"
    agent.multimodal_content = None
    agent.llm = _ns(
        _uses_responses_api=lambda: True,
        responses_chain_key=lambda: "chain-current",
    )
    agent.chat_history = [{
        "prompt": "old question",
        "response": "final answer",
        "tool_calls": [{
            "call_id": "call_1",
            "action_name": "search",
            "arguments": {"q": "x"},
            "result": "found",
        }],
        "metadata": {
            "responses_state": {
                "chain_key": "chain-other",
                "reasoning_items": [{"type": "reasoning", "id": "wrong"}],
            }
        },
    }]

    messages = agent._build_messages("system", "new question")
    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "tool", "assistant", "user"
    ]
    assert messages[2]["tool_calls"][0]["id"] == "call_1"
    assert messages[4]["content"] == "final answer"
    assert "responses_reasoning_items" not in messages[4]


@pytest.mark.unit
def test_build_messages_rewrites_duplicate_tool_call_ids(monkeypatch):
    agent = _bare_agent()
    agent.compressed_summary = None
    agent.model_id = "model"
    agent.model_user_id = None
    agent.user = "u"
    agent.multimodal_content = None
    agent.llm = _ns(
        _uses_responses_api=lambda: True,
        responses_chain_key=lambda: "chain-current",
    )
    agent.chat_history = [{
        "prompt": "old question",
        "response": "final answer",
        "tool_calls": [
            {
                "call_id": "functions.search:0",
                "action_name": "search",
                "arguments": {"q": "first"},
                "result": "first result",
            },
            {
                "call_id": "functions.search:0",
                "action_name": "search",
                "arguments": {"q": "second"},
                "result": "second result",
            },
        ],
    }]

    messages = agent._build_messages("system", "new question")
    assistant_calls = messages[2]["tool_calls"]
    tool_results = [message for message in messages if message["role"] == "tool"]

    replay_ids = [call["id"] for call in assistant_calls]
    assert replay_ids[0] == "functions.search:0"
    assert len(set(replay_ids)) == 2
    assert [message["tool_call_id"] for message in tool_results] == replay_ids
    assert [message["content"] for message in tool_results] == [
        "first result",
        "second result",
    ]


@pytest.mark.unit
def test_export_import_replays_encrypted_reasoning_on_fresh_instance(monkeypatch):
    first = _make_llm(monkeypatch, _responses_caps())
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "ciphertext",
        "summary": [],
    }
    first._last_response_id = "resp_1"
    first._last_reasoning_items = [reasoning]
    first._reasoning_for_calls = {"call_1": [reasoning]}

    second = _make_llm(monkeypatch, _responses_caps())
    assert second.import_responses_state(first.export_responses_state()) is True
    # Paired with its output: unpaired calls are dropped by the builder.
    items = second._to_responses_input([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "run", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ])
    assert items[0] == reasoning
    assert items[1]["type"] == "function_call"


@pytest.mark.unit
def test_import_rejects_state_for_different_target(monkeypatch):
    first = _make_llm(monkeypatch, _responses_caps())
    state = first.export_responses_state()
    second = _make_llm(monkeypatch, _responses_caps())
    second._effective_base_url = "https://different.example/v1"
    assert second.import_responses_state(state) is False


@pytest.mark.unit
def test_start_responses_turn_drops_prior_turn_call_map(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    llm._reasoning_for_calls = {"old_call": [{"id": "old_reasoning"}]}
    llm._last_reasoning_items = [{"id": "old_reasoning"}]
    llm._imported_response_id = "resp_old"

    llm.start_responses_turn()

    assert llm._reasoning_for_calls == {}
    assert llm._last_reasoning_items == []
    assert llm._imported_response_id is None


@pytest.mark.unit
def test_to_responses_input_multimodal_image(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
        ],
    }]
    items = llm._to_responses_input(messages)
    assert items == [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "look"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,xx",
                "detail": "auto",
            },
        ],
    }]


@pytest.mark.unit
def test_to_responses_tools_flatten(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    assert llm._to_responses_tools(tools) == [{
        "type": "function",
        "name": "search",
        "description": "Search",
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
    }]


@pytest.mark.unit
def test_responses_text_format_json_schema(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    rf = {
        "type": "json_schema",
        "json_schema": {"name": "out", "schema": {"type": "object"}, "strict": True},
    }
    assert llm._responses_text_format(rf) == {
        "type": "json_schema", "name": "out",
        "schema": {"type": "object"}, "strict": True,
    }


@pytest.mark.unit
def test_trim_for_previous_response(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old a"},
        {"role": "user", "content": "new q"},
    ]
    trimmed = llm._trim_for_previous_response(messages)
    # System stays; everything up to and including the last assistant text
    # is dropped (the server already holds it), leaving the new user turn.
    assert trimmed == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "new q"},
    ]


# ── request params ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_build_responses_params_stateless(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps(reasoning_effort="high"))
    params = llm._build_responses_params(
        "gpt-5.5", [{"role": "user", "content": []}], tools=None,
        response_format=None, previous_response_id=None, stream=True,
        kwargs={"max_completion_tokens": 256},
    )
    assert params["model"] == "gpt-5.5"
    assert params["stream"] is True
    assert params["max_output_tokens"] == 256
    assert params["reasoning"] == {"effort": "high", "summary": "auto"}
    assert params["store"] is False
    assert params["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in params


@pytest.mark.unit
def test_chat_path_drops_tool_controls_when_tools_are_unavailable(monkeypatch):
    caps = ModelCapabilities(
        supports_tools=False,
        api_flavor="chat_completions",
    )
    llm = _make_llm(monkeypatch, caps)
    llm.client.chat.completions.create = MagicMock(
        return_value=_ns(choices=[_ns(message=_ns(content="ok"))])
    )

    result = llm._raw_gen(
        llm,
        "model",
        [{"role": "user", "content": "hi"}],
        tools=[{
            "type": "function",
            "function": {"name": "search", "parameters": {}},
        }],
        tool_choice="required",
        parallel_tool_calls=True,
    )

    assert result == "ok"
    request = llm.client.chat.completions.create.call_args.kwargs
    assert "tools" not in request
    assert "tool_choice" not in request
    assert "parallel_tool_calls" not in request


@pytest.mark.unit
def test_chat_nonstream_records_provider_usage(monkeypatch):
    llm = _make_llm(monkeypatch, ModelCapabilities(api_flavor="chat_completions"))
    llm._last_usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    llm.client.chat.completions.create = MagicMock(
        return_value=_ns(
            choices=[_ns(message=_ns(content="ok"))],
            usage=_ns(prompt_tokens=11, completion_tokens=5, total_tokens=16),
        )
    )

    result = llm._raw_gen(llm, "model", [{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert llm._last_usage == {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "total_tokens": 16,
    }


@pytest.mark.unit
def test_chat_stream_requests_and_records_terminal_usage(monkeypatch):
    llm = _make_llm(monkeypatch, ModelCapabilities(api_flavor="chat_completions"))
    llm._last_usage = {"prompt_tokens": 9, "completion_tokens": 9, "total_tokens": 18}
    llm.client.chat.completions.create = MagicMock(
        return_value=[
            _ns(choices=[_ns(delta=_ns(content="Hi"), finish_reason=None)]),
            _ns(
                choices=[],
                usage=_ns(prompt_tokens=7, completion_tokens=2, total_tokens=9),
            ),
        ]
    )

    out = list(
        llm._raw_gen_stream(llm, "model", [{"role": "user", "content": "hi"}])
    )

    assert out == ["Hi"]
    request = llm.client.chat.completions.create.call_args.kwargs
    assert request["stream_options"] == {"include_usage": True}
    assert llm._last_usage == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }


@pytest.mark.unit
def test_chat_stream_without_usage_clears_stale_counts(monkeypatch):
    llm = _make_llm(monkeypatch, ModelCapabilities(api_flavor="chat_completions"))
    llm._last_usage = {"prompt_tokens": 9, "completion_tokens": 9, "total_tokens": 18}
    llm.client.chat.completions.create = MagicMock(
        return_value=[_ns(choices=[_ns(delta=_ns(content="Hi"), finish_reason=None)])]
    )

    assert list(
        llm._raw_gen_stream(llm, "model", [{"role": "user", "content": "hi"}])
    ) == ["Hi"]
    assert llm._last_usage is None


@pytest.mark.unit
def test_zeroed_chat_usage_does_not_clobber_estimates(monkeypatch):
    llm = _make_llm(monkeypatch, ModelCapabilities(api_flavor="chat_completions"))
    llm.client.chat.completions.create = MagicMock(
        return_value=_ns(
            choices=[_ns(message=_ns(content="ok"))],
            usage=_ns(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
    )

    llm._raw_gen(llm, "model", [{"role": "user", "content": "hi"}])

    assert llm._last_usage is None


@pytest.mark.unit
def test_record_chat_usage_captures_detail_bins(monkeypatch):
    llm = _make_llm(monkeypatch, ModelCapabilities(api_flavor="chat_completions"))
    llm._record_chat_usage(_ns(
        prompt_tokens=10,
        completion_tokens=7,
        total_tokens=17,
        prompt_tokens_details=_ns(cached_tokens=4),
        completion_tokens_details=_ns(reasoning_tokens=3),
    ))
    assert llm._last_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 7,
        "total_tokens": 17,
        "prompt_tokens_details": {"cached_tokens": 4},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }


@pytest.mark.unit
def test_build_responses_params_store_with_previous_id(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps(), store_responses=True)
    params = llm._build_responses_params(
        "gpt-5.5", [], tools=None, response_format=None,
        previous_response_id="resp_abc", stream=False, kwargs={},
    )
    assert params["store"] is True
    assert params["previous_response_id"] == "resp_abc"
    # Encrypted reasoning is always requested so in-turn carryover works
    # regardless of server-side retention.
    assert params["include"] == ["reasoning.encrypted_content"]


@pytest.mark.unit
def test_build_responses_params_maps_tool_controls(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    params = llm._build_responses_params(
        "gpt-5.5",
        [],
        tools=[{"type": "function", "function": {"name": "search"}}],
        response_format=None,
        previous_response_id=None,
        stream=False,
        kwargs={
            "tool_choice": {
                "type": "function",
                "function": {"name": "search"},
            },
            "parallel_tool_calls": False,
        },
    )
    assert params["tool_choice"] == {"type": "function", "name": "search"}
    assert params["parallel_tool_calls"] is False


@pytest.mark.unit
def test_record_responses_metadata_captures_usage_details(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    llm._record_responses_metadata(_ns(
        id="resp_1",
        usage=_ns(
            input_tokens=10,
            output_tokens=7,
            total_tokens=17,
            input_tokens_details=_ns(cached_tokens=4),
            output_tokens_details=_ns(reasoning_tokens=3),
        ),
    ))
    assert llm._last_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 7,
        "total_tokens": 17,
        "prompt_tokens_details": {"cached_tokens": 4},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }


# ── streaming normalization into the existing handler contract ───────────────


@pytest.mark.unit
def test_responses_gen_stream_text_and_tools(monkeypatch):
    from application.llm.handlers.openai import OpenAILLMHandler

    llm = _make_llm(monkeypatch, _responses_caps())
    events = [
        _ns(type="response.output_text.delta", delta="Hel"),
        _ns(type="response.output_text.delta", delta="lo"),
        _ns(type="response.reasoning_summary_text.delta", delta="thinking"),
        _ns(
            type="response.output_item.added",
            output_index=0,
            item=_ns(type="function_call", call_id="call_1", name="search", id="fc_1"),
        ),
        _ns(type="response.function_call_arguments.delta", output_index=0, delta='{"q":'),
        _ns(
            type="response.function_call_arguments.done",
            output_index=0,
            arguments='{"q":"hi"}',
        ),
        _ns(
            type="response.output_item.done",
            item=_ns(type="reasoning", id="rs_1", encrypted_content="enc", summary=[]),
        ),
        _ns(type="response.completed", response=_ns(id="resp_1")),
    ]
    llm.client.responses.create = MagicMock(return_value=events)

    out = list(llm._responses_gen_stream("gpt-5.5", [{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}]))

    assert "Hel" in out and "lo" in out
    assert {"type": "thought", "thought": "thinking"} in out
    choice = out[-1]
    parsed = OpenAILLMHandler().parse_response(choice)
    assert parsed.finish_reason == "tool_calls"
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "search"
    assert tc.arguments == '{"q":"hi"}'
    # Reasoning captured for in-turn carryover, last response id recorded.
    assert llm._reasoning_for_calls["call_1"][0]["encrypted_content"] == "enc"
    assert llm._last_response_id == "resp_1"


@pytest.mark.unit
def test_responses_gen_stream_text_only(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    events = [
        _ns(type="response.output_text.delta", delta="Answer"),
        _ns(type="response.completed", response=_ns(id="resp_2")),
    ]
    llm.client.responses.create = MagicMock(return_value=events)
    out = list(llm._responses_gen_stream("gpt-5.5", [{"role": "user", "content": "hi"}], tools=None))
    assert out == ["Answer"]
    assert llm._last_response_id == "resp_2"


@pytest.mark.unit
def test_responses_gen_stream_parallel_tool_calls(monkeypatch):
    from application.llm.handlers.openai import OpenAILLMHandler

    llm = _make_llm(monkeypatch, _responses_caps())
    events = [
        _ns(
            type="response.output_item.added", output_index=0,
            item=_ns(type="function_call", call_id="call_a", name="t1", id="fc_a"),
        ),
        _ns(
            type="response.output_item.added", output_index=1,
            item=_ns(type="function_call", call_id="call_b", name="t2", id="fc_b"),
        ),
        _ns(type="response.function_call_arguments.delta", output_index=0, delta='{"a":'),
        _ns(type="response.function_call_arguments.done", output_index=0, arguments='{"a":1}'),
        _ns(type="response.function_call_arguments.done", output_index=1, arguments='{"b":2}'),
        _ns(type="response.completed", response=_ns(id="resp_p")),
    ]
    llm.client.responses.create = MagicMock(return_value=events)
    out = list(llm._responses_gen_stream("gpt-5.5", [{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "t1", "parameters": {}}}]))
    parsed = OpenAILLMHandler().parse_response(out[-1])
    assert parsed.finish_reason == "tool_calls"
    assert [tc.id for tc in parsed.tool_calls] == ["call_a", "call_b"]
    assert [tc.index for tc in parsed.tool_calls] == [0, 1]
    assert parsed.tool_calls[0].arguments == '{"a":1}'
    assert parsed.tool_calls[1].arguments == '{"b":2}'


@pytest.mark.unit
def test_responses_gen_stream_error_event_raises(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    events = [
        _ns(type="response.output_text.delta", delta="partial"),
        _ns(type="response.failed", response=_ns(error="boom")),
    ]
    llm.client.responses.create = MagicMock(return_value=events)
    with pytest.raises(RuntimeError):
        list(llm._responses_gen_stream("gpt-5.5", [{"role": "user", "content": "hi"}], tools=None))


@pytest.mark.unit
def test_responses_gen_stream_incomplete_returns_partial_length(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    events = [
        _ns(type="response.output_text.delta", delta="partial"),
        _ns(
            type="response.incomplete",
            response=_ns(
                id="resp_incomplete",
                incomplete_details=_ns(reason="max_output_tokens"),
            ),
        ),
    ]
    llm.client.responses.create = MagicMock(return_value=events)

    out = list(
        llm._responses_gen_stream(
            "gpt-5.5", [{"role": "user", "content": "hi"}], tools=None
        )
    )
    from application.llm.handlers.openai import OpenAILLMHandler

    assert out[0] == "partial"
    assert OpenAILLMHandler().parse_response(out[-1]).finish_reason == "length"
    assert llm._last_response_id == "resp_incomplete"


@pytest.mark.unit
def test_responses_gen_stream_surfaces_refusal(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    events = [
        _ns(type="response.refusal.delta", delta="I cannot"),
        _ns(type="response.refusal.delta", delta=" help with that."),
        _ns(type="response.completed", response=_ns(id="resp_refusal")),
    ]
    llm.client.responses.create = MagicMock(return_value=events)

    assert list(
        llm._responses_gen_stream(
            "gpt-5.5", [{"role": "user", "content": "hi"}], tools=None
        )
    ) == ["I cannot", " help with that."]
    assert llm._last_response_id == "resp_refusal"


@pytest.mark.unit
def test_responses_gen_stream_surfaces_done_only_refusal(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    events = [
        _ns(type="response.refusal.done", refusal="I cannot help with that."),
        _ns(type="response.completed", response=_ns(id="resp_refusal")),
    ]
    llm.client.responses.create = MagicMock(return_value=events)

    assert list(
        llm._responses_gen_stream(
            "gpt-5.5", [{"role": "user", "content": "hi"}], tools=None
        )
    ) == ["I cannot help with that."]


@pytest.mark.unit
def test_responses_gen_nonstream_tools(monkeypatch):
    from application.llm.handlers.openai import OpenAILLMHandler

    llm = _make_llm(monkeypatch, _responses_caps())
    response = _ns(
        id="resp_3",
        output=[
            _ns(type="reasoning", id="rs", encrypted_content="e", summary=[]),
            _ns(type="message", content=[_ns(type="output_text", text="Answer")]),
            _ns(type="function_call", call_id="c1", name="t", arguments="{}", id="fc"),
        ],
    )
    llm.client.responses.create = MagicMock(return_value=response)
    choice = llm._responses_gen("gpt-5.5", [{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}])
    parsed = OpenAILLMHandler().parse_response(choice)
    assert parsed.finish_reason == "tool_calls"
    assert parsed.tool_calls[0].id == "c1"
    assert llm._reasoning_for_calls["c1"][0]["encrypted_content"] == "e"


@pytest.mark.unit
def test_responses_gen_nonstream_text(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    response = _ns(
        id="resp_4",
        output=[_ns(type="message", content=[_ns(type="output_text", text="Hi there")])],
    )
    llm.client.responses.create = MagicMock(return_value=response)
    result = llm._responses_gen("gpt-5.5", [{"role": "user", "content": "hi"}], tools=None)
    assert result == "Hi there"


@pytest.mark.unit
def test_responses_gen_nonstream_surfaces_refusal(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    response = _ns(
        id="resp_refusal",
        status="completed",
        output=[
            _ns(
                type="message",
                content=[_ns(type="refusal", refusal="I cannot help with that.")],
            )
        ],
    )
    llm.client.responses.create = MagicMock(return_value=response)

    result = llm._responses_gen(
        "gpt-5.5", [{"role": "user", "content": "hi"}], tools=None
    )
    assert result == "I cannot help with that."
    assert llm._last_response_id == "resp_refusal"


@pytest.mark.unit
def test_responses_gen_nonstream_incomplete_returns_partial_length(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    response = _ns(
        id="resp_incomplete",
        status="incomplete",
        incomplete_details=_ns(reason="max_output_tokens"),
        output=[
            _ns(type="message", content=[_ns(type="output_text", text="partial")])
        ],
    )
    llm.client.responses.create = MagicMock(return_value=response)

    result = llm._responses_gen(
        "gpt-5.5", [{"role": "user", "content": "hi"}], tools=None
    )
    assert result == "partial"
    assert llm._last_finish_reason == "length"
    assert llm._last_response_id == "resp_incomplete"


@pytest.mark.unit
def test_public_gen_keeps_plain_string_contract_for_incomplete_text(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    response = _ns(
        id="resp_incomplete",
        status="incomplete",
        incomplete_details=_ns(reason="max_output_tokens"),
        output=[
            _ns(type="message", content=[
                _ns(type="output_text", text="partial title")
            ])
        ],
    )
    llm.client.responses.create = MagicMock(return_value=response)
    monkeypatch.setattr("application.cache.get_redis_instance", lambda: None)

    result = llm.gen(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "make a title"}],
    )

    assert result == "partial title"
    assert isinstance(result, str)
    assert result.strip() == "partial title"
    assert llm._last_finish_reason == "length"


@pytest.mark.unit
def test_stream_error_event_preserves_upstream_message(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    llm.client.responses.create = MagicMock(
        return_value=[_ns(type="error", message="real upstream failure")]
    )
    with pytest.raises(RuntimeError, match="real upstream failure"):
        list(
            llm._responses_gen_stream(
                "gpt-5.5", [{"role": "user", "content": "hi"}], tools=None
            )
        )


@pytest.mark.unit
def test_previous_response_id_requires_immediately_preceding_matching_chain():
    agent = _bare_agent()
    agent.llm = _ns(responses_chain_key=lambda: "chain-current")

    agent.chat_history = [{
        "metadata": {
            "response_id": "resp_ok",
            "response_chain_key": "chain-current",
        }
    }]
    assert agent._previous_response_id() == "resp_ok"

    agent.chat_history.append({"metadata": {}})
    assert agent._previous_response_id() is None

    agent.chat_history[-1] = {
        "metadata": {
            "response_id": "resp_other",
            "response_chain_key": "chain-other",
        }
    }
    assert agent._previous_response_id() is None


@pytest.mark.unit
def test_responses_chain_key_scopes_model_endpoint_and_credential(monkeypatch):
    llm = _make_llm(monkeypatch, _responses_caps())
    llm._canonical_model_id = "model-a"
    initial = llm.responses_chain_key()

    llm._canonical_model_id = "model-b"
    different_model = llm.responses_chain_key()
    llm._canonical_model_id = "model-a"
    llm._effective_base_url = "https://other.example/v1"
    different_endpoint = llm.responses_chain_key()
    llm._effective_base_url = "https://api.openai.com/v1"
    llm.api_key = "different-key"
    different_credential = llm.responses_chain_key()

    llm.api_key = "k"
    monkeypatch.setattr(
        "application.llm.openai.settings.OPENAI_RESPONSES_STORE", True
    )
    different_store_mode = llm.responses_chain_key()

    assert len({
        initial,
        different_model,
        different_endpoint,
        different_credential,
        different_store_mode,
    }) == 5
    assert "different-key" not in different_credential


@pytest.mark.unit
def test_responses_metadata_persists_chain_key(monkeypatch):
    monkeypatch.setattr(
        "application.agents.base.settings.OPENAI_RESPONSES_STORE", True
    )
    agent = _bare_agent()
    agent.llm = _ns(
        _last_response_id="resp_1",
        responses_chain_key=lambda: "chain-current",
    )

    assert list(agent._emit_responses_metadata()) == [{
        "metadata": {
            "response_id": "resp_1",
            "response_chain_key": "chain-current",
        }
    }]


@pytest.mark.unit
def test_store_false_metadata_omits_unstored_response_id(monkeypatch):
    monkeypatch.setattr(
        "application.agents.base.settings.OPENAI_RESPONSES_STORE", False
    )
    agent = _bare_agent()
    agent.llm = _ns(
        _last_response_id="resp_unstored",
        _last_usage=None,
        _uses_responses_api=lambda: True,
        responses_chain_key=lambda: "chain-current",
        export_responses_state=lambda: {"reasoning_items": [{"id": "rs_1"}]},
    )

    metadata = list(agent._emit_responses_metadata())[0]["metadata"]
    assert "response_id" not in metadata
    assert "response_chain_key" not in metadata
    assert metadata["responses_state"]["reasoning_items"][0]["id"] == "rs_1"


# ── capability plumbing / yaml ───────────────────────────────────────────────


@pytest.mark.unit
def test_capability_field_rejects_bad_api_flavor():
    from application.core.model_yaml import _CapabilityFields

    with pytest.raises(ValueError):
        _CapabilityFields(api_flavor="grpc")


@pytest.mark.unit
def test_capability_field_rejects_bad_reasoning_effort():
    from application.core.model_yaml import _CapabilityFields

    with pytest.raises(ValueError):
        _CapabilityFields(reasoning_effort="extreme")


@pytest.mark.unit
def test_builtin_gpt55_opts_into_responses():
    from application.core.model_yaml import BUILTIN_MODELS_DIR, load_model_yamls

    catalogs = load_model_yamls([BUILTIN_MODELS_DIR])
    models = {m.id: m for c in catalogs for m in c.models}
    gpt = models["gpt-5.5"]
    assert gpt.capabilities.api_flavor == "responses"
    assert gpt.capabilities.reasoning_effort == "medium"


@pytest.mark.unit
def test_builtin_default_models_stay_chat_completions():
    from application.core.model_yaml import BUILTIN_MODELS_DIR, load_model_yamls

    catalogs = load_model_yamls([BUILTIN_MODELS_DIR])
    models = {m.id: m for c in catalogs for m in c.models}
    assert models["gpt-5.4-mini"].capabilities.api_flavor == "chat_completions"


@pytest.mark.unit
def test_to_responses_input_drops_unpaired_function_call(monkeypatch):
    """A function_call with no matching output must not reach the provider
    (it 400s with "No tool output found for function call ...")."""
    llm = _make_llm(monkeypatch, _responses_caps())
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_orphan",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                },
                {
                    "id": "call_ok",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_ok", "content": "result"},
    ]
    items = llm._to_responses_input(messages)
    calls = [i for i in items if i.get("type") == "function_call"]
    outputs = [i for i in items if i.get("type") == "function_call_output"]
    assert [c["call_id"] for c in calls] == ["call_ok"]
    assert [o["call_id"] for o in outputs] == ["call_ok"]


@pytest.mark.unit
def test_to_responses_input_drops_orphaned_output(monkeypatch):
    """A function_call_output whose call was never emitted is dropped."""
    llm = _make_llm(monkeypatch, _responses_caps())
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "call_ghost", "content": "result"},
        {"role": "assistant", "content": "final"},
    ]
    items = llm._to_responses_input(messages)
    assert not [i for i in items if i.get("type") == "function_call_output"]
    assert items[-1]["role"] == "assistant"


@pytest.mark.unit
def test_to_responses_input_skips_reasoning_of_dropped_calls(monkeypatch):
    """When every call on an assistant message is unpaired, its reasoning
    items are suppressed too — a trailing reasoning item with no following
    item is itself rejected by the Responses API."""
    llm = _make_llm(monkeypatch, _responses_caps())
    llm._reasoning_for_calls = {
        "call_orphan": [{"type": "reasoning", "id": "rs_1", "summary": []}]
    }
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "responses_reasoning_items": [
                {"type": "reasoning", "id": "rs_msg", "summary": []}
            ],
            "tool_calls": [{
                "id": "call_orphan",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }],
        },
    ]
    items = llm._to_responses_input(messages)
    assert not [i for i in items if i.get("type") == "reasoning"]
    assert not [i for i in items if i.get("type") == "function_call"]


@pytest.mark.unit
def test_to_responses_input_chained_keeps_bare_tool_output(monkeypatch):
    """Store-mode chaining (previous_response_id) deliberately sends a bare
    function_call_output whose call lives server-side — the pairing guard
    must not drop it (that would 400 every chained tool round)."""
    llm = _make_llm(monkeypatch, _responses_caps())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "tool_call_id": "call_server_side", "content": "result"},
    ]
    items = llm._to_responses_input(messages, chained=True)
    outputs = [i for i in items if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outputs] == ["call_server_side"]


@pytest.mark.unit
def test_to_responses_input_drops_out_of_order_pair_entirely(monkeypatch):
    """An output that precedes its call is malformed history: BOTH sides
    are dropped (keeping the call alone would produce the unpaired-call
    400 the guard exists to prevent)."""
    llm = _make_llm(monkeypatch, _responses_caps())
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "call_x", "content": "early result"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_x",
                "type": "function",
                "function": {"name": "t", "arguments": "{}"},
            }],
        },
    ]
    items = llm._to_responses_input(messages)
    assert not [i for i in items if i.get("type") == "function_call"]
    assert not [i for i in items if i.get("type") == "function_call_output"]
