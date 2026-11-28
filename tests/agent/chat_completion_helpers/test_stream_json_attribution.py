"""Issue #160764: only the SDK's own SSE decode failure is a provider stream error."""

import json

import httpx
import pytest
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

from agent.chat_completion_helpers import ProviderStreamError, _iter_provider_stream_chunks


def test_json_error_from_our_own_code_is_not_blamed_on_the_provider():
    def chunks():
        yield "first"
        json.loads("{not json")  # a bug in Hermes code running during iteration

    it = _iter_provider_stream_chunks(chunks())
    assert next(it) == "first"
    with pytest.raises(json.JSONDecodeError):
        next(it)


def test_sdk_sse_decode_error_is_still_a_provider_stream_error():
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    response = httpx.Response(200, request=request, content=b"data: not json\n\n")
    with OpenAI(api_key="test-key", max_retries=0) as client:
        stream = Stream(cast_to=ChatCompletionChunk, response=response, client=client)
        with pytest.raises(ProviderStreamError) as exc_info:
            list(_iter_provider_stream_chunks(stream))
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_sdk_error_keeps_the_stream_response_when_none_is_given():
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    response = httpx.Response(200, request=request, headers={"x-request-id": "req-1"},
                              content=b"data: not json\n\n")
    with OpenAI(api_key="test-key", max_retries=0) as client:
        stream = Stream(cast_to=ChatCompletionChunk, response=response, client=client)
        with pytest.raises(ProviderStreamError) as exc_info:
            list(_iter_provider_stream_chunks(stream, response=lambda: None))
    assert exc_info.value.response.headers["x-request-id"] == "req-1"
