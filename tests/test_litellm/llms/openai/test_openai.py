"""
Regression test: a provider host refusing the connection (httpx.ConnectError,
surfaced by the OpenAI SDK as openai.APIConnectionError with no status_code)
must map to litellm.APIConnectionError, not litellm.InternalServerError.

Before the fix, OpenAIChatCompletion.completion()/.acompletion() defaulted
any status-code-less exception to 500 and wrapped it in OpenAIError(500),
which exception_type() then mapped to InternalServerError - causing the
Router to wrongly cool down a deployment that was merely unreachable.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

import litellm


def _connection_refused_error() -> openai.APIConnectionError:
    request = httpx.Request(method="POST", url="https://api.openai.com/v1/chat/completions")
    return openai.APIConnectionError(request=request)


def _upstream_500_error() -> openai.InternalServerError:
    request = httpx.Request(method="POST", url="https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=500, request=request)
    return openai.InternalServerError(message="upstream server error", response=response, body=None)


class TestOpenAIConnectionErrorMapping:
    def test_sync_connection_refused_raises_api_connection_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = _connection_refused_error()

        with pytest.raises(litellm.APIConnectionError):
            litellm.completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-test",
                client=mock_client,
            )

    async def test_async_connection_refused_raises_api_connection_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(
            side_effect=_connection_refused_error()
        )

        with pytest.raises(litellm.APIConnectionError):
            await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-test",
                client=mock_client,
            )

    def test_sync_genuine_upstream_500_still_raises_internal_server_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = _upstream_500_error()

        with pytest.raises(litellm.InternalServerError):
            litellm.completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-test",
                client=mock_client,
            )

    async def test_async_genuine_upstream_500_still_raises_internal_server_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(
            side_effect=_upstream_500_error()
        )

        with pytest.raises(litellm.InternalServerError):
            await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-test",
                client=mock_client,
            )
