"""Tests for the Divyam LLM provider in LLMClient."""

import sys
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Load Divyam config
with open("conf/llm/divyam-pre-prod.json") as f:
    _divyam_config = json.load(f)
    DIVYAM_API_KEY = _divyam_config.get("llm_api_key")
    DIVYAM_MODEL = _divyam_config.get("llm_model")
    DIVYAM_API_ENDPOINT = _divyam_config.get("llm_api_endpoint")

# Patch target: the ChatOpenAI class as imported inside benchmark.llm_client
_PATCH_TARGET = "benchmark.llm_client.ChatOpenAI"


def _make_client(api_key="divyam-v1-test", model=None, **kwargs):
    """Instantiate LLMClient with the divyam provider, mocking ChatOpenAI."""
    if model is None:
        model = DIVYAM_MODEL
    # Force re-import so each test gets a fresh module state with the patch active
    if "benchmark.llm_client" in sys.modules:
        del sys.modules["benchmark.llm_client"]

    mock_llm = MagicMock()
    mock_cls = MagicMock(return_value=mock_llm)

    with patch("langchain_openai.ChatOpenAI", mock_cls):
        from benchmark.llm_client import LLMClient
        client = LLMClient(provider="divyam", model=model, api_key=api_key, **kwargs)

    return client, mock_cls, mock_llm


class TestDivyamProviderInit:
    def setup_method(self):
        # Ensure a clean import for every test
        sys.modules.pop("benchmark.llm_client", None)

    def _call_args_kwargs(self, mock_cls):
        """Return the kwargs ChatOpenAI was called with."""
        assert mock_cls.call_count == 1, "ChatOpenAI should be instantiated exactly once"
        return mock_cls.call_args[1]  # keyword arguments

    def test_uses_chatopenai(self):
        """divyam provider must initialize via ChatOpenAI."""
        _, mock_cls, _ = _make_client()
        assert mock_cls.call_count == 1

    def test_default_base_url(self):
        """When no api_endpoint is given, base URL must be https://api.divyam.ai/v1."""
        _, mock_cls, _ = _make_client()
        kwargs = self._call_args_kwargs(mock_cls)
        assert kwargs.get("openai_api_base") == "https://api.divyam.ai/v1"

    def test_custom_base_url_override(self):
        """api_endpoint parameter must override the default base URL."""
        custom_url = "https://custom.divyam.ai/v2"
        _, mock_cls, _ = _make_client(api_endpoint=custom_url)
        kwargs = self._call_args_kwargs(mock_cls)
        assert kwargs.get("openai_api_base") == custom_url

    def test_api_key_forwarded(self):
        """The API key must be forwarded as openai_api_key."""
        key = DIVYAM_API_KEY
        _, mock_cls, _ = _make_client(api_key=key)
        kwargs = self._call_args_kwargs(mock_cls)
        assert kwargs.get("openai_api_key") == key

    def test_required_header_accept(self):
        """Accept: application/json header must be present."""
        _, mock_cls, _ = _make_client()
        headers = self._call_args_kwargs(mock_cls).get("default_headers", {})
        assert headers.get("Accept") == "application/json"

    def test_required_header_traffic_allocation(self):
        """x-divyam-traffic-allocation-override: 8 header must be present."""
        _, mock_cls, _ = _make_client()
        headers = self._call_args_kwargs(mock_cls).get("default_headers", {})
        assert headers.get("x-divyam-traffic-allocation-override") == "8"

    def test_required_header_authorization(self):
        """Authorization: Bearer <key> header must be present and contain the API key."""
        key = DIVYAM_API_KEY
        _, mock_cls, _ = _make_client(api_key=key)
        headers = self._call_args_kwargs(mock_cls).get("default_headers", {})
        assert headers.get("Authorization") == f"Bearer {key}"

    def test_model_forwarded(self):
        """The model name must be passed through."""
        _, mock_cls, _ = _make_client(model=DIVYAM_MODEL)
        kwargs = self._call_args_kwargs(mock_cls)
        assert kwargs.get("model") == DIVYAM_MODEL

    def test_temperature_forwarded(self):
        _, mock_cls, _ = _make_client(temperature=0.7)
        kwargs = self._call_args_kwargs(mock_cls)
        assert kwargs.get("temperature") == 0.7

    def test_max_tokens_forwarded(self):
        _, mock_cls, _ = _make_client(max_tokens=1024)
        kwargs = self._call_args_kwargs(mock_cls)
        assert kwargs.get("max_tokens") == 1024

    def test_top_p_in_model_kwargs(self):
        """top_p must appear inside model_kwargs when provided."""
        _, mock_cls, _ = _make_client(top_p=0.9)
        kwargs = self._call_args_kwargs(mock_cls)
        assert kwargs.get("model_kwargs", {}).get("top_p") == 0.9

    def test_no_top_p_by_default(self):
        """model_kwargs must not contain top_p when it is not provided."""
        _, mock_cls, _ = _make_client()
        kwargs = self._call_args_kwargs(mock_cls)
        assert "top_p" not in kwargs.get("model_kwargs", {})

    def test_disable_divyam_selector_sends_selector_disabled_header(self):
        """disable_divyam_selector=True must set x-divyam-traffic-allocation-override to 'selector_disabled'."""
        _, mock_cls, _ = _make_client(disable_divyam_selector=True)
        headers = self._call_args_kwargs(mock_cls).get("default_headers", {})
        assert headers.get("x-divyam-traffic-allocation-override") == "selector_disabled"

    def test_disable_divyam_selector_false_keeps_default_header(self):
        """disable_divyam_selector=False (default) must keep x-divyam-traffic-allocation-override as '8'."""
        _, mock_cls, _ = _make_client(disable_divyam_selector=False)
        headers = self._call_args_kwargs(mock_cls).get("default_headers", {})
        assert headers.get("x-divyam-traffic-allocation-override") == "8"


class TestDivyamInvoke:
    def setup_method(self):
        sys.modules.pop("benchmark.llm_client", None)

    async def test_invoke_with_tools_calls_ainvoke(self):
        """invoke_with_tools must delegate to the underlying LLM's ainvoke."""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_retry = MagicMock()
        mock_response = MagicMock()

        mock_llm.bind_tools.return_value = mock_bound
        mock_bound.with_retry.return_value = mock_retry
        mock_retry.ainvoke = AsyncMock(return_value=mock_response)

        with patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
            from benchmark.llm_client import LLMClient
            client = LLMClient(provider="divyam", model=DIVYAM_MODEL, api_key="key")

        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            }
        ]
        messages = [MagicMock()]
        result = await client.invoke_with_tools(messages, tools)

        mock_llm.bind_tools.assert_called_once()
        mock_retry.ainvoke.assert_awaited_once_with(messages)
        assert result is mock_response


@pytest.mark.integration
class TestDivyamIntegration:
    async def test_live_response_with_selector_disabled(self):
        """With disable_divyam_selector=True the selector must route to exactly the requested model."""
        from langchain_core.messages import HumanMessage
        from benchmark.llm_client import LLMClient

        client = LLMClient(
            provider="divyam",
            model=DIVYAM_MODEL,
            api_key=DIVYAM_API_KEY,
            api_endpoint=DIVYAM_API_ENDPOINT,
            disable_divyam_selector=True,
        )

        response = await client.invoke_with_tools(
            messages=[HumanMessage(content="Say hello in one word.")],
            tools=[],
        )

        assert response is not None
        content = response.content if hasattr(response, "content") else str(response)
        assert isinstance(content, str) and len(content.strip()) > 0, (
            f"Expected non-empty string response, got: {content!r}"
        )

        routed_model = response.response_metadata.get("model_name")
        assert routed_model == DIVYAM_MODEL, (
            f"Selector chose '{routed_model}' but '{DIVYAM_MODEL}' was requested"
        )

    async def test_live_response(self):
        """Send a real request to Divyam and verify a non-empty text response is returned."""
        from langchain_core.messages import HumanMessage
        from benchmark.llm_client import LLMClient

        client = LLMClient(
            provider="divyam",
            model=DIVYAM_MODEL,
            api_key=DIVYAM_API_KEY,
            api_endpoint=DIVYAM_API_ENDPOINT,
        )

        response = await client.invoke_with_tools(
            messages=[HumanMessage(content="Say hello in one word.")],
            tools=[],
        )

        # Response must be a non-empty AI message
        assert response is not None
        content = response.content if hasattr(response, "content") else str(response)
        assert isinstance(content, str) and len(content.strip()) > 0, (
            f"Expected non-empty string response, got: {content!r}"
        )
