"""
Unit tests for the Claude integration.

These tests mock the Anthropic SDK so they run without a real API key or
network connection.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure the repo root is on the path regardless of where pytest is invoked.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestClaudeConfig(unittest.TestCase):
    """Tests for claude_integration.config."""

    def test_get_api_key_from_env(self):
        """get_api_key() returns the value of ANTHROPIC_API_KEY."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            from claude_integration.config import get_api_key
            self.assertEqual(get_api_key(), "test-key-123")

    def test_get_api_key_missing_raises(self):
        """get_api_key() raises EnvironmentError when the variable is unset."""
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from claude_integration.config import get_api_key
            with self.assertRaises(EnvironmentError):
                get_api_key()

    def test_get_api_key_empty_raises(self):
        """get_api_key() raises EnvironmentError when the variable is empty."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "  "}):
            from claude_integration.config import get_api_key
            with self.assertRaises(EnvironmentError):
                get_api_key()


class TestClaudeClient(unittest.TestCase):
    """Tests for claude_integration.client.ClaudeClient."""

    def _make_client(self, mock_anthropic):
        """Helper: create a ClaudeClient with a mocked Anthropic SDK."""
        from claude_integration.client import ClaudeClient
        return ClaudeClient(api_key="fake-key")

    def _mock_response(self, text: str):
        """Build a mock Anthropic Messages response object."""
        content_block = MagicMock()
        content_block.text = text
        response = MagicMock()
        response.content = [content_block]
        return response

    @patch("claude_integration.client.anthropic.Anthropic")
    def test_provider_name(self, mock_anthropic_cls):
        """provider_name returns the expected string."""
        client = self._make_client(mock_anthropic_cls)
        self.assertEqual(client.provider_name, "Anthropic Claude")

    @patch("claude_integration.client.anthropic.Anthropic")
    def test_complete_calls_chat(self, mock_anthropic_cls):
        """complete() wraps a single prompt in a user message and calls chat."""
        mock_instance = mock_anthropic_cls.return_value
        mock_instance.messages.create.return_value = self._mock_response(
            "Neural networks are computing systems."
        )

        client = self._make_client(mock_anthropic_cls)
        result = client.complete("What is a neural network?")

        self.assertEqual(result, "Neural networks are computing systems.")
        mock_instance.messages.create.assert_called_once()
        call_kwargs = mock_instance.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["messages"][0]["role"], "user")
        self.assertEqual(
            call_kwargs["messages"][0]["content"], "What is a neural network?"
        )

    @patch("claude_integration.client.anthropic.Anthropic")
    def test_chat_multi_turn(self, mock_anthropic_cls):
        """chat() passes all messages to the API."""
        mock_instance = mock_anthropic_cls.return_value
        mock_instance.messages.create.return_value = self._mock_response(
            "Sure, here is an example."
        )

        client = self._make_client(mock_anthropic_cls)
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Tell me more."},
        ]
        result = client.chat(messages)

        self.assertEqual(result, "Sure, here is an example.")
        call_kwargs = mock_instance.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["messages"], messages)

    @patch("claude_integration.client.anthropic.Anthropic")
    def test_custom_model_and_max_tokens(self, mock_anthropic_cls):
        """Per-call model and max_tokens overrides are forwarded to the API."""
        mock_instance = mock_anthropic_cls.return_value
        mock_instance.messages.create.return_value = self._mock_response(
            "Response"
        )

        client = self._make_client(mock_anthropic_cls)
        client.complete(
            "Hello", model="claude-3-haiku-20240307", max_tokens=128
        )

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "claude-3-haiku-20240307")
        self.assertEqual(call_kwargs["max_tokens"], 128)

    @patch("claude_integration.client.anthropic.Anthropic")
    def test_default_model_used_when_no_override(self, mock_anthropic_cls):
        """The default model is used when no override is supplied."""
        from claude_integration.client import DEFAULT_MODEL, DEFAULT_MAX_TOKENS

        mock_instance = mock_anthropic_cls.return_value
        mock_instance.messages.create.return_value = self._mock_response("OK")

        client = self._make_client(mock_anthropic_cls)
        client.complete("ping")

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(call_kwargs["max_tokens"], DEFAULT_MAX_TOKENS)


class TestBaseAIProvider(unittest.TestCase):
    """Tests for providers.base.BaseAIProvider."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseAIProvider cannot be instantiated directly."""
        from providers.base import BaseAIProvider
        with self.assertRaises(TypeError):
            BaseAIProvider()

    def test_concrete_subclass_must_implement_methods(self):
        """A subclass that omits abstract methods also cannot be instantiated."""
        from providers.base import BaseAIProvider

        class IncompleteProvider(BaseAIProvider):
            pass

        with self.assertRaises(TypeError):
            IncompleteProvider()

    def test_concrete_subclass_works(self):
        """A fully implemented subclass can be instantiated and called."""
        from providers.base import BaseAIProvider

        class DummyProvider(BaseAIProvider):
            @property
            def provider_name(self):
                return "Dummy"

            def complete(self, prompt, **kwargs):
                return f"echo: {prompt}"

            def chat(self, messages, **kwargs):
                return f"echo: {messages[-1]['content']}"

        provider = DummyProvider()
        self.assertEqual(provider.provider_name, "Dummy")
        self.assertEqual(provider.complete("hello"), "echo: hello")
        self.assertEqual(
            provider.chat([{"role": "user", "content": "hi"}]), "echo: hi"
        )


if __name__ == "__main__":
    unittest.main()
