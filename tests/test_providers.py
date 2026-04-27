"""
Unit tests for the AI provider integration.

All external API calls are mocked so the tests run without real API keys.
"""

import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers: build lightweight fake SDK modules so provider imports succeed
# even when the real packages are not installed.
# ---------------------------------------------------------------------------


def _make_anthropic_stub():
    stub = types.ModuleType("anthropic")
    stub.Anthropic = MagicMock()
    stub.APIError = Exception
    return stub


def _make_openai_stub():
    stub = types.ModuleType("openai")
    stub.OpenAI = MagicMock()
    stub.OpenAIError = Exception
    return stub


def _make_genai_stub():
    google_stub = types.ModuleType("google")
    generativeai_stub = types.ModuleType("google.generativeai")
    generativeai_stub.configure = MagicMock()
    generativeai_stub.GenerativeModel = MagicMock()
    google_stub.generativeai = generativeai_stub
    return google_stub, generativeai_stub


# ---------------------------------------------------------------------------
# Base provider tests
# ---------------------------------------------------------------------------


class TestBaseProvider(unittest.TestCase):
    def test_cannot_instantiate_directly(self):
        from ai_providers.base import BaseAIProvider

        with self.assertRaises(TypeError):
            BaseAIProvider()  # abstract — should raise

    def test_concrete_subclass_works(self):
        from ai_providers.base import BaseAIProvider

        class DummyProvider(BaseAIProvider):
            @property
            def provider_name(self):
                return "Dummy"

            def chat(self, message, *, system_prompt=None, **kwargs):
                return f"echo: {message}"

        p = DummyProvider()
        self.assertEqual(p.provider_name, "Dummy")
        self.assertEqual(p.chat("hi"), "echo: hi")
        self.assertIn("Dummy", repr(p))


# ---------------------------------------------------------------------------
# Claude provider tests
# ---------------------------------------------------------------------------


class TestClaudeProvider(unittest.TestCase):
    def setUp(self):
        self._stub = _make_anthropic_stub()
        sys.modules["anthropic"] = self._stub
        # Force reload so stub is picked up
        if "ai_providers.claude" in sys.modules:
            del sys.modules["ai_providers.claude"]

    def tearDown(self):
        sys.modules.pop("anthropic", None)
        sys.modules.pop("ai_providers.claude", None)

    def _make_provider(self, api_key="test-key", model=None):
        from ai_providers.claude import ClaudeProvider

        return ClaudeProvider(api_key=api_key, model=model)

    def test_provider_name(self):
        p = self._make_provider()
        self.assertEqual(p.provider_name, "Anthropic Claude")

    def test_chat_returns_text(self):
        # Configure mock response
        fake_text = "Hello from Claude!"
        fake_content = MagicMock()
        fake_content.text = fake_text
        fake_response = MagicMock()
        fake_response.content = [fake_content]
        self._stub.Anthropic.return_value.messages.create.return_value = fake_response

        p = self._make_provider()
        result = p.chat("Hi")
        self.assertEqual(result, fake_text)

    def test_chat_with_system_prompt(self):
        fake_content = MagicMock()
        fake_content.text = "response"
        fake_response = MagicMock()
        fake_response.content = [fake_content]
        self._stub.Anthropic.return_value.messages.create.return_value = fake_response

        p = self._make_provider()
        p.chat("Hello", system_prompt="Be concise.")
        call_kwargs = self._stub.Anthropic.return_value.messages.create.call_args[1]
        self.assertEqual(call_kwargs["system"], "Be concise.")

    def test_api_error_raises_runtime_error(self):
        self._stub.Anthropic.return_value.messages.create.side_effect = Exception("API down")
        p = self._make_provider()
        with self.assertRaises(RuntimeError):
            p.chat("Hello")

    def test_default_model(self):
        from ai_providers.claude import ClaudeProvider

        p = ClaudeProvider(api_key="x")
        self.assertEqual(p._model, ClaudeProvider.DEFAULT_MODEL)

    def test_custom_model(self):
        p = self._make_provider(model="claude-3-haiku-20240307")
        self.assertEqual(p._model, "claude-3-haiku-20240307")


# ---------------------------------------------------------------------------
# OpenAI provider tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider(unittest.TestCase):
    def setUp(self):
        self._stub = _make_openai_stub()
        sys.modules["openai"] = self._stub
        if "ai_providers.openai_provider" in sys.modules:
            del sys.modules["ai_providers.openai_provider"]

    def tearDown(self):
        sys.modules.pop("openai", None)
        sys.modules.pop("ai_providers.openai_provider", None)

    def _make_provider(self, api_key="test-key", model=None):
        from ai_providers.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=api_key, model=model)

    def test_provider_name(self):
        p = self._make_provider()
        self.assertEqual(p.provider_name, "OpenAI GPT")

    def test_chat_returns_text(self):
        fake_text = "Hello from GPT!"
        fake_message = MagicMock()
        fake_message.content = fake_text
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        self._stub.OpenAI.return_value.chat.completions.create.return_value = fake_response

        p = self._make_provider()
        result = p.chat("Hi")
        self.assertEqual(result, fake_text)

    def test_system_prompt_included(self):
        fake_message = MagicMock()
        fake_message.content = "ok"
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        self._stub.OpenAI.return_value.chat.completions.create.return_value = fake_response

        p = self._make_provider()
        p.chat("Hello", system_prompt="You are helpful.")
        call_kwargs = self._stub.OpenAI.return_value.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful.")
        self.assertEqual(messages[1]["role"], "user")

    def test_api_error_raises_runtime_error(self):
        self._stub.OpenAI.return_value.chat.completions.create.side_effect = Exception("fail")
        p = self._make_provider()
        with self.assertRaises(RuntimeError):
            p.chat("Hi")

    def test_default_model(self):
        from ai_providers.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="x")
        self.assertEqual(p._model, OpenAIProvider.DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Gemini provider tests
# ---------------------------------------------------------------------------


class TestGeminiProvider(unittest.TestCase):
    def setUp(self):
        google_stub, genai_stub = _make_genai_stub()
        sys.modules["google"] = google_stub
        sys.modules["google.generativeai"] = genai_stub
        self._genai_stub = genai_stub
        if "ai_providers.gemini" in sys.modules:
            del sys.modules["ai_providers.gemini"]

    def tearDown(self):
        sys.modules.pop("google", None)
        sys.modules.pop("google.generativeai", None)
        sys.modules.pop("ai_providers.gemini", None)

    def _make_provider(self, api_key="test-key", model=None):
        from ai_providers.gemini import GeminiProvider

        return GeminiProvider(api_key=api_key, model=model)

    def test_provider_name(self):
        p = self._make_provider()
        self.assertEqual(p.provider_name, "Google Gemini")

    def test_chat_returns_text(self):
        fake_response = MagicMock()
        fake_response.text = "Hello from Gemini!"
        self._genai_stub.GenerativeModel.return_value.generate_content.return_value = fake_response

        p = self._make_provider()
        result = p.chat("Hi")
        self.assertEqual(result, "Hello from Gemini!")

    def test_system_prompt_prepended(self):
        fake_response = MagicMock()
        fake_response.text = "ok"
        self._genai_stub.GenerativeModel.return_value.generate_content.return_value = fake_response

        p = self._make_provider()
        p.chat("Tell me something.", system_prompt="Context:")
        call_args = self._genai_stub.GenerativeModel.return_value.generate_content.call_args
        prompt_used = call_args[0][0]
        self.assertIn("Context:", prompt_used)
        self.assertIn("Tell me something.", prompt_used)

    def test_missing_api_key_raises(self):
        from ai_providers.gemini import GeminiProvider

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            with self.assertRaises(ValueError):
                GeminiProvider(api_key=None)

    def test_api_error_raises_runtime_error(self):
        self._genai_stub.GenerativeModel.return_value.generate_content.side_effect = Exception(
            "API error"
        )
        p = self._make_provider()
        with self.assertRaises(RuntimeError):
            p.chat("Hi")


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory(unittest.TestCase):
    def setUp(self):
        # Register stubs so imports in factory succeed
        self._anthropic_stub = _make_anthropic_stub()
        self._openai_stub = _make_openai_stub()
        google_stub, genai_stub = _make_genai_stub()
        sys.modules["anthropic"] = self._anthropic_stub
        sys.modules["openai"] = self._openai_stub
        sys.modules["google"] = google_stub
        sys.modules["google.generativeai"] = genai_stub
        self._genai_stub = genai_stub

        for mod in [
            "ai_providers.claude",
            "ai_providers.openai_provider",
            "ai_providers.gemini",
            "ai_providers.factory",
            "ai_providers",
        ]:
            sys.modules.pop(mod, None)

    def tearDown(self):
        for key in list(sys.modules.keys()):
            if key.startswith("ai_providers") or key in ("anthropic", "openai", "google"):
                sys.modules.pop(key, None)

    def test_list_providers(self):
        from ai_providers.factory import list_providers

        providers = list_providers()
        self.assertIn("claude", providers)
        self.assertIn("openai", providers)
        self.assertIn("gemini", providers)

    def test_get_provider_claude(self):
        from ai_providers.factory import get_provider

        p = get_provider("claude", api_key="test")
        self.assertEqual(p.provider_name, "Anthropic Claude")

    def test_get_provider_openai(self):
        from ai_providers.factory import get_provider

        p = get_provider("openai", api_key="test")
        self.assertEqual(p.provider_name, "OpenAI GPT")

    def test_get_provider_gemini(self):
        from ai_providers.factory import get_provider

        p = get_provider("gemini", api_key="test")
        self.assertEqual(p.provider_name, "Google Gemini")

    def test_get_provider_env_var(self):
        with patch.dict(os.environ, {"AI_PROVIDER": "openai"}):
            from ai_providers.factory import get_provider

            p = get_provider(api_key="test")
            self.assertEqual(p.provider_name, "OpenAI GPT")

    def test_get_provider_unknown_raises(self):
        from ai_providers.factory import get_provider

        with self.assertRaises(ValueError):
            get_provider("unknown_provider")

    def test_get_provider_case_insensitive(self):
        from ai_providers.factory import get_provider

        p = get_provider("Claude", api_key="test")
        self.assertEqual(p.provider_name, "Anthropic Claude")


if __name__ == "__main__":
    unittest.main()
