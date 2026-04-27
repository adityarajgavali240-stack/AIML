"""
Anthropic Claude client.

Implements the BaseAIProvider interface so this integration can be swapped
out for any other provider without changing the calling code.
"""

from typing import Optional

import anthropic

from claude_integration.config import get_api_key
from providers.base import BaseAIProvider

# Default model – can be overridden per-call via kwargs
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 1024


class ClaudeClient(BaseAIProvider):
    """
    Client for Anthropic's Claude models.

    Usage::

        from claude_integration import ClaudeClient

        client = ClaudeClient()
        response = client.complete("Explain neural networks in one paragraph.")
        print(response)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """
        Initialize the Claude client.

        Args:
            api_key:    Anthropic API key.  When *None* the key is read from
                        the ``ANTHROPIC_API_KEY`` environment variable.
            model:      Default Claude model to use for requests.
            max_tokens: Default maximum number of tokens in the response.
        """
        resolved_key = api_key if api_key is not None else get_api_key()
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # BaseAIProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "Anthropic Claude"

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Send a single-turn text-completion request to Claude.

        Args:
            prompt:     The user prompt.
            **kwargs:   Override ``model`` or ``max_tokens`` for this call.

        Returns:
            The model's text response.
        """
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages: list, **kwargs) -> str:
        """
        Send a multi-turn conversation request to Claude.

        Args:
            messages:   List of dicts with ``role`` (``"user"`` /
                        ``"assistant"``) and ``content`` keys.
            **kwargs:   Override ``model`` or ``max_tokens`` for this call.

        Returns:
            The model's text response.
        """
        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.content[0].text
