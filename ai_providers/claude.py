"""
Anthropic Claude provider adapter.

Requires the ``anthropic`` Python SDK::

    pip install anthropic

Set the ``ANTHROPIC_API_KEY`` environment variable (or pass ``api_key``
directly) before using this provider.
"""

from typing import Optional

try:
    import anthropic
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'anthropic' package is required for ClaudeProvider. "
        "Install it with: pip install anthropic"
    ) from exc

from .base import BaseAIProvider


class ClaudeProvider(BaseAIProvider):
    """AI provider adapter for Anthropic Claude models."""

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialise the Claude provider.

        Args:
            api_key: Anthropic API key.  Falls back to the
                ``ANTHROPIC_API_KEY`` environment variable when not provided.
            model: Claude model identifier (default: ``claude-3-5-sonnet-20241022``).
        """
        self._client = anthropic.Anthropic(api_key=api_key)  # uses env var if None
        self._model = model or self.DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return "Anthropic Claude"

    def chat(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """
        Send a message to Claude and return the reply.

        Args:
            message: The user message.
            system_prompt: Optional system instruction.
            max_tokens: Maximum tokens in the response (default: 1024).
            **kwargs: Additional keyword arguments forwarded to the
                ``anthropic.Anthropic.messages.create`` call.

        Returns:
            The assistant's response text.
        """
        create_kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": message}],
            **kwargs,
        )
        if system_prompt:
            create_kwargs["system"] = system_prompt

        try:
            response = self._client.messages.create(**create_kwargs)
        except anthropic.APIError as exc:
            raise RuntimeError(f"Claude API error: {exc}") from exc

        return response.content[0].text
