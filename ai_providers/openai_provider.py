"""
OpenAI GPT provider adapter.

Requires the ``openai`` Python SDK::

    pip install openai

Set the ``OPENAI_API_KEY`` environment variable (or pass ``api_key``
directly) before using this provider.
"""

from typing import Optional

try:
    import openai
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'openai' package is required for OpenAIProvider. "
        "Install it with: pip install openai"
    ) from exc

from .base import BaseAIProvider


class OpenAIProvider(BaseAIProvider):
    """AI provider adapter for OpenAI GPT models."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialise the OpenAI provider.

        Args:
            api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
                environment variable when not provided.
            model: GPT model identifier (default: ``gpt-4o``).
        """
        self._client = openai.OpenAI(api_key=api_key)  # uses env var if None
        self._model = model or self.DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return "OpenAI GPT"

    def chat(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Send a message to an OpenAI GPT model and return the reply.

        Args:
            message: The user message.
            system_prompt: Optional system instruction.
            max_tokens: Maximum tokens in the response (default: 1024).
            temperature: Sampling temperature (default: 0.7).
            **kwargs: Additional keyword arguments forwarded to
                ``openai.OpenAI.chat.completions.create``.

        Returns:
            The assistant's response text.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except openai.OpenAIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        return response.choices[0].message.content
