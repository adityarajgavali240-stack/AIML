"""
Google Gemini provider adapter.

Requires the ``google-generativeai`` Python SDK::

    pip install google-generativeai

Set the ``GOOGLE_API_KEY`` environment variable (or pass ``api_key``
directly) before using this provider.
"""

from typing import Optional

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'google-generativeai' package is required for GeminiProvider. "
        "Install it with: pip install google-generativeai"
    ) from exc

from .base import BaseAIProvider


class GeminiProvider(BaseAIProvider):
    """AI provider adapter for Google Gemini models."""

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialise the Gemini provider.

        Args:
            api_key: Google API key.  Falls back to the ``GOOGLE_API_KEY``
                environment variable when not provided.
            model: Gemini model identifier (default: ``gemini-1.5-flash``).
        """
        import os

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Google API key is required. Set the GOOGLE_API_KEY "
                "environment variable or pass api_key= to GeminiProvider."
            )
        genai.configure(api_key=resolved_key)
        self._model = genai.GenerativeModel(model or self.DEFAULT_MODEL)
        self._model_name = model or self.DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    def chat(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Send a message to Google Gemini and return the reply.

        Args:
            message: The user message.
            system_prompt: Optional system instruction prepended to the
                user message (Gemini does not have a dedicated system role
                in the basic API).
            **kwargs: Additional keyword arguments forwarded to
                ``GenerativeModel.generate_content``.

        Returns:
            The assistant's response text.
        """
        prompt = f"{system_prompt}\n\n{message}" if system_prompt else message

        try:
            response = self._model.generate_content(prompt, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc

        return response.text
