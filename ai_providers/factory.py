"""
Provider factory — resolves the active AI provider from configuration.

The provider is chosen according to the following priority:

1. The ``provider`` argument passed directly to :func:`get_provider`.
2. The ``AI_PROVIDER`` environment variable.
3. The ``config.py`` ``DEFAULT_PROVIDER`` setting.

Example::

    from ai_providers import get_provider

    ai = get_provider()          # uses config / env var
    ai = get_provider("claude")  # explicit override
    reply = ai.chat("Hello!")
"""

from typing import Optional

from .base import BaseAIProvider

_REGISTRY: dict[str, str] = {
    "claude": "ai_providers.claude.ClaudeProvider",
    "openai": "ai_providers.openai_provider.OpenAIProvider",
    "gemini": "ai_providers.gemini.GeminiProvider",
}


def list_providers() -> list[str]:
    """Return a list of all registered provider keys."""
    return list(_REGISTRY.keys())


def get_provider(provider: Optional[str] = None, **kwargs) -> BaseAIProvider:
    """
    Instantiate and return the requested AI provider.

    Args:
        provider: One of ``"claude"``, ``"openai"``, or ``"gemini"``.
            When *None*, the value is read from the ``AI_PROVIDER``
            environment variable, then from ``config.DEFAULT_PROVIDER``.
        **kwargs: Keyword arguments forwarded to the provider constructor
            (e.g. ``api_key``, ``model``).

    Returns:
        A :class:`~ai_providers.base.BaseAIProvider` instance.

    Raises:
        ValueError: If the provider name is unknown.
    """
    import importlib
    import os

    if provider is None:
        provider = os.environ.get("AI_PROVIDER")

    if provider is None:
        try:
            from config import DEFAULT_PROVIDER  # type: ignore[import]

            provider = DEFAULT_PROVIDER
        except ImportError:
            provider = "claude"

    provider = provider.lower().strip()

    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Available providers: {list(_REGISTRY.keys())}"
        )

    module_path, class_name = _REGISTRY[provider].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
