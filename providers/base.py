"""
Base interface for AI provider integrations.

To add a new AI provider:
1. Create a new module (e.g., openai_integration/).
2. Subclass BaseAIProvider and implement all abstract methods.
3. Register the provider in the appropriate factory or example code.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseAIProvider(ABC):
    """Abstract base class for AI provider clients."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Send a single-turn completion request.

        Args:
            prompt: The user prompt or instruction.
            **kwargs: Provider-specific parameters (e.g., max_tokens, temperature).

        Returns:
            The model's response as a plain string.
        """

    @abstractmethod
    def chat(self, messages: list, **kwargs) -> str:
        """
        Send a multi-turn conversation request.

        Args:
            messages: A list of message dicts with 'role' and 'content' keys.
                      Roles are typically 'user' and 'assistant'.
            **kwargs: Provider-specific parameters (e.g., max_tokens, temperature).

        Returns:
            The model's response as a plain string.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of the AI provider."""
