"""
Base AI provider interface.

All concrete provider implementations must subclass ``BaseAIProvider``
and implement the abstract methods defined here.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseAIProvider(ABC):
    """Abstract base class for AI provider adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable name for this provider."""

    @abstractmethod
    def chat(self, message: str, *, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Send a single user message and return the assistant's reply as a string.

        Args:
            message: The user message to send.
            system_prompt: Optional system-level instruction for the model.
            **kwargs: Provider-specific parameters (e.g. temperature, max_tokens).

        Returns:
            The assistant's response as a plain string.

        Raises:
            RuntimeError: If the API call fails.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name!r}>"
