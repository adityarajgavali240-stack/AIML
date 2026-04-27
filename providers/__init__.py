"""
providers package

Contains the base interface for AI provider integrations.
Each new provider should subclass BaseAIProvider.
"""

from providers.base import BaseAIProvider

__all__ = ["BaseAIProvider"]
