"""
AI Providers package — modular multi-provider AI integration.

Supported providers:
- claude   : Anthropic Claude (via anthropic SDK)
- openai   : OpenAI GPT (via openai SDK)
- gemini   : Google Gemini (via google-generativeai SDK)
"""

from .base import BaseAIProvider
from .factory import get_provider, list_providers

__all__ = ["BaseAIProvider", "get_provider", "list_providers"]
