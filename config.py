"""
Global configuration for the AIML multi-provider AI integration.

Values here act as defaults and can be overridden by environment variables.
Never store real API keys in this file — use a ``.env`` file or set
environment variables directly in your shell / CI pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Default AI provider
# Supported values: "claude", "openai", "gemini"
# Override with:  AI_PROVIDER=openai python your_script.py
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER: str = os.environ.get("AI_PROVIDER", "claude")

# ---------------------------------------------------------------------------
# Model overrides (optional)
# If empty the provider's own DEFAULT_MODEL constant is used.
# ---------------------------------------------------------------------------
CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "")
