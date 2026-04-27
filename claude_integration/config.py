"""
Configuration module for claude_integration.

Reads the Anthropic API key from the environment (or a .env file).
Never hardcode secrets in source code.
"""

import os

from dotenv import load_dotenv

# Load variables from a .env file if present (ignored in production environments
# where variables are injected directly into the process environment).
load_dotenv()


def get_api_key() -> str:
    """
    Return the Anthropic API key from the environment.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your API key, "
            "or set the variable in your environment."
        )
    return api_key
