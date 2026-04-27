"""
Example: Using the provider factory to switch providers via env var.

Set AI_PROVIDER to "claude", "openai", or "gemini" before running::

    AI_PROVIDER=openai python examples/example_factory.py

The factory reads AI_PROVIDER automatically (falls back to config.DEFAULT_PROVIDER).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_providers import get_provider, list_providers


def main() -> None:
    print(f"Available providers: {list_providers()}")

    ai = get_provider()
    print(f"Active provider   : {ai.provider_name}")

    reply = ai.chat(
        "Briefly explain what an AI provider abstraction layer is and why it is useful.",
        system_prompt="You are a software architect. Answer in 2–3 sentences.",
    )
    print(f"\nResponse:\n{reply}")


if __name__ == "__main__":
    main()
