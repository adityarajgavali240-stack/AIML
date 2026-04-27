"""
Example: Using the Anthropic Claude provider.

Prerequisites
-------------
1. Install dependencies::

       pip install -r requirements.txt

2. Set your Anthropic API key::

       export ANTHROPIC_API_KEY="sk-ant-..."

3. Run this script::

       python examples/example_claude.py
"""

import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_providers.claude import ClaudeProvider


def main() -> None:
    provider = ClaudeProvider()
    print(f"Provider : {provider.provider_name}")

    # --- Simple chat ---
    reply = provider.chat("What is machine learning in one sentence?")
    print(f"Reply    : {reply}")

    # --- Chat with a system prompt ---
    reply = provider.chat(
        "List three benefits of neural networks.",
        system_prompt="You are a concise AI/ML tutor. Keep every answer to 2–3 sentences.",
    )
    print(f"\nWith system prompt:\n{reply}")


if __name__ == "__main__":
    main()
