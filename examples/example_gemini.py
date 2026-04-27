"""
Example: Using the Google Gemini provider.

Prerequisites
-------------
1. Install dependencies::

       pip install -r requirements.txt

2. Set your Google API key::

       export GOOGLE_API_KEY="AIza..."

3. Run this script::

       python examples/example_gemini.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_providers.gemini import GeminiProvider


def main() -> None:
    provider = GeminiProvider()
    print(f"Provider : {provider.provider_name}")

    # --- Simple chat ---
    reply = provider.chat("What is reinforcement learning in one sentence?")
    print(f"Reply    : {reply}")

    # --- Chat with a system prompt (prepended as context) ---
    reply = provider.chat(
        "Explain the difference between supervised and unsupervised learning.",
        system_prompt="You are a knowledgeable AI/ML expert. Keep your answer short and structured.",
    )
    print(f"\nWith system prompt:\n{reply}")


if __name__ == "__main__":
    main()
