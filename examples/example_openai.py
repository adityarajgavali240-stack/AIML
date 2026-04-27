"""
Example: Using the OpenAI GPT provider.

Prerequisites
-------------
1. Install dependencies::

       pip install -r requirements.txt

2. Set your OpenAI API key::

       export OPENAI_API_KEY="sk-..."

3. Run this script::

       python examples/example_openai.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_providers.openai_provider import OpenAIProvider


def main() -> None:
    provider = OpenAIProvider()
    print(f"Provider : {provider.provider_name}")

    # --- Simple chat ---
    reply = provider.chat("What is deep learning in one sentence?")
    print(f"Reply    : {reply}")

    # --- Chat with a system prompt ---
    reply = provider.chat(
        "Give me three real-world applications of GPT models.",
        system_prompt="You are a helpful AI/ML teacher. Be concise and use bullet points.",
        temperature=0.5,
    )
    print(f"\nWith system prompt:\n{reply}")


if __name__ == "__main__":
    main()
