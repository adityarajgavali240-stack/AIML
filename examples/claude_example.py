"""
Example: Using the Claude AI client for text completion and conversation.

Before running this script:
1. Copy .env.example to .env in the project root.
2. Set ANTHROPIC_API_KEY in .env (or export it to your shell).
3. Install dependencies:  pip install -r requirements.txt
4. Run:  python examples/claude_example.py
"""

import sys
import os

# Allow running from the repo root or from this directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from claude_integration import ClaudeClient


def example_completion() -> None:
    """Single-turn text completion example."""
    print("=== Text Completion ===")
    client = ClaudeClient()
    response = client.complete(
        "Explain what a neural network is in two sentences."
    )
    print(f"Claude: {response}\n")


def example_conversation() -> None:
    """Multi-turn conversation example."""
    print("=== Multi-turn Conversation ===")
    client = ClaudeClient()

    messages = [
        {"role": "user", "content": "What is reinforcement learning?"},
    ]
    first_reply = client.chat(messages)
    print(f"User  : {messages[0]['content']}")
    print(f"Claude: {first_reply}\n")

    # Continue the conversation
    messages.append({"role": "assistant", "content": first_reply})
    messages.append(
        {"role": "user", "content": "Give me a real-world example of it."}
    )
    second_reply = client.chat(messages)
    print(f"User  : {messages[-1]['content']}")
    print(f"Claude: {second_reply}\n")


def example_custom_model() -> None:
    """Demonstrate overriding the model and max_tokens per call."""
    print("=== Custom Model / Max Tokens ===")
    client = ClaudeClient()
    response = client.complete(
        "List three benefits of using large language models.",
        model="claude-3-haiku-20240307",
        max_tokens=256,
    )
    print(f"Claude (Haiku): {response}\n")


if __name__ == "__main__":
    try:
        example_completion()
        example_conversation()
        example_custom_model()
    except EnvironmentError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
