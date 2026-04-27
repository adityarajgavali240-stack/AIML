# AIML

A repository for AI/ML experiments and integrations.  
Currently includes a modular **Anthropic Claude** integration that is designed to be extended to other AI providers.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration (API Key)](#configuration-api-key)
- [Quick Start](#quick-start)
- [Running the Example Script](#running-the-example-script)
- [Running Tests](#running-tests)
- [Adding a New AI Provider](#adding-a-new-ai-provider)
- [Security Best Practices](#security-best-practices)

---

## Project Structure

```
AIML/
├── .env.example              # Template for environment variables
├── .gitignore                # Excludes .env and build artifacts
├── requirements.txt          # Python dependencies
│
├── providers/
│   ├── __init__.py
│   └── base.py               # Abstract BaseAIProvider interface
│
├── claude_integration/
│   ├── __init__.py
│   ├── config.py             # Reads ANTHROPIC_API_KEY from the environment
│   └── client.py             # ClaudeClient – implements BaseAIProvider
│
├── examples/
│   └── claude_example.py     # Runnable demo (completion + conversation)
│
└── tests/
    └── test_claude_client.py # Unit tests (no API key required)
```

---

## Prerequisites

- Python 3.9 or higher
- An [Anthropic account](https://console.anthropic.com/) with an API key

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/adityarajgavali240-stack/AIML.git
cd AIML

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration (API Key)

The API key is read from the `ANTHROPIC_API_KEY` **environment variable**.  
**Never hardcode secrets in source code.**

```bash
# Option A – .env file (recommended for local development)
cp .env.example .env
# Edit .env and replace the placeholder with your real key:
#   ANTHROPIC_API_KEY=sk-ant-...

# Option B – export directly to your shell
export ANTHROPIC_API_KEY="sk-ant-..."

# Option C – CI/CD (GitHub Actions, etc.)
# Add ANTHROPIC_API_KEY as a repository secret and reference it in your workflow.
```

---

## Quick Start

```python
from claude_integration import ClaudeClient

# The key is loaded automatically from ANTHROPIC_API_KEY
client = ClaudeClient()

# Single-turn completion
response = client.complete("Explain neural networks in one paragraph.")
print(response)

# Multi-turn conversation
messages = [
    {"role": "user", "content": "What is reinforcement learning?"},
]
reply = client.chat(messages)
print(reply)

# Continue the conversation
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "Give me a real-world example."})
follow_up = client.chat(messages)
print(follow_up)
```

---

## Running the Example Script

```bash
python examples/claude_example.py
```

The script demonstrates:
1. Single-turn text completion
2. Multi-turn conversation
3. Overriding the model and `max_tokens` per call

---

## Running Tests

The tests mock the Anthropic SDK, so **no API key is needed** to run them.

```bash
pip install pytest          # if not already installed
pytest tests/ -v
```

---

## Adding a New AI Provider

The integration is intentionally modular.  To add, for example, an OpenAI client:

1. Create a new package: `openai_integration/`
2. Subclass `BaseAIProvider` from `providers.base`:

```python
from providers.base import BaseAIProvider

class OpenAIClient(BaseAIProvider):
    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def complete(self, prompt: str, **kwargs) -> str:
        ...

    def chat(self, messages: list, **kwargs) -> str:
        ...
```

3. Add the new dependency to `requirements.txt`.
4. Document the new `API_KEY` variable in `.env.example`.

---

## Security Best Practices

- **Do not commit `.env`** – it is listed in `.gitignore`.
- Use `.env.example` as a template; fill in real values only in `.env`.
- In production, inject secrets via your deployment platform's secret manager (e.g. GitHub Secrets, AWS Secrets Manager, Vault).
- Rotate your API key immediately if it is ever accidentally exposed.

