# AIML

A modular, multi-provider AI integration library supporting **Anthropic Claude**, **OpenAI GPT**, and **Google Gemini** — with a clean interface for adding more providers in the future.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Providers](#providers)
  - [Anthropic Claude](#anthropic-claude)
  - [OpenAI GPT](#openai-gpt)
  - [Google Gemini](#google-gemini)
- [Using the Provider Factory](#using-the-provider-factory)
- [Adding a New Provider](#adding-a-new-provider)
- [Running Tests](#running-tests)

---

## Features

- **Modular provider interface** — swap between Claude, OpenAI, and Gemini with one env-var change.
- **Secure API key handling** — keys are read from environment variables or a `.env` file; never hardcoded.
- **Provider factory** — select the active provider via `AI_PROVIDER` env var or `config.py`.
- **Example scripts** for each provider and for the factory-based approach.
- **Extensible** — adding a new provider requires only a single new file.

---

## Project Structure

```
AIML/
├── ai_providers/
│   ├── __init__.py          # Package exports
│   ├── base.py              # Abstract BaseAIProvider interface
│   ├── claude.py            # Anthropic Claude adapter
│   ├── openai_provider.py   # OpenAI GPT adapter
│   ├── gemini.py            # Google Gemini adapter
│   └── factory.py           # Provider factory (env-var / config driven)
├── examples/
│   ├── example_claude.py    # Claude usage demo
│   ├── example_openai.py    # OpenAI usage demo
│   ├── example_gemini.py    # Gemini usage demo
│   └── example_factory.py   # Factory / env-var switching demo
├── tests/
│   └── test_providers.py    # Unit tests (no real API keys needed)
├── config.py                # Default configuration
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
└── .gitignore
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/adityarajgavali240-stack/AIML.git
cd AIML
pip install -r requirements.txt
```

### 2. Set up environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Then edit .env with your favourite editor
```

Or export them directly in your shell:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
export AI_PROVIDER="claude"   # claude | openai | gemini
```

### 3. Run an example

```bash
# Claude
python examples/example_claude.py

# OpenAI
python examples/example_openai.py

# Gemini
python examples/example_gemini.py

# Factory — reads AI_PROVIDER env var
AI_PROVIDER=openai python examples/example_factory.py
```

---

## Configuration

| Environment Variable  | Description                                             | Default                         |
|-----------------------|---------------------------------------------------------|---------------------------------|
| `AI_PROVIDER`         | Active provider (`claude`, `openai`, `gemini`)          | `claude`                        |
| `ANTHROPIC_API_KEY`   | Anthropic API key                                       | *(required for Claude)*         |
| `OPENAI_API_KEY`      | OpenAI API key                                         | *(required for OpenAI)*         |
| `GOOGLE_API_KEY`      | Google API key                                          | *(required for Gemini)*         |
| `CLAUDE_MODEL`        | Override Claude model                                   | `claude-3-5-sonnet-20241022`    |
| `OPENAI_MODEL`        | Override OpenAI model                                   | `gpt-4o`                        |
| `GEMINI_MODEL`        | Override Gemini model                                   | `gemini-1.5-flash`              |

You can also edit `config.py` to set persistent defaults.

> **Security note:** API keys must _never_ be committed to source control.
> Add them to `.env` (which is git-ignored) or use CI/CD secrets.

---

## Providers

### Anthropic Claude

```python
from ai_providers.claude import ClaudeProvider

provider = ClaudeProvider()  # reads ANTHROPIC_API_KEY from env
reply = provider.chat("Explain transformers in one sentence.")
print(reply)

# With a system prompt
reply = provider.chat(
    "List three benefits of fine-tuning.",
    system_prompt="You are a concise ML tutor.",
    max_tokens=512,
)
```

Default model: `claude-3-5-sonnet-20241022`  
[Anthropic API docs →](https://docs.anthropic.com/)

---

### OpenAI GPT

```python
from ai_providers.openai_provider import OpenAIProvider

provider = OpenAIProvider()  # reads OPENAI_API_KEY from env
reply = provider.chat("What is GPT-4?")
print(reply)

# With a system prompt and temperature
reply = provider.chat(
    "Summarise the attention mechanism.",
    system_prompt="You are an AI research assistant.",
    temperature=0.3,
)
```

Default model: `gpt-4o`  
[OpenAI API docs →](https://platform.openai.com/docs/)

---

### Google Gemini

```python
from ai_providers.gemini import GeminiProvider

provider = GeminiProvider()  # reads GOOGLE_API_KEY from env
reply = provider.chat("What is reinforcement learning?")
print(reply)

# With a system prompt
reply = provider.chat(
    "Explain overfitting.",
    system_prompt="You are a helpful ML expert. Be concise.",
)
```

Default model: `gemini-1.5-flash`  
[Google AI docs →](https://ai.google.dev/)

---

## Using the Provider Factory

The factory lets you swap providers without changing your code:

```python
import os
from ai_providers import get_provider, list_providers

print(list_providers())   # ['claude', 'openai', 'gemini']

# Reads AI_PROVIDER env var (or config.DEFAULT_PROVIDER)
ai = get_provider()
reply = ai.chat("Hello!")

# Or override explicitly
ai = get_provider("gemini")
reply = ai.chat("Tell me about GANs.")
```

Switch providers at runtime:

```bash
AI_PROVIDER=gemini python your_script.py
```

---

## Adding a New Provider

1. Create `ai_providers/my_provider.py` with a class that inherits from `BaseAIProvider`:

```python
from .base import BaseAIProvider

class MyProvider(BaseAIProvider):
    @property
    def provider_name(self) -> str:
        return "My Custom Provider"

    def chat(self, message, *, system_prompt=None, **kwargs) -> str:
        # Call your API here
        return "response from my provider"
```

2. Register it in `ai_providers/factory.py`:

```python
_REGISTRY = {
    ...
    "myprovider": "ai_providers.my_provider.MyProvider",
}
```

3. Use it:

```bash
AI_PROVIDER=myprovider python your_script.py
```

---

## Running Tests

Tests use `unittest` and mock all external API calls — no real keys required:

```bash
pip install pytest
python -m pytest tests/ -v
```
