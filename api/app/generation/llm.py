# api/app/generation/llm.py
"""
LLM client with Groq (dev, free) and Anthropic (final demo) backends.

Provider is selected via LLM_PROVIDER env var:
  groq      → Llama 3.3 70B via Groq Cloud (free tier, fast)
  anthropic → Claude Sonnet (use for final demo / screenshots)

Both paths return a plain string (the full completion). Streaming is handled
separately in the SSE route — this module keeps generation logic simple and
testable without HTTP concerns.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "groq").lower()


# ---------------------------------------------------------------------------
# Groq path
# ---------------------------------------------------------------------------

def _call_groq(system: str, user: str) -> str:
    from groq import Groq  # type: ignore

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,   # low temperature: we want factual, grounded answers
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Anthropic path
# ---------------------------------------------------------------------------

def _call_anthropic(system: str, user: str) -> str:
    import anthropic  # type: ignore

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(system: str, user: str) -> str:
    """
    Call the configured LLM and return the response as a plain string.

    Args:
        system: System prompt (instructions + sources).
        user:   User turn (the question).

    Returns:
        Raw LLM response text, which will contain [SOURCE_X] citation tokens.
    """
    provider = _provider()

    if provider == "anthropic":
        return _call_anthropic(system, user)
    elif provider == "groq":
        return _call_groq(system, user)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Set to 'groq' or 'anthropic'."
        )
