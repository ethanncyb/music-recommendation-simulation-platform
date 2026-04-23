"""
LLM Provider abstraction layer.

Supports two backends:
  - Ollama (default): free, local, no API key needed
  - Anthropic (optional): cloud-based, requires ANTHROPIC_API_KEY env var

Usage:
    from src.llm_provider import get_provider

    llm = get_provider("ollama", model="llama3.1:8b")
    response = llm.generate("Say hello")

    llm = get_provider("anthropic")
    response = llm.generate("Say hello")
"""

from abc import ABC, abstractmethod
from typing import Optional
import json
import os


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Send a prompt and return the raw text response."""
        ...

    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict:
        """Send a prompt and parse the response as JSON.

        Strips markdown code fences if present before parsing.
        """
        raw = self.generate(prompt, system)
        cleaned = raw.strip()
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1:]
            if "```" in cleaned:
                cleaned = cleaned[:cleaned.rindex("```")]
            cleaned = cleaned.strip()
        return json.loads(cleaned)


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama (default provider)."""

    def __init__(self, model: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self):
        """Verify Ollama is reachable. Raises ConnectionError if not."""
        import urllib.request
        try:
            urllib.request.urlopen(self.base_url, timeout=5)
        except Exception:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Install Ollama (https://ollama.com) and start it with: ollama serve"
            )

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        import urllib.request
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        }
        if system:
            payload["system"] = system
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))["response"]


class AnthropicProvider(LLMProvider):
    """Cloud LLM via Anthropic API (optional provider)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Set ANTHROPIC_API_KEY environment variable to use the Anthropic provider"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return resp.content[0].text


def get_provider(backend: str = "ollama", **kwargs) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        backend: "ollama" (default) or "anthropic"
        **kwargs: passed to the provider constructor (e.g., model, base_url)
    """
    if backend == "ollama":
        return OllamaProvider(**kwargs)
    elif backend == "anthropic":
        return AnthropicProvider(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama' or 'anthropic'.")
