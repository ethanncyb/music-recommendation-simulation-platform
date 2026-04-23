"""
LLM Provider abstraction layer.

Supports three backends:
  - Ollama (default): free, local, no API key needed
  - Anthropic (optional): cloud-based, requires ANTHROPIC_API_KEY env var
  - Gemini (optional): Google AI, requires GEMINI_API_KEY env var

Usage:
    from src.llm_provider import get_provider

    llm = get_provider("ollama", model="llama3.2")
    response = llm.generate("Say hello")

    llm = get_provider("anthropic")
    response = llm.generate("Say hello")

    llm = get_provider("gemini")
    response = llm.generate("Say hello")
"""

from abc import ABC, abstractmethod
from typing import Optional
import json
import os
from .env_config import load_dotenv


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

    def __init__(self, model: Optional[str] = None,
                 base_url: Optional[str] = None):
        load_dotenv()
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
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


class GeminiProvider(LLMProvider):
    """Google Gemini API provider (free tier supported)."""

    def __init__(self, model: str = "gemma-4-27b-it"):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GEMINI_API_KEY environment variable to use the Gemini provider. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        contents = []
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System instructions]: {system}"}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow those instructions."}],
            })
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 4096,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            try:
                error_detail = json.loads(error_body)
                msg = error_detail.get("error", {}).get("message", error_body)
            except json.JSONDecodeError:
                msg = error_body
            if e.code == 429:
                raise ConnectionError(
                    f"Gemini API rate limit exceeded (HTTP 429). "
                    f"Free tier has limited requests per minute. "
                    f"Wait a moment and retry. Detail: {msg}"
                )
            elif e.code == 403:
                raise ConnectionError(
                    f"Gemini API access denied (HTTP 403). "
                    f"Check your GEMINI_API_KEY and ensure the model '{self.model}' "
                    f"is available. Detail: {msg}"
                )
            elif e.code == 400:
                raise ValueError(
                    f"Gemini API bad request (HTTP 400). Detail: {msg}"
                )
            else:
                raise ConnectionError(
                    f"Gemini API error (HTTP {e.code}): {msg}"
                )
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Gemini API: {e.reason}. Check your internet connection."
            )

        # Extract text from the response
        try:
            candidates = body.get("candidates", [])
            if not candidates:
                # Check for prompt-level blocking
                block_reason = body.get("promptFeedback", {}).get("blockReason", "")
                if block_reason:
                    raise ValueError(
                        f"Gemini blocked the prompt (reason: {block_reason}). "
                        "Try rephrasing your request."
                    )
                raise ValueError("Gemini returned no candidates in the response.")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                finish_reason = candidates[0].get("finishReason", "UNKNOWN")
                raise ValueError(
                    f"Gemini returned empty content (finishReason: {finish_reason})."
                )
            return parts[0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected Gemini API response structure: {e}")


def get_provider(backend: str = "ollama", **kwargs) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        backend: "ollama", "anthropic", or "gemini"
        **kwargs: passed to the provider constructor (e.g., model, base_url)
    """
    load_dotenv()
    if backend == "ollama":
        return OllamaProvider(**kwargs)
    elif backend == "anthropic":
        return AnthropicProvider(**kwargs)
    elif backend == "gemini":
        return GeminiProvider(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama', 'anthropic', or 'gemini'.")
