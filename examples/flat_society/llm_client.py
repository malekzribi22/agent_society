"""
Shared LLM client for the flat society.

This is the "shared brain" - a single LLM client that all agents can consult.
Uses a semaphore to limit concurrent requests.

Supports:
- Local Ollama (default)
- OpenAI (optional, if API key provided)
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

try:
    import urllib.request
    import urllib.error
except ImportError:
    urllib = None  # type: ignore

try:
    import requests
except ImportError:
    requests = None


class LLMClient:
    """
    Shared LLM client.

    Can talk to:
      - a local Ollama instance (default), or
      - OpenAI, if configured.

    Uses a semaphore to limit concurrent requests.
    """

    def __init__(
        self,
        use_local: bool = True,
        local_base_url: str = "http://localhost:11434",
        local_model: str = "llama3",
        max_concurrent: int = 2,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
    ):
        self.use_local = use_local
        self.local_base_url = local_base_url.rstrip("/")
        self.local_model = local_model
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key
        self.semaphore = threading.Semaphore(max_concurrent)

        if not use_local and not openai_api_key:
            raise ValueError("Either use_local=True or openai_api_key must be provided")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Call the configured backend and return the response text.
        Handle timeouts/errors gracefully and return a fallback string like
        '[LLM error]' if something goes wrong.
        """
        with self.semaphore:
            if self.use_local:
                return self._generate_local(prompt, max_tokens)
            else:
                return self._generate_openai(prompt, max_tokens)

    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        """Generate using local Ollama endpoint."""
        url = f"{self.local_base_url}/api/generate"
        payload = {
            "model": self.local_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        try:
            if requests is not None:
                # Use requests if available
                response = requests.post(
                    url, json=payload, timeout=30
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "[LLM error: no response]")
            elif urllib is not None:
                # Fallback to urllib
                req_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    return data.get("response", "[LLM error: no response]")
            else:
                return "[LLM error: no HTTP library available]"
        except Exception as e:
            return f"[LLM error: {type(e).__name__}]"

    def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        """Generate using OpenAI API."""
        if not self.openai_api_key:
            return "[LLM error: no OpenAI API key]"

        if requests is None:
            return "[LLM error: requests library required for OpenAI]"

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "[LLM error: no content]")
            return "[LLM error: no choices]"
        except Exception as e:
            return f"[LLM error: {type(e).__name__}]"
