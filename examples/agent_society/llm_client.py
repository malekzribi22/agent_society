"""
Thin wrapper around a shared Ollama-compatible LLM endpoint.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(slots=True)
class LLMClient:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """
        Call Ollama's /api/generate endpoint synchronously.
        Returns a fallback string if the call fails.
        """

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return "NO_RESPONSE"

        try:
            body = json.loads(data)
        except json.JSONDecodeError:
            return "NO_RESPONSE"

        return body.get("response", "").strip() or "NO_RESPONSE"

