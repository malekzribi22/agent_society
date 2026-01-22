"""Agent abstractions and LLM client helpers."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import config
from .metrics import AgentMetrics


class LLMClient:
    """Abstract async chat client."""

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Lightweight OpenAI ChatCompletions client."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or config.OPENAI_MODEL
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export the variable to use OpenAIClient."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai>=1.0.0 to use OpenAIClient") from exc
        self._client = OpenAI(api_key=self.api_key)

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._sync_chat, messages
        )

    def _sync_chat(self, messages: List[Dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


class OfflineEchoClient(LLMClient):
    """Fallback client that returns deterministic answers."""

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        user_content = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"Pretend reasoning for: {user_content}\n#### placeholder"


@dataclass
class Agent:
    agent_id: int
    name: str
    role: str  # "math" or "reasoning"
    skills: Dict[str, float]
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    history: List[Dict[str, Any]] = field(default_factory=list)


def create_default_agents(num_math: int, num_reasoning: int) -> List[Agent]:
    agents: List[Agent] = []
    for idx in range(1, num_math + 1):
        agents.append(
            Agent(
                agent_id=idx,
                name=f"MathAgent-{idx}",
                role="math",
                skills={"asdiv_math": 0.9, "strategyqa_reasoning": 0.4},
            )
        )
    for r_idx in range(num_math + 1, num_math + num_reasoning + 1):
        agents.append(
            Agent(
                agent_id=r_idx,
                name=f"ReasonAgent-{r_idx}",
                role="reasoning",
                skills={"asdiv_math": 0.3, "strategyqa_reasoning": 0.9},
            )
        )
    return agents


def build_llm_client() -> LLMClient:
    if config.USE_LOCAL_LLM:
        return OfflineEchoClient()
    if config.OPENAI_API_KEY:
        return OpenAIClient()
    return OfflineEchoClient()
