"""
Flat multi-agent society package.

A simple, scalable architecture with:
- 1 Supervisor managing N Agents directly (no hierarchy)
- Shared LLM brain for optional reasoning
- Lightweight agents with skills, tools, and credit scores
"""

from __future__ import annotations

from .models import Agent, Supervisor, Task, Tool, create_flat_society
from .simulation import Simulation
from .llm_client import LLMClient

__all__ = [
    "Agent",
    "Supervisor",
    "Task",
    "Tool",
    "Simulation",
    "LLMClient",
    "create_flat_society",
]
