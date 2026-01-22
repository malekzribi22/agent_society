"""
Core primitives for building an agent society simulation.

The package exposes the main data models plus the simulation engine
so consumers can compose their own behaviors without diving into
implementation details.
"""

from __future__ import annotations

from .llm_client import LLMClient
from .models import Agent, Supervisor, Task, create_society
from .simulation import Simulation

__all__ = [
    "Agent",
    "Supervisor",
    "Task",
    "Simulation",
    "create_society",
    "LLMClient",
]

