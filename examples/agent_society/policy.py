"""
Simple selection policy for matching tasks to agents.

The policy favors agents with higher credit scores, better skill overlap,
and lower average latency. It intentionally remains lightweight so the
simulation can scale to thousands of agents without complex heuristics.
"""

from __future__ import annotations

from typing import Callable

from .models import Agent, Supervisor, Task


def score_agent(agent: Agent, task: Task) -> float:
    """
    Compute a compatibility score for (agent, task).

    The score is composed of:
        - credit score (60%)
        - skill overlap ratio (30%)
        - inverse latency (10%)
    """

    if not agent.is_idle():
        return float("-inf")

    required = task.required_skills or set()
    overlap = len(required & agent.skills)
    overlap_ratio = overlap / len(required) if required else 1.0
    inverse_latency = 1.0 / (agent.avg_latency + 1e-6)

    return (
        agent.credit_score * 0.6
        + overlap_ratio * 0.3
        + inverse_latency * 0.1
    )


def select_agent(supervisor: Supervisor, task: Task) -> Agent | None:
    """
    Choose the best suited idle agent for the provided task.
    """

    best_agent: Agent | None = None
    best_score = float("-inf")

    for agent in supervisor.get_idle_agents():
        agent_score = score_agent(agent, task)
        if agent_score > best_score:
            best_agent = agent
            best_score = agent_score

    return best_agent


SelectionFn = Callable[[Supervisor, Task], Agent | None]

