"""
Pure selection logic for matching tasks to agents.

No LLM here; this is just scoring and ranking based on:
- Credit scores (Beta distribution means)
- Skill matching
- Tool latency
- Small random exploration
"""

from __future__ import annotations

import random
from typing import List

from .models import Agent, Task, Tool


def beta_mean(a: float, b: float) -> float:
    """Return the mean of a Beta distribution."""
    if a + b == 0:
        return 0.5
    return a / (a + b)


def skill_match(agent: Agent, task: Task) -> float:
    """
    Return a score in [0,1] for how well agent.skills cover task.required_skills.
    For example: fraction of required_skills with skill >= threshold.
    """
    if not task.required_skills:
        return 1.0

    threshold = 0.5
    matches = 0
    for skill_name in task.required_skills:
        # Map skill_name to task_type (simplified: assume they match)
        skill_value = agent.skills.get(task.task_type, 0.0)
        if skill_value >= threshold:
            matches += 1

    return matches / len(task.required_skills)


def best_tool_latency(agent: Agent, task_type: str, tools: List[Tool]) -> int:
    """
    Return minimum latency among tools the agent has that support this task_type,
    or a high default if none.
    """
    tool_dict = {tool.tool_id: tool for tool in tools}
    matching_latencies = []
    for tool_id in agent.tools:
        tool = tool_dict.get(tool_id)
        if tool and task_type in tool.tags:
            matching_latencies.append(tool.avg_latency_ms)

    if matching_latencies:
        return min(matching_latencies)
    return 1000  # High default if no matching tool


def score_agent(agent: Agent, task: Task, tools: List[Tool]) -> float:
    """
    Combine:
        - credit_mean for task_type (Beta)
        - skill_match score
        - tool latency (lower is better, normalized)
        - small random epsilon for exploration
    """
    credit_score = agent.credit_mean(task.task_type)
    skill_score = skill_match(agent, task)
    latency_ms = best_tool_latency(agent, task.task_type, tools)
    # Normalize latency: 0-1000ms -> 1.0 to 0.0
    latency_score = max(0.0, 1.0 - (latency_ms / 1000.0))

    # Weighted combination
    combined = (
        credit_score * 0.5
        + skill_score * 0.3
        + latency_score * 0.15
        + random.random() * 0.05  # exploration
    )

    # Penalty for malicious agents
    if agent.is_malicious:
        combined *= 0.7

    return combined


def select_best_agents(
    agents: List[Agent], task: Task, tools: List[Tool], k: int = 1
) -> List[Agent]:
    """
    Score each candidate agent and return the top-k in descending score order.
    """
    scored = [(score_agent(agent, task, tools), agent) for agent in agents]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [agent for _, agent in scored[:k]]
