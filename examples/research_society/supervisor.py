"""Supervisor policies for the research society."""

from __future__ import annotations

import asyncio
import random
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from . import config
from .agents import Agent, LLMClient
from .metrics import (
    AgentMetrics,
    exposure_fairness_penalty,
    ucb_exploration_term,
    update_task_metrics,
)
from .evaluator import Evaluator


SELECTION_POLICIES = [
    "greedy",
    "thompson",
    "ucb",
    "score_fair",
    "score_fair_softmax",
]


@dataclass
class TaskRecord:
    task_id: int
    dataset: str
    question_id: str
    task_type: str
    agent_id: int
    agent_role: str
    question: str
    answer_ground_truth: Any
    agent_answer: str
    numeric_reward: float
    process_reward: float
    total_reward: float
    timestamp: float


@dataclass
class SupervisorState:
    agents: List[Agent]
    total_tasks_per_type: Dict[str, int] = field(default_factory=dict)
    tasks: List[TaskRecord] = field(default_factory=list)


def _score_candidates(
    state: SupervisorState,
    agents: Sequence[Agent],
    task_type: str,
    policy: str,
) -> List[tuple[Agent, float]]:
    total_for_type = state.total_tasks_per_type.get(task_type, 0)
    exposures = {
        agent.agent_id: agent.metrics.get_or_create(task_type).n for agent in agents
    }
    total_exposure = sum(exposures.values()) or 1.0
    target_exposure = 1.0 / len(agents) if agents else 0.0

    scored: List[tuple[Agent, float]] = []
    for agent in agents:
        metrics = agent.metrics.get_or_create(task_type)

        # Base components
        credit = metrics.credit_mean
        explore = ucb_exploration_term(total_for_type + 1, metrics.n, config.UCB_C)
        exposure = exposures[agent.agent_id] / total_exposure
        fair_penalty = exposure_fairness_penalty(
            exposure, target_exposure, config.FAIRNESS_LAMBDA
        )

        # Sanitize any NaN / inf components
        def safe(x: float | None) -> float:
            if x is None or not math.isfinite(x):
                return 0.0
            return x

        credit = safe(credit)
        explore = safe(explore)
        fair_penalty = safe(fair_penalty)

        if policy == "thompson":
            # Thompson sampling on the success posterior + fairness term
            sampled = random.betavariate(metrics.alpha, metrics.beta)
            score = safe(sampled) + fair_penalty
        else:
            # Greedy / UCB / score_fair / score_fair_softmax all use this
            score = credit + explore + fair_penalty

        # Final safety: no NaN / inf scores
        score = safe(score)
        scored.append((agent, score))

    return scored


def select_agent_for_task(state: SupervisorState, task_type: str, policy: str) -> Agent:
    # Filter agents by role (e.g. "math" in task_type) – fallback to all if none match.
    candidates = [agent for agent in state.agents if agent.role in task_type]
    if not candidates:
        candidates = state.agents

    scored = _score_candidates(state, candidates, task_type, policy)

    # Softmax-based fair selection
    if policy == "score_fair_softmax":
        # Extract scores and sanitize again just in case
        scores = [score for _, score in scored]

        # If all scores are zero or list is empty, fallback to uniform choice
        if not scores or all(s == 0.0 for s in scores):
            return random.choice(candidates)

        # Standard softmax with numerical stability
        logits = [s / config.SOFTMAX_TEMPERATURE for s in scores]
        max_logit = max(logits)
        exp_scores = [math.exp(l - max_logit) for l in logits]

        total = sum(exp_scores)
        # If total is invalid, fallback to a simple greedy or uniform choice
        if total <= 0 or not math.isfinite(total):
            # Fallback: pure greedy on the (finite) scores
            return max(scored, key=lambda tup: tup[1])[0]

        probs = [s / total for s in exp_scores]
        choice = random.choices(scored, weights=probs, k=1)[0][0]
        return choice

    # Default: greedy on the score
    return max(scored, key=lambda tup: tup[1])[0]


async def run_single_task(
    state: SupervisorState,
    llm_client: LLMClient,
    dataset_name: str,
    task_type: str,
    question_id: str,
    question_text: str,
    ground_truth: Any,
    policy: str,
    evaluator: Evaluator,
    agent_override: Agent | None = None,
) -> TaskRecord:
    """Run a single task asynchronously."""
    agent = agent_override or select_agent_for_task(state, task_type, policy)
    state.total_tasks_per_type[task_type] = state.total_tasks_per_type.get(task_type, 0) + 1

    prompt = evaluator.build_prompt(task_type, question_text)
    messages = [
        {"role": "system", "content": evaluator.system_prompt(task_type)},
        {"role": "user", "content": prompt},
    ]

    agent_answer = await llm_client.chat(messages)
    numeric_reward, process_reward, total_reward = evaluator.evaluate(
        dataset_name, task_type, agent_answer, ground_truth
    )

    metrics = agent.metrics.get_or_create(task_type)
    update_task_metrics(metrics, total_reward, config.BETA_DECAY)

    record = TaskRecord(
        task_id=len(state.tasks) + 1,
        dataset=dataset_name,
        question_id=question_id,
        task_type=task_type,
        agent_id=agent.agent_id,
        agent_role=agent.role,
        question=question_text,
        answer_ground_truth=ground_truth,
        agent_answer=agent_answer,
        numeric_reward=numeric_reward,
        process_reward=process_reward,
        total_reward=total_reward,
        timestamp=time.time(),
    )
    state.tasks.append(record)

    agent.history.append(
        {
            "task_id": record.task_id,
            "dataset": dataset_name,
            "question_id": question_id,
            "task_type": task_type,
            "reward": total_reward,
            "timestamp": record.timestamp,
        }
    )
    return record

