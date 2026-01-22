"""Mathematically grounded metrics for agents and fairness."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TaskTypeMetrics:
    """Beta-Bernoulli credit tracker with exponential decay."""

    alpha: float = 1.0
    beta: float = 1.0
    n: int = 0
    successes: int = 0
    failures: int = 0

    @property
    def credit_mean(self) -> float:
        denom = self.alpha + self.beta
        return self.alpha / denom if denom > 0 else 0.5

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        denom = (a + b) ** 2 * (a + b + 1)
        return (a * b) / denom if denom > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n > 0 else 0.0


def update_task_metrics(metrics: TaskTypeMetrics, reward: float, decay: float) -> None:
    """Apply decayed Bayesian update with reward in [0, 1]."""
    metrics.alpha = metrics.alpha * decay + reward
    metrics.beta = metrics.beta * decay + (1.0 - reward)
    metrics.n += 1
    if reward >= 0.5:
        metrics.successes += 1
    else:
        metrics.failures += 1


@dataclass
class AgentMetrics:
    per_type: Dict[str, TaskTypeMetrics] = field(default_factory=dict)

    def get_or_create(self, task_type: str) -> TaskTypeMetrics:
        if task_type not in self.per_type:
            self.per_type[task_type] = TaskTypeMetrics()
        return self.per_type[task_type]


def ucb_exploration_term(total_tasks_for_type: int, n_for_agent: int, c: float) -> float:
    """Upper Confidence Bound exploration bonus."""
    if n_for_agent <= 0:
        return float("inf")
    return c * math.sqrt((2.0 * math.log(total_tasks_for_type + 1)) / n_for_agent)


def exposure_fairness_penalty(agent_exposure: float, target_exposure: float, fairness_lambda: float) -> float:
    """Penalize deviation from target exposure share."""
    return -fairness_lambda * abs(agent_exposure - target_exposure)


def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient (0 perfect equality, 1 maximal inequality)."""
    if not values:
        return 0.0
    sorted_vals = sorted(v for v in values if v >= 0)
    if not sorted_vals:
        return 0.0
    cumulative = 0.0
    total = sum(sorted_vals)
    for i, val in enumerate(sorted_vals, start=1):
        cumulative += val * i
    gini = (2 * cumulative) / (len(sorted_vals) * total) - (len(sorted_vals) + 1) / len(sorted_vals)
    return gini
