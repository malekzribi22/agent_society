"""
research_society
================

Research-grade experimentation toolkit for a compact multi-agent LLM society.

The package exposes utilities to:
* load benchmark math/reasoning datasets,
* maintain Bayesian credit metrics with fairness/exploration,
* coordinate supervisor policies (greedy, Thompson, UCB, fairness-aware),
* run large asynchronous experiments,
* evaluate answers, log metrics, and produce diagnostic plots.
"""

from .config import (
    ALPHA_NUMERIC,
    ALPHA_PROCESS,
    BETA_DECAY,
    FAIRNESS_LAMBDA,
    MAX_TASKS_PER_RUN,
    NUM_MATH_AGENTS,
    NUM_REASONING_AGENTS,
    OPENAI_MODEL,
    PARALLELISM,
    RANDOM_SEED,
    SOFTMAX_TEMPERATURE,
    UCB_C,
)

__all__ = [
    "ALPHA_NUMERIC",
    "ALPHA_PROCESS",
    "BETA_DECAY",
    "FAIRNESS_LAMBDA",
    "MAX_TASKS_PER_RUN",
    "NUM_MATH_AGENTS",
    "NUM_REASONING_AGENTS",
    "OPENAI_MODEL",
    "PARALLELISM",
    "RANDOM_SEED",
    "SOFTMAX_TEMPERATURE",
    "UCB_C",
]
