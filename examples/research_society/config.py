"""Configuration for the research_society package."""

from __future__ import annotations

import os

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MATH_DATASET_NAME = "MU-NLPC/Calc-asdiv_a"
REASONING_DATASET_NAME = "tasksource/strategy-qa"

BETA_DECAY = 0.995
UCB_C = 0.5
FAIRNESS_LAMBDA = 0.1
SOFTMAX_TEMPERATURE = 0.3

ALPHA_NUMERIC = 0.8
ALPHA_PROCESS = 0.2

NUM_MATH_AGENTS = 5
NUM_REASONING_AGENTS = 5
MAX_TASKS_PER_RUN = 2000
PARALLELISM = 20
RANDOM_SEED = 42

USE_LOCAL_LLM = False
