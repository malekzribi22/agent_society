"""
Configuration constants for the flat society simulation.
"""

NUM_AGENTS_DEFAULT = 10_000

TASK_TYPES = [
    "people_count",
    "news_summarize",
    "route_plan",
    "sensor_anomaly",
    "math_eval",
    "qa_fact",
    "math_word",
    "multi_step_reasoning",
]

RANDOM_SEED = 42

LLM_USE_LOCAL_DEFAULT = True
LLM_LOCAL_BASE_URL = "http://localhost:11434"
LLM_LOCAL_MODEL = "llama3"
LLM_MAX_CONCURRENT = 2

LLM_PROBABILITY_DEFAULT = 0.05  # 5% of decisions use LLM
MIN_CREDIT_FOR_LLM_DEFAULT = 0.7  # only agents with credit_mean >= this may use LLM

# History and scoring
CREDIT_DECAY = 0.995  # Decay factor for credit history (older tasks matter less)
EXPLORATION_RATE = 0.05  # Small random term for exploration in scoring
MAX_AGENT_MEMORY = 1000  # Maximum number of memory entries per agent (to avoid unbounded growth)
