"""Data models for the 10-agent mini demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


TASK_TYPE_MATH = "math"
TASK_TYPE_REASONING = "reasoning"
TASK_TYPE_UAV = "uav_car_count"
TASK_TYPE_UAV_MISSION = "uav_mission"
ALL_TASK_TYPES = (TASK_TYPE_MATH, TASK_TYPE_REASONING, TASK_TYPE_UAV)


@dataclass
class Agent:
    agent_id: int
    name: str
    role: str  # "math" or "reasoning"
    score: float = 0.0
    tasks_done: int = 0
    successes: int = 0
    failures: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    credits: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            TASK_TYPE_MATH: {"alpha": 1.0, "beta": 1.0, "credit_mean": 0.5},
            TASK_TYPE_REASONING: {"alpha": 1.0, "beta": 1.0, "credit_mean": 0.5},
            TASK_TYPE_UAV: {"alpha": 1.0, "beta": 1.0, "credit_mean": 0.5},
        }
    )
    exposure: Dict[str, int] = field(default_factory=dict)
    suspicious_count: int = 0
    task_types: List[str] = field(default_factory=list)

    def supports_task_type(self, task_type: str) -> bool:
        if task_type in self.task_types:
            return True
        # Backward-compatibility with legacy role-based routing
        return task_type == self.role


@dataclass
class GroupMetrics:
    n: int = 0
    successes: int = 0
    failures: int = 0
    credit_mean: float = 0.5
    exposure_count: int = 0
    total_latency: float = 0.0
    suspicious_events: int = 0

    def success_rate(self) -> float:
        total = self.successes + self.failures
        if total == 0:
            return 0.0
        return self.successes / total


@dataclass
class Leader:
    leader_id: int
    name: str
    role: str
    worker_ids: List[int]
    group_memory: List[Dict[str, Any]] = field(default_factory=list)
    group_metrics: Dict[str, GroupMetrics] = field(default_factory=dict)

    def get_or_create_group_metrics(self, task_type: str) -> GroupMetrics:
        metrics = self.group_metrics.get(task_type)
        if metrics is None:
            metrics = GroupMetrics()
            self.group_metrics[task_type] = metrics
        return metrics

    def update_group_metrics(
        self,
        task_type: str,
        success: bool | None,
        reward: float,
        latency: float | None = None,
        credit_mean: float | None = None,
        suspicious: bool = False,
    ) -> None:
        metrics = self.get_or_create_group_metrics(task_type)
        metrics.n += 1
        if success is True:
            metrics.successes += 1
        elif success is False:
            metrics.failures += 1
        metrics.exposure_count += 1
        if suspicious:
            metrics.suspicious_events += 1
        if latency is not None:
            metrics.total_latency += latency
        if credit_mean is not None:
            metrics.credit_mean = credit_mean

    def record_group_memory(self, entry: Dict[str, Any], limit: int = 500) -> None:
        self.group_memory.append(entry)
        if len(self.group_memory) > limit:
            # Keep the most recent entries
            self.group_memory = self.group_memory[-limit:]


@dataclass
class Task:
    task_id: int
    task_type: str
    input_text: str
    assigned_agent_id: int | None = None
    assigned_agent_name: str | None = None
    success: bool | None = None
    output: str = ""
    agent_answer: str | None = None
    question: str | None = None
    ground_truth: str | None = None
    numeric_reward: float | None = None
    process_reward: float | None = None
    total_reward: float | None = None
    abs_error: float | None = None
    image_path: str | None = None
    image_name: str | None = None
    timestamp: float = 0.0
    selection_scores: Dict[int, float] = field(default_factory=dict)
    selection_components: Dict[str, float] = field(default_factory=dict)
    auto_score: float | None = None
    is_correct: bool | None = None
    leader_id: int | None = None
    leader_name: str | None = None
    leader_selection_components: Dict[str, float] = field(default_factory=dict)
    used_llm: bool = False
    evaluation_reason: str = ""
    numeric_match: bool | None = None
    verdict_match: bool | None = None
    partial_credit: bool = False
    suspicious: bool = False


def create_ten_agents() -> List[Agent]:
    """
    Return a stable roster of 10 agents:
    - IDs 1-5 are math specialists
    - IDs 6-10 are reasoning specialists
    """
    agents: List[Agent] = []
    for i in range(1, 6):
        agents.append(
            Agent(
                agent_id=i,
                name=f"Math Agent {i}",
                role="math",
                score=5.0 - (i - 1) * 0.2,
                task_types=[TASK_TYPE_MATH, TASK_TYPE_UAV],
            )
        )
    for i in range(6, 11):
        agents.append(
            Agent(
                agent_id=i,
                name=f"Reasoning Agent {i}",
                role="reasoning",
                score=4.5 - (i - 6) * 0.15,
                task_types=[TASK_TYPE_REASONING, TASK_TYPE_UAV],
            )
        )
    return agents
