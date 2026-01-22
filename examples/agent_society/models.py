"""
Data models representing Agents, Tasks, and their Supervisor.

Only relies on the Python standard library to simplify portability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import choice, randint, random, sample
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_client import LLMClient


@dataclass(slots=True)
class Task:
    task_id: str | int
    task_type: str
    required_skills: set[str]
    estimated_duration: float | None = None
    metadata: dict[str, object] | None = None


@dataclass(slots=True)
class Agent:
    agent_id: int
    skills: set[str]
    tools: set[str]
    credit_score: float
    avg_latency: float
    status: str = "idle"
    current_task: Task | None = None

    def is_idle(self) -> bool:
        return self.status == "idle"

    def assign_task(self, task: Task) -> None:
        if not self.is_idle():
            raise RuntimeError(f"Agent {self.agent_id} is already busy.")
        self.current_task = task
        self.status = "busy"

    def complete_task(self, success: bool = True) -> None:
        if self.current_task is None:
            raise RuntimeError(f"Agent {self.agent_id} has no task to complete.")
        performance_delta = 0.05 if success else -0.1
        self.credit_score = max(0.0, min(1.0, self.credit_score + performance_delta))
        # Slightly nudge latency to simulate variability.
        self.avg_latency = max(0.1, self.avg_latency * (0.98 if success else 1.02))
        self.current_task = None
        self.status = "idle"

    def decide_action(
        self,
        task: Task,
        llm_client: "LLMClient | None" = None,
        use_llm: bool = False,
    ) -> str:
        """
        Decide how to approach a task.

        Agents can optionally leverage the shared LLM client; otherwise a
        deterministic rule-based response is returned to keep most agents
        lightweight.
        """

        summary = (
            f"Agent {self.agent_id} with skills {sorted(self.skills)} "
            f"handling task {task.task_id} ({task.task_type})"
        )

        if use_llm and llm_client is not None:
            prompt = (
                "You are a cooperative agent in a large society. "
                "Provide a concise action plan (<=3 steps) for the task.\n"
                f"Task type: {task.task_type}\n"
                f"Required skills: {', '.join(sorted(task.required_skills)) or 'none'}\n"
                f"Agent skills: {', '.join(sorted(self.skills))}\n"
                "Plan:"
            )
            response = llm_client.generate(prompt=prompt, max_tokens=96)
            return response if response != "NO_RESPONSE" else summary

        matched_skills = task.required_skills & self.skills
        if matched_skills:
            skill = sorted(matched_skills)[0]
            return f"{summary} using skill {skill}"
        return f"{summary} using heuristic fallback"


@dataclass(slots=True)
class Supervisor:
    supervisor_id: str
    agents: dict[int, Agent] = field(default_factory=dict)

    def register_agent(self, agent: Agent) -> None:
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent {agent.agent_id} already registered.")
        self.agents[agent.agent_id] = agent

    def get_agent(self, agent_id: int) -> Agent | None:
        return self.agents.get(agent_id)

    def get_all_agents(self) -> list[Agent]:
        return list(self.agents.values())

    def get_idle_agents(self) -> list[Agent]:
        return [agent for agent in self.agents.values() if agent.is_idle()]


def _random_subset(items: Iterable[str]) -> set[str]:
    items_list = list(items)
    if not items_list:
        return set()
    subset_size = randint(1, min(len(items_list), 3))
    return set(sample(items_list, subset_size))


def create_society(num_agents: int) -> Supervisor:
    """
    Create a Supervisor and populate it with num_agents Agents.

    Each agent receives skills, tools, and performance traits sampled
    from small, deterministic pools to keep the setup lightweight while
    still providing diversity across the swarm.
    """

    if num_agents <= 0:
        raise ValueError("num_agents must be positive.")

    skill_pool = {
        "planning",
        "navigation",
        "analysis",
        "vision",
        "control",
        "negotiation",
        "observation",
    }
    tool_pool = {
        "sim_logger",
        "map_builder",
        "diagnostics",
        "predictor",
        "planner",
    }

    supervisor = Supervisor(supervisor_id="supervisor-0")

    for agent_id in range(num_agents):
        agent = Agent(
            agent_id=agent_id,
            skills=_random_subset(skill_pool),
            tools=_random_subset(tool_pool),
            credit_score=round(random() * 0.5 + 0.5, 2),
            avg_latency=round(random() * 0.9 + 0.1, 2),
        )
        agent.skills.add(choice(tuple(skill_pool)))
        agent.tools.add(choice(tuple(tool_pool)))
        supervisor.register_agent(agent)

    return supervisor

