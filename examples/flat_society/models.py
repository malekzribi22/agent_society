"""
Data models for the flat society: Tool, Task, Agent, and Supervisor.

Agents are lightweight Python objects with skills, tools, and credit scores.
No per-agent LLM objects; LLM is centralized in a shared client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import TASK_TYPES


@dataclass
class Tool:
    tool_id: str
    purpose: str
    tags: List[str]  # which task types/capabilities it supports
    avg_latency_ms: int
    base_error_rate: float
    cost_units: float = 1.0


# Registry of available tools
def get_tool_registry() -> List[Tool]:
    """Return a list of tools covering the TASK_TYPES."""
    return [
        Tool(
            tool_id="people_counter",
            purpose="Count people in images/video",
            tags=["people_count"],
            avg_latency_ms=150,
            base_error_rate=0.05,
        ),
        Tool(
            tool_id="summarizer",
            purpose="Summarize news articles",
            tags=["news_summarize"],
            avg_latency_ms=200,
            base_error_rate=0.03,
        ),
        Tool(
            tool_id="route_planner",
            purpose="Plan optimal routes",
            tags=["route_plan"],
            avg_latency_ms=100,
            base_error_rate=0.02,
        ),
        Tool(
            tool_id="anomaly_detector",
            purpose="Detect sensor anomalies",
            tags=["sensor_anomaly"],
            avg_latency_ms=80,
            base_error_rate=0.08,
        ),
        Tool(
            tool_id="math_solver",
            purpose="Evaluate mathematical expressions",
            tags=["math_eval"],
            avg_latency_ms=50,
            base_error_rate=0.01,
        ),
        Tool(
            tool_id="fact_checker",
            purpose="Fact-check and answer questions",
            tags=["qa_fact"],
            avg_latency_ms=300,
            base_error_rate=0.04,
        ),
        Tool(
            tool_id="math_solver_v1",
            purpose="Solve structured math word problems",
            tags=["math_word", "math_eval"],
            avg_latency_ms=120,
            base_error_rate=0.02,
        ),
        Tool(
            tool_id="reasoning_tool_v1",
            purpose="Guide multi-step reasoning responses",
            tags=["multi_step_reasoning"],
            avg_latency_ms=130,
            base_error_rate=0.03,
        ),
    ]


@dataclass
class Task:
    task_id: str
    task_type: str  # e.g. "people_count", "news_summarize", ...
    required_skills: Set[str]
    metadata: Optional[Dict[str, object]] = None


@dataclass
class Agent:
    agent_id: int
    role: str = "agent"  # keep this string, do not use "worker"
    position: str = ""  # e.g. "attacker", "runner", "goalkeeper", "counter", ...
    skills: Dict[str, float] = field(default_factory=dict)  # e.g. {"shooting": 0.9, "running": 0.4, "goalkeeping": 0.1}
    credit: Dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )  # Beta(a,b) per task_type
    claims: List[str] = field(default_factory=list)  # claimed good task_types
    tools: List[str] = field(default_factory=list)  # tool_ids
    is_misreporter: bool = False
    is_malicious: bool = False
    load: int = 0  # how many tasks handled (for metrics)
    memory: List[Dict[str, Any]] = field(default_factory=list)  # History of tasks this agent performed

    def credit_mean(self, task_type: str) -> float:
        """Return the mean of the Beta distribution for this task_type."""
        if task_type not in self.credit:
            return 0.5  # default prior
        a, b = self.credit[task_type]
        if a + b == 0:
            return 0.5
        return a / (a + b)


@dataclass
class Supervisor:
    supervisor_id: str
    agents: Dict[int, Agent] = field(default_factory=dict)

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with this supervisor."""
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")
        self.agents[agent.agent_id] = agent

    def get_agent(self, agent_id: int) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List[Agent]:
        """Return all registered agents."""
        return list(self.agents.values())

    def get_candidates_for_task(self, task_type: str) -> List[Agent]:
        """
        Return a list of candidate agents for this task_type.
        Initially: all agents that claim this task_type; fallback: all agents.
        """
        candidates = [agent for agent in self.agents.values() if task_type in agent.claims]
        if not candidates:
            # Fallback: all agents
            candidates = list(self.agents.values())
        return candidates

    def choose_agents_for_task(
        self, task: Task, tools: List[Tool], k: int = 1
    ) -> List[Agent]:
        """
        Use the selection policy to choose top-k agents.
        Note: policy module will be imported here to avoid circular deps.
        """
        from .policy import select_best_agents

        candidates = self.get_candidates_for_task(task.task_type)
        return select_best_agents(candidates, task, tools, k=k)

    def assign_task(self, task: Task, tools: List[Tool]) -> Optional[Agent]:
        """
        Choose a single best agent for this task and return it.
        Do not execute the task here; just selection.
        """
        chosen = self.choose_agents_for_task(task, tools, k=1)
        return chosen[0] if chosen else None


def create_flat_society(num_agents: int, seed: int = 42) -> Supervisor:
    """
    Create one Supervisor with id 'supervisor-0' and num_agents Agents.
    Initialize skills, credit, claims and tools using reasonable random logic.
    Assign each agent a position (attacker, runner, goalkeeper, counter, etc.)
    and bias their skills according to position.
    Return the Supervisor instance.
    """
    rng = Random(seed)
    supervisor = Supervisor(supervisor_id="supervisor-0")
    tool_registry = get_tool_registry()
    all_task_types = list(TASK_TYPES)

    # Position definitions with skill biases
    positions = ["attacker", "runner", "goalkeeper", "counter", "planner", "analyzer"]
    position_skills = {
        "attacker": {"shooting": 0.9, "running": 0.5, "goalkeeping": 0.1, "counting": 0.3, "planning": 0.4, "analysis": 0.3},
        "runner": {"shooting": 0.4, "running": 0.9, "goalkeeping": 0.2, "counting": 0.5, "planning": 0.6, "analysis": 0.4},
        "goalkeeper": {"shooting": 0.2, "running": 0.5, "goalkeeping": 0.9, "counting": 0.4, "planning": 0.3, "analysis": 0.5},
        "counter": {"shooting": 0.3, "running": 0.4, "goalkeeping": 0.2, "counting": 0.9, "planning": 0.5, "analysis": 0.6},
        "planner": {"shooting": 0.4, "running": 0.5, "goalkeeping": 0.3, "counting": 0.5, "planning": 0.9, "analysis": 0.7},
        "analyzer": {"shooting": 0.3, "running": 0.4, "goalkeeping": 0.2, "counting": 0.6, "planning": 0.7, "analysis": 0.9},
    }

    # Map task types to skills
    task_to_skill = {
        "people_count": "counting",
        "news_summarize": "analysis",
        "route_plan": "planning",
        "sensor_anomaly": "analysis",
        "math_eval": "analysis",
        "qa_fact": "analysis",
        "math_word": "analysis",
        "multi_step_reasoning": "analysis",
    }

    def init_credit(preferred: Optional[str] = None, boost: float = 0.1) -> Dict[str, Tuple[float, float]]:
        credit_dict: Dict[str, Tuple[float, float]] = {}
        for task_type in all_task_types:
            mean = rng.uniform(0.5, 0.7)
            if preferred == task_type:
                mean = min(0.9, mean + boost)
            total = rng.uniform(8, 12)
            a = mean * total
            b = total - a
            credit_dict[task_type] = (a, b)
        return credit_dict

    for idx in range(num_agents):
        agent_id = idx + 1

        if agent_id <= 50:
            # Math specialists: handle expressions + word problems
            position = "math_specialist"
            combined_skills = {
                task_type: rng.uniform(0.35, 0.6) for task_type in all_task_types
            }
            combined_skills["math_word"] = rng.uniform(0.9, 0.98)
            combined_skills["math_eval"] = rng.uniform(0.9, 0.98)
            combined_skills["skill_math_word"] = combined_skills["math_word"]
            combined_skills["skill_math_eval"] = combined_skills["math_eval"]
            credit = init_credit(preferred="math_word", boost=0.2)
            # Boost math_eval credit as well
            a_eval, b_eval = credit.get("math_eval", (5.0, 5.0))
            credit["math_eval"] = (a_eval + 2.0, b_eval)
            claims = ["math_word", "math_eval"]
            tools = ["math_solver_v1"]
            is_misreporter = False
            is_malicious = rng.random() < 0.02
        elif agent_id <= 100:
            # Multi-step reasoning specialists
            position = "reasoning_specialist"
            combined_skills = {
                task_type: rng.uniform(0.35, 0.6) for task_type in all_task_types
            }
            combined_skills["multi_step_reasoning"] = rng.uniform(0.9, 0.98)
            combined_skills["math_word"] = rng.uniform(0.1, 0.2)
            combined_skills["math_eval"] = rng.uniform(0.1, 0.2)
            combined_skills["skill_multi_step_reasoning"] = combined_skills["multi_step_reasoning"]
            credit = init_credit(preferred="multi_step_reasoning", boost=0.2)
            claims = ["multi_step_reasoning"]
            tools = ["reasoning_tool_v1"]
            is_misreporter = False
            is_malicious = rng.random() < 0.02
        else:
            # Generalist agents use the legacy initialization
            position = rng.choice(positions)
            position_skill_template = position_skills[position]

            position_skills_dict: Dict[str, float] = {}
            for skill_name, base_value in position_skill_template.items():
                skill = max(0.0, min(1.0, rng.gauss(base_value, 0.15)))
                position_skills_dict[skill_name] = skill

            task_skills: Dict[str, float] = {}
            focus_task = None
            best_task_score = 0.0
            for task_type in all_task_types:
                skill_name = task_to_skill.get(task_type, "analysis")
                if skill_name in position_skills_dict:
                    task_skill = position_skills_dict[skill_name]
                else:
                    task_skill = rng.uniform(0.3, 0.65)
                task_skills[task_type] = task_skill
                if task_skill > best_task_score:
                    best_task_score = task_skill
                    focus_task = task_type

            if focus_task is None:
                focus_task = rng.choice(all_task_types)

            credit = init_credit()
            claims = [focus_task]
            other_tasks = [t for t in all_task_types if t != focus_task]
            num_extra_claims = rng.randint(0, 2)
            claims.extend(rng.sample(other_tasks, min(num_extra_claims, len(other_tasks))))

            is_misreporter = rng.random() < 0.1
            if is_misreporter:
                extra_claims = rng.sample(other_tasks, min(len(other_tasks), rng.randint(1, 3)))
                claims.extend(extra_claims)

            is_malicious = rng.random() < 0.05

            tools = []
            for tool in tool_registry:
                if any(tag in claims for tag in tool.tags):
                    if rng.random() < 0.6:
                        tools.append(tool.tool_id)

            if not tools:
                tools.append(rng.choice(tool_registry).tool_id)

            combined_skills = {**task_skills, **{f"skill_{k}": v for k, v in position_skills_dict.items()}}

        # Ensure at least one tool for specialists as well
        if not tools:
            tools.append(rng.choice(tool_registry).tool_id)

        agent = Agent(
            agent_id=agent_id,
            role="agent",
            position=position,
            skills=combined_skills,
            credit=credit,
            claims=list(dict.fromkeys(claims)),
            tools=tools,
            is_misreporter=is_misreporter,
            is_malicious=is_malicious,
        )

        supervisor.register_agent(agent)

    return supervisor
