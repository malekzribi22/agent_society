"""
Simulation engine for the flat society.

This module implements the main simulation loop that:
- Generates tasks
- Selects agents using the policy
- Optionally consults shared LLM for some decisions
- Updates agent credit/skills based on outcomes
- Collects metrics

Architecture notes:
- Agents are lightweight Python objects (no per-agent threads or LLM instances)
- 10,000 agents is feasible because they're just dataclasses with small dicts/lists
- Only a small subset of decisions call the shared LLM (controlled by probability and credit thresholds)
- The system is designed so asyncio or Ray could later parallelize:
  * LLM calls (async HTTP requests)
  * Multiple independent simulations
  * Task execution across agents
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import (
    LLM_LOCAL_BASE_URL,
    LLM_LOCAL_MODEL,
    LLM_MAX_CONCURRENT,
    LLM_PROBABILITY_DEFAULT,
    MIN_CREDIT_FOR_LLM_DEFAULT,
    NUM_AGENTS_DEFAULT,
    RANDOM_SEED,
    TASK_TYPES,
)
from .llm_client import LLMClient
from .models import Agent, Supervisor, Task, Tool, create_flat_society, get_tool_registry
from .policy import select_best_agents


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    total_tasks: int = 0
    total_assigned: int = 0
    total_successful: int = 0
    total_failed: int = 0
    success_by_type: Dict[str, int] = field(default_factory=dict)
    failed_by_type: Dict[str, int] = field(default_factory=dict)
    tasks_per_agent: Dict[int, int] = field(default_factory=dict)
    num_llm_calls: int = 0
    num_llm_errors: int = 0


class Simulation:
    def __init__(
        self,
        num_agents: int = NUM_AGENTS_DEFAULT,
        num_steps: int = 1_000,
        tasks_per_step: int = 10,
        use_llm: bool = False,
        llm_probability: float = LLM_PROBABILITY_DEFAULT,
        min_credit_for_llm: float = MIN_CREDIT_FOR_LLM_DEFAULT,
        seed: int = RANDOM_SEED,
    ):
        random.seed(seed)
        self.num_steps = num_steps
        self.tasks_per_step = tasks_per_step
        self.llm_probability = llm_probability
        self.min_credit_for_llm = min_credit_for_llm

        # Create society
        self.supervisor = create_flat_society(num_agents, seed=seed)

        # Create shared LLM client if requested
        if use_llm:
            self.llm_client = LLMClient(
                use_local=True,
                local_base_url=LLM_LOCAL_BASE_URL,
                local_model=LLM_LOCAL_MODEL,
                max_concurrent=LLM_MAX_CONCURRENT,
            )
        else:
            self.llm_client = None

        # Tool registry
        self.tools = get_tool_registry()

        # Metrics
        self.metrics = SimulationMetrics()

    def generate_task(self, step: int, idx: int) -> Task:
        """
        Pick a task_type from TASK_TYPES using some distribution,
        create a Task with a unique id (e.g. f"{step}-{idx}"),
        and set required_skills accordingly.
        """
        task_type = random.choice(TASK_TYPES)
        task_id = f"{step}-{idx}"

        # Map task_type to required skills (simplified mapping)
        skill_mapping = {
            "people_count": {"vision", "counting"},
            "news_summarize": {"nlp", "summarization"},
            "route_plan": {"planning", "optimization"},
            "sensor_anomaly": {"analysis", "detection"},
            "math_eval": {"computation", "math"},
            "qa_fact": {"reasoning", "knowledge"},
        }
        required_skills = skill_mapping.get(task_type, {"general"})

        return Task(
            task_id=task_id,
            task_type=task_type,
            required_skills=required_skills,
            metadata={"step": step, "priority": random.randint(1, 5)},
        )

    def agent_should_use_llm(self, agent: Agent, task_type: str) -> bool:
        """
        Return True if this agent is allowed to consult the LLM for this task.
        Conditions:
          - self.llm_client is not None
          - random.random() < self.llm_probability
          - agent.credit_mean(task_type) >= min_credit_for_llm
        """
        if self.llm_client is None:
            return False
        if random.random() >= self.llm_probability:
            return False
        if agent.credit_mean(task_type) < self.min_credit_for_llm:
            return False
        return True

    def _execute_task_with_agent(
        self, agent: Agent, task: Task
    ) -> tuple[bool, Optional[str]]:
        """
        Simulate task execution by the agent.
        Returns (success, llm_response_or_none).
        """
        llm_response = None

        # Check if agent should use LLM
        if self.agent_should_use_llm(agent, task.task_type):
            self.metrics.num_llm_calls += 1
            prompt = f"""You are a cooperative agent (ID: {agent.agent_id}) with skills: {list(agent.skills.keys())}.
Task: {task.task_type}
Required skills: {task.required_skills}
Provide a brief action plan (1-2 sentences)."""
            llm_response = self.llm_client.generate(prompt, max_tokens=128)
            if "[LLM error" in llm_response:
                self.metrics.num_llm_errors += 1

        # Determine success based on:
        # - Agent's skill for this task type
        # - Tool error rates
        # - Malicious agents sometimes fail intentionally
        skill_value = agent.skills.get(task.task_type, 0.0)
        base_success_prob = skill_value

        # Find best tool for this task
        best_tool = None
        for tool in self.tools:
            if tool.tool_id in agent.tools and task.task_type in tool.tags:
                best_tool = tool
                break

        if best_tool:
            base_success_prob *= 1.0 - best_tool.base_error_rate
        else:
            base_success_prob *= 0.5  # Penalty for no matching tool

        # Malicious agents have 30% chance to sabotage
        if agent.is_malicious and random.random() < 0.3:
            success = False
        else:
            success = random.random() < base_success_prob

        return success, llm_response

    def _update_agent_credit(self, agent: Agent, task: Task, success: bool) -> None:
        """Update agent's credit (Beta distribution) based on outcome."""
        task_type = task.task_type
        if task_type not in agent.credit:
            # Initialize with default prior
            agent.credit[task_type] = (5.0, 5.0)

        a, b = agent.credit[task_type]
        if success:
            # Positive update: increase a
            a += 1.0
        else:
            # Negative update: increase b
            b += 1.0

        agent.credit[task_type] = (a, b)

    def run(self) -> Dict[str, Any]:
        """
        Run the simulation and return metrics.

        For each step:
          - generate tasks_per_step tasks
          - for each task:
             * get candidates from supervisor
             * select best agent (using policy)
             * decide if agent uses LLM or heuristic
             * simulate success/failure and latency
             * update agent credit/skills
          - collect aggregate metrics
        """
        for step in range(self.num_steps):
            # Generate tasks for this step
            tasks = [
                self.generate_task(step, idx) for idx in range(self.tasks_per_step)
            ]

            for task in tasks:
                self.metrics.total_tasks += 1

                # Select best agent
                agent = self.supervisor.assign_task(task, self.tools)
                if agent is None:
                    continue  # No agent available

                self.metrics.total_assigned += 1
                agent.load += 1
                self.metrics.tasks_per_agent[agent.agent_id] = (
                    self.metrics.tasks_per_agent.get(agent.agent_id, 0) + 1
                )

                # Execute task
                success, _ = self._execute_task_with_agent(agent, task)

                # Update metrics
                if success:
                    self.metrics.total_successful += 1
                    self.metrics.success_by_type[task.task_type] = (
                        self.metrics.success_by_type.get(task.task_type, 0) + 1
                    )
                else:
                    self.metrics.total_failed += 1
                    self.metrics.failed_by_type[task.task_type] = (
                        self.metrics.failed_by_type.get(task.task_type, 0) + 1
                    )

                # Update agent credit
                self._update_agent_credit(agent, task, success)

        # Compute final metrics
        all_loads = list(self.metrics.tasks_per_agent.values())
        min_load = min(all_loads) if all_loads else 0
        max_load = max(all_loads) if all_loads else 0
        avg_load = sum(all_loads) / len(all_loads) if all_loads else 0.0

        overall_success_rate = (
            self.metrics.total_successful / self.metrics.total_assigned
            if self.metrics.total_assigned > 0
            else 0.0
        )

        return {
            "total_tasks": self.metrics.total_tasks,
            "total_assigned": self.metrics.total_assigned,
            "total_successful": self.metrics.total_successful,
            "total_failed": self.metrics.total_failed,
            "overall_success_rate": overall_success_rate,
            "success_by_type": dict(self.metrics.success_by_type),
            "failed_by_type": dict(self.metrics.failed_by_type),
            "min_tasks_per_agent": min_load,
            "avg_tasks_per_agent": avg_load,
            "max_tasks_per_agent": max_load,
            "num_llm_calls": self.metrics.num_llm_calls,
            "num_llm_errors": self.metrics.num_llm_errors,
        }
