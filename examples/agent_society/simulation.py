"""
Simulation helpers for running task assignments across large agent pools.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from random import random
from typing import Deque, Dict, Iterable, List

from .models import Supervisor, Task
from .policy import SelectionFn, select_agent


@dataclass
class SimulationStats:
    steps: int = 0
    assigned_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0


@dataclass
class Simulation:
    supervisor: Supervisor
    selection_fn: SelectionFn = select_agent
    active_assignments: Dict[int, int] = field(default_factory=dict)
    pending_tasks: Deque[Task] = field(default_factory=deque)
    stats: SimulationStats = field(default_factory=SimulationStats)

    def add_tasks(self, tasks: Iterable[Task]) -> None:
        for task in tasks:
            self.pending_tasks.append(task)

    def step(self) -> None:
        self._progress_active_tasks()
        self._assign_new_tasks()
        self.stats.steps += 1

    def run(self, max_steps: int | None = None) -> SimulationStats:
        """
        Run the simulation until all tasks complete or max_steps reached.
        """

        while self.pending_tasks or self.active_assignments:
            if max_steps is not None and self.stats.steps >= max_steps:
                break
            self.step()
        return self.stats

    def _progress_active_tasks(self) -> None:
        to_remove: List[int] = []
        for agent_id, remaining in list(self.active_assignments.items()):
            remaining -= 1
            if remaining <= 0:
                self._complete_assignment(agent_id)
                to_remove.append(agent_id)
            else:
                self.active_assignments[agent_id] = remaining

        for agent_id in to_remove:
            self.active_assignments.pop(agent_id, None)

    def _complete_assignment(self, agent_id: int) -> None:
        agent = self.supervisor.get_agent(agent_id)
        if agent is None:
            return
        success = random() > 0.05
        agent.complete_task(success=success)
        if success:
            self.stats.completed_tasks += 1
        else:
            self.stats.failed_tasks += 1

    def _assign_new_tasks(self) -> None:
        while self.pending_tasks:
            task = self.pending_tasks[0]
            agent = self.selection_fn(self.supervisor, task)
            if agent is None:
                break

            self.pending_tasks.popleft()
            agent.assign_task(task)
            duration = task.estimated_duration if task.estimated_duration else 1.0
            self.active_assignments[agent.agent_id] = max(1, int(round(duration)))
            self.stats.assigned_tasks += 1

