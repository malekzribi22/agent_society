"""Matplotlib plotting helpers."""

from __future__ import annotations

import pathlib
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt

from .metrics import gini_coefficient
from .supervisor import SupervisorState


def plot_agent_credit_over_time(state: SupervisorState, task_type: str, output_path: str) -> None:
    plt.figure(figsize=(8, 4))
    per_agent_y = defaultdict(list)
    per_agent_x = defaultdict(list)
    for idx, task in enumerate(state.tasks, start=1):
        if task.task_type != task_type:
            continue
        agent = next((a for a in state.agents if a.agent_id == task.agent_id), None)
        if not agent:
            continue
        metrics = agent.metrics.get_or_create(task_type)
        per_agent_x[agent.agent_id].append(idx)
        per_agent_y[agent.agent_id].append(metrics.credit_mean)
    for agent_id, xs in per_agent_x.items():
        plt.plot(xs, per_agent_y[agent_id], label=f"Agent {agent_id}")
    plt.xlabel("Task index")
    plt.ylabel("Credit mean")
    plt.title(f"Credit trajectories ({task_type})")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_over_time(state: SupervisorState, output_path: str, window: int = 50) -> None:
    accuracy = []
    cumulative = 0
    for idx, task in enumerate(state.tasks, start=1):
        cumulative += task.total_reward >= 0.5
        accuracy.append(cumulative / idx)
    plt.figure(figsize=(8, 4))
    plt.plot(accuracy)
    plt.xlabel("Task index")
    plt.ylabel("Cumulative accuracy")
    plt.title("Global accuracy over time")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_exposure_bar(state: SupervisorState, task_type: str, output_path: str) -> None:
    exposures = []
    labels = []
    for agent in state.agents:
        metrics = agent.metrics.per_type.get(task_type)
        if metrics:
            exposures.append(metrics.n)
            labels.append(str(agent.agent_id))
    plt.figure(figsize=(8, 4))
    plt.bar(labels, exposures)
    plt.xlabel("Agent ID")
    plt.ylabel("Tasks handled")
    plt.title(f"Exposure distribution ({task_type})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_gini_over_time(state: SupervisorState, task_type: str, output_path: str, window: int = 100) -> None:
    window_values: List[float] = []
    for idx in range(len(state.tasks)):
        sub = state.tasks[max(0, idx - window) : idx + 1]
        exposures = defaultdict(int)
        for task in sub:
            if task.task_type == task_type:
                exposures[task.agent_id] += 1
        window_values.append(gini_coefficient(list(exposures.values())))
    plt.figure(figsize=(8, 4))
    plt.plot(window_values)
    plt.xlabel("Task index")
    plt.ylabel("Gini coefficient")
    plt.title(f"Fairness over time ({task_type})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
