"""
ten_agent_web_demo
==================

A miniature 10-agent society demo with a lightweight web UI.

Dependencies: install `pip install ultralytics opencv-python Pillow` to enable UAV vision mode.
"""

from .models import Agent, Task, create_ten_agents
from .simulation import SimulationState, execute_task, infer_task_type, select_agent

__all__ = [
    "Agent",
    "Task",
    "create_ten_agents",
    "SimulationState",
    "execute_task",
    "infer_task_type",
    "select_agent",
]
