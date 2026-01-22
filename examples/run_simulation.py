"""
Entry point for running the agent society simulation.
"""

from __future__ import annotations

from random import choice, randint

from agent_society import Simulation, Task, create_society


def _generate_tasks(total: int) -> list[Task]:
    task_types = [
        "analysis",
        "coordination",
        "planning",
        "monitoring",
        "exploration",
    ]
    skill_sets = [
        {"planning", "analysis"},
        {"navigation", "observation"},
        {"control", "vision"},
        {"negotiation"},
        {"analysis", "vision"},
    ]

    tasks: list[Task] = []
    for idx in range(total):
        task_type = choice(task_types)
        required = choice(skill_sets)
        estimated_duration = float(randint(1, 5))
        task = Task(
            task_id=f"task-{idx}",
            task_type=task_type,
            required_skills=set(required),
            estimated_duration=estimated_duration,
            metadata={"priority": randint(1, 5)},
        )
        tasks.append(task)
    return tasks


def main() -> None:
    supervisor = create_society(num_agents=10_000)
    tasks = _generate_tasks(total=2_000)

    simulation = Simulation(supervisor=supervisor)
    simulation.add_tasks(tasks)
    stats = simulation.run()

    print("Simulation complete")
    print(f"Steps: {stats.steps}")
    print(f"Assigned tasks: {stats.assigned_tasks}")
    print(f"Completed tasks: {stats.completed_tasks}")
    print(f"Failed tasks: {stats.failed_tasks}")
    remaining = len(simulation.pending_tasks) + len(simulation.active_assignments)
    print(f"Unfinished tasks: {remaining}")


if __name__ == "__main__":
    main()

