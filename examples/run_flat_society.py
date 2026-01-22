"""
Entry point for running the flat society simulation.

Simple CLI that instantiates Simulation and runs it, printing metrics.
"""

from __future__ import annotations

import argparse
import json
from pprint import pprint

from flat_society import Simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a flat multi-agent society simulation"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=10_000,
        help="Number of agents in the society (default: 10000)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1_000,
        help="Number of simulation steps (default: 1000)",
    )
    parser.add_argument(
        "--tasks-per-step",
        type=int,
        default=10,
        help="Number of tasks generated per step (default: 10)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable shared LLM client (Ollama)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Flat Multi-Agent Society Simulation")
    print("=" * 60)
    print(f"Agents: {args.num_agents:,}")
    print(f"Steps: {args.num_steps:,}")
    print(f"Tasks per step: {args.tasks_per_step}")
    print(f"Total tasks: {args.num_steps * args.tasks_per_step:,}")
    print(f"LLM enabled: {args.use_llm}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    print()

    simulation = Simulation(
        num_agents=args.num_agents,
        num_steps=args.num_steps,
        tasks_per_step=args.tasks_per_step,
        use_llm=args.use_llm,
        seed=args.seed,
    )

    print("Running simulation...")
    metrics = simulation.run()
    print("Simulation complete!")
    print()

    print("=" * 60)
    print("METRICS")
    print("=" * 60)
    pprint(metrics, width=80)
    print()

    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tasks generated: {metrics['total_tasks']:,}")
    print(f"Tasks assigned: {metrics['total_assigned']:,}")
    print(f"Tasks successful: {metrics['total_successful']:,}")
    print(f"Tasks failed: {metrics['total_failed']:,}")
    print(f"Overall success rate: {metrics['overall_success_rate']:.2%}")
    print()
    print("Tasks per agent:")
    print(f"  Min: {metrics['min_tasks_per_agent']}")
    print(f"  Avg: {metrics['avg_tasks_per_agent']:.2f}")
    print(f"  Max: {metrics['max_tasks_per_agent']}")
    print()
    if args.use_llm:
        print(f"LLM calls: {metrics['num_llm_calls']:,}")
        print(f"LLM errors: {metrics['num_llm_errors']:,}")
        if metrics['num_llm_calls'] > 0:
            error_rate = metrics['num_llm_errors'] / metrics['num_llm_calls']
            print(f"LLM error rate: {error_rate:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
