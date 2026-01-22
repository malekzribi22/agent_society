"""CLI entry point for the research_society experiments."""

from __future__ import annotations

import argparse
import asyncio
import pathlib

from .experiment_runner import run_experiment, save_task_log
from .plots import plot_accuracy_over_time, plot_agent_credit_over_time, plot_exposure_bar, plot_gini_over_time
from .supervisor import SELECTION_POLICIES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research society experiments.")
    parser.add_argument("--policy", type=str, default="score_fair_softmax", choices=SELECTION_POLICIES)
    parser.add_argument("--math-examples", type=int, default=200)
    parser.add_argument("--reasoning-examples", type=int, default=200)
    parser.add_argument("--cooperative", action="store_true")
    parser.add_argument("--output-dir", type=str, default="research_results")
    args = parser.parse_args()

    state = asyncio.run(
        run_experiment(
            policy=args.policy,
            num_math_examples=args.math_examples,
            num_reasoning_examples=args.reasoning_examples,
            cooperative=args.cooperative,
        )
    )
    output_dir = pathlib.Path(args.output_dir)
    save_task_log(state, output_dir)
    plot_accuracy_over_time(state, str(output_dir / "accuracy_over_time.png"))
    plot_agent_credit_over_time(state, "asdiv_math", str(output_dir / "credit_math.png"))
    plot_agent_credit_over_time(
        state, "strategyqa_reasoning", str(output_dir / "credit_reasoning.png")
    )
    plot_exposure_bar(state, "asdiv_math", str(output_dir / "exposure_math.png"))
    plot_exposure_bar(state, "strategyqa_reasoning", str(output_dir / "exposure_reasoning.png"))
    plot_gini_over_time(
        state, "asdiv_math", str(output_dir / "gini_math.png"), window=100
    )


if __name__ == "__main__":
    main()
