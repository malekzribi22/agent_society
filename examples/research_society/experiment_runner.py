"""Experiment orchestration for research_society."""

from __future__ import annotations

import asyncio
import csv
import pathlib
import random
from typing import Any, Dict, List

from . import config
from .agents import Agent, build_llm_client, create_default_agents
from .datasets import (
    extract_final_number_from_asdiv_answer,
    load_math_dataset,
    load_reasoning_dataset,
    normalize_strategyqa_answer,
)
from .evaluator import Evaluator
from .supervisor import SELECTION_POLICIES, SupervisorState, run_single_task


async def run_experiment(
    policy: str,
    num_math_examples: int,
    num_reasoning_examples: int,
    cooperative: bool = False,
) -> SupervisorState:
    """
    Run one experiment with a given supervisor policy.

    - Loads a math dataset (Calc-asdiv_a) and a reasoning dataset (StrategyQA).
    - Builds a unified queue of tasks.
    - Runs them asynchronously through the supervisor + agents.
    """
    # Fix random seed for reproducibility
    random.seed(config.RANDOM_SEED)

    # --- Load datasets (HF Dataset objects) ---
    # Calc-asdiv_a only has a 'test' split; we request 'test' explicitly.
    math_ds = load_math_dataset(split="test").shuffle(seed=config.RANDOM_SEED)[
        :num_math_examples
    ]
    # StrategyQA has 'train' split; we use that.
    reasoning_ds = load_reasoning_dataset(split="train").shuffle(
        seed=config.RANDOM_SEED
    )[:num_reasoning_examples]

    # ------------------------------------------------------------------
    # Build unified task queue: math tasks + reasoning tasks
    # ------------------------------------------------------------------
    tasks_queue: List[Dict[str, Any]] = []

    # -------------------- MATH (Calc-asdiv_a) -------------------------
    #
    # Expected fields from MU-NLPC/Calc-asdiv_a:
    #   - id
    #   - question          (may include some metadata, but still text)
    #   - source_question   (clean original question text)
    #   - chain             (solution steps)
    #   - result / result_float / result_unit
    #
    # For our purposes:
    #   - question_text  := source_question (if present) else question
    #   - ground_truth   := str(result or result_float or answer)
    # ------------------------------------------------------------------
    for row in math_ds:
        if isinstance(row, str):
            # Very defensive fallback (should not normally happen)
            question_text = row
            question_id = ""
            ground_truth = ""
        else:
            # Prefer clean source_question if available
            question_text = (
                row.get("source_question")
                or row.get("question")
                or row.get("Question")
                or ""
            )
            question_id = str(row.get("id", row.get("question_id", "")))

            # Ground truth: use numeric result fields first, or any answer-like field
            gt_raw = (
                row.get("result")
                or row.get("result_float")
                or row.get("Answer")
                or row.get("answer")
                or ""
            )
            ground_truth = str(gt_raw)

        tasks_queue.append(
            {
                "dataset": "asdiv_math",
                "task_type": "asdiv_math",
                "question_id": question_id,
                "question": question_text,
                "ground_truth": ground_truth,
            }
        )

    # -------------------- REASONING (StrategyQA) ----------------------
    #
    # Expected fields from tasksource/strategy-qa:
    #   - qid         : string id
    #   - question    : question text
    #   - answer      : boolean True/False
    #   - decomposition : list[str] (optional reasoning steps)
    #
    # For our purposes:
    #   - question_text  := question
    #   - ground_truth   := answer (bool or yes/no string)
    # ------------------------------------------------------------------
    reasoning_tasks: List[Dict[str, Any]] = []
    for row in reasoning_ds:
        if isinstance(row, str):
            # Defensive fallback in case of unexpected string rows
            question_text = row
            question_id = ""
            answer = None
            decomposition = []
        else:
            question_text = row.get("question", "")
            question_id = (
                row.get("qid")
                or row.get("id")
                or row.get("question_id", "")
            )
            answer = row.get("answer")
            decomposition = row.get("decomposition", [])

        reasoning_tasks.append(
            {
                "dataset": config.REASONING_DATASET_NAME,
                "task_type": "strategyqa_reasoning",
                "question_id": str(question_id),
                "question": question_text,
                "ground_truth": answer,
                "extra": {"decomposition": decomposition},
            }
        )

    # Add reasoning tasks and shuffle the global queue
    tasks_queue.extend(reasoning_tasks)
    random.shuffle(tasks_queue)

    # ------------------------------------------------------------------
    # Build agents, supervisor state, LLM client, evaluator
    # ------------------------------------------------------------------
    agents: List[Agent] = create_default_agents(
        config.NUM_MATH_AGENTS, config.NUM_REASONING_AGENTS
    )
    state = SupervisorState(agents=agents)
    llm_client = build_llm_client()
    evaluator = Evaluator()
    semaphore = asyncio.Semaphore(config.PARALLELISM)

    # ------------------------------------------------------------------
    # Worker coroutine: run a single entry through the supervisor
    # ------------------------------------------------------------------
    async def worker(entry: Dict[str, Any]) -> None:
        async with semaphore:
            await run_single_task(
                state=state,
                llm_client=llm_client,
                dataset_name=entry["dataset"],
                task_type=entry["task_type"],
                question_id=entry["question_id"],
                question_text=entry["question"],
                ground_truth=entry["ground_truth"],
                policy=policy,
                evaluator=evaluator,
            )

    # Run all tasks concurrently under the parallelism limit
    await asyncio.gather(*(worker(entry) for entry in tasks_queue))
    return state


async def run_all_policies(
    num_math_examples: int, num_reasoning_examples: int
) -> Dict[str, SupervisorState]:
    """
    Convenience helper: run all policies and return a dict of
    policy_name -> SupervisorState.
    """
    results: Dict[str, SupervisorState] = {}
    for policy in SELECTION_POLICIES:
        results[policy] = await run_experiment(
            policy, num_math_examples, num_reasoning_examples
        )
    return results


def save_task_log(state: SupervisorState, output_dir: pathlib.Path) -> None:
    """
    Save a CSV log of all tasks executed in an experiment.

    Columns:
        task_id, dataset, question_id, task_type,
        agent_id, agent_role, question,
        ground_truth, agent_answer,
        numeric_reward, process_reward, total_reward, timestamp
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "task_log.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_id",
                "dataset",
                "question_id",
                "task_type",
                "agent_id",
                "agent_role",
                "question",
                "ground_truth",
                "agent_answer",
                "numeric_reward",
                "process_reward",
                "total_reward",
                "timestamp",
            ]
        )
        for task in state.tasks:
            writer.writerow(
                [
                    task.task_id,
                    task.dataset,
                    task.question_id,
                    task.task_type,
                    task.agent_id,
                    task.agent_role,
                    task.question,
                    task.answer_ground_truth,
                    task.agent_answer,
                    task.numeric_reward,
                    task.process_reward,
                    task.total_reward,
                    task.timestamp,
                ]
            )

