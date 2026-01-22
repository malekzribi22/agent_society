"""Automatic grading utilities for math and reasoning tasks."""

from __future__ import annotations

import re
from typing import Any, Tuple

from . import config
from .datasets import extract_final_number_from_asdiv_answer, normalize_strategyqa_answer


def extract_final_number_from_answer(answer_str: str) -> float | None:
    if not answer_str:
        return None
    lines = [line.strip() for line in answer_str.splitlines() if line.strip()]
    final_line = next((ln for ln in reversed(lines) if ln.startswith("####")), "")
    match = re.search(r"-?\d+(?:\.\d+)?", final_line)
    return float(match.group()) if match else None


def evaluate_math_answer(agent_answer: str, ground_truth: str) -> Tuple[float, float, float]:
    gold = extract_final_number_from_asdiv_answer(ground_truth)
    pred = extract_final_number_from_answer(agent_answer)
    if gold is None or pred is None:
        numeric_reward = 0.0
    else:
        numeric_reward = 1.0 if abs(gold - pred) <= 1e-2 else 0.0
    process_reward = 0.0
    total_reward = config.ALPHA_NUMERIC * numeric_reward + config.ALPHA_PROCESS * process_reward
    return numeric_reward, process_reward, total_reward


def normalize_yes_no(text: str) -> str:
    if not text:
        return "unknown"
    lowered = text.strip().lower()
    if lowered.startswith("####"):
        lowered = lowered[4:].strip()
    mapping = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
    }
    return mapping.get(lowered, lowered)


def evaluate_reasoning_answer(agent_answer: str, ground_truth_label: str) -> Tuple[float, float, float]:
    gt = normalize_strategyqa_answer(ground_truth_label)
    pred = normalize_yes_no(agent_answer.splitlines()[-1] if agent_answer else "")
    numeric_reward = 1.0 if gt == pred and gt in {"yes", "no"} else 0.0
    process_reward = 0.0
    total_reward = config.ALPHA_NUMERIC * numeric_reward + config.ALPHA_PROCESS * process_reward
    return numeric_reward, process_reward, total_reward


class Evaluator:
    """Dispatch evaluator for math vs reasoning datasets."""

    def system_prompt(self, task_type: str) -> str:
        if "math" in task_type:
            return "You are a precise math solver. Always show steps then output final number as '#### value'."
        return "You reason carefully about yes/no questions. End with '#### yes' or '#### no'."

    def build_prompt(self, task_type: str, question_text: str) -> str:
        return f"Task Type: {task_type}\nQuestion: {question_text}\nProvide reasoning then final answer."

    def evaluate(
        self,
        dataset_name: str,
        task_type: str,
        agent_answer: str,
        ground_truth: Any,
    ) -> Tuple[float, float, float]:
        if "math" in task_type:
            return evaluate_math_answer(agent_answer, ground_truth)
        return evaluate_reasoning_answer(agent_answer, ground_truth)
