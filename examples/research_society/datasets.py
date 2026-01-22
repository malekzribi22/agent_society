"""Dataset utilities for the research_society package."""

from __future__ import annotations

from typing import Any, Dict

from . import config


def _load_dataset(name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required. Install with 'pip install datasets'."
        ) from exc
    try:
        return load_dataset(name, split=split)
    except ValueError as err:
        if "Unknown split" not in str(err):
            raise
        dataset = load_dataset(name)
        available_splits = list(dataset.keys())
        if not available_splits:
            raise RuntimeError(f"No splits available for dataset {name}") from err
        fallback_split = available_splits[0]
        print(
            f"[research_society] Split '{split}' not found for dataset {name}. "
            f"Falling back to '{fallback_split}'."
        )
        return dataset[fallback_split]


def load_math_dataset(split: str = "train"):
    """
    Load the Calc-asdiv_a subset.
    Calc-asdiv_a only ships a 'test' split on HF, so requesting 'train'
    automatically falls back to 'test' via _load_dataset.
    Expected fields: 'question', 'answer'.
    """
    return _load_dataset(config.MATH_DATASET_NAME, split=split)


def load_reasoning_dataset(split: str = "train"):
    """
    Load StrategyQA (tasksource/strategy-qa).

    Each row is a dict with keys such as 'qid', 'question', 'answer', 'decomposition'.
    """
    return _load_dataset(config.REASONING_DATASET_NAME, split=split)


def normalize_strategyqa_answer(answer: Any) -> str:
    """Normalize StrategyQA labels to 'yes'/'no' strings."""
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if isinstance(answer, (int, float)):
        return "yes" if answer else "no"
    if isinstance(answer, str):
        lowered = answer.strip().lower()
        if lowered in {"yes", "true", "y"}:
            return "yes"
        if lowered in {"no", "false", "n"}:
            return "no"
    return "unknown"


def extract_final_number_from_asdiv_answer(answer_str: str) -> float | None:
    """Extract numeric ground-truth from ASDiv answer text."""
    import re

    if not answer_str:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer_str.replace(",", ""))
    return float(matches[-1]) if matches else None
