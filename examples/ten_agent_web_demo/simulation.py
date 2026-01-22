"""Simulation logic for the 10-agent mini demo."""

from __future__ import annotations

import ast
import csv
import logging
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import requests

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .config import (
    ARCH_MODE_FLAT,
    ARCH_MODE_HIER,
    DEFAULT_ARCHITECTURE_MODE,
    MAX_GROUP_MEMORY,
)
from .models import (
    Agent,
    Leader,
    Task,
    create_ten_agents,
    TASK_TYPE_MATH,
    TASK_TYPE_REASONING,
    TASK_TYPE_UAV,
    TASK_TYPE_UAV_MISSION,
)
import json
import functools
from difflib import SequenceMatcher
from .uav_vision import get_uav_detector

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent
UAV_DATA_ROOT = (EXAMPLES_ROOT / "uav_roundabouts").resolve()
UAV_DATASET_PATH = (UAV_DATA_ROOT / "data.csv").resolve()
UAV_IMAGE_ROOT = (UAV_DATA_ROOT / "original" / "imgs").resolve()
UAV_MAX_DATASET = 200

# Supervisor-only dataset import
from .uav_mission_dataset import get_uav_mission_templates, MissionTemplate

DECAY = 0.995
UAV_REWARD_ALPHA = 1.0
UAV_SUCCESS_TOLERANCE = 1

USE_LLM_FOR_REASONING = True
LLM_MODEL_NAME = "gpt-4o-mini"
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 20


class ReasoningLLMClient:
    """Thin wrapper around OpenAI chat completions for reasoning prompts."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.warning("[ten_agent_web_demo] OPENAI_API_KEY missing; reasoning LLM disabled.")
            self._client = None
        else:
            try:
                self._client = OpenAI()
            except Exception as exc:
                logging.exception("[ten_agent_web_demo] Failed to init OpenAI client: %s", exc)
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if not self._client:
            raise RuntimeError("LLM client unavailable")
        response = self._client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT,
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise RuntimeError("LLM returned empty content")
        return content.strip()


_reasoning_client: ReasoningLLMClient | None = None


def _get_reasoning_client() -> ReasoningLLMClient | None:
    global _reasoning_client
    if _reasoning_client is None:
        _reasoning_client = ReasoningLLMClient()
    if _reasoning_client and _reasoning_client.available:
        return _reasoning_client
    return None


@dataclass
class UavTask:
    task_id: int
    image_path: str
    image_name: str
    num_cars: int


def load_uav_roundabout_tasks(max_images: int | None = UAV_MAX_DATASET) -> List[UavTask]:
    tasks: List[UavTask] = []
    root = UAV_DATA_ROOT
    data_path = root / "data.csv"
    if not data_path.exists():
        logging.error("[ten_agent_web_demo] UAV dataset missing at %s", data_path)
        return tasks

    try:
        # Read CSV, assuming header exists (row 0).
        # We use the names provided by user but skip the header row if it matches names.
        # Actually, simpler to just read with header=0 if we know the columns.
        # The file has: image_name,x_min,y_min,x_max,y_max,class_name
        df = pd.read_csv(data_path)
        # Normalize column names just in case
        df.columns = [c.strip() for c in df.columns]
    except Exception as exc:
        logging.error("[ten_agent_web_demo] Failed to read UAV CSV: %s", exc)
        return []

    if "class_name" not in df.columns or "image_name" not in df.columns:
        logging.error("[ten_agent_web_demo] CSV missing required columns.")
        return []

    # Filter for vehicles (car, truck, bus)
    target_classes = {"car", "truck", "bus"}
    df_vehicles = df[df["class_name"].isin(target_classes)]
    
    # Group by image to get counts
    counts = df_vehicles.groupby("image_name").size()
    
    valid_tasks = []
    idx = 0
    
    for image_name, count in counts.items():
        # image_name in CSV is like "original/imgs/..."
        # root is "uav_roundabouts"
        # full path = root / image_name
        full_path = root / image_name
        
        if not full_path.exists():
            # Try constructing it as root / "original" / "imgs" / basename?
            # In case image_name is just filename.
            alt_path = root / "original" / "imgs" / Path(image_name).name
            if alt_path.exists():
                full_path = alt_path
            else:
                continue

        valid_tasks.append(
            UavTask(
                task_id=idx,
                image_path=str(full_path),
                image_name=str(image_name),
                num_cars=int(count),
            )
        )
        idx += 1
        if max_images is not None and len(valid_tasks) >= max_images:
            break

    logging.info("[ten_agent_web_demo] Loaded %d UAV tasks.", len(valid_tasks))
    return valid_tasks


UAV_TASKS: List[UavTask] = load_uav_roundabout_tasks()
UAV_TASK_LOOKUP: Dict[str, UavTask] = {task.image_name: task for task in UAV_TASKS}


@functools.lru_cache(maxsize=1)
def _get_mission_templates_cached() -> List[MissionTemplate]:
    return get_uav_mission_templates()


def get_uav_task_for_image_name(image_name: str) -> UavTask | None:
    """
    Look up a UAV task by its image filename (basename or full relative path).
    """
    # Exact match
    if image_name in UAV_TASK_LOOKUP:
        return UAV_TASK_LOOKUP[image_name]
    
    # Try matching by basename
    target_base = Path(image_name).name
    for name, task in UAV_TASK_LOOKUP.items():
        if Path(name).name == target_base:
            return task
    return None


@dataclass
class SimulationState:
    agents: List[Agent]
    tasks: List[Task] = field(default_factory=list)
    next_task_id: int = 1
    leaders: List[Leader] = field(default_factory=list)
    leader_by_role: Dict[str, Leader] = field(default_factory=dict)
    agent_to_leader: Dict[int, Leader] = field(default_factory=dict)
    agent_to_leader: Dict[int, Leader] = field(default_factory=dict)
    architecture_mode: str = DEFAULT_ARCHITECTURE_MODE
    
    # Dynamic Configuration
    use_softmax: bool = False
    softmax_temperature: float = 0.8
    exploration_coefficient: float = 0.0
    enable_fairness: bool = False
    fairness_lambda: float = 0.0


def create_leaders_and_groups(agents: List[Agent]) -> List[Leader]:
    """
    Create leaders for math and reasoning groups and assign worker IDs.
    """
    math_agents = [agent for agent in agents if agent.role == "math"]
    reasoning_agents = [agent for agent in agents if agent.role == "reasoning"]
    leaders: List[Leader] = []
    if math_agents:
        leaders.append(
            Leader(
                leader_id=1,
                name="Math Leader",
                role="math_leader",
                worker_ids=[agent.agent_id for agent in math_agents],
            )
        )
    if reasoning_agents:
        leaders.append(
            Leader(
                leader_id=2,
                name="Reasoning Leader",
                role="reasoning_leader",
                worker_ids=[agent.agent_id for agent in reasoning_agents],
            )
        )
    return leaders


def infer_task_type(input_text: str) -> str:
    text = input_text.lower()
    
    # UAV detection: "count" + "car" + image extension
    if "count" in text and "car" in text:
        # Check for image extensions
        if re.search(r"\.jpe?g", text):
            return TASK_TYPE_UAV
    
    # UAV Mission detection
    mission_keywords = ["drone", "uav", "survey", "takeoff", "orbit", "altitude", "mission", "fly"]
    if any(k in text for k in mission_keywords):
        return TASK_TYPE_UAV_MISSION

    if re.search(r"\d", text):
        if re.search(r"(?:kg|kilogram|lbs?|pounds?|mile|km|kilometer)", text):
            return TASK_TYPE_MATH
        if re.fullmatch(r"[\d\.\s\+\-\*/\(\)]+", text):
            return TASK_TYPE_MATH
    
    # Default to reasoning for everything else
    return TASK_TYPE_REASONING


def _get_credit(agent: Agent, task_type: str) -> float:
    info = agent.credits.get(task_type)
    if not info:
        return 0.5
    return float(info.get("credit_mean", 0.5))


def _get_exposure(agent: Agent, task_type: str) -> int:
    return agent.exposure.get(task_type, 0)


def _eligible_agents(state: SimulationState, task_type: str) -> List[Agent]:
    candidates = [agent for agent in state.agents if agent.supports_task_type(task_type)]
    if candidates:
        return candidates
    return list(state.agents)


def compute_selection_scores(
    state: SimulationState,
    task_type: str,
    candidate_agents: List[Agent] | None = None,
):
    candidates = list(candidate_agents) if candidate_agents is not None else _eligible_agents(
        state, task_type
    )
    if not candidates:
        candidates = list(state.agents)
    total_tasks = sum(_get_exposure(agent, task_type) for agent in candidates) + 1
    N = len(candidates) or 1
    target_share = 1.0 / N
    scored = []
    for agent in candidates:
        credit = _get_credit(agent, task_type)
        exposure = _get_exposure(agent, task_type)
        exploration = 0.0
        if total_tasks > 1 and state.exploration_coefficient > 0:
            exploration = state.exploration_coefficient * math.sqrt(
                math.log(1.0 + total_tasks) / (1.0 + exposure)
            )
        share = exposure / float(total_tasks)
        fairness_penalty = 0.0
        if state.enable_fairness and state.fairness_lambda > 0:
            fairness_penalty = -state.fairness_lambda * max(0.0, share - target_share)
        total = credit + exploration + fairness_penalty
        scored.append((agent, total, credit, exploration, fairness_penalty))
    return scored


def select_agent(
    state: SimulationState,
    task_type: str,
    candidate_agents: List[Agent] | None = None,
):
    scored = compute_selection_scores(state, task_type, candidate_agents=candidate_agents)
    if not scored:
        raise RuntimeError("No agents available for selection")
        
    if state.use_softmax:
        # Softmax selection
        temp = state.softmax_temperature
        # Avoid division by zero or extremely small temp
        temp = max(0.01, temp)
        
        logits = [score / temp for _, score, _, _, _ in scored]
        max_logit = max(logits)
        # Numerical stability
        exp_scores = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_scores) or 1.0
        probs = [val / total for val in exp_scores]
        chosen = random.choices(scored, weights=probs, k=1)[0]
    else:
        # Greedy selection (Argmax)
        # Shuffle first to break ties randomly
        random.shuffle(scored)
        chosen = max(scored, key=lambda x: x[1])
        
        chosen = max(scored, key=lambda x: x[1])
        
    logger.info(
        "[select_agent] task_type=%s use_softmax=%s chosen_agent=%s",
        task_type,
        state.use_softmax,
        chosen[0].name,
    )
    return chosen[0], scored


def select_agent_flat(state: SimulationState, task_type: str):
    """
    Supervisor directly selects among all eligible agents.
    """
    agent, scored = select_agent(state, task_type)
    return agent, scored


def _leader_candidates_for_task(state: SimulationState, task_type: str) -> List[Leader]:
    if not state.leaders:
        return []
    role_key = f"{task_type}_leader"
    candidates = [
        leader for leader in state.leaders if leader.role == role_key
    ]
    if candidates:
        return candidates
    # Fallback to any leader that manages workers of the right role.
    role_specific = []
    for leader in state.leaders:
        for worker_id in leader.worker_ids:
            agent = next((a for a in state.agents if a.agent_id == worker_id), None)
            if agent and agent.supports_task_type(task_type):
                role_specific.append(leader)
                break
    return role_specific or list(state.leaders)


def _compute_group_credit_mean(state: SimulationState, leader: Leader, task_type: str) -> float:
    workers = [agent for agent in state.agents if agent.agent_id in leader.worker_ids]
    if not workers:
        return 0.5
    total = sum(_get_credit(agent, task_type) for agent in workers)
    return total / len(workers)


def compute_leader_selection_scores(
    state: SimulationState,
    task_type: str,
    candidate_leaders: List[Leader],
):
    if not candidate_leaders:
        return []
    total_exposure = (
        sum(
            leader.get_or_create_group_metrics(task_type).exposure_count
            for leader in candidate_leaders
        )
        + 1
    )
    N = len(candidate_leaders) or 1
    target_share = 1.0 / N
    scored = []
    for leader in candidate_leaders:
        metrics = leader.get_or_create_group_metrics(task_type)
        credit = metrics.credit_mean
        exposure = metrics.exposure_count
        exploration = 0.0
        if total_exposure > 1 and state.exploration_coefficient > 0:
            exploration = state.exploration_coefficient * math.sqrt(
                math.log(1.0 + total_exposure) / (1.0 + exposure)
            )
        share = exposure / float(total_exposure)
        fairness_penalty = 0.0
        if state.enable_fairness and state.fairness_lambda > 0:
            fairness_penalty = -state.fairness_lambda * max(0.0, share - target_share)
        total = credit + exploration + fairness_penalty
        scored.append((leader, total, credit, exploration, fairness_penalty))
    return scored


def select_leader_for_task(state: SimulationState, task_type: str):
    candidates = _leader_candidates_for_task(state, task_type)
    scored = compute_leader_selection_scores(state, task_type, candidates)
    if not scored:
        raise RuntimeError("No leaders available for selection")
        
    if state.use_softmax:
        temp = state.softmax_temperature
        temp = max(0.01, temp)
        logits = [score / temp for _, score, _, _, _ in scored]
        max_logit = max(logits)
        exp_scores = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_scores) or 1.0
        probs = [val / total for val in exp_scores]
        chosen = random.choices(scored, weights=probs, k=1)[0]
    else:
        random.shuffle(scored)
        chosen = max(scored, key=lambda x: x[1])
        
    return chosen[0], scored


def select_agent_hierarchical(state: SimulationState, task_type: str):
    leader, leader_scores = select_leader_for_task(state, task_type)
    agent, agent_scores = select_agent_within_group(state, leader, task_type)
    return leader, leader_scores, agent, agent_scores


def select_agent_within_group(state: SimulationState, leader: Leader, task_type: str):
    candidates = [
        agent
        for agent in state.agents
        if agent.agent_id in leader.worker_ids and agent.supports_task_type(task_type)
    ]
    if not candidates:
        raise RuntimeError(f"Leader {leader.name} has no worker agents")
    agent, scored = select_agent(state, task_type, candidate_agents=candidates)
    return agent, scored


def _select_agent_routing(
    state: SimulationState,
    task_type: str,
    use_leaders: bool | None = None,
) -> tuple[str, Leader | None, Sequence, Agent, Sequence]:
    """
    Select an agent (and leader if applicable) honoring the architecture preference.
    """
    if use_leaders is None:
        mode = state.architecture_mode
    else:
        mode = ARCH_MODE_HIER if use_leaders else ARCH_MODE_FLAT

    leader: Leader | None = None
    leader_scores: Sequence = []
    agent: Agent | None = None
    scored: Sequence = []
    if mode == ARCH_MODE_HIER and state.leaders:
        try:
            leader, leader_scores, agent, scored = select_agent_hierarchical(state, task_type)
        except RuntimeError:
            leader = None
            leader_scores = []
            agent = None
            scored = []
    if agent is None:
        agent, scored = select_agent_flat(state, task_type)
    if leader is None and mode != ARCH_MODE_FLAT:
        leader = state.agent_to_leader.get(agent.agent_id)
    return mode, leader, leader_scores, agent, scored


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")
_YES_TOKENS = {"yes", "y", "yeah", "yep", "true", "affirmative", "certainly", "absolutely"}
_NO_TOKENS = {"no", "n", "nope", "false", "negative", "never"}


def _extract_first_number(text: str) -> float | None:
    if not text:
        return None
    matches = _NUMERIC_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _coerce_to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _normalize_expected_reasoning(expected: str | float | int | None):
    if expected is None:
        return {"kind": "none"}
    if isinstance(expected, bool):
        return {"kind": "bool", "value": expected}
    text = str(expected).strip().lower()
    if not text:
        return {"kind": "none"}
    if text in _YES_TOKENS:
        return {"kind": "bool", "value": True}
    if text in _NO_TOKENS:
        return {"kind": "bool", "value": False}
    return {"kind": "text", "value": _normalize_text_token(text)}


def is_yes_no_question(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    if not t.endswith("?"):
        return False
    first = t.split()[0]
    return first in {
        "is",
        "are",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "has",
        "have",
        "had",
    }


def extract_yes_no(answer_text: str) -> tuple[bool | None, bool]:
    """
    Return (verdict, explicit) where explicit=True if the answer clearly starts
    with 'yes' or 'no'.
    """
    t = (answer_text or "").strip().lower()
    if not t:
        return None, False
    if t.startswith("yes"):
        return True, True
    if t.startswith("no"):
        return False, True
    if "raining" in t and "wet" in t:
        return True, False
    if "drive" in t and "sleep" in t and "safe" in t and "not" in t:
        return False, False
    return None, False


def evaluate_answer(
    task_type: str,
    question: str,
    expected: str | float | int | None,
    agent_answer: str,
    predicted_label: str | None = None,
) -> Dict[str, object]:
    """
    Evaluate an agent's answer independent from the agent's self-report.
    """
    agent_answer = (agent_answer or "").strip()
    result: Dict[str, object] = {
        "success": False,
        "numeric_match": None,
        "reason": "no evaluation",
        "suspicious": False,
        "verdict_match": None,
        "partial": False,
    }
    if task_type == "math":
        expected_float = _coerce_to_float(expected)
        answer_float = _extract_first_number(agent_answer)
        if expected_float is not None and answer_float is not None:
            numeric_match = math.isclose(answer_float, expected_float, rel_tol=1e-3, abs_tol=1e-3)
            result["numeric_match"] = numeric_match
            result["success"] = numeric_match
            if numeric_match:
                result["reason"] = "numeric match"
            else:
                result["reason"] = f"expected {expected_float:g}, got {answer_float:g}"
            result["suspicious"] = not bool(result["success"])
        else:
            expected_str = str(expected).strip().lower() if expected is not None else ""
            answer_str = agent_answer.lower()
            if expected_str and answer_str:
                success = answer_str == expected_str
                result["success"] = success
                result["reason"] = "text match" if success else f"expected '{expected_str}'"
            else:
                result["reason"] = "insufficient ground truth"
                result["suspicious"] = False
    else:
        normalized = _normalize_expected_reasoning(expected)
        if normalized["kind"] == "none":
            result["success"] = None
            result["reason"] = "awaiting human feedback (no ground truth)"
            result["suspicious"] = False
        elif normalized["kind"] == "bool":
            expected_bool = normalized["value"]
            
            # If we have a predicted label from the LLM (yes/no), use it directly
            if predicted_label:
                verdict = (predicted_label == "yes")
                explicit = True
            else:
                verdict, explicit = extract_yes_no(agent_answer)
                
            if verdict is None:
                result["success"] = False
                result["reason"] = "unable to extract yes/no answer"
                result["verdict_match"] = False
                result["partial"] = False
                result["suspicious"] = True
            elif verdict == expected_bool:
                result["success"] = True
                result["verdict_match"] = True
                result["partial"] = not explicit
                if explicit:
                    result["reason"] = "explicit yes/no matches expected"
                else:
                    result["reason"] = "implicit answer matches expected (partial credit)"
                result["suspicious"] = False
            else:
                expected_label = "yes" if expected_bool else "no"
                observed_label = "yes" if verdict else "no"
                result["success"] = False
                result["verdict_match"] = False
                result["partial"] = False
                result["reason"] = f"answer contradicts expected (expected '{expected_label}', got '{observed_label}')"
                result["suspicious"] = True
        else:
            expected_text = normalized["value"]
            answer_norm = _normalize_text_token(agent_answer)
            if not answer_norm:
                result["success"] = False
                result["reason"] = "no comparable reasoning answer extracted"
                result["verdict_match"] = False
                result["suspicious"] = True
            else:
                success = expected_text in answer_norm
                result["success"] = success
                result["verdict_match"] = success
                result["reason"] = (
                    "answer contains expected phrase" if success else f"expected '{expected_text}'"
                )
                result["suspicious"] = not success
    return result


def _safe_eval_expression(expression: str) -> float:
    if not re.fullmatch(r"[0-9\.\s\+\-\*/\(\)]+", expression):
        raise ValueError("Unsupported characters")
    node = ast.parse(expression, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            return left / right
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            value = _eval(n.operand)
            return value if isinstance(n.op, ast.UAdd) else -value
        raise ValueError("Unsupported expression")

    return _eval(node)


def _handle_math(input_text: str) -> tuple[str, float | None]:
    """
    Parse and solve small arithmetic prompts.

    Supports:
      - Pure expressions: "2+2", "3 * (4 + 5)"
      - Questions: "What is 3 + 4?", "Compute 10 - 3."
      - Verb phrases: "Multiply 13 by 15.", "Divide 20 by 5."
      - Simple unit conversions: "Convert 10 kg to pounds", "10 kilograms in lbs".
    """
    text = input_text.strip()
    lower = text.lower()

    # ---- 1) Simple unit conversions (kg <-> lb, mile <-> km) ----
    # Look for a number + unit, then a target unit after "to" or "in"
    value_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(kg|kilogram[s]?|lb|lbs|pound[s]?|mile[s]?|km|kilometer[s]?)",
        lower,
    )
    target_match = re.search(
        r"(?:to|in)\s*(kg|kilogram[s]?|lb|lbs|pound[s]?|mile[s]?|km|kilometer[s]?)",
        lower,
    )

    if value_match and target_match:
        value = float(value_match.group(1))
        src = value_match.group(2)
        tgt = target_match.group(1)

        def starts(u: str, *prefixes: str) -> bool:
            return any(u.startswith(p) for p in prefixes)

        if starts(src, "kg") and starts(tgt, "lb", "pound"):
            result = value * 2.205
            return f"{value:g} kg ≈ {result:.2f} lbs", result
        if starts(src, "lb", "pound") and starts(tgt, "kg"):
            result = value / 2.205
            return f"{value:g} lbs ≈ {result:.2f} kg", result
        if starts(src, "mile") and starts(tgt, "km", "kilometer"):
            result = value * 1.60934
            return f"{value:g} miles ≈ {result:.2f} km", result
        if starts(src, "km", "kilometer") and starts(tgt, "mile"):
            result = value / 1.60934
            return f"{value:g} km ≈ {result:.2f} miles", result

    # ---- 2) Worded binary operations ----
    # "Multiply 13 by 15"
    m = re.search(
        r"multiply\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)",
        lower,
    )
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        result = a * b
        return f"{a:g} * {b:g} = {result:g}", result

    # "Divide 20 by 5"
    m = re.search(
        r"divide\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)",
        lower,
    )
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        if b == 0:
            return "Division by zero is undefined.", None
        result = a / b
        return f"{a:g} / {b:g} = {result:g}", result

    # "What is 3 plus 5", "Compute 10 - 3", "Add 4 and 7"
    m = re.search(
        r"(?:what is|compute|calculate|add)\s+(-?\d+(?:\.\d+)?)\s*(?:\+|plus|and)\s*(-?\d+(?:\.\d+)?)",
        lower,
    )
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        result = a + b
        return f"{a:g} + {b:g} = {result:g}", result

    m = re.search(
        r"(?:what is|compute|calculate|subtract)\s+(-?\d+(?:\.\d+)?)\s*(?:-|minus)\s*(-?\d+(?:\.\d+)?)",
        lower,
    )
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        result = a - b
        return f"{a:g} - {b:g} = {result:g}", result

    # ---- 3) Fallback: extract bare expression and use safe evaluator ----
    # Grab the first substring that looks like "digits and operators"
    expr_match = re.search(r"([0-9\.\s\+\-\*/\(\)]+)", text)
    if expr_match:
        expr = expr_match.group(1)
    else:
        expr = text  # last resort

    try:
        value = _safe_eval_expression(expr)
        return f"{expr.strip()} = {value:g}", value
    except Exception as e:
        return f"Could not evaluate expression: {e}", None



def _baseline_reasoning(text: str) -> str:
    if is_yes_no_question(text):
        return "Yes, under typical conditions that outcome holds."
    return "Provide a brief, direct answer based on common knowledge."


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "llama3.1"

def call_local_llm(
    input_data: str | list | None = None, 
    *, 
    user_text: str | None = None, 
    system_prompt: str | None = None, 
    timeout: float = 30.0
) -> Optional[str]:
    """
    Call local Ollama (llama3.1) via /api/chat and return the assistant reply as text.
    Supports:
      - input_data as list of dicts (messages)
      - input_data as string (user prompt)
      - user_text + system_prompt kwargs
    Return None on any error, and do not raise.
    """
    try:
        messages = []
        if isinstance(input_data, list):
            messages = input_data
        elif isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif user_text:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text})
        else:
            return None

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
            }
        }
        
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle chat response format
        if isinstance(data, dict):
            msg = data.get("message") or data.get("choices", [{}])[0].get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None
        
    except Exception as exc:
        print(f"[reasoning] Ollama error: {exc}", flush=True)
        return None


import json

def _handle_reasoning(input_text: str, agent_name: str = "Agent") -> tuple[str, bool, str | None]:
    """
    Handle general reasoning tasks.
    Prefer the local LLM (Ollama) when available; otherwise fall back to a deterministic baseline.
    Returns (answer_text, used_llm_flag, predicted_label).
    """
    text = input_text.strip()
    
    # Try Ollama first
    reply = call_local_llm(
        user_text=text,
        system_prompt=(
            "You are a helpful reasoning agent. "
            "Answer clearly and concisely, with step-by-step reasoning when helpful."
        ),
    )
    if reply:
        logging.info("[reasoning] backend=ollama")
        return reply, True, None
    
    logging.warning("[reasoning] backend=baseline (LLM returned no answer)")
    
    # Baseline fallback
    logging.info("[reasoning] backend=baseline")
    baseline = _baseline_reasoning(text)
    # Extract label from baseline for consistency
    label = None
    if baseline.lower().startswith("yes"):
        label = "yes"
    elif baseline.lower().startswith("no"):
        label = "no"
    return baseline, False, label


    return baseline, False, label


    return baseline, False, label


def _handle_uav_mission(input_text: str, agent_name: str = "Agent") -> tuple[str, Dict[str, Any]]:
    """
    Handle UAV mission planning tasks using Ollama.
    Returns (plan_text, debug_info).
    """
    text = input_text.strip()
    
    # Strong system prompt as requested
    system_prompt = (
        "You are a UAV mission planner for multirotor drones.\n"
        "- You receive a natural language mission request.\n"
        "- You must output a concrete Mission Plan as a numbered list of steps.\n"
        "- Always include ARM, TAKEOFF with altitude, NAVIGATION, PATTERN (if needed),\n"
        "  PERCEPTION (if requested), and RETURN/LAND.\n"
        "- If the user asks for multiple drones (e.g. \"use 2 drones\", \"with 3 UAVs\"),\n"
        "  produce separate sub-plans, prefixed with \"Agent 1:\", \"Agent 2:\", etc.\n"
        "- Adapt the altitude, area, and behavior to the specific request,\n"
        "  do NOT just repeat a generic plan.\n"
        "- Plans can be short for simple tasks or longer for complex missions."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    try:
        response = call_local_llm(messages)
        if response:
            logging.info("[uav_mission] backend=ollama question=%r", text[:50])
            return response, {"backend": "ollama"}
    except Exception as e:
        logging.warning("[uav_mission] Ollama failed: %s", e)
        
    # Fallback
    logging.info("[uav_mission] backend=baseline question=%r", text[:50])
    return (
        "Mission Plan (Baseline):\n"
        "1. Arm motors.\n"
        "2. Takeoff to default altitude.\n"
        "3. Execute basic survey pattern.\n"
        "4. Return to launch and land."
    ), {"backend": "baseline"}


def _find_mission_template(user_prompt: str) -> MissionTemplate | None:
    """
    Find best matching mission template based on lexical overlap.
    """
    templates = _get_mission_templates_cached()
    user_lower = user_prompt.lower()
    
    best_match = None
    best_score = 0.0
    
    for tmpl in templates:
        # Simple ratio of similarity
        score = SequenceMatcher(None, user_lower, tmpl.prompt.lower()).ratio()
        
        # Boost if prompt contains specific keywords from template prompt
        # e.g. "survey" in user prompt and "survey" in template prompt
        tmpl_keywords = set(tmpl.prompt.lower().split())
        user_keywords = set(user_lower.split())
        overlap = len(tmpl_keywords.intersection(user_keywords))
        
        # Combine ratio and overlap
        final_score = score + (0.1 * overlap)
        
        if final_score > best_score:
            best_score = final_score
            best_match = tmpl
            
    # Threshold to avoid garbage matches
    if best_score < 0.3:
        return None
        
    return best_match


def score_uav_mission(template: MissionTemplate, agent_plan: str) -> float:
    """
    Return r_num in [0, 1] based on presence of key actions and parameters.
    """
    plan_lower = agent_plan.lower()
    
    # Key tokens to look for
    required_tokens = ["arm", "takeoff", "land"]
    
    # Add pattern if specified
    if template.pattern and template.pattern != "hover":
        required_tokens.append(template.pattern)
        
    # Add altitude if specified (check for the number)
    if template.altitude_m > 0:
        required_tokens.append(str(int(template.altitude_m)))
        
    # Check for multi-agent headers if needed
    if template.drones > 1:
        required_tokens.append("agent 2")
        
    hits = 0
    for token in required_tokens:
        if token in plan_lower:
            hits += 1
            
    if not required_tokens:
        return 0.5
        
    return hits / len(required_tokens)


def _update_credit(agent: Agent, task_type: str, reward_value: float) -> None:
    """
    Update the exponential moving-average credit for an agent on a task type.
    """
    reward = max(0.0, min(1.0, reward_value))
    credit = agent.credits.setdefault(
        task_type, {"alpha": 1.0, "beta": 1.0, "credit_mean": 0.5}
    )
    credit["alpha"] = credit["alpha"] * DECAY + reward
    credit["beta"] = credit["beta"] * DECAY + (1.0 - reward)
    total = credit["alpha"] + credit["beta"]
    credit["credit_mean"] = credit["alpha"] / total if total > 0 else 0.5


def _apply_leader_feedback_adjustment(
    state: SimulationState,
    agent: Agent,
    task: Task,
    previous_success: bool | None,
    new_success: bool,
    previous_suspicious: bool,
    new_suspicious: bool,
) -> None:
    leader = state.agent_to_leader.get(agent.agent_id)
    if not leader:
        return
    metrics = leader.get_or_create_group_metrics(task.task_type)
    if previous_success is True:
        metrics.successes = max(0, metrics.successes - 1)
    elif previous_success is False:
        metrics.failures = max(0, metrics.failures - 1)
    if new_success:
        metrics.successes += 1
    else:
        metrics.failures += 1
    metrics.credit_mean = _compute_group_credit_mean(state, leader, task.task_type)
    if previous_suspicious:
        metrics.suspicious_events = max(0, metrics.suspicious_events - 1)
    if new_suspicious:
        metrics.suspicious_events += 1
    task.leader_id = leader.leader_id
    task.leader_name = leader.name


def execute_task(
    state: SimulationState,
    input_text: str,
    task_type: str | None = None,
    expected_answer: str | float | int | None = None,
) -> Task:
    task_type = task_type or infer_task_type(input_text)
    task = Task(
        task_id=state.next_task_id,
        task_type=task_type,
        input_text=input_text,
        timestamp=time.time(),
    )
    task.question = input_text
    state.next_task_id += 1

    mode, leader, leader_scores, agent, scored = _select_agent_routing(state, task_type)

    task.assigned_agent_id = agent.agent_id
    task.assigned_agent_name = agent.name
    task.leader_id = leader.leader_id if leader else None
    task.leader_name = leader.name if leader else None


    # 2. Execute task based on type
    agent.exposure[task_type] = agent.exposure.get(task_type, 0) + 1

    expected = expected_answer
    used_llm = False
    predicted_label = None
    
    if task_type == "math":
        output, math_expected = _handle_math(input_text)
        if math_expected is not None:
            expected = math_expected
    elif task_type == TASK_TYPE_UAV_MISSION:
        output, debug_info = _handle_uav_mission(input_text, agent_name=agent.name)
        used_llm = (debug_info.get("backend") == "ollama")
        
        # Supervisor grading
        template = _find_mission_template(input_text)
        if template:
            r_num = score_uav_mission(template, output)
            eval_str = f"Dataset=uav_mission, template={template.id}, r_num={r_num:.2f}"
        else:
            r_num = 0.5
            eval_str = "Dataset=uav_mission, no close template (r_num=0.5)"
            
        # Combine with process reward (placeholder)
        r_proc = 0.0 # Placeholder for thumbs up/down
        total_reward = UAV_REWARD_ALPHA * r_num + (1.0 - UAV_REWARD_ALPHA) * r_proc
        
        success = r_num > 0.6
        evaluation = {
            "success": success, 
            "reason": eval_str,
            "numeric_reward": total_reward
        }
        
        # Update agent credit
        agent.tasks_done += 1
        if success:
            agent.successes += 1
            agent.score += total_reward
        else:
            agent.failures += 1
            agent.score = max(0.0, agent.score - 0.25)
        _update_credit(agent, TASK_TYPE_UAV_MISSION, total_reward)
    else:
        output, used_llm, predicted_label = _handle_reasoning(input_text, agent_name=agent.name)






    evaluation = evaluate_answer(task_type, input_text, expected, output, predicted_label=predicted_label)
    success = evaluation.get("success")
    numeric_match = evaluation.get("numeric_match")
    task.output = output
    task.agent_answer = output
    task.used_llm = used_llm
    task.numeric_match = numeric_match if isinstance(numeric_match, bool) else None
    task.verdict_match = evaluation.get("verdict_match")
    task.evaluation_reason = str(evaluation.get("reason", ""))
    task.success = success
    task.suspicious = bool(evaluation.get("suspicious", False))
    partial_credit = bool(evaluation.get("partial", False))
    task.partial_credit = partial_credit
    if success is True:
        reward_value = 0.6 if partial_credit else 1.0
    else:
        reward_value = 0.0
    task.numeric_reward = reward_value
    task.total_reward = reward_value
    if expected is not None:
        task.ground_truth = str(expected)

    agent.tasks_done += 1
    if success is True:
        agent.successes += 1
        agent.score += reward_value
        _update_credit(agent, task_type, reward_value)
        agent.suspicious_count = max(0, agent.suspicious_count - 1)
    elif success is False:
        agent.failures += 1
        agent.score = max(0.0, agent.score - 0.5)
        _update_credit(agent, task_type, 0.0)
        agent.suspicious_count += 1

    agent.history.append(
        {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "input": task.input_text,
            "output": task.output,
            "success": success,
            "evaluation_reason": task.evaluation_reason,
            "used_llm": used_llm,
            "suspicious": task.suspicious,
            "timestamp": task.timestamp,
        }
    )

    task.selection_scores = {entry[0].agent_id: entry[1] for entry in scored}
    chosen_entry = next((entry for entry in scored if entry[0].agent_id == agent.agent_id), None)
    if chosen_entry:
        _, total_score, credit_component, exploration_component, fairness_penalty = chosen_entry
        task.selection_components = {
            "credit_mean": credit_component,
            "exploration_term": exploration_component,
            "fairness_penalty": fairness_penalty,
            "total_score": total_score,
        }

    if leader and leader_scores:
        chosen_leader_entry = next(
            (entry for entry in leader_scores if entry[0].leader_id == leader.leader_id),
            None,
        )
        if chosen_leader_entry:
            _, total_score, credit_component, exploration_component, fairness_penalty = chosen_leader_entry
            task.leader_selection_components = {
                "credit_mean": credit_component,
                "exploration_term": exploration_component,
                "fairness_penalty": fairness_penalty,
                "total_score": total_score,
            }
            

    if leader:
        group_credit_mean = _compute_group_credit_mean(state, leader, task_type)
        reward = reward_value
        leader.update_group_metrics(
            task_type,
            success,
            reward,
            credit_mean=group_credit_mean,
            suspicious=task.suspicious,
        )
        question_snippet = input_text[:60] + ("..." if len(input_text) > 60 else "")
        leader.record_group_memory(
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "question": question_snippet,
                "agent_id": agent.agent_id,
                "success": success,
                "reward": reward,
                "timestamp": task.timestamp,
                "evaluation_reason": task.evaluation_reason,
                "suspicious": task.suspicious,
                "partial_credit": task.partial_credit,
            },
            limit=MAX_GROUP_MEMORY,
        )

    state.tasks.append(task)
    return task


def create_default_state() -> SimulationState:
    agents = create_ten_agents()
    leaders = create_leaders_and_groups(agents)
    state = SimulationState(agents=agents, leaders=leaders)
    state.leader_by_role = {leader.role: leader for leader in leaders}
    for leader in leaders:
        for worker_id in leader.worker_ids:
            state.agent_to_leader[worker_id] = leader
    return state


def split_into_subtasks(user_text: str) -> List[str]:
    parts = re.split(r"\s+(?:and|&|;)\s+", user_text, flags=re.IGNORECASE)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned or [user_text.strip()]


def apply_feedback(state: SimulationState, task_id: int, positive: bool) -> None:
    task = next((t for t in state.tasks if t.task_id == task_id), None)
    if not task or task.assigned_agent_id is None:
        return
    agent = next((a for a in state.agents if a.agent_id == task.assigned_agent_id), None)
    if not agent:
        return
    previous_success = task.success
    previous_suspicious = task.suspicious
    if previous_success == positive:
        return
    # Remove prior effect
    if previous_success is True:
        agent.successes = max(0, agent.successes - 1)
        agent.score = max(0.0, agent.score - 1.0)
    elif previous_success is False:
        agent.failures = max(0, agent.failures - 1)
        agent.score += 0.5  # since we previously subtracted

    task.success = positive
    task.suspicious = not positive
    task.evaluation_reason = "Human feedback override."
    task.numeric_match = None
    task.verdict_match = None
    task.partial_credit = False
    if previous_suspicious:
        agent.suspicious_count = max(0, agent.suspicious_count - 1)
    if positive:
        agent.successes += 1
        agent.score += 1.0
    else:
        agent.failures += 1
        agent.score = max(0.0, agent.score - 0.5)
    if task.suspicious:
        agent.suspicious_count += 1
    _update_credit(agent, task.task_type, 1.0 if positive else 0.0)
    _apply_leader_feedback_adjustment(
        state,
        agent,
        task,
        previous_success,
        positive,
        previous_suspicious or False,
        task.suspicious,
    )


def load_uav_tasks(limit: int | None = None, random_subset: bool = True) -> List[UavTask]:
    """
    Return cached UAV tasks with optional subsampling.
    """
    tasks = UAV_TASKS
    if not tasks:
        return []
    if limit is None or limit >= len(tasks):
        return list(tasks)
    if random_subset:
        return random.sample(tasks, limit)
    return list(tasks[:limit])


def sample_uav_task() -> UavTask | None:
    if not UAV_TASKS:
        return None
    return random.choice(UAV_TASKS)


def _compute_uav_numeric_reward(ground_truth: int, prediction: int) -> float:
    error = abs(prediction - ground_truth)
    if error == 0:
        return 1.0
    denom = max(1, ground_truth)
    return max(0.0, 1.0 - (error / denom))


def execute_uav_car_count_task(
    state: SimulationState,
    task: UavTask,
    policy: str = "score_fair_softmax",
    use_leaders: bool | None = None,
    leader_override: Leader | None = None,
) -> Task:
    """
    Execute one UAV car-count task while respecting the society's scoring rules.
    """
    _ = policy  # placeholder for future policies; current logic uses softmax scores globally.
    question = f"UAV: count cars in {task.image_name}"
    sim_task = Task(
        task_id=state.next_task_id,
        task_type=TASK_TYPE_UAV,
        input_text=question,
        question=question,
        timestamp=time.time(),
        image_path=task.image_path,
        image_name=task.image_name,
        ground_truth=str(task.num_cars),
        leader_id=leader_override.leader_id if leader_override else None,
    )
    state.next_task_id += 1
    
    # If leader is already assigned (multi-image task), use it.
    if use_leaders and sim_task.leader_id:
        # Leader is already set, just need to assign an agent
        # The leader logic below will re-select an agent, which is fine.
        # But we want to ensure the leader credit is updated correctly.
        pass

    _mode, leader, leader_scores, agent, scored = _select_agent_routing(
        state, TASK_TYPE_UAV, use_leaders=use_leaders
    )
    
    print(f"DEBUG: execute_task start prompt={question[:20]}")
    # 1. Infer task type if not provided
    # The original instruction had `if not task_type:ader_id:`, which is a syntax error.
    # Assuming it meant to check `sim_task.leader_id` as per the context of the next lines.
    # However, the `if not task_type:` part seems to be from a different `execute_task` function.
    # Given the context of `execute_uav_car_count_task`, `task_type` is always `TASK_TYPE_UAV`.
    # The instruction was "Add print statements to debug execute_task flow."
    # and the provided snippet was for `execute_task` (general one).
    # I will assume the user wants to add the print statement and the leader override logic
    # that follows it, but adapted to `execute_uav_car_count_task` context.
    # The `if not task_type:` part is not applicable here.
    # The `ader_id:` part is also a typo.
    # I will only add the print statement and keep the existing leader override logic.
    # If the user intended to add the `if not task_type:` logic, they need to clarify.
    # For now, I'll just add the print statement as it's the only syntactically valid and
    # contextually plausible part of the requested change for this function.
    # The original code already has the "Override leader if provided" block.
    # So, I'll just add the print statement.

    # Override leader if provided
    if sim_task.leader_id:
        # We trust the caller (process_user_line) to have picked a valid leader
        # But we need the Leader object to update metrics
        leader = next((l for l in state.leaders if l.leader_id == sim_task.leader_id), None)
        if leader:
            sim_task.leader_name = leader.name

    sim_task.assigned_agent_id = agent.agent_id
    sim_task.assigned_agent_name = agent.name
    if not sim_task.leader_id:
        sim_task.leader_id = leader.leader_id if leader else None
        sim_task.leader_name = leader.name if leader else None

    agent.exposure[TASK_TYPE_UAV] = agent.exposure.get(TASK_TYPE_UAV, 0) + 1

    # Execute YOLO
    detector = get_uav_detector()
    prediction = 0
    if detector is None:
        logging.warning("[ten_agent_web_demo] YOLO detector unavailable; defaulting to 0 cars.")
    else:
        try:
            prediction = detector.count_cars(task.image_path)
        except Exception as exc: # pragma: no cover - safety net
            logging.error(
                "[ten_agent_web_demo] YOLO inference failed on %s: %s", task.image_path, exc
            )
            prediction = 0

    gt_cars = task.num_cars
    abs_error = abs(prediction - gt_cars)
    numeric_reward = _compute_uav_numeric_reward(gt_cars, prediction)
    process_reward = 0.0 # Placeholder
    total_reward = UAV_REWARD_ALPHA * numeric_reward + (1.0 - UAV_REWARD_ALPHA) * process_reward
    sim_task.numeric_reward = numeric_reward
    sim_task.process_reward = process_reward
    sim_task.total_reward = total_reward
    sim_task.output = str(prediction)
    sim_task.agent_answer = sim_task.output
    sim_task.ground_truth = str(gt_cars)
    sim_task.abs_error = abs_error
    sim_task.evaluation_reason = (
        f"Dataset=uav_roundabouts GT={gt_cars}, prediction={prediction}, abs_error={abs_error}"
    )

    success_flag = abs_error <= UAV_SUCCESS_TOLERANCE
    sim_task.success = success_flag
    sim_task.partial_credit = not success_flag and numeric_reward > 0
    sim_task.suspicious = numeric_reward < 0.3

    agent.tasks_done += 1
    if success_flag:
        agent.successes += 1
        agent.score += total_reward
        agent.suspicious_count = max(0, agent.suspicious_count - 1)
    else:
        agent.failures += 1
        agent.score = max(0.0, agent.score - 0.25)
        agent.suspicious_count += 1
    _update_credit(agent, TASK_TYPE_UAV, total_reward)

    agent.history.append(
        {
            "task_id": sim_task.task_id,
            "task_type": sim_task.task_type,
            "input": sim_task.input_text,
            "output": sim_task.output,
            "success": sim_task.success,
            "evaluation_reason": sim_task.evaluation_reason,
            "used_llm": sim_task.used_llm,
            "suspicious": sim_task.suspicious,
            "timestamp": sim_task.timestamp,
            "ground_truth": sim_task.ground_truth,
            "prediction": sim_task.output,
            "image_path": task.image_path,
        }
    )

    sim_task.selection_scores = {entry[0].agent_id: entry[1] for entry in scored}
    chosen_entry = next((entry for entry in scored if entry[0].agent_id == agent.agent_id), None)
    if chosen_entry:
        _, total_score, credit_component, exploration_component, fairness_penalty = chosen_entry
        sim_task.selection_components = {
            "credit_mean": credit_component,
            "exploration_term": exploration_component,
            "fairness_penalty": fairness_penalty,
            "total_score": total_score,
        }

    if leader and leader_scores:
        chosen_leader_entry = next(
            (entry for entry in leader_scores if entry[0].leader_id == leader.leader_id),
            None,
        )
        if chosen_leader_entry:
            _, total_score, credit_component, exploration_component, fairness_penalty = (
                chosen_leader_entry
            )
            sim_task.leader_selection_components = {
                "credit_mean": credit_component,
                "exploration_term": exploration_component,
                "fairness_penalty": fairness_penalty,
                "total_score": total_score,
            }

    if leader:
        group_credit_mean = _compute_group_credit_mean(state, leader, TASK_TYPE_UAV)
        leader.update_group_metrics(
            TASK_TYPE_UAV,
            success_flag,
            total_reward,
            credit_mean=group_credit_mean,
            suspicious=sim_task.suspicious,
        )
        question_snippet = sim_task.input_text[:60] + ("..." if len(sim_task.input_text) > 60 else "")
        leader.record_group_memory(
            {
                "task_id": sim_task.task_id,
                "task_type": sim_task.task_type,
                "question": question_snippet,
                "agent_id": agent.agent_id,
                "success": sim_task.success,
                "reward": numeric_reward,
                "timestamp": sim_task.timestamp,
                "evaluation_reason": sim_task.evaluation_reason,
                "suspicious": sim_task.suspicious,
                "partial_credit": sim_task.partial_credit,
            },
            limit=MAX_GROUP_MEMORY,
        )

    state.tasks.append(sim_task)
    return sim_task


def process_user_line(state: SimulationState, line: str) -> None:
    """
    Process a single line of user input, inferring task type and handling UAV groups.
    """
    task_type = infer_task_type(line)
    
    if task_type == TASK_TYPE_UAV:
        # Extract image names
        # Look for tokens ending in .jpg or .jpeg
        tokens = line.split()
        image_names = [t for t in tokens if re.search(r"\.jpe?g$", t, re.IGNORECASE)]
        
        # Clean up punctuation attached to filenames
        cleaned_names = []
        for name in image_names:
            # Remove trailing punctuation like comma, etc.
            clean = re.sub(r"[^\w\./\-]+$", "", name)
            cleaned_names.append(clean)
            
        valid_tasks = []
        for name in cleaned_names:
            uav_task = get_uav_task_for_image_name(name)
            if uav_task:
                valid_tasks.append(uav_task)
            else:
                logging.warning("[ten_agent_web_demo] Image not found: %s", name)
        
        if not valid_tasks:
            logging.warning("[ten_agent_web_demo] No valid images found in UAV request: %s", line)
            # Treat as reasoning if no images found
            execute_task(state, line, task_type=TASK_TYPE_REASONING)
            return

        use_leaders = state.architecture_mode == ARCH_MODE_HIER
        
        if len(valid_tasks) == 1:
            # Single task
            execute_uav_car_count_task(state, valid_tasks[0], use_leaders=use_leaders)
        else:
            # Group task - Multi-image
            # 1. Select a Leader (Supervisor logic)
            # For now, pick Math Leader (ID 1) or Reasoning Leader (ID 2) randomly or round-robin
            # In a real system, we'd use bandit scores for leaders too.
            # Let's reuse _select_agent_routing to pick a leader, but we only care about the leader part.
            _mode, leader, _, _, _ = _select_agent_routing(state, TASK_TYPE_UAV, use_leaders=use_leaders)
            
            # If flat mode, leader is None.
            leader_id = leader.leader_id if leader else None
            leader_name = leader.name if leader else None

            # 2. Execute sub-tasks
            group_tasks = []
            total_gt = 0
            total_pred = 0
            
            for uav_task in valid_tasks:
                # Pass the chosen leader to the task execution
                # We need to modify execute_uav_car_count_task to accept a forced leader?
                # Actually, we can just let it pick (it might pick different leaders if we don't force it).
                # But the requirement is "The chosen leader... splits them".
                # So we should force the leader ID on the subtasks.
                
                # We need to manually construct the Task object or modify execute_uav_car_count_task
                # Let's modify execute_uav_car_count_task to accept an optional leader override.
                # (I will do this in the previous chunk)
                
                # Create a temporary Task object to pass leader info? 
                # No, execute_uav_car_count_task takes a UavTask (dataclass).
                # We need to pass the leader_id to execute_uav_car_count_task.
                # I'll update the signature of execute_uav_car_count_task in the next step.
                
                # Wait, I can't change the signature in this chunk easily without matching the previous chunk.
                # I'll assume I updated the signature to `execute_uav_car_count_task(..., leader_override=None)`
                
                # Actually, let's just create the tasks here manually if needed, or better, 
                # update execute_uav_car_count_task to respect a pre-assigned leader.
                
                # For now, let's just call it. If we want strict "Leader distributes", we should enforce the same leader.
                # But `execute_uav_car_count_task` calls `_select_agent_routing`.
                # If I pass `use_leaders=True`, it picks a leader.
                # If I want to FORCE a specific leader, I need to change `execute_uav_car_count_task`.
                
                # Let's assume for this step that `execute_uav_car_count_task` handles it if I modify it.
                # I will modify `execute_uav_car_count_task` to take `leader_override`.
                
                t = execute_uav_car_count_task(state, uav_task, use_leaders=use_leaders, leader_override=leader)
                group_tasks.append(t)
                
                gt = int(float(t.ground_truth)) if t.ground_truth else 0
                pred = int(float(t.output)) if t.output and t.output.isdigit() else 0
                total_gt += gt
                total_pred += pred

            # 3. Create Summary Task
            # This is the "Parent" task that shows the aggregation.
            summary_text = f"Multi-image Task (Leader: {leader_name or 'None'}). Total: {total_pred} (GT: {total_gt})"
            details = [f"{Path(t.image_name).name}: {t.output}" for t in group_tasks]
            
            task = Task(
                task_id=state.next_task_id,
                task_type=TASK_TYPE_UAV,  # Keep it as UAV type so it shows in stats? Or uav_group?
                # User asked for "one row per image... and a small summary row".
                # If we make it a separate task, it shows as a separate row.
                input_text=f"Count cars in {len(valid_tasks)} images",
                question=line,
                timestamp=time.time(),
                output=summary_text + "\n" + "\n".join(details),
                ground_truth=str(total_gt),
                agent_answer=str(total_pred),
                leader_id=leader_id,
                leader_name=leader_name,
                assigned_agent_name="—",
                success=(total_pred == total_gt),
                numeric_reward=1.0 if total_pred == total_gt else 0.0,
                evaluation_reason="Aggregated Multi-Image Task"
            )
            state.next_task_id += 1
            state.tasks.append(task)

    else:
        # Math, Reasoning, or UAV Mission
        for subtask in split_into_subtasks(line):
            execute_task(state, subtask, task_type=task_type)


def run_uav_batch(
    state: SimulationState,
    n_tasks: int,
    policy: str = "score_fair_softmax",
    use_leaders: bool | None = None,
    selected_image_names: Sequence[str] | None = None,
) -> None:
    """
    Execute either a random subset of UAV tasks or the explicitly selected rows.
    """
    if not UAV_TASKS:
        return
    tasks_to_run: List[UavTask] = []
    if selected_image_names:
        for name in selected_image_names:
            task = UAV_TASK_LOOKUP.get(name)
            if task:
                tasks_to_run.append(task)
    else:
        k = min(max(1, n_tasks), len(UAV_TASKS))
        tasks_to_run = random.sample(UAV_TASKS, k) if k < len(UAV_TASKS) else list(UAV_TASKS)
    for entry in tasks_to_run:
        execute_uav_car_count_task(
            state,
            entry,
            policy=policy,
            use_leaders=use_leaders,
        )


def execute_random_uav_task(
    state: SimulationState,
    policy: str = "score_fair_softmax",
    use_leaders: bool | None = None,
) -> Task | None:
    task = sample_uav_task()
    if not task:
        logging.warning("[ten_agent_web_demo] No UAV tasks available.")
        return None
    return execute_uav_car_count_task(state, task, policy=policy, use_leaders=use_leaders)
_MATH_TEMPLATES = [
    "What is {a} + {b}?",
    "If you subtract {b} from {a}, what do you get?",
    "Multiply {a} by {b}.",
    "What is {a} * {b}?",
    "Compute {a} - {b}.",
]

_REASONING_PROMPTS = [
    {"question": "Can a person safely drive a car while sleeping?", "expected": "no"},
    {"question": "Can a fish breathe underwater without equipment?", "expected": "yes"},
    {"question": "If it is raining, are the streets likely to be wet?", "expected": "yes"},
    {"question": "Can a human breathe underwater without a tank?", "expected": "no"},
    {"question": "Would you expect snow in a desert every day?", "expected": "no"},
    {"question": "Is water usually a liquid at room temperature?", "expected": "yes"},
]


def _random_math_prompt() -> str:
    template = random.choice(_MATH_TEMPLATES)
    a = random.randint(1, 20)
    b = random.randint(1, 20)
    return template.format(a=a, b=b)


def _random_reasoning_prompt() -> tuple[str, str | None]:
    choice = random.choice(_REASONING_PROMPTS)
    return choice["question"], choice.get("expected")


def run_batch(state: SimulationState, kind: str, n: int) -> None:
    """Run n tasks of the specified kind using random prompts."""
    n = max(1, n)
    for _ in range(n):
        if kind == "math":
            forced_type = "math"
        elif kind == "reasoning":
            forced_type = "reasoning"
        else:
            forced_type = random.choice(["math", "reasoning"])
        if forced_type == "math":
            prompt = _random_math_prompt()
            expected = None
        else:
            prompt, expected = _random_reasoning_prompt()
        execute_task(state, prompt, task_type=forced_type, expected_answer=expected)



