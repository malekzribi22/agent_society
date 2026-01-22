#!/usr/bin/env python3
"""
High-level multi-agent “drone society” coordinator.

This script layers a Supervisor/Lead/Worker hierarchy on top of the existing
PX4 + MAVSDK workers provided by autogen_drones.py. You can speak to the
Supervisor using natural prompts (e.g. “I need two drones to fly a survey at
5 meters” or “Put two drones side-by-side at 3 m”) and it will:

  * interpret the request (optionally with OpenAI if OPENAI_API_KEY is set)
  * consult Lead agents that maintain skills / recent state for their drones
  * delegate concrete missions to the individual DroneWorker objects imported
    from autogen_drones

PX4 connectivity, MAVSDK execution, and worker behaviors remain unchanged;
this script only reasons about task allocation and memory.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - OpenAI SDK optional
    OpenAI = None

import autogen_drones as legacy
from autogen_drones import DroneWorker, build_workers


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


@dataclass
class DroneMemory:
    worker: DroneWorker
    lead_name: str
    last_task: Optional[str] = None
    status: str = "idle"
    notes: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.worker.name

    def is_available(self) -> bool:
        return not self.worker.busy and self.status in {"idle", "ready"}


# ---------------------------------------------------------------------------
# Lead agents
# ---------------------------------------------------------------------------


class LeadAgent:
    def __init__(self, name: str, drones: List[DroneMemory]):
        self.name = name
        self.drones = drones

    def pretty_status(self) -> str:
        busy = sum(1 for d in self.drones if not d.is_available())
        return f"{self.name}: {len(self.drones)} drones ({busy} busy)"

    # Score drones based on skills + recency and pick best candidate
    def propose_candidate(self, task_key: str, params: Dict) -> Optional[DroneMemory]:
        best_score = -1.0
        best_drone = None
        for memory in self.drones:
            if not memory.is_available():
                continue
            skill = memory.worker.skills.get(task_key, 0.4)
            freshness = 0.9 if memory.last_task == task_key else 1.0
            randomness = random.uniform(0.0, 0.05)
            score = 0.7 * skill * freshness + randomness
            if score > best_score:
                best_score = score
                best_drone = memory
        return best_drone

    def dispatch(self, memory: DroneMemory, task_key: str, params: Dict) -> None:
        memory.status = f"running {task_key}"
        memory.last_task = task_key
        memory.notes.append(f"Assigned {task_key} with {params}")
        if task_key == "survey":
            memory.worker.execute_survey(params)
        elif task_key in {"oscillation_vertical", "oscillation_horizontal"}:
            params = dict(params)
            params["axis"] = "vertical" if task_key.endswith("vertical") else "horizontal"
            memory.worker.execute_oscillation(params)
        else:
            print(f"[{self.name}] Task '{task_key}' not supported; instructing survey fallback.")
            memory.worker.execute_survey(params)


# ---------------------------------------------------------------------------
# Supervisor agent
# ---------------------------------------------------------------------------


class SupervisorAgent:
    def __init__(self, leads: List[LeadAgent], client: Optional[OpenAI] = None):
        self.leads = leads
        self.client = client
        self._default_survey = {"diameter": 10.0, "velocity": 1.5, "altitude": 3.0, "loops": 2}
        self._default_osc = {"axis": "vertical", "low": 1.0, "high": 3.0, "repeats": 3, "velocity": 1.0}

    # ----------------- Public API -----------------
    def repl(self) -> None:
        self._print_banner()
        while True:
            try:
                text = input("\nAsk the supervisor (or 'status'/'exit'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                print("Bye!")
                break
            if text.lower() == "status":
                self._print_status()
                continue
            tasks = self._interpret(text)
            if not tasks:
                print("Supervisor: I couldn't parse that request.")
                continue
            self._execute_tasks(tasks)

    # ----------------- Internals -----------------
    def _print_banner(self) -> None:
        print(textwrap.dedent(
            """
            ===============================================================
            Drone Society Supervisor
            ---------------------------------------------------------------
            - Natural commands: "Send two drones to survey 5m altitude"
                                 "oscillate three drones vertically"
                                 "two drones side by side at 3m"
            - Type 'status' to inspect leads/drones, 'exit' to quit.
            ===============================================================
            """
        ))

    def _print_status(self) -> None:
        for lead in self.leads:
            print(lead.pretty_status())
            for drone in lead.drones:
                print(
                    f"   {drone.name:<8} status={drone.status:<15} "
                    f"skill(survey)={drone.worker.skills.get('survey', 0):.2f}"
                )

    def _interpret(self, text: str) -> List[Dict]:
        if self.client and os.getenv("OPENAI_API_KEY"):
            try:
                return self._interpret_with_llm(text)
            except Exception as exc:
                print(f"[LLM parser failed: {exc}] Falling back to heuristic parser.")
        return self._interpret_heuristic(text)

    def _interpret_with_llm(self, text: str) -> List[Dict]:
        prompt = textwrap.dedent(f"""
        You coordinate PX4/MAVSDK drones. Extract tasks from the user request.
        Return JSON with a list of objects: {{"task":"survey","count":2,"params":{{...}}}}
        Tasks supported: survey, oscillation_vertical, oscillation_horizontal.
        Default params: survey diam=10m alt=3m vel=1.5, oscillation low=1 high=3 vel=1.
        User request: "{text}"
        JSON:
        """)
        resp = self.client.responses.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            input=prompt,
            temperature=0.2,
        )
        raw = resp.output[0].content[0].text  # type: ignore[attr-defined]
        data = json.loads(raw)
        return data if isinstance(data, list) else []

    def _interpret_heuristic(self, text: str) -> List[Dict]:
        text_low = text.lower()
        count = self._extract_count(text_low)
        tasks: List[Dict] = []
        if "survey" in text_low:
            params = dict(self._default_survey)
            params["altitude"] = self._extract_number(text_low, "meters", default=params["altitude"])
            tasks.append({"task": "survey", "count": count, "params": params})
        if "oscillation" in text_low or "oscillate" in text_low:
            axis = "horizontal" if "horizontal" in text_low else "vertical"
            params = dict(self._default_osc)
            params["axis"] = axis
            task_key = "oscillation_horizontal" if axis == "horizontal" else "oscillation_vertical"
            tasks.append({"task": task_key, "count": count, "params": params})
        if not tasks:
            # default to survey to avoid silent drops
            tasks.append({"task": "survey", "count": count, "params": dict(self._default_survey)})
        return tasks

    def _extract_count(self, text: str) -> int:
        match = re.search(r"(\d+)\s+drones?", text)
        if match:
            return max(1, min(5, int(match.group(1))))
        if "pair" in text or "two" in text:
            return 2
        return 1

    def _extract_number(self, text: str, keyword: str, default: float) -> float:
        pattern = rf"(\d+(?:\.\d+)?)\s*{keyword}"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else default

    def _execute_tasks(self, tasks: List[Dict]) -> None:
        for spec in tasks:
            task_key = spec["task"]
            params = spec["params"]
            remaining = spec.get("count", 1)
            print(f"\nSupervisor: assigning {remaining} x {task_key} ({params})")
            while remaining > 0:
                lead = self._pick_lead(task_key)
                if not lead:
                    print("Supervisor: no available lead/drone for this task right now.")
                    break
                candidate = lead.propose_candidate(task_key, params)
                if candidate is None:
                    print(f"{lead.name}: all drones busy; try later.")
                    break
                lead.dispatch(candidate, task_key, params)
                remaining -= 1

    def _pick_lead(self, task_key: str) -> Optional[LeadAgent]:
        available = [lead for lead in self.leads if any(d.is_available() for d in lead.drones)]
        return random.choice(available) if available else None


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------


def build_leads(workers: List[DroneWorker], num_leads: int = 2) -> List[LeadAgent]:
    memories = [DroneMemory(worker=w, lead_name=f"Lead{i}") for i, w in enumerate(workers)]
    leads: List[LeadAgent] = []
    chunk_size = (len(workers) + num_leads - 1) // num_leads
    for i in range(num_leads):
        chunk = memories[i * chunk_size:(i + 1) * chunk_size]
        if chunk:
            leads.append(LeadAgent(name=f"Lead{i+1}", drones=chunk))
    return leads


def init_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def main():
    workers = build_workers()
    leads = build_leads(workers, num_leads=3)
    supervisor = SupervisorAgent(leads, client=init_openai_client())
    supervisor.repl()


if __name__ == "__main__":
    if not os.getenv("MAVSDK_SERVER"):
        # Reminder for the operator
        print("Ensure ten_drones_ready.py is running and PX4 reports Ready for takeoff.")
    main()
