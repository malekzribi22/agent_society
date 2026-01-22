#!/usr/bin/env python3
"""Interactive supervisor/worker demo for 10 Isaac Sim drones.

This script does **not** modify any of the existing simulation launchers. It
implements the AutoGen-style hierarchy entirely in Python so you can assign
survey/oscillation tasks to the ten drones that already exist in
`ten_drones_ready.py` (or any other world). Each worker represents one drone
and knows how to request the parameters it needs before "executing" the
mission. Hook the execute methods up to your actual MAVSDK control code when
ready.
"""

from __future__ import annotations

import random
import sys
import textwrap
import asyncio
import math
import threading
import re
from concurrent.futures import Future

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Coroutine

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

NUM_DRONES = 10
MAVSDK_PORT_BASE = 14540


# ---------------------------------------------------------------------------
# Shared background asyncio loop so multiple drones can run simultaneously
# ---------------------------------------------------------------------------

_ASYNC_LOOP = asyncio.new_event_loop()


def _loop_worker(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


_ASYNC_THREAD = threading.Thread(target=_loop_worker, args=(_ASYNC_LOOP,), daemon=True)
_ASYNC_THREAD.start()


# ---------------------------------------------------------------------------
# Drone/worker definitions
# ---------------------------------------------------------------------------


@dataclass
class DroneWorker:
    name: str
    mavsdk_url: str
    ros_namespace: str
    skills: Dict[str, float]
    busy: bool = False
    completed_tasks: int = 0
    _current_future: Optional[Future] = field(default=None, repr=False, compare=False)

    def availability_score(self) -> float:
        if self.busy:
            return 0.0
        # Light load gets a boost so the supervisor prefers idle drones
        load_penalty = 0.05 * self.completed_tasks
        return max(0.1, 1.0 - load_penalty)

    # ---- Execution hooks ----
    def execute_survey(self, params: Dict[str, float]) -> None:
        self._start_task(self._survey_async(params), label="survey")

    def execute_oscillation(self, params: Dict[str, float]) -> None:
        self._start_task(self._oscillation_async(params), label="oscillation")

    def _start_task(self, coro: Coroutine, label: str) -> None:
        if self.busy:
            print(f"[{self.name}] Busy; skipping new {label} assignment.")
            return
        self.busy = True

        def _finish(fut: Future):
            exc = fut.exception()
            if exc:
                print(f"[{self.name}] {label} task error: {exc}")
            else:
                self.completed_tasks += 1
            self.busy = False
            self._current_future = None

        future = asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
        future.add_done_callback(_finish)
        self._current_future = future

    # ---- Async implementations ----
    async def _survey_async(self, params: Dict[str, float]) -> None:
        altitude = params.get("altitude", 3.0)
        loops = max(1, int(params.get("loops", 1)))
        radius = max(1.0, params.get("diameter", 5.0) / 2.0)
        velocity = max(0.5, params.get("velocity", 1.0))

        drone = await self._connect()
        print(f"[{self.name}] Starting survey: altitude={altitude}m radius={radius}m loops={loops}")
        try:
            await self._arm_and_takeoff(drone, altitude)
            await self._start_offboard(drone, altitude)
            await self._fly_circle(drone, radius, altitude, loops, velocity)
            await self._stop_offboard(drone)
            await self._land_and_disarm(drone)
        except Exception as exc:
            print(f"[{self.name}] Survey error: {exc}")
            await self._safe_recover(drone)
        else:
            print(f"[{self.name}] Survey finished.")

    async def _oscillation_async(self, params: Dict[str, float]) -> None:
        axis = params.get("axis", "vertical")
        low = params.get("low", 1.0)
        high = params.get("high", 3.0)
        repeats = max(1, int(params.get("repeats", 2)))
        velocity = max(0.2, params.get("velocity", 1.0))

        drone = await self._connect()
        print(f"[{self.name}] Starting {axis} oscillation low={low} high={high} repeats={repeats}")
        try:
            base_alt = max(2.0, high if axis == "vertical" else 3.0)
            await self._arm_and_takeoff(drone, base_alt)
            await self._start_offboard(drone, base_alt)
            if axis == "vertical":
                await self._oscillate_vertical(drone, low, high, repeats, velocity)
            else:
                await self._oscillate_horizontal(drone, low, high, base_alt, repeats, velocity)
            await self._stop_offboard(drone)
            await self._land_and_disarm(drone)
        except Exception as exc:
            print(f"[{self.name}] Oscillation error: {exc}")
            await self._safe_recover(drone)
        else:
            print(f"[{self.name}] Oscillation finished.")

    # ---- MAVSDK helpers ----
    async def _connect(self) -> System:
        print(f"[{self.name}] Connecting to {self.mavsdk_url}")
        drone = System()
        await drone.connect(system_address=self.mavsdk_url)
        async for state in drone.core.connection_state():
            if state.is_connected:
                print(f"[{self.name}] MAVSDK connected.")
                break
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print(f"[{self.name}] PX4 health ready.")
                break
        return drone

    async def _arm_and_takeoff(self, drone: System, altitude: float) -> None:
        print(f"[{self.name}] Arming + takeoff to {altitude} m")
        await drone.action.arm()
        await drone.action.set_takeoff_altitude(altitude)
        await drone.action.takeoff()
        await asyncio.sleep(5)

    async def _start_offboard(self, drone: System, altitude: float) -> None:
        initial = PositionNedYaw(0.0, 0.0, -altitude, 0.0)
        print(f"[{self.name}] Priming offboard at altitude {altitude} m")
        for _ in range(20):
            await drone.offboard.set_position_ned(initial)
            await asyncio.sleep(0.05)
        try:
            await drone.offboard.start()
            print(f"[{self.name}] Offboard started.")
        except OffboardError:
            print(f"[{self.name}] Offboard start failed, retrying...")
            for _ in range(20):
                await drone.offboard.set_position_ned(initial)
                await asyncio.sleep(0.05)
            await drone.offboard.start()
            print(f"[{self.name}] Offboard started on retry.")

    async def _stop_offboard(self, drone: System) -> None:
        try:
            await drone.offboard.stop()
            print(f"[{self.name}] Offboard stopped.")
        except OffboardError:
            print(f"[{self.name}] Offboard stop raised, ignoring.")

    async def _land_and_disarm(self, drone: System) -> None:
        print(f"[{self.name}] Landing...")
        await drone.action.land()
        await asyncio.sleep(5)

    async def _safe_recover(self, drone: System) -> None:
        print(f"[{self.name}] Attempting safe recovery...")
        try:
            await self._stop_offboard(drone)
        except Exception:
            pass
        try:
            await drone.action.land()
        except Exception:
            pass
        await asyncio.sleep(3)

    async def _hold_position(self, drone: System, pos: PositionNedYaw, duration: float) -> None:
        end = asyncio.get_event_loop().time() + max(duration, 0.05)
        while asyncio.get_event_loop().time() < end:
            await drone.offboard.set_position_ned(pos)
            await asyncio.sleep(0.05)

    async def _fly_circle(self, drone: System, radius: float, altitude: float, loops: int, velocity: float) -> None:
        circumference = 2 * math.pi * radius
        period = max(5.0, circumference / max(velocity, 0.5))
        segments = max(60, int(period / 0.2))
        step_time = max(0.1, period / segments)
        total_steps = loops * segments
        for step in range(total_steps):
            angle = 2 * math.pi * (step / segments)
            north = radius * math.cos(angle)
            east = radius * math.sin(angle)
            pos = PositionNedYaw(north, east, -altitude, math.degrees(angle))
            await self._hold_position(drone, pos, step_time)

    async def _oscillate_vertical(self, drone: System, low: float, high: float, repeats: int, velocity: float) -> None:
        low_alt = min(low, high)
        high_alt = max(low, high)
        travel = max(0.5, high_alt - low_alt)
        step_time = max(0.5, travel / max(velocity, 0.2))
        for _ in range(repeats):
            await self._hold_position(drone, PositionNedYaw(0.0, 0.0, -high_alt, 0.0), step_time)
            await self._hold_position(drone, PositionNedYaw(0.0, 0.0, -low_alt, 0.0), step_time)

    async def _oscillate_horizontal(self, drone: System, low: float, high: float, altitude: float, repeats: int, velocity: float) -> None:
        travel = abs(high - low)
        step_time = max(0.5, travel / max(velocity, 0.2))
        for _ in range(repeats):
            await self._hold_position(drone, PositionNedYaw(high, 0.0, -altitude, 0.0), step_time)
            await self._hold_position(drone, PositionNedYaw(low, 0.0, -altitude, 0.0), step_time)


def build_workers() -> List[DroneWorker]:
    workers: List[DroneWorker] = []
    for idx in range(NUM_DRONES):
        url = f"udpin://0.0.0.0:{MAVSDK_PORT_BASE + idx}"
        ns = f"drone{idx:02d}"
        # Give each drone slightly different strengths so the supervisor can
        # choose who is "best" for a task.
        skills = {
            "survey": round(random.uniform(0.55, 0.95), 2),
            "oscillation_vertical": round(random.uniform(0.45, 0.9), 2),
            "oscillation_horizontal": round(random.uniform(0.45, 0.9), 2),
        }
        workers.append(DroneWorker(f"Drone{idx}", url, ns, skills))
    return workers


# ---------------------------------------------------------------------------
# Supervisor logic
# ---------------------------------------------------------------------------


class DroneSupervisor:
    def __init__(self, workers: List[DroneWorker]):
        self.workers = workers
        self._default_survey = {
            "diameter": 10.0,
            "velocity": 1.5,
            "altitude": 3.0,
            "loops": 2,
        }
        self._default_oscillation = {
            "axis": "vertical",
            "low": 1.0,
            "high": 3.0,
            "repeats": 3,
            "velocity": 1.0,
        }

    # --- CLI entry point ---
    def interactive(self) -> None:
        self._print_banner()
        while True:
            try:
                raw = input("\nCommand (survey|oscillation|status|exit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting supervisor.")
                break

            if not raw:
                continue
            if raw.lower() in {"exit", "quit"}:
                print("Bye!")
                break
            if raw.lower() == "status":
                self._print_status()
                continue

            multi_parts = None
            if "&&" in raw:
                multi_parts = [p.strip() for p in raw.split("&&") if p.strip()]
            elif re.search(r"\band\b", raw, flags=re.IGNORECASE):
                multi_parts = [p.strip() for p in re.split(r"\band\b", raw, flags=re.IGNORECASE) if p.strip()]

            if multi_parts:
                self._handle_quick_sequence(multi_parts)
                continue

            cmd = raw.lower()
            if cmd == "survey":
                params = self._prompt_survey_params()
                self._assign_task("survey", params)
            elif cmd == "oscillation":
                params = self._prompt_oscillation_params()
                task_key = (
                    "oscillation_vertical"
                    if params.get("axis") == "vertical"
                    else "oscillation_horizontal"
                )
                self._assign_task(task_key, params)
            elif self._handle_quick_command(raw):
                continue
            else:
                print("Unknown command. Use survey, oscillation, status, exit.")

    # --- User prompts ---
    def _prompt_survey_params(self) -> Dict[str, float]:
        print("Survey parameters:")
        diameter = self._ask_float("  Diameter (meters)", default=self._default_survey["diameter"])
        velocity = self._ask_float("  Velocity (m/s)", default=self._default_survey["velocity"])
        altitude = self._ask_float("  Altitude (meters)", default=self._default_survey["altitude"])
        loops = int(self._ask_float("  Number of loops", default=self._default_survey["loops"]))
        return {
            "diameter": diameter,
            "velocity": velocity,
            "altitude": altitude,
            "loops": loops,
        }

    def _prompt_oscillation_params(self) -> Dict[str, float]:
        print("Oscillation parameters:")
        axis = self._ask_choice("  Axis", ["vertical", "horizontal"], default=self._default_oscillation["axis"])
        low = self._ask_float("  Low bound (meters)", default=self._default_oscillation["low"])
        high = self._ask_float("  High bound (meters)", default=self._default_oscillation["high"])
        repeats = int(self._ask_float("  Number of repetitions", default=self._default_oscillation["repeats"]))
        velocity = self._ask_float("  Velocity (m/s)", default=self._default_oscillation["velocity"])
        if axis == "vertical":
            low_val, high_val = (min(low, high), max(low, high))
        else:
            low_val, high_val = low, high
        return {
            "axis": axis,
            "low": low_val,
            "high": high_val,
            "repeats": repeats,
            "velocity": velocity,
        }

    def _ask_float(self, label: str, default: float) -> float:
        while True:
            raw = input(f"{label} [{default}]: ").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                print("  Please enter a numeric value.")

    def _ask_choice(self, label: str, options: List[str], default: str) -> str:
        opts = "/".join(options)
        while True:
            raw = input(f"{label} ({opts}) [{default}]: ").strip().lower()
            if not raw:
                return default
            if raw in options:
                return raw
            print(f"  Invalid choice. Pick one of {options}.")

    # --- Assignment logic ---
    def _assign_task(self, task_key: str, params: Dict[str, float]) -> None:
        worker = self._select_best_worker(task_key)
        if not worker:
            print("No available drones right now; try again soon.")
            return

        print(f"Assigning {task_key} to {worker.name}...")
        if task_key == "survey":
            worker.execute_survey(params)
        else:
            worker.execute_oscillation(params)

    def _handle_quick_sequence(self, commands: List[str]) -> None:
        for cmd in commands:
            if not self._handle_quick_command(cmd):
                print(f"[supervisor] Could not parse '{cmd}'. Use survey/oscillation keywords.")

    def _handle_quick_command(self, text: str) -> bool:
        if not text:
            return False
        raw = text.strip()
        lower = raw.lower()
        if lower.startswith("survey"):
            params = dict(self._default_survey)
            params.update(self._parse_overrides(raw, {"diameter", "velocity", "altitude", "loops"}))
            params["loops"] = int(params["loops"])
            print(f"[supervisor] Quick survey task queued (alt={params['altitude']}m, diameter={params['diameter']}m).")
            self._assign_task("survey", params)
            return True
        if lower.startswith("oscillation") or lower.startswith("osc"):
            params = dict(self._default_oscillation)
            overrides = self._parse_overrides(raw, {"axis", "low", "high", "repeats", "velocity"})
            params.update({k: overrides[k] for k in overrides if k in params})
            if isinstance(params["repeats"], float):
                params["repeats"] = int(max(1, round(params["repeats"])))
            axis = str(params.get("axis", "vertical")).lower()
            if "horizontal" in lower:
                axis = "horizontal"
            elif "vertical" in lower:
                axis = "vertical"
            params["axis"] = axis if axis in {"vertical", "horizontal"} else "vertical"
            task_key = "oscillation_vertical" if params["axis"] == "vertical" else "oscillation_horizontal"
            print(f"[supervisor] Quick oscillation ({params['axis']}) task queued.")
            self._assign_task(task_key, params)
            return True
        return False

    def _parse_overrides(self, text: str, allowed: set) -> Dict[str, float]:
        overrides: Dict[str, float] = {}
        for token in text.split():
            if "=" not in token:
                continue
            key, val = token.split("=", 1)
            key = key.lower()
            if key not in allowed:
                continue
            overrides[key] = self._coerce_number(val)
        return overrides

    def _coerce_number(self, value: str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def _select_best_worker(self, task_key: str) -> Optional[DroneWorker]:
        scored: List[tuple[float, DroneWorker]] = []
        for worker in self.workers:
            base_skill = worker.skills.get(task_key, 0.4)
            availability = worker.availability_score()
            if availability <= 0.0:
                continue
            score = 0.7 * base_skill + 0.3 * availability + random.uniform(0.0, 0.05)
            scored.append((score, worker))
        if not scored:
            return None
        scored.sort(key=lambda s: s[0], reverse=True)
        return scored[0][1]

    # --- Status helpers ---
    def _print_status(self) -> None:
        print("\nCurrent drone status:")
        for w in self.workers:
            print(
                f"  {w.name:<8} | busy={w.busy!s:<5} | completed={w.completed_tasks:<3} |"
                f" survey={w.skills['survey']:.2f}"
                f" vert={w.skills['oscillation_vertical']:.2f}"
                f" horiz={w.skills['oscillation_horizontal']:.2f}"
            )

    def _print_banner(self) -> None:
        print(textwrap.dedent(
            """
            ==============================================================
            Drone Supervisor (AutoGen-style demo)
            --------------------------------------------------------------
            - 10 worker drones, each with survey & oscillation skills.
            - Commands:
                survey       -> prompts for diameter/velocity/altitude/loops
                oscillation   -> prompts for axis, bounds, repeats
                status        -> show worker skill/availability
                exit          -> quit
            - Replace the execute_* methods with real MAVSDK commands to
              integrate with your running Isaac Sim world.
            ==============================================================
            """
        ))


def main() -> None:
    workers = build_workers()
    supervisor = DroneSupervisor(workers)
    supervisor.interactive()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
