#!/usr/bin/env python3
"""
Supervisor agent that sequentially takes off multiple PX4 drones while keeping them flying.

It reuses the same MAVSDK endpoint mapping as two_drones_dual_agents.py (ports 14540+idx) and
prints every socket action so you can verify which PX4 instance received which command.

Example:
    python3 supervisor_dual_agents.py --drones 0 1 --takeoff-alt 3.5 \
        --hover-sec 12 --stagger-sec 5 --auto-land --log-file sockets.log
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

from mavsdk import System


DEFAULT_BASE_PORT = 14540
DEFAULT_PORT_STEP = 1


def port_for(idx: int, base: int, step: int) -> int:
    return base + idx * step


class SocketJournal:
    """Collects every MAVSDK action with the associated UDP endpoint."""

    def __init__(self) -> None:
        self.entries: List[str] = []

    def log(self, idx: int, url: str, action: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] drone{idx}@{url}: {action}"
        print(line)
        self.entries.append(line)


@dataclass
class DroneEndpoint:
    idx: int
    url: str
    journal: SocketJournal
    drone: Optional[System] = field(default=None)

    async def connect(self, timeout: float) -> None:
        self.journal.log(self.idx, self.url, "connecting via MAVSDK")
        self.drone = System()
        await self.drone.connect(system_address=self.url)

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                try:
                    uuid = await self.drone.core.get_uuid()
                except Exception:
                    uuid = "unknown"
                self.journal.log(
                    self.idx,
                    self.url,
                    f"connected (UUID={uuid})",
                )
                return
            if loop.time() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for MAVSDK connection on {self.url}"
                )

    async def wait_until_ready(self, timeout: float) -> None:
        assert self.drone is not None
        self.journal.log(self.idx, self.url, "waiting for PX4 health")
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                self.journal.log(self.idx, self.url, "PX4 ready to fly")
                return
            if loop.time() > deadline:
                self.journal.log(
                    self.idx,
                    self.url,
                    "PX4 health timeout (continuing anyway)",
                )
                return

    async def arm(self) -> None:
        assert self.drone is not None
        self.journal.log(self.idx, self.url, "arm()")
        await self.drone.action.arm()

    async def takeoff(self, altitude: float) -> None:
        assert self.drone is not None
        try:
            await self.drone.action.set_takeoff_altitude(altitude)
        except Exception:
            pass
        self.journal.log(self.idx, self.url, f"takeoff to ~{altitude:.1f} m")
        await self.drone.action.takeoff()

    async def hover_and_optionally_land(
        self,
        hover_time: float,
        auto_land: bool,
    ) -> None:
        assert self.drone is not None
        self.journal.log(
            self.idx,
            self.url,
            f"hovering for {hover_time:.1f} s",
        )
        await asyncio.sleep(max(hover_time, 0.0))
        if auto_land:
            self.journal.log(self.idx, self.url, "land()")
            await self.drone.action.land()


async def run_sequence(args: argparse.Namespace) -> None:
    journal = SocketJournal()
    indices = sorted(set(args.drones))
    for idx in indices:
        if idx < 0:
            raise ValueError("Drone indices must be non-negative")

    endpoints = [
        DroneEndpoint(
            idx=i,
            url=f"udpin://0.0.0.0:{port_for(i, args.base_port, args.port_step)}",
            journal=journal,
        )
        for i in indices
    ]

    # Connect + wait for PX4 health on every drone first so later commands are crisp.
    for ep in endpoints:
        await ep.connect(args.connect_timeout)
        await ep.wait_until_ready(args.health_timeout)

    hover_tasks = []
    for ep in endpoints:
        await ep.arm()
        await ep.takeoff(args.takeoff_alt)
        hover_tasks.append(
            asyncio.create_task(
                ep.hover_and_optionally_land(args.hover_sec, args.auto_land)
            )
        )
        await asyncio.sleep(max(args.stagger_sec, 0.0))

    await asyncio.gather(*hover_tasks)

    if args.log_file:
        with open(args.log_file, "w", encoding="ascii") as log_file:
            log_file.write("\n".join(journal.entries) + "\n")
        print(f"[supervisor] Socket log saved to {args.log_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervisor that sequentially commands multiple PX4 drones via MAVSDK."
    )
    parser.add_argument(
        "--drones",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Drone indices to manage (default: 0 1).",
    )
    parser.add_argument(
        "--takeoff-alt",
        type=float,
        default=3.0,
        help="Target takeoff altitude in meters.",
    )
    parser.add_argument(
        "--hover-sec",
        type=float,
        default=12.0,
        help="How long each drone should stay airborne before landing (seconds).",
    )
    parser.add_argument(
        "--stagger-sec",
        type=float,
        default=5.0,
        help="Delay between starting commands for consecutive drones.",
    )
    parser.add_argument(
        "--auto-land",
        action="store_true",
        help="If set, send a land command after hover time elapses.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for each MAVSDK connection.",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for PX4 to report a healthy state.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=DEFAULT_BASE_PORT,
        help="Base MAVSDK UDP port (default 14540). Matches two_drones_dual_agents.py.",
    )
    parser.add_argument(
        "--port-step",
        type=int,
        default=DEFAULT_PORT_STEP,
        help="Port increment between drones (default 1).",
    )
    parser.add_argument(
        "--log-file",
        help="Optional path to store the socket action log.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_sequence(args))


if __name__ == "__main__":
    main()
