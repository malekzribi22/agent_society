#!/usr/bin/env python3
"""
Launch two MAVSDK agents (drone0 + drone1) inside the same process so they arm,
take off, hover, and land at the exact same time.

Usage:
    PEGASUS_AUTO_AGENTS=0 isaacpy two_drones_dual_agents_ready.py   # in one terminal
    python3 dual_parallel_agents.py --takeoff-alt 3.0 --hover-sec 12
"""

import argparse
import asyncio
from dataclasses import dataclass
from typing import Dict, Tuple

from mavsdk import System


BASE_PORT = 14540
PORT_STEP = 1
DRONES = (0, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control two Pegasus PX4 drones simultaneously via MAVSDK."
    )
    parser.add_argument(
        "--takeoff-alt",
        type=float,
        default=3.0,
        help="Target takeoff altitude in meters (default: 3.0).",
    )
    parser.add_argument(
        "--hover-sec",
        type=float,
        default=10.0,
        help="Hover duration before landing (seconds).",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=25.0,
        help="Seconds to wait for each MAVSDK connection.",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=25.0,
        help="Seconds to wait for PX4 health before continuing.",
    )
    return parser


@dataclass
class DroneHandle:
    idx: int
    url: str
    system: System


async def connect_and_ready(
    idx: int, connect_timeout: float, health_timeout: float
) -> DroneHandle:
    url = f"udpin://0.0.0.0:{BASE_PORT + idx * PORT_STEP}"
    label = f"drone{idx}"
    print(f"[{label}] Connecting to {url}")
    system = System()
    await system.connect(system_address=url)

    loop = asyncio.get_event_loop()
    deadline = loop.time() + connect_timeout
    async for state in system.core.connection_state():
        if state.is_connected:
            try:
                uuid = await system.core.get_uuid()
            except Exception:
                uuid = "unknown"
            print(f"[{label}] MAVSDK connected (UUID={uuid})")
            break
        if loop.time() > deadline:
            raise TimeoutError(f"[{label}] Timed out waiting for MAVSDK connection")

    deadline = loop.time() + health_timeout
    async for health in system.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print(f"[{label}] PX4 health OK — ready to fly")
            break
        if loop.time() > deadline:
            print(f"[{label}] PX4 health timeout, continuing anyway")
            break

    return DroneHandle(idx=idx, url=url, system=system)


async def synchronized_sequence(handles: Dict[int, DroneHandle], args: argparse.Namespace):
    # Apply takeoff altitude first so PX4 uses the same target.
    await asyncio.gather(
        *[
            set_takeoff_alt(handle.system, handle.idx, args.takeoff_alt)
            for handle in handles.values()
        ]
    )

    # Arm both drones at once.
    await asyncio.gather(
        *[run_action(handle.system.action.arm, handle.idx, "arm()") for handle in handles.values()]
    )

    # Take off together.
    await asyncio.gather(
        *[
            run_action(handle.system.action.takeoff, handle.idx, "takeoff()")
            for handle in handles.values()
        ]
    )

    hover_time = max(args.hover_sec, 0.0)
    print(f"[both] Hovering at ~{args.takeoff_alt:.1f} m for {hover_time:.1f} s")
    await asyncio.sleep(hover_time)

    # Land together.
    await asyncio.gather(
        *[
            run_action(handle.system.action.land, handle.idx, "land()")
            for handle in handles.values()
        ]
    )
    await asyncio.sleep(3.0)
    print("[both] Mission complete")


async def set_takeoff_alt(system: System, idx: int, altitude: float):
    try:
        await system.action.set_takeoff_altitude(altitude)
    except Exception:
        pass


async def run_action(coro_fn, idx: int, action: str):
    label = f"drone{idx}"
    try:
        await coro_fn()
        print(f"[{label}] {action} sent")
        return True
    except Exception as exc:
        print(f"[{label}] {action} failed: {exc}")
        return False


async def main_async(args: argparse.Namespace):
    handles_list = await asyncio.gather(
        *[
            connect_and_ready(idx, args.connect_timeout, args.health_timeout)
            for idx in DRONES
        ]
    )
    handles = {handle.idx: handle for handle in handles_list}
    await synchronized_sequence(handles, args)


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
