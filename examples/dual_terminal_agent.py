#!/usr/bin/env python3
"""
Simple MAVSDK agent that controls exactly one PX4 drone based on its index.

Usage example (two separate terminals):
    # Terminal A -> drone0 (MAVSDK port 14540)
    python3 dual_terminal_agent.py --drone 0

    # Terminal B -> drone1 (MAVSDK port 14541)
    python3 dual_terminal_agent.py --drone 1
"""

import argparse
import asyncio
from typing import Optional

from mavsdk import System


MAVSDK_BASE_PORT = 14540
PORT_STEP = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control a single Pegasus/PX4 drone via MAVSDK."
    )
    parser.add_argument(
        "--drone",
        type=int,
        required=True,
        help="Drone index (0 for the first PX4 instance, 1 for the second, etc.).",
    )
    parser.add_argument(
        "--takeoff-alt",
        type=float,
        default=3.0,
        help="Takeoff altitude in meters (default: 3.0).",
    )
    parser.add_argument(
        "--hover-sec",
        type=float,
        default=8.0,
        help="Hover duration before landing (seconds).",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=25.0,
        help="Seconds to wait for MAVSDK connection.",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=25.0,
        help="Seconds to wait for PX4 health before continuing.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="Extra seconds to wait after connection/health before arming (sync multiple agents).",
    )
    return parser


async def wait_for_connection(
    drone: System, label: str, timeout: float
) -> Optional[str]:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    async for state in drone.core.connection_state():
        if state.is_connected:
            try:
                uuid = await drone.core.get_uuid()
            except Exception:
                uuid = "unknown"
            print(f"[{label}] MAVSDK connected (UUID={uuid})")
            return uuid
        if loop.time() > deadline:
            print(f"[{label}] Timed out waiting for MAVSDK connection")
            return None
    return None


async def wait_for_health(drone: System, label: str, timeout: float) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print(f"[{label}] PX4 health OK — ready to fly")
            return True
        if loop.time() > deadline:
            print(f"[{label}] PX4 health timeout, proceeding anyway")
            return False
    return False


async def run_agent(args: argparse.Namespace) -> None:
    idx = args.drone
    if idx < 0:
        raise SystemExit("Drone index must be non-negative.")

    url = f"udpin://0.0.0.0:{MAVSDK_BASE_PORT + idx * PORT_STEP}"
    label = f"agent:drone{idx}"
    print(f"[{label}] Connecting to {url}")

    drone = System()
    await drone.connect(system_address=url)

    if not await wait_for_connection(drone, label, args.connect_timeout):
        return
    await wait_for_health(drone, label, args.health_timeout)

    try:
        await drone.action.set_takeoff_altitude(args.takeoff_alt)
    except Exception:
        pass

    if args.start_delay > 0.0:
        print(
            f"[{label}] Waiting {args.start_delay:.1f} s before arming "
            "(sync start with other agents)"
        )
        await asyncio.sleep(args.start_delay)

    if not await send_action(drone.action.arm, label, "arm()"):
        return
    if not await send_action(drone.action.takeoff, label, "takeoff()"):
        return

    hover_time = max(args.hover_sec, 0.0)
    print(
        f"[{label}] Hovering at ~{args.takeoff_alt:.1f} m for {hover_time:.1f} s "
        "(this drone only)"
    )
    await asyncio.sleep(hover_time)
    await send_action(drone.action.land, label, "land()")
    await asyncio.sleep(3.0)
    print(f"[{label}] Mission complete")


async def send_action(coro, label: str, name: str) -> bool:
    try:
        await coro()
        print(f"[{label}] {name} sent")
        return True
    except Exception as exc:
        print(f"[{label}] {name} failed: {exc}")
        return False


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
