#!/usr/bin/env python3
"""
Interactive MAVSDK console: type commands like "drone 1 takeoff" and only that
drone (PX4 instance) will take off.

Workflow:
    1. Launch the Isaac Sim scene without auto-agents:
         PEGASUS_AUTO_AGENTS=0 isaacpy two_drones_dual_agents_ready.py
    2. In another terminal, run:
         python3 dual_command_agent.py --takeoff-alt 3.0
    3. Type commands:
         > drone 1 takeoff
         > drone 2 land
         > exit
"""

import argparse
import asyncio
import re
from dataclasses import dataclass
from typing import Dict, Optional

from mavsdk import System


BASE_PORT = 14540
PORT_STEP = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive controller for multiple Pegasus PX4 drones."
    )
    parser.add_argument(
        "--takeoff-alt",
        type=float,
        default=3.0,
        help="Takeoff altitude in meters when issuing 'takeoff' (default: 3.0).",
    )
    parser.add_argument(
        "--hover-sec",
        type=float,
        default=8.0,
        help="Hover duration after takeoff before auto-landing (default: 8).",
    )
    parser.add_argument(
        "--drones",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Drone indices to connect (default: 0 1).",
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
    parser.add_argument(
        "--command-base",
        type=int,
        default=1,
        help=(
            "Interpret 'drone N ...' commands as referring to the N-th connected drone "
            "starting at this base (set to 0 if you prefer zero-based numbering)."
        ),
    )
    return parser


@dataclass
class DroneHandle:
    idx: int
    label: str
    url: str
    system: System


async def connect_drone(
    idx: int, *, connect_timeout: float, health_timeout: float
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

    return DroneHandle(idx=idx, label=label, url=url, system=system)


async def interactive_loop(
    by_idx: Dict[int, DroneHandle],
    ordinal_map: Dict[int, DroneHandle],
    args: argparse.Namespace,
):
    print("\nCommand mappings:")
    for num, handle in ordinal_map.items():
        print(
            f"  'drone {num}' -> MAVSDK {handle.url} (vehicle_id {handle.idx})"
        )
    print(
        "You can also address the PX4 vehicle_id directly (e.g., 'drone 0').\n"
        "Type commands like 'drone 1 takeoff', 'drone 2 land', 'quit'."
    )
    print(
        "Takeoff automatically arms, hovers, and lands after "
        f"{args.hover_sec:.1f} seconds.\n"
    )

    loop = asyncio.get_event_loop()
    while True:
        cmd = await loop.run_in_executor(None, input, "> ")
        if cmd is None:
            continue
        cmd = cmd.strip()
        if not cmd:
            continue

        lower = cmd.lower()
        if lower in {"exit", "quit"}:
            print("Exiting interactive controller.")
            return
        if lower == "help":
            print("Examples: 'drone 1 takeoff', 'drone 0 land', 'quit'")
            continue

        target_idx, action = parse_command(lower)
        if target_idx is None or action is None:
            print("Unrecognized command. Try 'drone 1 takeoff'.")
            continue
        handle = ordinal_map.get(target_idx)
        if handle is None:
            handle = by_idx.get(target_idx)
        if handle is None:
            print(f"No drone index {target_idx} is connected.")
            continue

        await execute_action(handle, action, args)


def parse_command(text: str) -> tuple[Optional[int], Optional[str]]:
    match = re.match(r"drone\s*(\d+)\s+(.+)", text)
    if not match:
        return None, None
    idx = int(match.group(1))
    action = match.group(2).strip()
    return idx, action


async def execute_action(handle: DroneHandle, action: str, args: argparse.Namespace):
    action_lower = action.lower()
    if action_lower == "takeoff":
        await run_takeoff_sequence(handle, args)
    elif action_lower == "land":
        await safe_call(handle.system.action.land, handle.label, "land()")
    elif action_lower == "arm":
        await safe_call(handle.system.action.arm, handle.label, "arm()")
    elif action_lower == "disarm":
        await safe_call(handle.system.action.disarm, handle.label, "disarm()")
    else:
        print(f"[{handle.label}] Unsupported action '{action}'. "
              "Use takeoff, land, arm, disarm.")


async def run_takeoff_sequence(handle: DroneHandle, args: argparse.Namespace):
    try:
        await handle.system.action.set_takeoff_altitude(args.takeoff_alt)
    except Exception:
        pass

    print(f"[{handle.label}] Takeoff requested (target ~{args.takeoff_alt:.1f} m)")
    if not await safe_call(handle.system.action.arm, handle.label, "arm()"):
        return
    if not await safe_call(handle.system.action.takeoff, handle.label, "takeoff()"):
        return

    hover_time = max(args.hover_sec, 0.0)
    print(
        f"[{handle.label}] Hovering for {hover_time:.1f} s before automatic landing"
    )
    await asyncio.sleep(hover_time)
    await safe_call(handle.system.action.land, handle.label, "land()")


async def safe_call(coro, label: str, name: str) -> bool:
    try:
        await coro()
        print(f"[{label}] {name} sent")
        return True
    except Exception as exc:
        print(f"[{label}] {name} failed: {exc}")
        return False


async def main_async(args: argparse.Namespace):
    handles_list = await asyncio.gather(
        *[
            connect_drone(
                idx,
                connect_timeout=args.connect_timeout,
                health_timeout=args.health_timeout,
            )
            for idx in args.drones
        ]
    )
    by_idx = {handle.idx: handle for handle in handles_list}
    ordinal_map = {
        args.command_base + offset: handle
        for offset, handle in enumerate(handles_list)
    }
    await interactive_loop(by_idx, ordinal_map, args)


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
