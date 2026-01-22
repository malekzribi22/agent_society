#!/usr/bin/env python3

import argparse
import asyncio
import os
import subprocess
import sys
from typing import Iterable, List, Optional

from mavsdk import System


def build_drone_map(total: int = 10):
    base_port = 14540
    return {
        f"drone{i}": {
            "cli_id": i + 1,
            "url": f"udpin://0.0.0.0:{base_port + i}",
            "sys_id": i + 1,
        }
        for i in range(total)
    }


DRONES = build_drone_map(10)
ID_LOOKUP = {cfg["cli_id"]: name for name, cfg in DRONES.items()}


async def connect_and_run(
    drone_name: str,
    mavsdk_url: str,
    *,
    expected_sysid: Optional[int] = None,
):
    """
    Connect to exactly ONE PX4 instance via its own UDP port
    and run a simple arm / takeoff / hover / land sequence.
    """
    print(f"\n[agent:{drone_name}] Connecting to {mavsdk_url} ...")

    drone = System()
    await drone.connect(system_address=mavsdk_url)

    # Wait for MAVSDK connection with timeout
    print(f"[agent:{drone_name}] Waiting for MAVSDK connection...")
    connection_timeout = 15  # seconds
    start_time = asyncio.get_event_loop().time()
    
    connected = False
    async for state in drone.core.connection_state():
        if state.is_connected:
            try:
                uuid = await drone.core.get_uuid()
            except Exception:
                uuid = "unknown"
            print(f"[agent:{drone_name}] ✅ MAVSDK connected. UUID: {uuid}")
            connected = True
            break
            
        # Check timeout
        if asyncio.get_event_loop().time() - start_time > connection_timeout:
            print(f"[agent:{drone_name}] ❌ CONNECTION TIMEOUT")
            print(f"[agent:{drone_name}] Make sure simulation is running and PX4 instances are ready")
            return

    if not connected:
        print(f"[agent:{drone_name}] ❌ Failed to connect")
        return

    if expected_sysid is not None:
        ok = await verify_sysid(drone, expected_sysid, drone_name)
        if not ok:
            return

    # Wait until PX4 is ready to fly
    print(f"[agent:{drone_name}] Waiting for PX4 to be ready...")
    health_timeout = 15
    start_time = asyncio.get_event_loop().time()
    
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print(f"[agent:{drone_name}] ✅ PX4 READY TO FLY 🚀")
            break
            
        # Check timeout
        if asyncio.get_event_loop().time() - start_time > health_timeout:
            print(f"[agent:{drone_name}] ⚠️  PX4 health check timeout, continuing anyway...")
            break

    # ARM
    print(f"[agent:{drone_name}] Arming...")
    try:
        await drone.action.arm()
        print(f"[agent:{drone_name}] ✅ Armed!")
    except Exception as e:
        print(f"[agent:{drone_name}] ❌ Arming failed: {e}")
        return

    # TAKEOFF
    print(f"[agent:{drone_name}] Taking off...")
    try:
        await drone.action.takeoff()
    except Exception as e:
        print(f"[agent:{drone_name}] ❌ Takeoff failed: {e}")
        return

    # Hover a bit
    print(f"[agent:{drone_name}] Hovering for 10 seconds...")
    await asyncio.sleep(10)
    print(f"[agent:{drone_name}] Hovering complete, landing...")

    # LAND
    try:
        await drone.action.land()
        print(f"[agent:{drone_name}] ✅ Land command sent. Waiting a bit...")
        await asyncio.sleep(5)
    except Exception as e:
        print(f"[agent:{drone_name}] ❌ Landing failed: {e}")

    print(f"[agent:{drone_name}] Mission complete. Exiting.")


async def verify_sysid(drone: System, expected: int, drone_name: str) -> bool:
    try:
        sysid = await drone.param.get_param_int("SYSID_THISMAV")
    except Exception as exc:
        print(f"[agent:{drone_name}] ⚠️  Could not read SYSID_THISMAV ({exc}), continuing without check.")
        return True

    if sysid != expected:
        print(
            f"[agent:{drone_name}] ❌ SYSID_THISMAV mismatch "
            f"(expected {expected}, got {sysid}). "
            "Check which drone is connected before sending commands."
        )
        return False
    print(f"[agent:{drone_name}] Verified SYSID_THISMAV={sysid}")
    return True


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Control one or both Pegasus drones via MAVSDK"
    )
    parser.add_argument(
        "drone_name",
        nargs="?",
        help="Friendly name or key (drone0/drone1) when controlling a single drone",
    )
    parser.add_argument(
        "drone_id",
        nargs="?",
        type=int,
        choices=list(ID_LOOKUP.keys()),
        help="Drone numeric ID (1 for drone0, 2 for drone1) when running a single agent",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Legacy shortcut for drone0 + drone1",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Launch agents for every known drone",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help="List of drone names or IDs to launch together",
    )
    parser.add_argument(
        "--url",
        help="Override MAVSDK connection URL (use when the defaults do not match)",
    )
    parser.add_argument(
        "--sys-id",
        type=int,
        help="Expected SYSID_THISMAV (optional sanity check for single-drone runs)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the known drone/port mapping and exit",
    )
    return parser


def spawn_agents(targets: Iterable[str]):
    """
    Launch one subprocess per drone (same behavior as separate terminals). This avoids UDP
    bind conflicts because each MAVSDK System runs in its own process.
    """
    processes = []
    script_path = os.path.abspath(__file__)
    for name in targets:
        cmd = [sys.executable, script_path, name]
        print(f"[manager] Spawning {' '.join(cmd)}")
        processes.append(subprocess.Popen(cmd))

    exit_code = 0
    for proc in processes:
        ret = proc.wait()
        if exit_code == 0 and ret != 0:
            exit_code = ret

    return exit_code


def print_mapping():
    print("Known drone endpoints (matches two_drones_9_people.py):")
    for name, cfg in DRONES.items():
        print(f"  {name}: id={cfg['cli_id']} url={cfg['url']} expected_sysid={cfg['sys_id']}")


def resolve_target(name: Optional[str], cli_id: Optional[int]) -> Optional[str]:
    if name:
        if name in DRONES:
            return name
        try:
            parsed = int(name)
            if parsed in ID_LOOKUP:
                return ID_LOOKUP[parsed]
        except ValueError:
            pass
    if cli_id in ID_LOOKUP:
        return ID_LOOKUP[cli_id]
    return None


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list:
        print_mapping()
        sys.exit(0)

    if args.all:
        sys.exit(spawn_agents(DRONES.keys()))

    combined: List[str] = []

    if args.both:
        combined.extend(["drone0", "drone1"])

    if args.targets:
        for token in args.targets:
            resolved = resolve_target(token, None)
            if not resolved:
                print(f"Unknown drone target: {token}")
                sys.exit(1)
            combined.append(resolved)

    if combined:
        sys.exit(spawn_agents(combined))

    target_name = resolve_target(args.drone_name, args.drone_id)

    if target_name is None and not args.url:
        print_mapping()
        parser.print_help()
        sys.exit(1)

    if target_name:
        cfg = DRONES[target_name]
        url = cfg["url"]
        expected_sysid = args.sys_id  # only check if user supplied override
    else:
        target_name = args.drone_name or "custom"
        url = args.url
        expected_sysid = args.sys_id

    asyncio.run(
        connect_and_run(
            target_name,
            url,
            expected_sysid=expected_sysid,
        )
    )
