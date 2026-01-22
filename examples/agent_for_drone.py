#!/usr/bin/env python3
"""
Generic MAVSDK agent launcher for multi-drone PX4 / Pegasus setups.

Usage:
    python3 agent_for_drone.py <drone_index>

Where:
    drone_index = 0..9  (for your 10 drones)

This script:
  - Computes the MAVSDK UDP port for that drone.
  - Sets MAVSDK_URL and AGENT_NAME environment variables.
  - Imports and runs agent_mavsdk_pro.main().
"""

import asyncio
import os
import sys

# Match the two_drones_dual_agents.py layout: 14540, 14541, ...
MAVSDK_BASE_PORT = 14540
PORT_INCREMENT = 1  # PX4 increments each instance by 1 UDP port


async def main() -> None:
    # ---- parse CLI arg ----
    if len(sys.argv) != 2:
        print("Usage: python3 agent_for_drone.py <drone_index 0-9>")
        sys.exit(1)

    try:
        idx = int(sys.argv[1])
    except ValueError:
        print("Error: <drone_index> must be an integer between 0 and 9.")
        sys.exit(1)

    if not 0 <= idx <= 9:
        print("Error: <drone_index> must be between 0 and 9.")
        sys.exit(1)

    # Use standard PX4 port mapping: 14540, 14541, 14542...
    port = MAVSDK_BASE_PORT + (idx * PORT_INCREMENT)
    url = f"udpin://0.0.0.0:{port}"

    # CRITICAL: Set environment variables BEFORE importing agent_mavsdk_pro
    # This ensures each agent process gets the correct URL
    os.environ["MAVSDK_URL"] = url
    os.environ["AGENT_NAME"] = f"Agent{idx}"  # Use direct assignment, not setdefault
    os.environ["DRONE_INDEX"] = str(idx)  # Store index for verification
    os.environ["EXPECTED_SYSTEM_ID"] = str(idx + 1)  # PX4 system_id = vehicle_id + 1
    agent_name = os.environ["AGENT_NAME"]

    print(
        f"[launcher] Starting {agent_name} for drone{idx} on {url} "
        "(matches two_drones_dual_agents PX4 mapping)"
    )
    print(f"[launcher] Environment MAVSDK_URL={os.environ.get('MAVSDK_URL')}")
    print(f"[launcher] Environment AGENT_NAME={os.environ.get('AGENT_NAME')}")
    print(f"[launcher] Expected System ID: {idx + 1} (for vehicle_id {idx})")

    # Import AFTER env is set so agent_mavsdk_pro can read it on import
    from agent_mavsdk_pro import main as agent_main

    await agent_main()


if __name__ == "__main__":
    asyncio.run(main())
