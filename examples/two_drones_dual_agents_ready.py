#!/usr/bin/env python3
"""
Spawn two PX4-backed drones inside Isaac Sim and launch one MAVSDK agent per drone.

Each agent connects to its assigned PX4 instance only, so both drones can be commanded
in parallel without sharing control logic.
"""

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import List

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__VK_LAYER_NV_optimus"] = "NVIDIA_only"

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting",
})

import carb
from isaacsim.core.utils.extensions import enable_extension
import omni.timeline
import omni.usd
from scipy.spatial.transform import Rotation
from mavsdk import System

from omni.isaac.core.world import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig


NUM_DRONES = 2
PX4_TCP_BASE = 4560
MAVSDK_BASE = 14540
AUTO_AGENT_SUPERVISOR = os.getenv("PEGASUS_AUTO_AGENTS", "1").lower() not in {
    "0",
    "false",
    "no",
}


@dataclass(frozen=True)
class DroneSlot:
    name: str
    stage_prim: str
    vehicle_id: int
    spawn: List[float]
    px4_port: int
    mavsdk_port: int
    ros_namespace: str


def build_layout() -> List[DroneSlot]:
    spacing = 3.0
    slots = []
    for idx in range(NUM_DRONES):
        slots.append(
            DroneSlot(
                name=f"drone{idx}",
                stage_prim=f"/World/drone{idx:02d}",
                vehicle_id=idx,
                spawn=[(idx * spacing) - 0.5 * spacing, 0.0, 0.07],
                px4_port=PX4_TCP_BASE + idx,
                mavsdk_port=MAVSDK_BASE + idx,
                ros_namespace=f"drone{idx:02d}",
            )
        )
    return slots


DRONE_LAYOUT = build_layout()


settings = carb.settings.get_settings()
settings.set("/app/renderer/activeRenderer", "rtx")
settings.set("/rtx/enabled", True)
settings.set("/app/asyncRendering", True)
settings.set("/app/runLoops/main/rateLimitEnabled", True)
settings.set("/app/runLoops/main/timeStepsPerSecond", 60)
print("Active renderer:", settings.get_as_string("/app/renderer/activeRenderer"))

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
omni.usd.get_context().new_stage()


class DualDroneApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")
        self.pg.set_viewport_camera([6.0, 8.0, 5.0], [0.0, 0.0, 0.0])

        cam_cfg = {
            "update_rate": 30.0,
            "width": 1280,
            "height": 720,
            "hfov": 90.0,
            "enable_distortion": False,
            "noise_mean": 0.0,
            "noise_stddev": 0.002,
            "rgb": True,
            "depth": False,
            "semantic": False,
        }

        self.drones = []
        for slot in DRONE_LAYOUT:
            self.drones.append(self._spawn_drone(slot, cam_cfg))

        self.world.reset()
        self._print_mapping()
        self.supervisor = (
            DualAgentSupervisor(DRONE_LAYOUT) if AUTO_AGENT_SUPERVISOR else None
        )

    def _spawn_drone(self, slot: DroneSlot, cam_cfg):
        cfg = MultirotorConfig()
        px4_cfg = PX4MavlinkBackendConfig({
            "vehicle_id": slot.vehicle_id,
            "px4_autolaunch": True,
            "use_tcp": True,
            "tcp_port": slot.px4_port,
            "px4_host": "127.0.0.1",
        })
        ros_backend = ROS2Backend(
            vehicle_id=slot.vehicle_id,
            config={
                "namespace": slot.ros_namespace,
                "pub_sensors": False,
                "pub_graphical_sensors": True,
                "pub_state": True,
                "pub_tf": False,
                "sub_control": False,
            },
        )
        cfg.backends = [PX4MavlinkBackend(px4_cfg), ros_backend]
        cfg.graphical_sensors = [MonocularCamera(f"cam_{slot.name}", config=cam_cfg)]
        quat = Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()
        return Multirotor(
            slot.stage_prim,
            ROBOTS["Iris"],
            slot.vehicle_id,
            slot.spawn,
            quat,
            config=cfg,
        )

    def _print_mapping(self):
        print("\n" + "=" * 80)
        print("Dual-drone layout with dedicated MAVSDK agents")
        print("Name   | MAVSDK URL              | PX4 TCP port | ROS namespace")
        for slot in DRONE_LAYOUT:
            url = f"udpin://0.0.0.0:{slot.mavsdk_port}"
            print(
                f"{slot.name:<6} | {url:<24} | {slot.px4_port:<12} | {slot.ros_namespace}"
            )
        print("=" * 80 + "\n")

    def run(self):
        self.timeline.play()
        if self.supervisor:
            self.supervisor.start()
        try:
            while simulation_app.is_running():
                self.world.step(render=True)
        finally:
            self.timeline.stop()
            simulation_app.close()


class DualAgentSupervisor:
    """Launch one MAVSDK agent per drone so each drone is controlled by its own task."""

    def __init__(
        self,
        layout: List[DroneSlot],
        *,
        takeoff_alt: float = 3.0,
        hover_sec: float = 10.0,
        startup_delay: float = 8.0,
    ):
        self.layout = layout
        self.takeoff_alt = takeoff_alt
        self.hover_sec = hover_sec
        self.startup_delay = max(0.0, startup_delay)
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        print(
            f"[supervisor] Launching one agent per drone after {self.startup_delay:.1f}s "
            "to give PX4 time to boot."
        )
        self._thread = threading.Thread(target=self._thread_entry, daemon=True)
        self._thread.start()

    def _thread_entry(self):
        asyncio.run(self._async_entry())

    async def _async_entry(self):
        if self.startup_delay > 0.0:
            await asyncio.sleep(self.startup_delay)
        tasks = [
            asyncio.create_task(
                DroneAgent(
                    slot,
                    takeoff_alt=self.takeoff_alt,
                    hover_sec=self.hover_sec,
                ).execute()
            )
            for slot in self.layout
        ]
        await asyncio.gather(*tasks)


class DroneAgent:
    """Simple MAVSDK agent dedicated to a single drone."""

    def __init__(self, slot: DroneSlot, *, takeoff_alt: float, hover_sec: float):
        self.slot = slot
        self.url = f"udpin://0.0.0.0:{slot.mavsdk_port}"
        self.takeoff_alt = takeoff_alt
        self.hover_sec = hover_sec

    async def execute(self):
        label = f"agent:{self.slot.name}"
        print(f"[{label}] Connecting to {self.url}")
        drone = System()
        await drone.connect(system_address=self.url)

        if not await self._wait_for_connection(drone, label, timeout=30.0):
            return
        await self._wait_for_health(drone, label, timeout=25.0)

        try:
            await drone.action.set_takeoff_altitude(self.takeoff_alt)
        except Exception:
            pass

        if not await self._safe_call(drone.action.arm, label, "arm()"):
            return
        if not await self._safe_call(drone.action.takeoff, label, "takeoff()"):
            return

        print(
            f"[{label}] Hovering for {self.hover_sec:.1f} s at ~{self.takeoff_alt:.1f} m"
        )
        await asyncio.sleep(max(self.hover_sec, 0.0))
        await self._safe_call(drone.action.land, label, "land()")
        await asyncio.sleep(3.0)
        print(f"[{label}] Sequence complete")

    async def _wait_for_connection(self, drone: System, label: str, *, timeout: float):
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        async for state in drone.core.connection_state():
            if state.is_connected:
                try:
                    uuid = await drone.core.get_uuid()
                except Exception:
                    uuid = "unknown"
                print(f"[{label}] MAVSDK connected (UUID={uuid})")
                return True
            if loop.time() > deadline:
                print(f"[{label}] Timed out waiting for MAVSDK connection")
                return False
        return False

    async def _wait_for_health(self, drone: System, label: str, *, timeout: float):
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print(f"[{label}] PX4 reports it is ready to fly")
                return True
            if loop.time() > deadline:
                print(f"[{label}] Health timeout, continuing anyway")
                return False
        return False

    async def _safe_call(self, coro, label: str, action: str):
        try:
            await coro()
            print(f"[{label}] {action} sent")
            return True
        except Exception as exc:
            print(f"[{label}] Failed to execute {action}: {exc}")
            return False


def main():
    DualDroneApp().run()


if __name__ == "__main__":
    main()
