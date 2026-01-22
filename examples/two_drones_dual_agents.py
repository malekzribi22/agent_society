#!/usr/bin/env python3
"""
Minimal setup: Launch two PX4-controlled drones with automatic PX4 autostart.
Updated to use standard PX4 MAVLink ports for proper multi-agent control.
"""

import os

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
import numpy as np
from scipy.spatial.transform import Rotation

from omni.isaac.core.world import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackendConfig, PX4MavlinkBackend
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig


NUM_DRONES = 2
PX4_TCP_BASE = 4560
# Use standard PX4 MAVLink UDP ports
MAVSDK_BASE = 14540  # drone0→14540, drone1→14541 (standard PX4 increment is 1)


def build_layout():
    spacing = 3.0
    layout = []
    for idx in range(NUM_DRONES):
        layout.append({
            "name": f"drone{idx}",
            "stage_prim": f"/World/drone{idx:02d}",
            "vehicle_id": idx,
            "spawn": [
                (idx * spacing) - spacing * 0.5,
                0.0,
                0.07,
            ],
            "px4_port": PX4_TCP_BASE + idx,
            # Standard PX4 port mapping: 14540, 14541...
            "mavsdk_port": MAVSDK_BASE + idx,  # This is key: 14540, 14541
            "ros_namespace": f"drone{idx:02d}",
        })
    return layout


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


class TwoDroneApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

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

        self.drones = {}
        for entry in DRONE_LAYOUT:
            self.drones[entry["name"]] = self._spawn_drone(entry, cam_cfg)

        self.pg.set_viewport_camera([6.0, 8.0, 5.0], [0.0, 0.0, 0.0])
        self.world.reset()
        self._print_mapping()

    def _spawn_drone(self, cfg: dict, cam_cfg: dict):
        mult_cfg = MultirotorConfig()
        px4_cfg = PX4MavlinkBackendConfig({
            "vehicle_id": cfg["vehicle_id"],
            "px4_autolaunch": True,
            "use_tcp": True,
            "tcp_port": cfg["px4_port"],
            "px4_host": "127.0.0.1",
        })
        ros_backend = ROS2Backend(
            vehicle_id=cfg["vehicle_id"],
            config={
                "namespace": cfg["ros_namespace"],
                "pub_sensors": False,
                "pub_graphical_sensors": True,
                "pub_state": True,
                "pub_tf": False,
                "sub_control": False,
            },
        )
        mult_cfg.backends = [PX4MavlinkBackend(px4_cfg), ros_backend]
        mult_cfg.graphical_sensors = [MonocularCamera(f"cam_{cfg['name']}", config=cam_cfg)]

        quat = Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()
        return Multirotor(
            cfg["stage_prim"],
            ROBOTS["Iris"],
            cfg["vehicle_id"],
            cfg["spawn"],
            quat,
            config=mult_cfg,
        )

    def _print_mapping(self):
        print("\n" + "=" * 80)
        print("TWO DRONES READY — PX4 + MAVSDK mapping")
        print("Name   | MAVSDK URL              | PX4 TCP port | ROS namespace")
        for entry in DRONE_LAYOUT:
            print(
                f"{entry['name']:<6} | udpin://0.0.0.0:{entry['mavsdk_port']:<5} "
                f"| {entry['px4_port']:<12} | {entry['ros_namespace']}"
            )
        print("PX4 is autostarted by this script.")
        print("Control each drone with: python3 agent_for_drone.py 0 and python3 agent_for_drone.py 1")
        print("=" * 80 + "\n")

    def run(self):
        self.timeline.play()
        try:
            while simulation_app.is_running():
                self.world.step(render=True)
        finally:
            self.timeline.stop()
            simulation_app.close()


def main():
    TwoDroneApp().run()


if __name__ == "__main__":
    main()
