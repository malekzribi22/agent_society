#!/usr/bin/env python3
"""
ten_drones_manual_px4.py

Spawns 10 PX4-controlled Iris drones in Pegasus/Isaac Sim.

- Drones:      drone0 ... drone9
- Vehicle IDs: 0 ... 9
- PX4 sim TCP: 4560 + vehicle_id  => 4560..4569
- NO px4_autolaunch: you start PX4 manually in 10 terminals.

After running this script, start each PX4 instance with the shell
commands given below.
"""

import os

# GPU env (safe to keep; remove if not needed)
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
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# ------------------------------
# Layout / ports
# ------------------------------

NUM_DRONES = 10
SIM_PORT_BASE = 4560      # PX4 TCP sim port (4560 + vehicle_id)


def build_layout():
    layout = []
    cols = 5
    spacing = 2.5
    base_x = -(cols - 1) * spacing * 0.5
    base_y = -spacing * 0.5

    for idx in range(NUM_DRONES):
        row = idx // cols
        col = idx % cols
        layout.append(
            {
                "name": f"drone{idx}",
                "stage_prim": f"/World/drone{idx:02d}",
                "vehicle_id": idx,
                "spawn": [
                    base_x + col * spacing,
                    base_y + row * spacing,
                    0.07,
                ],
                "sim_port": SIM_PORT_BASE + idx,   # must match PX4 TCP
                "ros_namespace": f"drone{idx:02d}",
            }
        )
    return layout


DRONE_LAYOUT = build_layout()

# Renderer settings
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


class TenDroneApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Environment
        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # Simple RGB camera config
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

        # Nice top view
        self.pg.set_viewport_camera([7.0, 12.0, 7.0], [0.0, 0.0, 0.0])

        self.world.reset()
        self._print_mapping()

    def _spawn_drone(self, cfg: dict, cam_cfg: dict):
        mult_cfg = MultirotorConfig()

        # PX4 backend: no autolaunch, we will run PX4 in terminals
        px4_cfg = PX4MavlinkBackendConfig(
            {
                "vehicle_id": cfg["vehicle_id"],
                "px4_autolaunch": False,      # IMPORTANT: manual PX4 launch
                "use_tcp": True,
                "tcp_port": cfg["sim_port"],  # 4560 + vehicle_id
                "px4_host": "127.0.0.1",
            }
        )

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
        mult_cfg.graphical_sensors = [
            MonocularCamera(f"cam_{cfg['name']}", config=cam_cfg),
        ]

        quat = Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()

        drone = Multirotor(
            cfg["stage_prim"],
            ROBOTS["Iris"],
            cfg["vehicle_id"],
            cfg["spawn"],
            quat,
            config=mult_cfg,
        )

        return drone

    def _print_mapping(self):
        print("\n" + "=" * 80)
        print(" TEN DRONES — PX4 TCP mapping ".center(80, "="))
        print("Instance | Name   | PX4 TCP sim port | ROS namespace")
        print("-" * 80)
        for entry in DRONE_LAYOUT:
            inst = entry["vehicle_id"]
            name = entry["name"]
            port = entry["sim_port"]
            ns = entry["ros_namespace"]
            print(f"{inst:^8} | {name:<6} | tcp://127.0.0.1:{port:<5}   | {ns}")
        print("-" * 80)
        print(
            "Start PX4 from ~/PX4-Autopilot for each instance i=0..9 using:\n"
            "  PX4_SYS_AUTOSTART=10015 PX4_SIM_MODEL=none PX4_SIM_HOSTNAME=localhost \\\n"
            "    ./build/px4_sitl_default/bin/px4 ./ROMFS/px4fmu_common -i <i> -s etc/init.d-posix/rcS"
        )
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
    TenDroneApp().run()


if __name__ == "__main__":
    main()

