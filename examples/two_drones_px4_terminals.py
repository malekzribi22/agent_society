#!/usr/bin/env python3
"""
two_drones_px4_terminals.py

Spawn 2 Iris multirotors in Isaac Sim (Pegasus) and connect each to a different
PX4 SITL instance running in its own terminal.

- Drone0 ↔ PX4 instance 0 on TCP 4560
- Drone1 ↔ PX4 instance 1 on TCP 4561

PX4 is **NOT** autolaunched here so that you can run it manually in terminals
and use the px4 shell (pxh>) for direct control (commander takeoff, land, etc.).
"""

import os

# --- GPU / Optimus hints (same style you were using) ---
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__VK_LAYER_NV_optimus"] = "NVIDIA_only"

from isaacsim import SimulationApp

# Start Isaac Sim with GUI
simulation_app = SimulationApp(
    {
        "headless": False,
        "renderer": "RayTracedLighting",
    }
)

import carb
import omni.usd
import omni.timeline
from isaacsim.core.utils.extensions import enable_extension
from omni.isaac.core.world import World

from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera


# --------------------------------------------------------------------------
# Basic layout: 2 drones on the ground, a few meters apart
# --------------------------------------------------------------------------

DRONES = [
    {
        "name": "drone0",
        "stage_prim": "/World/drone00",
        "vehicle_id": 0,               # PX4 instance 0, SYSID 1 by default
        "spawn": [-2.0, 0.0, 0.07],    # x, y, z in meters
        "sim_port": 4560,              # PX4 simulator TCP port
        "ros_namespace": "drone00",
    },
    {
        "name": "drone1",
        "stage_prim": "/World/drone01",
        "vehicle_id": 1,               # PX4 instance 1, SYSID 2 by default
        "spawn": [ 2.0, 0.0, 0.07],
        "sim_port": 4561,
        "ros_namespace": "drone01",
    },
]


# --------------------------------------------------------------------------
# Isaac / Pegasus setup
# --------------------------------------------------------------------------

def configure_renderer():
    settings = carb.settings.get_settings()
    settings.set("/app/renderer/activeRenderer", "rtx")
    settings.set("/rtx/enabled", True)
    settings.set("/app/asyncRendering", True)
    settings.set("/app/runLoops/main/rateLimitEnabled", True)
    settings.set("/app/runLoops/main/timeStepsPerSecond", 60)
    print("[two_drones] Active renderer:", settings.get_as_string("/app/renderer/activeRenderer"))


class TwoDronePx4App:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        # Pegasus world
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Simple environment (you can change this to any env in SIMULATION_ENVIRONMENTS)
        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # Simple camera so you can see the drones
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

        # Enable ROS bridge (optional, but matches your previous setups)
        enable_extension("isaacsim.ros2.bridge")

        self.drones = {}
        for cfg in DRONES:
            self.drones[cfg["name"]] = self._spawn_drone(cfg, cam_cfg)

        # Place the viewport camera so you see both drones
        self.pg.set_viewport_camera([8.0, 8.0, 6.0], [0.0, 0.0, 0.0])

        self.world.reset()
        self._print_mapping()

    def _spawn_drone(self, cfg: dict, cam_cfg: dict):
        """
        Create one multirotor with:
          - a PX4MavlinkBackend (NO autolaunch)
          - a ROS2 backend (optional)
          - a graphical monocular camera
        """
        mult_cfg = MultirotorConfig()

        # PX4 backend: we do NOT autolaunch, so you can run PX4 manually
        px4_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": cfg["vehicle_id"],   # 0 / 1
                "px4_autolaunch": False,           # <-- MANUAL PX4 LAUNCH
                "use_tcp": True,                   # Pegasus ↔ PX4 via TCP
                "tcp_port": cfg["sim_port"],       # 4560 / 4561
                "px4_host": "127.0.0.1",
            }
        )

        px4_backend = PX4MavlinkBackend(px4_config)

        # ROS2 backend (optional, for visualization)
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

        mult_cfg.backends = [px4_backend, ros_backend]

        mult_cfg.graphical_sensors = [
            MonocularCamera(f"cam_{cfg['name']}", config=cam_cfg)
        ]

        # Orientation as quaternion (no rotation)
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
        print("TWO DRONES READY — PX4 TERMINAL CONTROL")
        print("Name    | PX4 vehicle_id | PX4 sim TCP port | PX4 instance suggested")
        for cfg in DRONES:
            inst = cfg["vehicle_id"]  # 0 / 1
            print(
                f"{cfg['name']:<7} | {cfg['vehicle_id']:^14} | {cfg['sim_port']:^16} | instance {inst}"
            )
        print(
            """
How to connect PX4:

  - Start this script in one terminal:
      isaacpy two_drones_px4_terminals.py

  - Then in two *separate* terminals, start PX4 like:

      # Terminal B (instance 0 → drone0 on TCP 4560)
      cd ~/PX4-Autopilot
      PX4_SYS_AUTOSTART=10015 PX4_SIM_MODEL=none PX4_SIM_HOSTNAME=localhost \\
        ./build/px4_sitl_default/bin/px4 ./ROMFS/px4fmu_common -i 0 -s etc/init.d-posix/rcS

      # Terminal C (instance 1 → drone1 on TCP 4561)
      cd ~/PX4-Autopilot
      PX4_SYS_AUTOSTART=10015 PX4_SIM_MODEL=none PX4_SIM_HOSTNAME=localhost \\
        ./build/px4_sitl_default/bin/px4 ./ROMFS/px4fmu_common -i 1 -s etc/init.d-posix/rcS

    Then in each PX4 shell (pxh>), you can do, independently:

      commander arm
      commander takeoff
      commander land

    and both drones should fly in parallel in Isaac Sim.
"""
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
    configure_renderer()
    app = TwoDronePx4App()
    app.run()


if __name__ == "__main__":
    main()

