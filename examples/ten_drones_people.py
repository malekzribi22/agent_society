#!/usr/bin/env python3
import os

# Force discrete GPU path if available (matches other examples)
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

from pathlib import Path
import subprocess
import time

from omni.isaac.core.world import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig


def build_layout(num_drones: int = 10):
    """
    Generate 10 spawn spots on a 5x2 grid so drones do not overlap.
    """
    layout = []
    cols = 5
    spacing_xy = 2.5
    base_x = -(cols - 1) / 2.0 * spacing_xy
    base_y = -spacing_xy * 0.5

    for idx in range(num_drones):
        row = idx // cols
        col = idx % cols
        layout.append({
            "name": f"drone{idx}",
            "stage_prim": f"/World/drone{idx:02d}",
            "vehicle_id": idx,
            "spawn": [
                base_x + col * spacing_xy,
                base_y + row * spacing_xy,
                0.07,
            ],
            "mavsdk_port": 14540 + idx,
            # PX4 SITL feeds the simulator via UDP port 14560, 14570, ...
            "px4_udp": 14560 + idx * 10,
        })
    return tuple(layout)


DRONE_LAYOUT = build_layout(10)


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

# PX4 SITL binaries (same layout as two_drones_sitl_clean.py)
PX4_ROOT = Path("/home/px4/PX4-Autopilot").resolve()
PX4_BUILD = PX4_ROOT / "build" / "px4_sitl_default"
PX4_BIN = PX4_BUILD / "bin" / "px4"
PX4_ROOTFS = PX4_BUILD / "rootfs"


def launch_px4_instance(instance: int):
    """
    Launch one PX4 SITL instance in the background.
    - Instance i listens on sim UDP 14560 + 10*i (default PX4 pattern).
    """
    if not PX4_BIN.exists():
        raise FileNotFoundError(f"PX4 binary not found at {PX4_BIN}")
    if not PX4_ROOTFS.exists():
        raise FileNotFoundError(f"PX4 rootfs not found at {PX4_ROOTFS}")

    workdir = PX4_BUILD / f"instance_{instance}"
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PX4_BIN),
        "-i",
        str(instance),
        "-w",
        str(workdir),
    ]
    print(f"[PX4-{instance}] Launching: {' '.join(cmd)} (cwd={PX4_ROOTFS})")
    return subprocess.Popen(
        cmd,
        cwd=str(PX4_ROOTFS),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


class CirclePersonController(PersonController):
    def __init__(self, radius=5.0, omega=0.2):
        super().__init__()
        self._radius = radius
        self.gamma = 0.0
        self.gamma_dot = omega

    def update(self, dt: float):
        self.gamma += self.gamma_dot * dt
        self._person.update_target_position(
            [self._radius * np.cos(self.gamma), self._radius * np.sin(self.gamma), 0.0]
        )


class TenDroneApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        # Start PX4 instances first so Pegasus connects to running SITL
        self.px4_procs = []
        for idx, _ in enumerate(DRONE_LAYOUT):
            try:
                proc = launch_px4_instance(idx)
                self.px4_procs.append(proc)
            except Exception as exc:
                carb.log_error(f"Failed to launch PX4 instance {idx}: {exc}")
        # Give PX4 time to boot before Pegasus connects
        time.sleep(4.0)

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        ctrl = CirclePersonController()
        _ = Person(
            "person1",
            "original_male_adult_construction_05",
            init_pos=[3.0, 0.0, 0.0],
            init_yaw=1.0,
            controller=ctrl,
        )
        p2 = Person(
            "person2",
            "original_female_adult_business_02",
            init_pos=[-2.0, 1.0, 0.0],
        )
        p2.update_target_position([10.0, 0.0, 0.0], 1.0)

        camera_cfg = {
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
            self.drones[entry["name"]] = self._spawn_drone(entry, camera_cfg)

        self.pg.set_viewport_camera([8.0, 12.0, 8.5], [0.0, 0.0, 0.0])
        self.world.reset()
        self._log_table()

    def _spawn_drone(self, cfg: dict, camera_cfg: dict):
        mult_cfg = MultirotorConfig()
        px4_cfg = PX4MavlinkBackendConfig({
            "vehicle_id": cfg["vehicle_id"],
            "px4_autolaunch": False,
            # Using UDP exactly like the two-drone example. PX4 instance i listens on 14560 + 10*i.
            "udp_ip": "127.0.0.1",
            "udp_out": cfg["px4_udp"],
            "udp_in": cfg["px4_udp"],
        })
        ros_backend = ROS2Backend(
            vehicle_id=cfg["vehicle_id"],
            config={
                "namespace": cfg["name"],
                "pub_sensors": False,
                "pub_graphical_sensors": True,
                "pub_state": True,
                "pub_tf": False,
                "sub_control": False,
            },
        )
        mult_cfg.backends = [PX4MavlinkBackend(px4_cfg), ros_backend]
        mult_cfg.graphical_sensors = [
            MonocularCamera(f"cam_{cfg['name']}", config=camera_cfg)
        ]

        quat = Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()
        return Multirotor(
            cfg["stage_prim"],
            ROBOTS["Iris"],
            cfg["vehicle_id"],
            cfg["spawn"],
            quat,
            config=mult_cfg,
        )

    def _log_table(self):
        print("=" * 72)
        print("=== TEN DRONES READY ===")
        print("Name   | MAVSDK port       | PX4 UDP (sim) | ROS namespace")
        for entry in DRONE_LAYOUT:
            print(
                f"{entry['name']:<6} | udpin://0.0.0.0:{entry['mavsdk_port']:<5} "
                f"| {entry['px4_udp']:<12} | {entry['name']}"
            )
        print("\nLaunch agent per drone, e.g.:")
        print("  isaacpy agent_mavsdk_pro_two_drone.py drone0")
        print("  isaacpy agent_mavsdk_pro_two_drone.py drone7")
        print("=" * 72)

    def run(self):
        self.timeline.play()
        try:
            while simulation_app.is_running():
                self.world.step(render=True)
        finally:
            self.timeline.stop()
            for idx, proc in enumerate(self.px4_procs):
                if proc and proc.poll() is None:
                    print(f"[PX4-{idx}] Terminating PX4 process")
                    proc.terminate()
            simulation_app.close()


def main():
    TenDroneApp().run()


if __name__ == "__main__":
    main()
