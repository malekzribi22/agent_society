#!/usr/bin/env python3
"""Spawn 10 drones connected to external PX4 SITL instances.

When this script starts it launches PX4's multi-instance helper in headless
mode (so no xterms are required). Each PX4 runs in
`/home/px4/PX4-Autopilot/build/px4_sitl_default/instance_*` and listens on the
standard port pairs (sim: 14560, 14570, … / MAVSDK: 14540, 14541, …). The
Pegasus drones use `px4_autolaunch=False` and connect to those ports. When you
close Isaac Sim the PX4 helper is terminated automatically.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"})

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


PX4_ROOT = Path("/home/px4/PX4-Autopilot").resolve()
PX4_BUILD = PX4_ROOT / "build" / "px4_sitl_default"
PX4_BIN = PX4_BUILD / "bin" / "px4"
PX4_ROMFS = PX4_ROOT / "ROMFS" / "px4fmu_common"
NUM_DRONES = 10


def prepare_instance_dirs(instances: int) -> None:
    for idx in range(instances):
        inst_dir = PX4_BUILD / f"instance_{idx}"
        if inst_dir.exists():
            shutil.rmtree(inst_dir)
        shutil.copytree(PX4_ROMFS, inst_dir)


def launch_px4_instances(instances: int):
    prepare_instance_dirs(instances)
    procs = []
    for idx in range(instances):
        inst_dir = PX4_BUILD / f"instance_{idx}"
        env = os.environ.copy()
        env["PX4_SIM_MODEL"] = "iris"
        log = open(inst_dir / "px4.log", "w", buffering=1)
        cmd = [
            str(PX4_BIN),
            "-i", str(idx),
            "-d",
            "-s", "etc/init.d-posix/rcS",
            ".",
        ]
        print(f"[PX4] Launching instance {idx} -> {inst_dir}")
        proc = subprocess.Popen(cmd, cwd=str(inst_dir), env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((proc, log))
    return procs


def build_layout():
    layout = []
    cols = 5
    spacing = 2.5
    base_x = -(cols - 1) * spacing * 0.5
    base_y = -spacing * 0.5
    for idx in range(NUM_DRONES):
        row = idx // cols
        col = idx % cols
        layout.append({
            "name": f"drone{idx}",
            "stage_prim": f"/World/drone{idx:02d}",
            "vehicle_id": idx,
            "spawn": [base_x + col * spacing, base_y + row * spacing, 0.07],
            "sim_port": 14560 + idx * 10,
            "mavsdk_port": 14540 + idx,
            "ros_namespace": f"drone{idx:02d}",
        })
    return layout


DRONE_LAYOUT = build_layout()


class TenDroneExternal:
    def __init__(self):
        self.px4_procs = launch_px4_instances(NUM_DRONES)
        time.sleep(6.0)

        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        self.drones = {}
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

        for entry in DRONE_LAYOUT:
            self.drones[entry["name"]] = self._spawn(entry, cam_cfg)

        self.pg.set_viewport_camera([6.5, 11.5, 7.0], [0.0, 0.0, 0.0])
        self.world.reset()
        self._print_mapping()

    def _spawn(self, cfg, cam_cfg):
        mult_cfg = MultirotorConfig()
        px4_cfg = PX4MavlinkBackendConfig({
            "vehicle_id": cfg["vehicle_id"],
            "px4_autolaunch": False,
            "udp_ip": "127.0.0.1",
            "udp_out": cfg["sim_port"],
            "udp_in": cfg["sim_port"],
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
        print("Ten drones connected to external PX4 instances")
        print("Name   | MAVSDK URL              | PX4 sim port | ROS namespace")
        for entry in DRONE_LAYOUT:
            print(
                f"{entry['name']:<6} | udpin://0.0.0.0:{entry['mavsdk_port']:<5} "
                f"| {entry['sim_port']:<12} | {entry['ros_namespace']}"
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
            for proc, log in self.px4_procs:
                if proc.poll() is None:
                    print(f"[PX4] Terminating instance PID {proc.pid}")
                    proc.terminate()
                log.close()


def main():
    settings = carb.settings.get_settings()
    settings.set("/app/renderer/activeRenderer", "rtx")
    settings.set("/rtx/enabled", True)
    settings.set("/app/asyncRendering", True)
    settings.set("/app/runLoops/main/rateLimitEnabled", True)
    settings.set("/app/runLoops/main/timeStepsPerSecond", 60)

    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()
    omni.usd.get_context().new_stage()

    app = TenDroneExternal()
    app.run()


if __name__ == "__main__":
    main()
