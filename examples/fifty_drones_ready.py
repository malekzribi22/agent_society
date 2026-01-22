#!/usr/bin/env python3
"""
Spin up fifty PX4-controlled drones in Pegasus with automatic PX4 launch per drone.
Each drone prints "Ready for takeoff!" once PX4 completes boot.
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
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pymavlink import mavutil


class PX4ReadyBackend(PX4MavlinkBackend):
    """PX4 backend that notifies when PX4 prints 'Ready for takeoff!'."""

    def __init__(self, config, ready_callback=None):
        super().__init__(config)
        self._ready_callback = ready_callback
        self._ready_announced = False

    def _handle_ready_text(self, text: str):
        if self._ready_announced:
            return
        self._ready_announced = True
        if self._ready_callback:
            self._ready_callback(text)

    def _decode_text(self, msg):
        text = getattr(msg, "text", "")
        if isinstance(text, bytes):
            text = text.decode(errors="ignore")
        return text.replace("\x00", "").strip()

    def poll_mavlink_messages(self):
        if not self._received_first_hearbeat:
            return

        needs_to_wait_for_actuator = self._received_first_actuator and self._enable_lockstep
        self._received_actuator = False

        while True:
            msg = self._connection.recv_match(blocking=needs_to_wait_for_actuator)

            if msg is not None:
                if msg.id in (
                    mavutil.mavlink.MAVLINK_MSG_ID_STATUSTEXT,
                    getattr(mavutil.mavlink, "MAVLINK_MSG_ID_STATUSTEXT_LONG", -1),
                ):
                    text = self._decode_text(msg)
                    if text and "ready for takeoff" in text.lower():
                        self._handle_ready_text(text)

                if msg.id == mavutil.mavlink.MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS:
                    self._received_first_actuator = True
                    self._received_actuator = True
                    self.handle_control(msg.time_usec, msg.controls, msg.mode, msg.flags)

            if not needs_to_wait_for_actuator or self._received_actuator:
                break


NUM_DRONES = 50
SIM_PORT_BASE = 4560      # PX4 TCP ports (4560 + vehicle_id)
MAVSDK_PORT_BASE = 14540  # MAVSDK side increments by 1


def build_layout():
    layout = []
    cols = 10
    spacing = 3.0
    base_x = -(cols - 1) * spacing * 0.5
    rows = int(np.ceil(NUM_DRONES / cols))
    base_y = -(rows - 1) * spacing * 0.5
    for idx in range(NUM_DRONES):
        row = idx // cols
        col = idx % cols
        layout.append({
            "name": f"drone{idx}",
            "stage_prim": f"/World/drone{idx:02d}",
            "vehicle_id": idx,
            "spawn": [
                base_x + col * spacing,
                base_y + row * spacing,
                0.07,
            ],
            "sim_port": SIM_PORT_BASE + idx,
            "mavsdk_port": MAVSDK_PORT_BASE + idx,
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


class FiftyDroneApp:
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
        self._ready_messages = {}
        for entry in DRONE_LAYOUT:
            self.drones[entry["name"]] = self._spawn_drone(entry, cam_cfg)

        self.pg.set_viewport_camera([12.0, 18.0, 10.0], [0.0, 0.0, 0.0])
        self.world.reset()
        self._print_mapping()

    def _spawn_drone(self, cfg: dict, cam_cfg: dict):
        mult_cfg = MultirotorConfig()
        px4_cfg = PX4MavlinkBackendConfig({
            "vehicle_id": cfg["vehicle_id"],
            "px4_autolaunch": True,
            "use_tcp": True,
            "tcp_port": cfg["sim_port"],
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
        px4_backend = PX4ReadyBackend(
            px4_cfg,
            ready_callback=lambda text, entry=cfg: self._on_drone_ready(entry, text),
        )
        mult_cfg.backends = [px4_backend, ros_backend]
        mult_cfg.graphical_sensors = [
            MonocularCamera(f"cam_{cfg['name']}", config=cam_cfg)
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

    def _on_drone_ready(self, entry: dict, text: str):
        name = entry["name"]
        if name in self._ready_messages:
            return
        self._ready_messages[name] = text
        print(f"[{name}] {text} — connect via udpin://0.0.0.0:{entry['mavsdk_port']}")
        if len(self._ready_messages) == NUM_DRONES:
            print("All drones report PX4 Ready for takeoff. MAVSDK agents can arm now.")

    def _print_mapping(self):
        print("\n" + "=" * 80)
        print("FIFTY DRONES READY — PX4 + MAVSDK mapping")
        print("Name   | MAVSDK URL              | PX4 TCP port | ROS namespace")
        for entry in DRONE_LAYOUT:
            print(
                f"{entry['name']:<6} | udpin://0.0.0.0:{entry['mavsdk_port']:<5} "
                f"| {entry['sim_port']:<12} | {entry['ros_namespace']}"
            )
        print("Launch agents: python3 autogen_drones.py (now 50 workers).")
        print("Wait for fifty '[droneX] Ready for takeoff!' lines before connecting.")
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
    FiftyDroneApp().run()


if __name__ == "__main__":
    main()
