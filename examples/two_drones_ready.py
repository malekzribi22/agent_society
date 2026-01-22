#!/usr/bin/env python3
import os

# Force dGPU on hybrid laptops (same pattern as 9_people.py)
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

# IMPORTANT: use the old omni.isaac.core World, as in 9_people.py
from omni.isaac.core.world import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# Basic renderer / timing settings (copied from 9_people.py)
settings = carb.settings.get_settings()
settings.set("/app/renderer/activeRenderer", "rtx")
settings.set("/rtx/enabled", True)
settings.set("/app/asyncRendering", True)
settings.set("/app/runLoops/main/rateLimitEnabled", True)
settings.set("/app/runLoops/main/timeStepsPerSecond", 60)
print("Active renderer:", settings.get_as_string("/app/renderer/activeRenderer"))

# Enable ROS2 bridge (same as example)
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
# New empty stage
omni.usd.get_context().new_stage()


# ---------------------------------------------------------------------------
#  Simple circular person controller (same as 9_people)
# ---------------------------------------------------------------------------
class CirclePersonController(PersonController):
    def __init__(self):
        super().__init__()
        self._radius = 5.0
        self.gamma = 0.0
        self.gamma_dot = 0.3

    def update(self, dt: float):
        self.gamma += self.gamma_dot * dt
        self._person.update_target_position(
            [self._radius * np.cos(self.gamma), self._radius * np.sin(self.gamma), 0.0]
        )


# ---------------------------------------------------------------------------
#  Main Pegasus App: Curved Gridroom + 2 people + 2 PX4 drones
# ---------------------------------------------------------------------------
class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        # Pegasus + Isaac World (old omni.isaac.core API, matches 9_people.py)
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Environment
        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # -------------------------------------------------------------------
        #  People (just two examples so you see motion)
        # -------------------------------------------------------------------
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
            init_pos=[2.0, 0.0, 0.0],
        )
        p2.update_target_position([10.0, 0.0, 0.0], 1.0)

        # -------------------------------------------------------------------
        #  Helper: make a PX4-backed multirotor (USING SAME PATTERN AS 9_people)
        # -------------------------------------------------------------------
        def make_px4_multirotor(
            usd_path: str,
            vehicle_id: int,
            init_pos,
            init_yaw_deg: float,
            ros_namespace: str,
        ):
            cfg = MultirotorConfig()

            # Use UDP + px4_autolaunch EXACTLY like 9_people.py.
            #  - 14560 is PX4 "simulator" port for the first vehicle.
            #  - For the second drone we offset to 14570 to avoid clashes.
            #  - Offboard/MAVSDK still uses 14540/14541 automatically (PX4 default).
            udp_base = 14560 if vehicle_id == 0 else 14570

            mavlink_cfg = PX4MavlinkBackendConfig({
                "vehicle_id": vehicle_id,
                "px4_autolaunch": True,
                "use_tcp": False,
                "udp_ip": "127.0.0.1",
                "udp_out": udp_base,
                "udp_in": udp_base,
            })

            ros2_backend = ROS2Backend(
                vehicle_id=vehicle_id,
                config={
                    "namespace": ros_namespace,
                    "pub_sensors": False,
                    "pub_graphical_sensors": True,
                    "pub_state": True,
                    "pub_tf": False,
                    "sub_control": False,
                },
            )

            cfg.backends = [PX4MavlinkBackend(mavlink_cfg), ros2_backend]

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
            cfg.graphical_sensors = [
                MonocularCamera("camera", config=camera_cfg)
            ]

            quat = Rotation.from_euler(
                "XYZ", [0.0, 0.0, init_yaw_deg], degrees=True
            ).as_quat()

            drone = Multirotor(
                usd_path,
                ROBOTS["Iris"],
                vehicle_id,
                init_pos,
                quat,
                config=cfg,
            )
            return drone

        # -------------------------------------------------------------------
        #  Drone 0: vehicle_id = 0, ROS namespace "drone0"
        # -------------------------------------------------------------------
        self.drone0 = make_px4_multirotor(
            "/World/drone00",
            vehicle_id=0,
            init_pos=[0.0, 0.0, 0.07],
            init_yaw_deg=0.0,
            ros_namespace="drone0",
        )

        # -------------------------------------------------------------------
        #  Drone 1: vehicle_id = 1, ROS namespace "drone1"
        # -------------------------------------------------------------------
        self.drone1 = make_px4_multirotor(
            "/World/drone01",
            vehicle_id=1,
            init_pos=[2.0, 0.0, 0.07],
            init_yaw_deg=0.0,
            ros_namespace="drone1",
        )

        # Camera view
        self.pg.set_viewport_camera([5.0, 9.0, 6.5], [0.0, 0.0, 0.0])

        # Reset so everything initializes
        self.world.reset()
        self.stop_sim = False

        print("=== PegasusApp initialized with TWO PX4 drones ===")
        print("Drone0 namespace: drone0  (PX4 vehicle_id 0)")
        print("Drone1 namespace: drone1  (PX4 vehicle_id 1)")
        print("For MAVSDK connect to:")
        print("  - Drone0 Offboard port: 14540 (PX4 default first instance)")
        print("  - Drone1 Offboard port: 14541 (PX4 default second instance)")

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
        carb.log_warn("PegasusApp closing")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = PegasusApp()
    app.run()


if __name__ == "__main__":
    main()

