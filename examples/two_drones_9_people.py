#!/usr/bin/env python3
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
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera

# SAME imports as in 9_people.py
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig


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


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # People
        ctrl = CirclePersonController()
        Person("person1", "original_male_adult_construction_05",
               init_pos=[3.0, 0.0, 0.0], init_yaw=1.0, controller=ctrl)

        p2 = Person("person2", "original_female_adult_business_02",
                    init_pos=[2.0, 0.0, 0.0])
        p2.update_target_position([10.0, 0.0, 0.0], 1.0)

        # -------------------------
        # COMMON CAMERA CONFIG
        # -------------------------
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

        # -------------------------
        # DRONE 0
        # -------------------------
        cfg0 = MultirotorConfig()
        mav0 = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "use_tcp": True,
            "tcp_port": 4560,  # PX4 instance 0 waits for tcp://localhost:4560
            "px4_host": "127.0.0.1",
        })
        ros0 = ROS2Backend(
            vehicle_id=0,
            config={
                "namespace": "drone0",
                "pub_sensors": False,
                "pub_graphical_sensors": True,
                "pub_state": True,
                "pub_tf": False,
                "sub_control": False,
            }
        )
        cfg0.backends = [PX4MavlinkBackend(mav0), ros0]
        cfg0.graphical_sensors = [MonocularCamera("cam0", config=camera_cfg)]

        self.drone0 = Multirotor(
            "/World/drone00",
            ROBOTS["Iris"],
            0,
            [-2.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=cfg0,
        )

        # -------------------------
        # DRONE 1
        # -------------------------
        cfg1 = MultirotorConfig()
        mav1 = PX4MavlinkBackendConfig({
            "vehicle_id": 1,
            "px4_autolaunch": True,
            "use_tcp": True,
            "tcp_port": 4561,  # PX4 instance 1 waits for tcp://localhost:4561
            "px4_host": "127.0.0.1",
        })
        ros1 = ROS2Backend(
            vehicle_id=1,
            config={
                "namespace": "drone1",
                "pub_sensors": False,
                "pub_graphical_sensors": True,
                "pub_state": True,
                "pub_tf": False,
                "sub_control": False,
            }
        )
        cfg1.backends = [PX4MavlinkBackend(mav1), ros1]
        cfg1.graphical_sensors = [MonocularCamera("cam1", config=camera_cfg)]

        self.drone1 = Multirotor(
            "/World/drone01",
            ROBOTS["Iris"],
            1,
            [2.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=cfg1,
        )

        self.pg.set_viewport_camera([5.0, 9.0, 6.5], [0.0, 0.0, 0.0])
        self.world.reset()

        print("=== TWO DRONES READY ===")
        print("Drone0 → MAVSDK: udpin://0.0.0.0:14540 → PX4 port14560")
        print("Drone1 → MAVSDK: udpin://0.0.0.0:14541 → PX4 port14570")

    def run(self):
        self.timeline.play()
        while simulation_app.is_running():
            self.world.step(render=True)
        self.timeline.stop()
        simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()
