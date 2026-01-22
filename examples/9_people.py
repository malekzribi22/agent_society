#!/usr/bin/env python3
import os

# -------------------------------------------------------------------
# GPU / NVIDIA env
# -------------------------------------------------------------------
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__VK_LAYER_NV_optimus"] = "NVIDIA_only"

# Tell Pegasus where PX4 SITL is
os.environ["PX4_PATH"] = "/home/px4/PX4-Autopilot/build/px4_sitl_default/bin/px4"

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "renderer": "RayTracedLighting",
    }
)

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
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# -------------------------------------------------------------------
# Renderer config
# -------------------------------------------------------------------
settings = carb.settings.get_settings()
settings.set("/app/renderer/activeRenderer", "rtx")
settings.set("/rtx/enabled", True)
settings.set("/app/asyncRendering", True)
settings.set("/app/runLoops/main/rateLimitEnabled", True)
settings.set("/app/runLoops/main/timeStepsPerSecond", 60)
print("[SIM] Active renderer:", settings.get_as_string("/app/renderer/activeRenderer"))

# ROS2 bridge
enable_extension("isaacsim.ros2.bridge")

simulation_app.update()
omni.usd.get_context().new_stage()


# -------------------------------------------------------------------
# Simple circular controller for one person
# -------------------------------------------------------------------
class CirclePersonController(PersonController):
    def __init__(self):
        super().__init__()
        self._radius = 5.0
        self.gamma = 0.0
        self.gamma_dot = 0.3

    def update(self, dt: float):
        self.gamma += self.gamma_dot * dt
        self._person.update_target_position(
            [
                self._radius * np.cos(self.gamma),
                self._radius * np.sin(self.gamma),
                0.0,
            ]
        )


# -------------------------------------------------------------------
# Main Pegasus app
# -------------------------------------------------------------------
class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        # World / interface
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Environment
        self.pg.load_asset(
            SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout"
        )

        # People
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
        # Multirotor + PX4 MAVLink backend (TCP 4560)
        # -------------------------------------------------------------------
        cfg = MultirotorConfig()

        # IMPORTANT:
        # PX4 log says: "Waiting for simulator to accept connection on TCP port 4560"
        # => PX4 is a TCP CLIENT trying to connect to a SERVER on port 4560.
        # Pegasus must be that server. So we use TCP with tcp_port=4560.
        mavlink_cfg = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "use_tcp": True,
                "tcp_port": 4560,        # Pegasus listens here, PX4 connects
                "px4_host": "127.0.0.1", # where PX4 runs
            }
        )

        print("[SIM] PX4 autolaunch enabled (TCP 4560).")
        print("[SIM] PX4_PATH:", os.environ.get("PX4_PATH"))
        print("[SIM] PX4 will connect as client to Pegasus on tcp://127.0.0.1:4560")

        px4_backend = PX4MavlinkBackend(mavlink_cfg)

        # Optional ROS2 backend
        ros2_backend = ROS2Backend(
            vehicle_id=0,
            config={
                "namespace": "drone0",
                "pub_sensors": False,
                "pub_graphical_sensors": True,
                "pub_state": True,
                "pub_tf": False,
                "sub_control": False,
            },
        )

        cfg.backends = [px4_backend, ros2_backend]

        # Camera on the drone (optional)
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
        cfg.graphical_sensors = [MonocularCamera("camera", config=camera_cfg)]

        # Spawn drone
        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=cfg,
        )

        # Camera view
        self.pg.set_viewport_camera([5.0, 9.0, 6.5], [0.0, 0.0, 0.0])

        # Reset world
        self.world.reset()
        self.stop_sim = False

        print("=== PegasusApp initialized ===")
        print("Drone namespace: drone0")
        print("Watch for PX4 log messages like 'Ready for takeoff!'.")

    def run(self):
        self.timeline.play()
        try:
            while simulation_app.is_running() and not self.stop_sim:
                self.world.step(render=True)
        finally:
            carb.log_warn("PegasusApp closing")
            self.timeline.stop()
            simulation_app.close()


def main():
    app = PegasusApp()
    app.run()


if __name__ == "__main__":
    main()

