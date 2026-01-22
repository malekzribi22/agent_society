#!/usr/bin/env python3
import os

# If you have a hybrid laptop, keep these to force dGPU
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__VK_LAYER_NV_optimus"] = "NVIDIA_only"

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting",
    # You can set fixed resolution if you want:
    # "width": 1600, "height": 900,
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
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# --- Make RTX the active renderer and enable async + main-loop cap ---
settings = carb.settings.get_settings()
settings.set("/app/renderer/activeRenderer", "rtx")
settings.set("/rtx/enabled", True)
settings.set("/app/asyncRendering", True)
settings.set("/app/runLoops/main/rateLimitEnabled", True)
settings.set("/app/runLoops/main/timeStepsPerSecond", 60)
print("Active renderer:", settings.get_as_string("/app/renderer/activeRenderer"))

# Enable ROS 2 bridge so Pegasus can publish images if configured to
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# start with a fresh stage
omni.usd.get_context().new_stage()

class CirclePersonController(PersonController):
    def __init__(self, radius=5.0, omega=0.3):
        super().__init__()
        self._radius = float(radius)
        self.gamma = 0.0
        self.gamma_dot = float(omega)

    def update(self, dt: float):
        self.gamma += self.gamma_dot * dt
        self._person.update_target_position(
            [self._radius * np.cos(self.gamma), self._radius * np.sin(self.gamma), 0.0]
        )

def make_drone(
    usd_prim_path: str,
    robot_key: str,
    vehicle_id: int,
    pose_xyz,
    pose_quat,
    px4_tcp_port: int,
    ros_namespace: str,
):
    """Create a Multirotor with distinct MAVLink TCP port and ROS 2 namespace."""
    cfg = MultirotorConfig()

    mavlink_cfg = PX4MavlinkBackendConfig({
        "vehicle_id": vehicle_id,
        "px4_autolaunch": False,   # External PX4 (SITL or HITL)
        "use_tcp": True,
        "tcp_port": int(px4_tcp_port),
        "px4_host": "127.0.0.1",
    })

    ros2_backend = ROS2Backend(
        vehicle_id=vehicle_id,
        config={
            "namespace": ros_namespace,      # e.g., "drone0", "drone1"
            "pub_sensors": False,
            "pub_graphical_sensors": True,   # publish RGB frames
            "pub_state": True,               # pose/twist/etc.
            "pub_tf": False,
            "sub_control": False,            # no control subscriptions yet
        }
    )

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

    cfg.backends = [PX4MavlinkBackend(mavlink_cfg), ros2_backend]
    cfg.graphical_sensors = [MonocularCamera("camera", config=camera_cfg)]

    drone = Multirotor(
        usd_prim_path,
        ROBOTS[robot_key],
        vehicle_id,
        pose_xyz,
        pose_quat,
        config=cfg,
    )
    return drone

class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Environment
        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # People (one circling, one walking straight then stopping)
        ctrl = CirclePersonController(radius=5.0, omega=0.3)
        _ = Person("person1", "original_male_adult_construction_05",
                   init_pos=[3.0, 0.0, 0.0], init_yaw=1.0, controller=ctrl)

        p2 = Person("person2", "original_female_adult_business_02",
                    init_pos=[2.0, 0.0, 0.0])
        p2.update_target_position([10.0, 0.0, 0.0], 1.0)

        # Two drones with distinct IDs/ports/namespaces/poses
        q0 = Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()
        q1 = Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()

        self.drone0 = make_drone(
            usd_prim_path="/World/quadrotor0",
            robot_key="Iris",              # swap to your X500 mapping if available
            vehicle_id=0,
            pose_xyz=[0.0, 0.0, 0.07],
            pose_quat=q0,
            px4_tcp_port=4560,
            ros_namespace="drone0"
        )

        self.drone1 = make_drone(
            usd_prim_path="/World/quadrotor1",
            robot_key="Iris",
            vehicle_id=1,
            pose_xyz=[2.0, 0.0, 0.07],     # small offset so both are visible
            pose_quat=q1,
            px4_tcp_port=4570,
            ros_namespace="drone1"
        )

        # Camera view
        self.pg.set_viewport_camera([6.5, 10.0, 6.5], [1.0, 0.0, 0.0])

        self.world.reset()
        self.stop_sim = False

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

