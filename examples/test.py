#!/usr/bin/env python
import os

# If you have a hybrid laptop, keep these to force dGPU
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
from pxr import Usd, UsdGeom

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

# Enable ROS 2 bridge BEFORE creating world
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# start with a fresh stage
omni.usd.get_context().new_stage()

class CirclePersonController(PersonController):
    def __init__(self):
        super().__init__()
        self._radius = 8.0
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

        # ===== Load City Massing environment with proper USD reference =====
        try:
            print("Loading City Massing environment...")
            stage = omni.usd.get_context().get_stage()
            
            # Create a reference to load the USD
            prim_path = "/World/CityEnvironment"
            city_prim = stage.DefinePrim(prim_path, "Xform")
            city_prim.GetReferences().AddReference(
                "/home/px4/Downloads/AECO_CityMassingDemoPack/Demos/AEC/TowerDemo/CityMassingDemopack/World_CityMassingDemopack.usd"
            )
            print("City environment loaded successfully!")
            simulation_app.update()
            
        except Exception as e:
            print(f"Error loading city environment: {e}")
            print("Falling back to default environment...")
            self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # ===== Pedestrians =====
        ctrl = CirclePersonController()
        _ = Person("person1", "original_male_adult_construction_05",
                   init_pos=[5.0, 5.0, 0.0], init_yaw=1.0, controller=ctrl)

        p2 = Person("person2", "original_female_adult_business_02",
                    init_pos=[8.0, 3.0, 0.0])
        p2.update_target_position([15.0, 8.0, 0.0], 1.0)

        # ===== Drone backends =====
        cfg = MultirotorConfig()

        mavlink_cfg = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": False,
            "use_tcp": True,
            "tcp_port": 4560,
            "px4_host": "127.0.0.1",
        })

        ros2_backend = ROS2Backend(
            vehicle_id=0,
            config={
                "namespace": "drone",
                "pub_sensors": False,
                "pub_graphical_sensors": True,
                "pub_state": True,
                "pub_tf": False,
                "sub_control": False,
            }
        )

        cfg.backends = [PX4MavlinkBackend(mavlink_cfg), ros2_backend]

        # camera parameters
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

        # ===== Drone =====
        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 3.0],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=cfg,
        )

        self.pg.set_viewport_camera([15.0, 20.0, 10.0], [5.0, 5.0, 0.0])
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
    try:
        app = PegasusApp()
        app.run()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
