#!/usr/bin/env python
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni.usd
import numpy as np
from scipy.spatial.transform import Rotation

from omni.isaac.core.world import World
from isaacsim.core.utils.extensions import enable_extension

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
omni.usd.get_context().new_stage()

class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        cfg = MultirotorConfig()

        ros2_backend = ROS2Backend(
            vehicle_id=0,
            config={
                "namespace": "drone0",
                "pub_graphical_sensors": True,
                "pub_state": True,
            }
        )

        cfg.backends = [ros2_backend]

        # Test different camera configurations
        print("=== TESTING CAMERA CONFIGURATIONS ===")
        
        # Test 1: Default camera
        camera1 = MonocularCamera("camera_default", config={})
        print("Default camera config:", camera1.config)
        
        # Test 2: Camera with depth enabled
        camera2 = MonocularCamera("camera_depth", config={"depth": True})
        print("Depth camera config:", camera2.config)
        
        # Test 3: Camera with explicit parameters
        camera3 = MonocularCamera("camera_explicit", config={
            "width": 640,
            "height": 480,
            "rgb": True,
            "depth": True
        })
        print("Explicit camera config:", camera3.config)

        cfg.graphical_sensors = [camera1, camera2, camera3]

        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.5],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=cfg,
        )

        self.pg.set_viewport_camera([5.0, 9.0, 6.5], [0.0, 0.0, 0.0])
        self.world.reset()
        self.stop_sim = False

    def run(self):
        self.timeline.play()
        # Run for just 2 seconds to see the output
        for i in range(120):
            self.world.step(render=True)
        self.timeline.stop()
        simulation_app.close()

def main():
    app = PegasusApp()
    app.run()

if __name__ == "__main__":
    main()
