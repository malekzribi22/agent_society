#!/usr/bin/env python
import os

# force discrete GPU on hybrid systems
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__VK_LAYER_NV_optimus"] = "NVIDIA_only"
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"})

import carb
from isaacsim.core.utils.extensions import enable_extension
import omni.timeline
import omni.usd
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, UsdLux, Gf

from omni.isaac.core.world import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# user switches
CITY_USD = "/home/px4/Downloads/AECO_CityMassingDemoPack/Demos/AEC/TowerDemo/CityMassingDemopack/World_CityMassingDemopack.usd"
PUBLISH_GRAPHICAL_TO_ROS2 = False     # set True later after it is smooth

# global render and app settings
s = carb.settings.get_settings()
s.set_string("/app/renderer/activeRenderer", "rtx")
s.set_bool("/rtx/enabled", True)
s.set_bool("/app/asyncRendering", True)
s.set_bool("/app/framerateLimitEnabled", True)
s.set_int("/app/framerateLimitFrequency", 60)
s.set_int("/app/runLoops/main/timeStepsPerSecond", 60)

# RTX real time path kept light
s.set_string("/rtx/rendermode", "RayTracedLighting")
s.set_bool("/rtx/denoiser/enabled", True)
s.set_bool("/rtx/post/denoiser/enabled", True)
s.set_bool("/rtx/taa/enabled", True)
s.set_int("/rtx/raytracing/spp", 1)
s.set_int("/rtx/raytracing/secondaryRays/spp", 1)
s.set_int("/rtx/raytracing/reflections/spp", 1)
s.set_int("/rtx/raytracing/shadows/spp", 1)

# keep viewport reasonable
s.set_int("/app/renderer/resolution/width", 1600)
s.set_int("/app/renderer/resolution/height", 900)

# enable ROS 2 bridge before creating the world
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# new stage
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

def enable_gpu_physx(stage):
    if stage.GetDefaultPrim() is None:
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Get(stage, "/World/physicsScene")
    if not scene:
        scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    p = scene.GetPrim()
    p.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
    p.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
    p.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")
    p.CreateAttribute("physxScene:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    p.CreateAttribute("physxScene:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(1)
    p.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.Float).Set(60.0)
    stage.SetTimeCodesPerSecond(60.0)
    return scene

def ensure_lights(stage):
    if not UsdLux.DomeLight.Get(stage, "/World/Dome"):
        dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
        dome.CreateIntensityAttr(1500.0)
    if not UsdLux.DistantLight.Get(stage, "/World/Sun"):
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(10000.0)
        sun.CreateAngleAttr(0.5)
        sun.AddTranslateOp().Set(Gf.Vec3d(0, 0, 10))

class CirclePersonController(PersonController):
    def __init__(self):
        super().__init__()
        self._radius = 8.0
        self.gamma = 0.0
        self.gamma_dot = 0.3
        self._acc = 0.0
        self._dt_target = 0.05  # update at about twenty hertz

    def update(self, dt: float):
        self._acc += dt
        if self._acc < self._dt_target:
            return
        self._acc = 0.0
        self.gamma += self.gamma_dot * self._dt_target
        self._person.update_target_position(
            [self._radius * np.cos(self.gamma), self._radius * np.sin(self.gamma), 0.0]
        )

class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # GPU PhysX and light rig
        enable_gpu_physx(stage)
        ensure_lights(stage)

        # city environment
        try:
            print("Loading city USD")
            prim_path = "/World/CityEnvironment"
            city_prim = stage.DefinePrim(prim_path, "Xform")
            city_prim.GetReferences().AddReference(CITY_USD)
            simulation_app.update()
            print("City environment loaded")
        except Exception as e:
            print(f"City load error: {e}")
            print("Falling back to Curved Gridroom")
            self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # pedestrians
        ctrl = CirclePersonController()
        _ = Person("person1", "original_male_adult_construction_05",
                   init_pos=[5.0, 5.0, 0.0], init_yaw=1.0, controller=ctrl)
        p2 = Person("person2", "original_female_adult_business_02",
                    init_pos=[8.0, 3.0, 0.0])
        p2.update_target_position([15.0, 8.0, 0.0], 1.0)

        # backends
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
                "pub_graphical_sensors": PUBLISH_GRAPHICAL_TO_ROS2,  # start False
                "pub_state": True,
                "pub_tf": False,
                "sub_control": False,
            }
        )
        cfg.backends = [PX4MavlinkBackend(mavlink_cfg), ros2_backend]

        # light camera profile for low CPU
        camera_cfg = {
            "update_rate": 15.0,
            "width": 640,
            "height": 360,
            "hfov": 90.0,
            "enable_distortion": False,
            "noise_mean": 0.0,
            "noise_stddev": 0.002,
            "rgb": True,
            "depth": False,
            "semantic": False,
        }
        cfg.graphical_sensors = [MonocularCamera("camera", config=camera_cfg)]

        # drone
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

