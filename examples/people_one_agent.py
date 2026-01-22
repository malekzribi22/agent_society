import os
import threading

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
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TwistStamped

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


class DroneTelemetry(Node):
    def __init__(self, namespace: str = "drone0"):
        super().__init__(f"{namespace}_telemetry")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image, f"/{namespace}/camera/color/image_raw", self._on_image, sensor_qos)
        self.create_subscription(PoseStamped, f"/{namespace}/state/pose", self._on_pose, 10)
        self.create_subscription(TwistStamped, f"/{namespace}/state/twist", self._on_twist, 10)

        self.create_timer(3.0, self._print_topics_periodic)
        self.get_logger().info(f"DroneTelemetry subscribers ready for namespace {namespace}")

    def _on_image(self, msg: Image):
        pass

    def _on_pose(self, msg: PoseStamped):
        pass

    def _on_twist(self, msg: TwistStamped):
        pass

    def _print_topics_periodic(self):
        names_types = self.get_topic_names_and_types()
        interesting = [nt for nt in names_types if "/drone0/" in nt[0] or "drone0" == nt[0].strip("/")]
        if interesting:
            self.get_logger().info("Discovered ros 2 topics under drone0:")
            for name, types in interesting:
                self.get_logger().info(f"  {name}  {types}")


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        ctrl = CirclePersonController()
        _ = Person("person1", "original_male_adult_construction_05",
                   init_pos=[3.0, 0.0, 0.0], init_yaw=1.0, controller=ctrl)

        p2 = Person("person2", "original_female_adult_business_02",
                    init_pos=[2.0, 0.0, 0.0])
        p2.update_target_position([10.0, 0.0, 0.0], 1.0)

        cfg = MultirotorConfig()

        ros2_backend = ROS2Backend(
            vehicle_id=0,
            config={
                "namespace": "drone0",
                "pub_sensors": False,
                "pub_graphical_sensors": True,
                "pub_state": True,
                "pub_tf": False,
                "sub_control": True,
            }
        )

        cfg.backends = [ros2_backend]

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

        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=cfg,
        )

        self.pg.set_viewport_camera([5.0, 9.0, 6.5], [0.0, 0.0, 0.0])
        self.world.reset()
        self.stop_sim = False

        self._start_ros_subscribers()

    def _start_ros_subscribers(self):
        def spin_ros():
            rclpy.init()
            node = DroneTelemetry(namespace="drone0")
            try:
                rclpy.spin(node)
            finally:
                node.destroy_node()
                rclpy.shutdown()

        self.ros_thread = threading.Thread(target=spin_ros, daemon=True)
        self.ros_thread.start()

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

