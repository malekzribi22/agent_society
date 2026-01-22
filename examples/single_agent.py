# single_agent.py
import asyncio
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TwistStamped

from mavsdk_adapter import MavsdkAdapter

class RosSubscriber(Node):
    def __init__(self, ns: str):
        super().__init__(f"{ns}_telemetry")
        self.create_subscription(Image, f"/{ns}/camera/color/image_raw", self.on_img, 10)
        self.create_subscription(PoseStamped, f"/{ns}/state/pose", self.on_pose, 10)
        self.create_subscription(TwistStamped, f"/{ns}/state/twist", self.on_twist, 10)

    def on_img(self, msg: Image):
        pass

    def on_pose(self, msg: PoseStamped):
        pass

    def on_twist(self, msg: TwistStamped):
        pass

async def try_connect(url: str, timeout_s: float = 10.0) -> MavsdkAdapter:
    print(f"[connect] trying {url}")
    ctrl = MavsdkAdapter(system_url=url)
    try:
        await asyncio.wait_for(ctrl.connect(), timeout=timeout_s)
        print(f"[connect] connected on {url}")
        return ctrl
    except asyncio.TimeoutError:
        print(f"[connect] timeout waiting for heartbeat on {url}")
        return None
    except Exception as e:
        print(f"[connect] error on {url}: {e}")
        return None

async def fly_vertical_slice(ns: str):
    rclpy.init()
    node = RosSubscriber(ns)

    ctrl = None
    # try the usual PX4 UDP first
    for url in ["udp://:14540", "udp://:14550", "tcp://127.0.0.1:4560"]:
        ctrl = await try_connect(url)
        if ctrl:
            break
    if not ctrl:
        print("[connect] could not connect on any endpoint, exiting")
        rclpy.shutdown()
        sys.exit(1)

    print("[flight] arming and starting offboard")
    await ctrl.takeoff_slow(climb_rate=-0.5, seconds=3.0)
    print("[flight] hover")
    await ctrl.move_vel(0.0, 0.0, 0.0, 0.0, duration_s=3.0)
    print("[flight] landing")
    await ctrl.land_slow(descend_rate=0.5, seconds=3.0)
    print("[flight] done")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    asyncio.run(fly_vertical_slice(ns="drone0"))

