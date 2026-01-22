#!/usr/bin/env python3
import asyncio
import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String as StringMsg

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.telemetry import FlightMode


class CommandBridge(Node):
    def __init__(self, loop: asyncio.AbstractEventLoop, ns: str, mav_conn: str = "udp://:14540"):
        super().__init__(f"{ns}_mavsdk_bridge")
        self.loop = loop
        self.ns = ns
        self.drone = System()
        self.mav_conn = mav_conn

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub = self.create_subscription(
            StringMsg, f"/{ns}/agent/command", self.on_cmd, qos
        )

        self.get_logger().info("[bridge] start")
        self.get_logger().info(f"[bridge] connecting {self.mav_conn}")

        # connect MAVSDK on the asyncio loop
        fut = asyncio.run_coroutine_threadsafe(self._connect_and_report(), self.loop)
        fut.result()

    async def _connect_and_report(self):
        # connect and wait for heartbeat plus armability
        await self.drone.connect(system_address=self.mav_conn)

        # wait for connection
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.get_logger().info("[bridge] MAVSDK connected")
                break

        # wait until the vehicle reports a valid home and estimates
        self.get_logger().info("[bridge] waiting for global position estimate")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                self.get_logger().info("[bridge] EKF ready")
                break

    def on_cmd(self, msg: StringMsg):
        # run the async handler on the MAVSDK loop
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"[bridge] bad JSON {e}")
            return
        asyncio.run_coroutine_threadsafe(self._on_cmd_async(data), self.loop)

    async def _on_cmd_async(self, data: dict):
        try:
            action = data.get("action", "").lower()
            if action == "takeoff":
                alt = float(data.get("target_altitude", 2.0))
                await self.takeoff(alt)
            elif action == "land":
                await self.land()
            elif action == "hold" or action == "hover":
                await self.hold()
            else:
                self.get_logger().warn(f"[bridge] unknown action {action}")
        except Exception as e:
            self.get_logger().error(f"[bridge] command failed {e}")

    async def arm_if_needed(self):
        # arm only if not already armed
        async for armed in self.drone.telemetry.armed():
            if armed:
                return
            break
        try:
            await self.drone.action.arm()
            self.get_logger().info("[bridge] armed")
        except ActionError as e:
            self.get_logger().error(f"[bridge] arm failed {e}")
            raise

    async def takeoff(self, altitude: float):
        self.get_logger().info(f"[bridge] takeoff to {altitude}")
        # allow takeoff without GPS if you configured PX4 accordingly
        try:
            await self.arm_if_needed()
            await self.drone.action.set_takeoff_altitude(altitude)
            await self.drone.action.takeoff()
        except ActionError as e:
            self.get_logger().error(f"[bridge] takeoff failed {e}")
            raise

        # wait until altitude crosses about eighty percent of target
        async for pos in self.drone.telemetry.position():
            if pos.relative_altitude_m >= 0.8 * altitude:
                self.get_logger().info("[bridge] reached climb altitude")
                break

    async def land(self):
        self.get_logger().info("[bridge] land")
        try:
            await self.drone.action.land()
        except ActionError as e:
            self.get_logger().error(f"[bridge] land failed {e}")
            raise
        # wait until landed flight mode
        async for fm in self.drone.telemetry.flight_mode():
            if fm == FlightMode.LAND or fm == FlightMode.HOLD:
                self.get_logger().info("[bridge] landing sequence active")
                break

    async def hold(self):
        self.get_logger().info("[bridge] hold")
        try:
            await self.drone.action.hold()
        except ActionError as e:
            self.get_logger().error(f"[bridge] hold failed {e}")
            raise


def spin_ros(node: Node):
    rclpy.spin(node)


async def main_async():
    print("[bridge] async main")
    loop = asyncio.get_running_loop()

    rclpy.init()
    node = CommandBridge(loop, ns="drone00", mav_conn="udp://:14540")
    print(f"[bridge] listening on /{node.ns}/agent/command")

    # spin ROS in a thread so MAVSDK can keep its own loop
    t = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    t.start()

    # keep the asyncio loop alive
    try:
        while rclpy.ok():
            await asyncio.sleep(0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main_async())

