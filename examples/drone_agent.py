#!/usr/bin/env python3
import json
import time
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

class DroneAgent(Node):
    """
    Minimal per-drone agent with agent-to-agent comms over /swarm/intents.
    - Subscribes to its own drone's pose:   /<ns>/state/pose
    - Publishes intents to shared bus:      /swarm/intents  (std_msgs/String JSON)
    - Subscribes to peer intents on same bus
    """
    def __init__(self, ns: str, broadcast_period: float = 2.5):
        super().__init__(f"{ns}_agent")
        self.ns = ns
        self.pose = None
        self.broadcast_period = float(broadcast_period)
        self.last_broadcast = 0.0

        # Reliable QoS for coordination intents
        intent_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=16
        )

        # Subscribe to our own drone's pose (Pegasus publishes PoseStamped here)
        self.create_subscription(
            PoseStamped,
            f"/{ns}/state/pose",
            self._on_pose,
            10
        )

        # Pub/Sub intents on shared bus
        self.intent_pub = self.create_publisher(String, "/swarm/intents", intent_qos)
        self.create_subscription(String, "/swarm/intents", self._on_intent, intent_qos)

        # Periodic agent behavior
        self.timer = self.create_timer(1.0, self._tick)

        self.get_logger().info(f"Agent started for namespace '{ns}'")

    # === Callbacks ===
    def _on_pose(self, msg: PoseStamped):
        self.pose = msg

    def _on_intent(self, msg: String):
        try:
            intent = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("Received malformed intent (non-JSON)")
            return

        # Ignore our own broadcasts
        if intent.get("from") == self.ns:
            return

        # Example tiny policy:
        # If peer declares FOLLOW on person1, we avoid duplicate FOLLOW and switch posture to OBSERVE.
        if intent.get("cmd") == "FOLLOW" and intent.get("target") == "person1":
            ack = {"from": self.ns, "cmd": "OBSERVE", "about": "person1", "ts": time.time()}
            self.intent_pub.publish(String(data=json.dumps(ack)))
            self.get_logger().info(f"Peer FOLLOW detected; acknowledging OBSERVE: {ack}")

    # === Periodic tick ===
    def _tick(self):
        now = time.time()
        if self.pose and (now - self.last_broadcast) >= self.broadcast_period:
            # Broadcast a simple 'PATROL' intent with current position
            pos = [
                float(self.pose.pose.position.x),
                float(self.pose.pose.position.y),
                float(self.pose.pose.position.z),
            ]
            msg = {"from": self.ns, "cmd": "PATROL", "pos": pos, "ts": now}
            self.intent_pub.publish(String(data=json.dumps(msg)))
            self.last_broadcast = now
            self.get_logger().debug(f"Broadcast intent: {msg}")

def main():
    parser = argparse.ArgumentParser(description="Per-drone ROS 2 agent")
    parser.add_argument("--ns", required=True, help="Drone namespace (e.g., drone0 or drone1)")
    parser.add_argument("--period", type=float, default=2.5, help="Broadcast period seconds")
    args = parser.parse_args()

    rclpy.init()
    node = DroneAgent(ns=args.ns, broadcast_period=args.period)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

