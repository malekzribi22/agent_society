# agent_main.py
import os, sys, asyncio, threading, time, traceback, json
from typing import Optional, TypedDict, Deque
from collections import deque

print(">>> boot agent_main.py")

# ROS2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String as StringMsg
    print(">>> ROS2 imports OK")
except Exception as e:
    print("!!! ROS2 import error:", repr(e))
    traceback.print_exc()
    sys.exit(1)

# LangGraph
try:
    from langgraph.graph import StateGraph, END
    print(">>> LangGraph imports OK")
except Exception as e:
    print("!!! LangGraph import error:", repr(e))
    traceback.print_exc()
    sys.exit(1)

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

# shared pose cache
class PoseCache:
    def __init__(self):
        self._lock = threading.Lock()
        self.last_pose: Optional[PoseStamped] = None
        self.last_time = 0.0

    def set_pose(self, msg: PoseStamped):
        with self._lock:
            self.last_pose = msg
            self.last_time = time.time()

    def altitude(self) -> float:
        with self._lock:
            if self.last_pose is None:
                return 0.0
            return float(self.last_pose.pose.position.z)

    def age(self) -> float:
        with self._lock:
            if self.last_time == 0.0:
                return 1e9
            return time.time() - self.last_time

POSE_CACHE = PoseCache()
COMMAND_PUB = None  # Will be set by the ROS node

# ROS subscriber and publisher
class AgentNode(Node):
    def __init__(self, ns: str):
        super().__init__(f"{ns}_agent_node")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # Subscribe to pose
        topic = f"/{ns}/state/pose"
        print(f">>> creating subscription to {topic} (BEST_EFFORT)")
        self.create_subscription(PoseStamped, topic, self._on_pose, qos)
        
        # Publish commands to bridge
        cmd_topic = f"/{ns}/agent/command"
        print(f">>> creating publisher to {cmd_topic}")
        self.cmd_pub = self.create_publisher(StringMsg, cmd_topic, 10)
        
        global COMMAND_PUB
        COMMAND_PUB = self.cmd_pub

    def _on_pose(self, msg: PoseStamped):
        POSE_CACHE.set_pose(msg)

# graph state
class AgentState(TypedDict, total=False):
    altitude: float
    decision: str
    steps: int
    flat_steps: int

TARGET_ALT = 2.0
MAX_STEPS = 200
FLAT_EPS = 0.01
FLAT_WINDOW = 10

# window for flatline check
alt_window: Deque[float] = deque(maxlen=FLAT_WINDOW)

def sense(state: AgentState) -> AgentState:
    alt = POSE_CACHE.altitude()
    alt_window.append(alt)
    if POSE_CACHE.age() > 1.0:
        print("[sense] no fresh pose, keeping last reading")
    print(f"[sense] altitude {alt:.2f}")
    return {"altitude": alt}

def think(state: AgentState) -> AgentState:
    alt = state.get("altitude", 0.0)
    if alt < 0.1:
        decision = "takeoff"
    elif alt < TARGET_ALT * 0.95:
        decision = "ascend"
    elif alt <= TARGET_ALT * 1.2:
        decision = "hover"
    else:
        decision = "land"
    print(f"[think] decision {decision}")
    return {"decision": decision}

async def act_async(_decision: str):
    await asyncio.sleep(0.1)

def act(state: AgentState) -> AgentState:
    decision = state.get("decision", "hover")
    print(f"[act] performing {decision}")
    
    # Publish command to bridge
    if COMMAND_PUB is not None:
        cmd = {"action": decision}
        if decision == "takeoff":
            cmd["target_altitude"] = TARGET_ALT
        msg = StringMsg()
        msg.data = json.dumps(cmd)
        COMMAND_PUB.publish(msg)
        print(f"[act] published command: {cmd}")
    else:
        print("[act] WARNING: COMMAND_PUB is None, cannot publish")
    
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(act_async(decision))
    except RuntimeError:
        pass

    steps = int(state.get("steps", 0)) + 1

    # flatline detector
    flat_steps = int(state.get("flat_steps", 0))
    if len(alt_window) == alt_window.maxlen:
        if max(alt_window) - min(alt_window) < FLAT_EPS:
            flat_steps += 1
        else:
            flat_steps = 0

    return {"steps": steps, "flat_steps": flat_steps}

def should_stop(state: AgentState) -> bool:
    steps = int(state.get("steps", 0))
    flat_steps = int(state.get("flat_steps", 0))

    if steps >= MAX_STEPS:
        print(f"[stop] reached max steps {steps}")
        return True

    # if altitude has been flat for several windows, exit with notice
    if flat_steps >= 5:
        print("[stop] altitude flat for extended time, exiting to avoid recursion")
        return True

    # stop after first stable hover
    if state.get("decision") == "hover":
        print("[stop] reached hover, stopping")
        return True

    return False

def build_app():
    g = StateGraph(AgentState)
    g.add_node("sense", sense)
    g.add_node("think", think)
    g.add_node("act", act)

    g.set_entry_point("sense")
    g.add_edge("sense", "think")
    g.add_edge("think", "act")

    def router(state: AgentState):
        return END if should_stop(state) else "sense"

    g.add_conditional_edges("act", router, {"sense": "sense", END: END})
    return g.compile()

def start_ros_spin(ns: str):
    print(">>> initializing ROS2")
    rclpy.init(args=None)
    node = AgentNode(ns)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    def spin():
        try:
            print(">>> ROS2 spin thread started")
            executor.spin()
        except Exception:
            print("!!! exception in ROS spin thread")
            traceback.print_exc()
        finally:
            print(">>> shutting down ROS2")
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    return th

async def main():
    print(">>> main() starting")
    ns = os.environ.get("DRONE_NS", "drone00")
    print(f">>> using namespace: {ns}")

    start_ros_spin(ns)

    t0 = time.time()
    while POSE_CACHE.last_pose is None and time.time() - t0 < 5.0:
        await asyncio.sleep(0.1)
    if POSE_CACHE.last_pose is None:
        print(f"warning no pose received from /{ns}/state/pose after 5 seconds, proceeding anyway")

    # Give the command publisher time to initialize
    await asyncio.sleep(0.5)
    
    app = build_app()
    cfg = {"configurable": {"thread_id": "run1"}, "recursion_limit": 500}
    print(">>> LangGraph app compiled, starting stream")

    state: AgentState = {"steps": 0, "flat_steps": 0}
    async for _ in app.astream(state, stream_mode="values", config=cfg):
        pass

    print(">>> graph finished")
    await asyncio.sleep(0.2)
    print(">>> done")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("!!! top level exception:", repr(e))
        traceback.print_exc()
        sys.exit(1)

