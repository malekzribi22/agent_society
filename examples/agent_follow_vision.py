#!/usr/bin/env python3
import os, sys, time, math, threading, asyncio, json
import numpy as np
from typing import Optional

# -------- MAVSDK ----------
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed

# -------- ROS 2 / Image ----
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import cv2

# -------- YOLO (Ultralytics) ----
try:
    from ultralytics import YOLO
except Exception as e:
    print("[agent] Ultralytics not available. Install with: pip install ultralytics")
    raise

# ------------- Params (tweak as needed) -------------
CAMERA_TOPIC = os.environ.get("FOLLOW_CAM_TOPIC", "/drone0/camera/rgb")  # adjust to your actual topic
TARGET_CLASS = 0   # YOLO 'person' class in COCO
TARGET_ALT_M = float(os.environ.get("FOLLOW_ALT_M", "6.0"))  # hold ~6m alt (takeoff)
DESIRED_BOX_FRAC = 0.25  # desired bbox height fraction of image (controls distance)
# Control gains
K_LAT = 1.0       # lateral speed gain (m/s per normalized pixel error)
K_FWD = 3.0       # forward/back gain to regulate distance via box height
K_YAW = 1.5       # yaw rate gain (deg/s per normalized pixel error)
MAX_V   = 3.0     # m/s limit
MAX_YAW = 60.0    # deg/s limit
NO_DET_STOP_SEC = 1.5  # if no detection for this time, hover

# ------------- Shared state -------------
latest_frame = None
latest_box   = None  # (cx, cy, w, h) in pixel coords
img_w = 0
img_h = 0
last_det_t = 0.0

# ------------- ROS2 subscriber -------------
class ImageSub(Node):
    def __init__(self, topic):
        super().__init__("follow_vision_sub")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.sub = self.create_subscription(Image, topic, self.cb, qos)

    def cb(self, msg: Image):
        global latest_frame, img_w, img_h
        # Convert ROS2 Image (assume rgb8) -> OpenCV BGR
        img_w, img_h = msg.width, msg.height
        if msg.encoding not in ("rgb8", "bgr8"):
            # Try to interpret as raw
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            # best effort reshape; may need cv_bridge for other encodings
            try:
                frame = arr.reshape((msg.height, msg.width, 3))
            except Exception:
                return
        else:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = arr.reshape((msg.height, msg.width, 3))
            if msg.encoding == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        latest_frame = frame

def ros_spin_thread():
    rclpy.init(args=None)
    node = ImageSub(CAMERA_TOPIC)
    print(f"[agent] Subscribed to camera topic: {CAMERA_TOPIC}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

# ------------- YOLO inference thread -------------
def detector_thread():
    global latest_frame, latest_box, last_det_t, img_w, img_h
    # Download model on first run; uses small model for speed
    model = YOLO("yolov8n.pt")
    print("[agent] YOLOv8n loaded.")
    while True:
        frame = latest_frame
        if frame is None:
            time.sleep(0.02)
            continue

        # Run detection at ~10 Hz to save CPU
        t0 = time.time()
        results = model.predict(frame, imgsz=img_w or 640, conf=0.4, verbose=False)
        best = None
        best_area = 0
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                if cls != TARGET_CLASS:
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                area = w * h
                if area > best_area:
                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0
                    best = (cx, cy, w, h)
                    best_area = area

        if best is not None:
            latest_box = best
            last_det_t = time.time()
        time.sleep(max(0.0, 0.1 - (time.time() - t0)))

# ------------- MAVSDK helpers -------------
async def wait_connected(drone: System):
    # Wait for connection & home
    async for s in drone.core.connection_state():
        if s.is_connected:
            break
    print("[agent] MAVSDK connected.")
    # Arm & takeoff to target alt
    try:
        await drone.action.arm()
    except Exception:
        pass
    await asyncio.sleep(0.3)
    try:
        await drone.action.set_takeoff_altitude(TARGET_ALT_M)
    except Exception:
        pass
    await drone.action.takeoff()
    print(f"[agent] Takeoff to ~{TARGET_ALT_M} m")
    await asyncio.sleep(2.0)

async def ensure_offboard(drone: System):
    # Start offboard with zero velocities
    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        print("[agent] Offboard started.")
    except Exception as e:
        print("[agent] Offboard start failed, retrying:", e)
        await asyncio.sleep(0.5)
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ------------- Control loop -------------
async def control_loop(drone: System):
    global latest_box, img_w, img_h, last_det_t
    target_h_frac = DESIRED_BOX_FRAC

    while True:
        now = time.time()
        box = latest_box

        if box is None or (now - last_det_t) > NO_DET_STOP_SEC:
            # No detection recently: hover (zero velocities)
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)
            continue

        cx, cy, bw, bh = box
        if img_w <= 0 or img_h <= 0:
            await asyncio.sleep(0.02)
            continue

        # Pixel errors normalized to [-1, 1] (0 = centered)
        ex = (cx - img_w / 2.0) / (img_w / 2.0)   # right positive
        ey = (cy - img_h / 2.0) / (img_h / 2.0)   # down positive (we mostly ignore vertical in image)
        h_frac = bh / img_h                       # bbox height fraction

        # Control logic:
        # - Lateral speed from ex (strafe left/right): vy
        # - Forward speed from distance error (bbox size): vx (positive forward)
        # - Yaw rate from ex to turn towards target
        dist_err = (target_h_frac - h_frac)       # + if too far (wants to move forward)
        vx = clamp(K_FWD * dist_err, -MAX_V, MAX_V)           # forward/back
        vy = clamp(K_LAT * (-ex),    -MAX_V, MAX_V)           # strafe to center
        yaw_rate = clamp(K_YAW * (-ex) * 45.0, -MAX_YAW, MAX_YAW)  # deg/s

        # (Optionally) small vertical correction from ey if you want to keep target vertically centered.
        vz = 0.0

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, vz, yaw_rate))
        await asyncio.sleep(0.05)

# ------------- Main -------------
async def main():
    # Start ROS2 subscriber thread
    t_ros = threading.Thread(target=ros_spin_thread, daemon=True)
    t_ros.start()

    # Start detector thread
    t_det = threading.Thread(target=detector_thread, daemon=True)
    t_det.start()

    # MAVSDK
    url = os.environ.get("MAVSDK_URL", "udp://:14540")  # LISTEN on 14540
    drone = System()
    print(f"[agent] Connecting MAVSDK {url} ...")
    await drone.connect(system_address=url)
    await wait_connected(drone)
    await ensure_offboard(drone)

    print("\n[agent] FOLLOW mode active. Ctrl+C to stop.\n")
    try:
        await control_loop(drone)
    except asyncio.CancelledError:
        pass
    finally:
        print("[agent] Stopping offboard & landing…")
        try:
            await drone.offboard.stop()
        except Exception:
            pass
        try:
            await drone.action.land()
        except Exception:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[agent] bye")

