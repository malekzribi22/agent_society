#!/usr/bin/env python3
"""
Agent with:
- arm / takeoff / land / survey circle / survey rect  (same as before)
- follow (vision-based)
- detect (vision-only, logs detections to JSONL)

Run once:
  pip install mavsdk ultralytics opencv-python openai

ROS 2 in this terminal:
  source /opt/ros/humble/setup.bash

Env:
  export MAVSDK_URL="udpin://0.0.0.0:14540"
  export FOLLOW_CAM_TOPIC="/drone0/camera/rgb"   # adjust if needed
  export DETECT_OUT="/tmp/detections.jsonl"      # optional
"""

import asyncio, os, re, json, sys, math, time, threading
from typing import Optional, Tuple, Dict, Any

# ----------------- Optional LLM parsing -----------------
def call_llm(nl: str) -> Optional[dict]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        sys_prompt = (
            "Convert user's drone request to JSON (no extra text). Actions:\n"
            "- arm -> {\"action\":\"arm\"}\n"
            "- takeoff -> {\"action\":\"takeoff\",\"altitude\":<float>}\n"
            "- land -> {\"action\":\"land\"}\n"
            "- survey circle -> {\"action\":\"survey\",\"shape\":\"circle\",\"diameter_m\":<float>,\"altitude_m\":<float>,\"speed_mps\":<float>}\n"
            "- survey rectangle -> {\"action\":\"survey\",\"shape\":\"rect\",\"width_m\":<float>,\"height_m\":<float>,\"spacing_m\":<float>,\"altitude_m\":<float>,\"speed_mps\":<float>}\n"
            "- follow person -> {\"action\":\"follow\",\"altitude_m\":<float>}\n"
            "- detect (vision only) -> {\"action\":\"detect\"}\n"
            "If fields missing, omit them."
        )
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":nl}],
            temperature=0
        )
        txt = rsp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", txt, flags=re.S)
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

# ----------------- Offline parser (typo tolerant) -----------------
def parse_offline(nl: str) -> Optional[dict]:
    s = nl.lower().strip()

    if "arm" in s and ("takeoff" in s or "take off" in s):
        m = re.search(r"([-+]?\d*\.?\d+)", s)
        alt = float(m.group(1)) if m else 4.0
        return {"action": "combo_arm_takeoff", "altitude": alt}

    if re.search(r"\barm\b", s):           return {"action": "arm"}
    if re.search(r"\bland\b|\brtl\b", s):  return {"action": "land"}

    if re.search(r"take[\s-]?off|tkof|launch|fly up", s):
        m = re.search(r"([-+]?\d*\.?\d+)", s)
        alt = float(m.group(1)) if m else 4.0
        return {"action": "takeoff", "altitude": alt}

    if "survey" in s or "grid" in s or "pattern" in s:
        intent: Dict[str, Any] = {"action": "survey"}
        if "circle" in s or "circular" in s or "orbit" in s:
            intent["shape"] = "circle"
            m = re.search(r"(?:diam(?:eter)?|rad(?:ius)?)\s*([-+]?\d*\.?\d+)", s)
            if m:
                val = float(m.group(1))
                intent["diameter_m"] = 2.0 * val if "rad" in s else val
        else:
            intent["shape"] = "rect"
            w = re.search(r"(?:width|w)\s*([-+]?\d*\.?\d+)", s)
            h = re.search(r"(?:height|h)\s*([-+]?\d*\.?\d+)", s)
            sp = re.search(r"(?:spacing|strip|lane)\s*([-+]?\d*\.?\d+)", s)
            if w: intent["width_m"] = float(w.group(1))
            if h: intent["height_m"] = float(h.group(1))
            if sp: intent["spacing_m"] = float(sp.group(1))
        alt = re.search(r"(?:alt(?:itude)?)\s*([-+]?\d*\.?\d+)", s)
        spd = re.search(r"(?:speed|vel(?:ocity)?)\s*([-+]?\d*\.?\d+)", s)
        if alt: intent["altitude_m"] = float(alt.group(1))
        if spd: intent["speed_mps"] = float(spd.group(1))
        return intent

    # follow
    if ("follow" in s) or ("track" in s and "person" in s) or ("follow me" in s) or ("follow the pedestrian" in s):
        m = re.search(r"(?:alt|altitude)\s*([-+]?\d*\.?\d+)", s)
        alt = float(m.group(1)) if m else None
        return {"action": "follow", "altitude_m": alt}

    # detect
    if "detect" in s or "detection only" in s or "vision only" in s:
        return {"action": "detect"}

    return None

# ----------------- Geo helpers -----------------
def meters_to_deg_offsets(north_m: float, east_m: float, ref_lat_deg: float):
    lat_m = 111_320.0
    lon_m = math.cos(math.radians(ref_lat_deg)) * 111_320.0
    return (north_m / lat_m, east_m / lon_m)

# ----------------- MAVSDK high-level -----------------
async def wait_until_connected(drone, timeout_s=20.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    async for state in drone.core.connection_state():
        if state.is_connected:
            try: uuid = await drone.core.get_uuid()
            except Exception: uuid = "unknown"
            print(f"[agent] Connected (UUID: {uuid})")
            return True
        if asyncio.get_running_loop().time() > deadline:
            print("[agent] Still waiting for heartbeat...")
            return False

async def do_arm(drone):
    print("[agent] Arming..."); await drone.action.arm(); print("[agent] Armed.")

async def do_takeoff(drone, alt: float):
    try: await drone.action.set_takeoff_altitude(alt)
    except Exception as e: print("[agent] set_takeoff_altitude failed (continuing):", e)
    print(f"[agent] Takeoff to ~{alt} m..."); await drone.action.takeoff(); print("[agent] Takeoff commanded.")

async def do_land(drone):
    print("[agent] Landing..."); await drone.action.land(); print("[agent] Land commanded.")

async def get_home(drone):
    print("[agent] Waiting for home position...")
    async for hp in drone.telemetry.home():
        return hp.latitude_deg, hp.longitude_deg, hp.absolute_altitude_m

async def goto(drone, lat: float, lon: float, rel_alt: float):
    async for hp in drone.telemetry.home():
        abs_alt = hp.absolute_altitude_m + rel_alt
        break
    await drone.action.goto_location(lat, lon, abs_alt, 0.0)

# ----------------- Survey -----------------
async def run_survey_circle(drone, diameter_m: float, altitude_m: float, speed_mps: float):
    lat0, lon0, _ = await get_home(drone)
    radius = diameter_m / 2.0
    N = 8
    wps = []
    for k in range(N):
        th = 2.0 * math.pi * k / N
        north, east = radius * math.cos(th), radius * math.sin(th)
        dlat, dlon = meters_to_deg_offsets(north, east, lat0)
        wps.append((lat0 + dlat, lon0 + dlon))
    try: await do_arm(drone)
    except Exception: pass
    await asyncio.sleep(0.5)
    await do_takeoff(drone, altitude_m)
    try: await drone.action.set_current_speed(speed_mps)
    except Exception: pass
    for (la, lo) in wps:
        await goto(drone, la, lo, altitude_m)
        await asyncio.sleep(max(1.0, radius / max(0.1, speed_mps)))
    print("[agent] Circle survey complete.")

async def run_survey_rect(drone, width_m, height_m, spacing_m, altitude_m, speed_mps):
    lat0, lon0, _ = await get_home(drone)
    half_w = width_m / 2.0
    lines = max(1, int(height_m / max(1.0, spacing_m)))
    north_step = height_m / max(1, lines)
    path = []
    for i in range(lines + 1):
        north = -height_m/2.0 + i * north_step
        east_vals = ([-half_w, +half_w] if i % 2 == 0 else [+half_w, -half_w])
        for east in east_vals:
            dlat, dlon = meters_to_deg_offsets(north, east, lat0)
            path.append((lat0 + dlat, lon0 + dlon))
    try: await do_arm(drone)
    except Exception: pass
    await asyncio.sleep(0.5)
    await do_takeoff(drone, altitude_m)
    try: await drone.action.set_current_speed(speed_mps)
    except Exception: pass
    for (la, lo) in path:
        await goto(drone, la, lo, altitude_m)
        await asyncio.sleep(max(1.0, spacing_m / max(0.1, speed_mps)))
    print("[agent] Rectangle survey complete.")

# ----------------- Vision stack (shared) -----------------
latest_frame = None
latest_box   = None  # (cx, cy, w, h)
img_w = 0
img_h = 0
last_det_t = 0.0

FOLLOW_ALT_DEFAULT = 6.0
DESIRED_BOX_FRAC = 0.25

# Controller gains / limits
K_LAT = 1.0
K_FWD = 3.0
K_YAW = 1.5
MAX_V = 3.0
MAX_YAW = 60.0

# New: deadbands + minimum command
EX_DEADBAND = 0.06          # ignore small horizontal error
MIN_V_CMD = 0.3             # send at least this speed when outside deadband
NO_DET_STOP_SEC = 1.5
SHOW_PREVIEW = True

def _ros_spin_thread(topic: str):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image
    import numpy as np
    import cv2
    global latest_frame, img_w, img_h

    class ImageSub(Node):
        def __init__(self, topic):
            super().__init__("follow_vision_sub")
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )
            self.sub = self.create_subscription(Image, topic, self.cb, qos)

        def cb(self, msg: Image):
            global latest_frame, img_w, img_h
            img_w, img_h = msg.width, msg.height
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            try:
                frame = arr.reshape((msg.height, msg.width, -1))
            except Exception:
                return
            if msg.encoding == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "rgba8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif msg.encoding == "bgr8":
                pass
            else:
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
            latest_frame = frame

    rclpy.init(args=None)
    node = ImageSub(topic)
    print(f"[agent] Subscribed to camera topic: {topic}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

def _detector_loop(preview: bool, write_jsonl: Optional[str]=None, class_filter:int=0):
    """Runs YOLOv8 continuously. If write_jsonl is set, append detections there."""
    import cv2, time
    from ultralytics import YOLO
    global latest_frame, latest_box, last_det_t, img_w, img_h
    model = YOLO("yolov8n.pt")
    print("[agent] YOLOv8n loaded.")
    fps_target = 10.0
    dt = 1.0 / fps_target
    _preview_on = preview
    out_path = write_jsonl

    while True:
        t0 = time.time()
        frame = latest_frame
        best = None
        if frame is not None:
            results = model.predict(frame, imgsz=img_w or 640, conf=0.4, verbose=False)
            best_area = 0
            dets = []  # for JSON logging
            for r in results:
                for b in r.boxes:
                    cls = int(b.cls[0])
                    conf = float(b.conf[0]) if hasattr(b, "conf") else None
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
                    cx, cy = x1 + w/2.0, y1 + h/2.0
                    dets.append({"cls": cls, "conf": conf, "bbox_xywh": [x1, y1, w, h], "cx": cx, "cy": cy,
                                 "img_wh": [img_w, img_h], "t": time.time()})
                    if cls == class_filter:
                        area = w * h
                        if area > best_area:
                            best_area = area
                            best = (cx, cy, w, h)

            # JSON logging for ALL detections (not only best)
            if out_path and dets:
                try:
                    with open(out_path, "a") as f:
                        for d in dets:
                            f.write(json.dumps(d) + "\n")
                except Exception as e:
                    print("[agent] JSONL write error:", e)

        if best is not None:
            latest_box = best
            last_det_t = time.time()

        if _preview_on and frame is not None:
            disp = frame.copy()
            if latest_box is not None:
                cx, cy, w, h = latest_box
                x1, y1, x2, y2 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(disp, "person", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("vision", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                _preview_on = not _preview_on
                print(f"[agent] Preview {'ON' if _preview_on else 'OFF'}")

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

async def _ensure_offboard(drone):
    from mavsdk.offboard import VelocityBodyYawspeed
    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        print("[agent] Offboard started.")
    except Exception as e:
        print("[agent] Offboard start failed, retrying:", e)
        await asyncio.sleep(0.5)
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()

# ----------------- MODES -----------------
async def run_follow(drone, altitude_m: float):
    cam_topic = os.environ.get("FOLLOW_CAM_TOPIC", "/drone0/camera/rgb")
    # Start ROS & detector (no JSON logging in follow)
    threading.Thread(target=_ros_spin_thread, args=(cam_topic,), daemon=True).start()
    threading.Thread(target=_detector_loop, args=(True, None, 0), daemon=True).start()

    try: await do_arm(drone)
    except Exception: pass
    await asyncio.sleep(0.3)
    try: await drone.action.set_takeoff_altitude(altitude_m)
    except Exception: pass
    await drone.action.takeoff()
    print(f"[agent] Takeoff to ~{altitude_m} m")
    await asyncio.sleep(2.0)
    await _ensure_offboard(drone)

    from mavsdk.offboard import VelocityBodyYawspeed
    last_dbg = 0.0
    while True:
        now = time.time()
        box = latest_box
        if box is None or (now - last_det_t) > NO_DET_STOP_SEC:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)
            continue

        cx, cy, bw, bh = box
        if img_w <= 0 or img_h <= 0:
            await asyncio.sleep(0.02)
            continue

        # normalized horizontal error and size
        ex = (cx - img_w / 2.0) / (img_w / 2.0)   # right positive
        h_frac = bh / img_h
        dist_err = (DESIRED_BOX_FRAC - h_frac)

        # deadband + min command
        def with_deadband_and_min(cmd, sign, deadband, min_cmd):
            if abs(sign) < deadband:
                return 0.0
            mag = max(min_cmd, abs(cmd))
            return math.copysign(mag, cmd)

        vx_raw  = K_FWD * dist_err
        vy_raw  = K_LAT * (-ex)
        yaw_raw = K_YAW * (-ex) * 45.0

        vx = max(-MAX_V,  min(MAX_V,  with_deadband_and_min(vx_raw, dist_err, 0.02, MIN_V_CMD)))
        vy = max(-MAX_V,  min(MAX_V,  with_deadband_and_min(vy_raw,  -ex,     EX_DEADBAND, MIN_V_CMD)))
        yaw_rate = max(-MAX_YAW, min(MAX_YAW, with_deadband_and_min(yaw_raw, -ex, EX_DEADBAND, 10.0)))

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, 0.0, yaw_rate))

        if now - last_dbg > 0.5:
            print(f"[agent] det h={h_frac:.2f} ex={ex:.2f} -> vx={vx:.2f} vy={vy:.2f} yaw={yaw_rate:.1f}")
            last_dbg = now

        await asyncio.sleep(0.05)

def run_detect_only():
    """Vision only; writes JSONL to DETECT_OUT; no MAVSDK needed."""
    cam_topic = os.environ.get("FOLLOW_CAM_TOPIC", "/drone0/camera/rgb")
    out = os.environ.get("DETECT_OUT", "/tmp/detections.jsonl")
    print(f"[agent] DETECT mode. Writing detections to: {out}")
    threading.Thread(target=_ros_spin_thread, args=(cam_topic,), daemon=True).start()
    # detector with JSON logging
    _detector_loop(preview=True, write_jsonl=out, class_filter=0)

# ----------------- Interactive helpers -----------------
def need(val: Optional[Any], prompt: str, cast=float):
    if val is not None:
        return val
    while True:
        s = input(prompt).strip()
        try:
            return cast(s)
        except Exception:
            print("Please enter a number.")

async def handle_survey(drone, intent: dict):
    shape = intent.get("shape")
    if shape not in ("circle", "rect"):
        shape = input("[agent] Survey shape? (circle/rect) > ").strip().lower()
        if shape not in ("circle", "rect"): print("[agent] Cancelled."); return
    if shape == "circle":
        d = need(intent.get("diameter_m"), "[agent] Diameter (m)? > ")
        alt = need(intent.get("altitude_m"), "[agent] Altitude (m)? > ")
        spd = need(intent.get("speed_mps"), "[agent] Speed (m/s)? > ")
        print(f"[agent] Confirm circle: diameter={d}m alt={alt}m speed≈{spd} m/s")
        if input("[agent] Proceed? (y/n) > ").strip().lower() != "y": return
        await run_survey_circle(drone, d, alt, spd)
    else:
        w  = need(intent.get("width_m"),   "[agent] Width (m)? > ")
        h  = need(intent.get("height_m"),  "[agent] Height (m)? > ")
        sp = need(intent.get("spacing_m"), "[agent] Track spacing (m)? > ")
        alt= need(intent.get("altitude_m"),"[agent] Altitude (m)? > ")
        sd = need(intent.get("speed_mps"), "[agent] Speed (m/s)? > ")
        print(f"[agent] Confirm rect: {w}x{h}m spacing={sp}m alt={alt}m speed≈{sd} m/s")
        if input("[agent] Proceed? (y/n) > ").strip().lower() != "y": return
        await run_survey_rect(drone, w, h, sp, alt, sd)

async def handle_follow(drone, intent: dict):
    alt = intent.get("altitude_m")
    if alt is None:
        alt = need(None, "[agent] Follow altitude (m)? [6] > ", float) or 6.0
    print(f"[agent] Starting FOLLOW at ~{float(alt)} m. Ctrl+C to stop.")
    await run_follow(drone, altitude_m=float(alt))

# ----------------- Main loop -----------------
async def run_agent(connection_url="udpin://0.0.0.0:14540"):
    try:
        from mavsdk import System
    except ModuleNotFoundError:
        print("\n[agent] Install MAVSDK with: pip install mavsdk\n"); sys.exit(1)

    print(f"[agent] LLM: {'ENABLED' if os.environ.get('OPENAI_API_KEY') else 'OFF (offline parser)'}")
    print(f"[agent] Connecting to {connection_url} ...")

    drone = System()
    await drone.connect(system_address=connection_url)
    await asyncio.sleep(2.0)
    ok = await wait_until_connected(drone, timeout_s=20.0)
    while not ok:
        await asyncio.sleep(2.0)
        ok = await wait_until_connected(drone, timeout_s=4.0)

    print("\nCommands:")
    print("  arm")
    print("  takeoff 4")
    print("  land")
    print("  survey circle       (asks diameter/alt/speed)")
    print("  survey rect         (asks width/height/spacing/alt/speed)")
    print("  follow              (vision-based person following; asks altitude)")
    print("  detect              (vision-only; logs detections to JSONL)")
    print("Type 'q' to quit.\n")

    while True:
        try:
            line = input("[you] > ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "q"
        if line.lower() in ("q","quit","exit"):
            print("[agent] bye"); break

        # "detect" works even if not connected to MAVSDK, but we already connected above.
        if line.lower() in ("detect", "vision", "detect only", "detection only"):
            run_detect_only()
            # blocks until Ctrl+C; if you want it non-blocking, run in a thread
            continue

        intent = call_llm(line) or parse_offline(line)
        if not intent:
            print("[agent] I couldn’t parse that. Try: arm | takeoff 4 | land | survey circle | survey rect | follow | detect")
            continue

        try:
            act = intent.get("action")
            if act == "arm":
                await do_arm(drone)

            elif act == "takeoff":
                alt = float(intent.get("altitude", 4.0))
                try: await do_arm(drone)
                except Exception: pass
                await asyncio.sleep(0.5)
                await do_takeoff(drone, alt)

            elif act == "combo_arm_takeoff":
                alt = float(intent.get("altitude", 4.0))
                try: await do_arm(drone)
                except Exception: pass
                await asyncio.sleep(0.5)
                await do_takeoff(drone, alt)

            elif act == "land":
                await do_land(drone)

            elif act == "survey":
                await handle_survey(drone, intent)

            elif act == "follow":
                await handle_follow(drone, intent)

            elif act == "detect":
                run_detect_only()

            else:
                print(f"[agent] Unsupported intent: {intent}")

        except Exception as e:
            print("[agent] Command error:", e)

def main():
    url = os.environ.get("MAVSDK_URL", "udpin://0.0.0.0:14540")
    asyncio.run(run_agent(url))

if __name__ == "__main__":
    main()

