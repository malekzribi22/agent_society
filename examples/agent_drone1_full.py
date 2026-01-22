#!/usr/bin/env python3
"""
MAVSDK agent (LLM-first, strong rule fallback) with robust Offboard + multi-step parsing.

Improvements vs last version:
- Offboard priming + retry to avoid NO_SETPOINT_SET / TIMEOUT.
- After mission, we do action.hold() and small settle to re-enter offboard cleanly.
- Faster, tolerant takeoff wait.
- Much better fuzzy parsing (typos & freeform multi-step).
- Executes steps strictly in order; each blocks until done.

NOTE: This copy is dedicated to drone1 (MAVSDK port 14541). No need to set
MAVSDK_URL; just run `python3 agent_drone1_full.py`. Optional: set
OPENAI_API_KEY to enable LLM parsing.
"""

import asyncio, os, re, sys, math, difflib, json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------- Optional LLM ----------
USE_LLM = False
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        USE_LLM = True
except Exception:
    USE_LLM = False

# ---------- Globals ----------
ARMED_CACHE = False

# ---------- Utilities ----------
async def wait_until(predicate_coro, timeout_s: float, interval_s: float = 0.1) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        try:
            if await predicate_coro():
                return True
        except Exception:
            pass
        await asyncio.sleep(interval_s)
    return False

# ---------- MAVSDK helpers ----------
async def connect(url: str):
    from mavsdk import System
    drone = System()
    print(f"[agent] Connecting to {url} ...")
    await drone.connect(system_address=url)
    async for st in drone.core.connection_state():
        if st.is_connected:
            try:
                uuid = await drone.core.get_uuid()
            except Exception:
                uuid = "unknown"
            print(f"[agent] MAVSDK connected. UUID: {uuid}")
            break
    asyncio.create_task(_armed_tracker(drone))
    return drone

async def _armed_tracker(drone):
    global ARMED_CACHE
    try:
        async for a in drone.telemetry.armed():
            ARMED_CACHE = bool(a)
    except Exception:
        pass

async def _get_position_once(drone):
    async for p in drone.telemetry.position():
        return p

async def _get_in_air_once(drone) -> Optional[bool]:
    async for s in drone.telemetry.in_air():
        return bool(s)

async def _get_velocity_ned_once(drone):
    async for v in drone.telemetry.velocity_ned():
        return v

async def ensure_armed(drone):
    global ARMED_CACHE
    if ARMED_CACHE:
        return
    try:
        async for a in drone.telemetry.armed():
            if a:
                ARMED_CACHE = True
                return
            break
    except Exception:
        pass
    try:
        print("[agent] Arming...")
        await drone.action.arm()
        ARMED_CACHE = True
        print("[agent] Armed.")
    except Exception as e:
        print("[agent] arm():", e)

# ---------- Robust Offboard manager ----------
class OffboardManager:
    def __init__(self, drone):
        self.drone = drone

    async def start(self) -> bool:
        """
        Prime with a few zero setpoints, then start offboard.
        Retry if needed.
        """
        from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed)
        # Prime 8–10 setpoints (PX4 requirement)
        try:
            for _ in range(10):
                await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0,0.0,0.0,0.0))
                await asyncio.sleep(0.05)
            await self.drone.offboard.start()
            return True
        except OffboardError as e:
            # Retry once with a longer prime if NO_SETPOINT_SET or TIMEOUT
            print(f"[agent] offboard.start() failed ({e}); retrying...")
            try:
                for _ in range(20):
                    await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0,0.0,0.0,0.0))
                    await asyncio.sleep(0.05)
                await self.drone.offboard.start()
                return True
            except Exception as e2:
                print(f"[agent] offboard.start() retry failed: {e2}")
                return False
        except Exception as e:
            print(f"[agent] offboard.init failed: {e}")
            return False

    async def stop(self):
        from mavsdk.offboard import VelocityBodyYawspeed
        try:
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0,0.0,0.0,0.0))
            await self.drone.offboard.stop()
        except Exception:
            pass

OFFBOARD: Optional[OffboardManager] = None

async def ensure_offboard(drone) -> bool:
    global OFFBOARD
    if OFFBOARD is None:
        OFFBOARD = OffboardManager(drone)
    return await OFFBOARD.start()

# ---------- Actions ----------
async def do_takeoff(drone, alt: float):
    """Arm if needed, takeoff, and wait fast/robust until airborne & near target."""
    await ensure_armed(drone)
    try:
        await drone.action.set_takeoff_altitude(alt)
    except Exception as e:
        print("[agent] set_takeoff_altitude:", e)
    print(f"[agent] Takeoff to ~{alt:.1f} m")
    await drone.action.takeoff()

    TOL = 0.5
    MAX_WAIT = 18.0
    SETTLE_V_DOWN = 0.18
    SETTLE_TIME = 1.6

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    settled_since = None

    while True:
        if loop.time() - t0 > MAX_WAIT:
            print("[agent] Takeoff: timeout waiting; continuing.")
            break

        pos = await _get_position_once(drone)
        in_air = await _get_in_air_once(drone)
        rel = getattr(pos, "relative_altitude_m", None)

        if (rel is not None) and (rel >= alt - TOL) and in_air:
            break

        v = await _get_velocity_ned_once(drone)
        v_up = -v.down_m_s
        if in_air and abs(v_up) <= SETTLE_V_DOWN:
            if settled_since is None:
                settled_since = loop.time()
            elif loop.time() - settled_since >= SETTLE_TIME:
                break
        else:
            settled_since = None

        await asyncio.sleep(0.15)

async def do_land(drone):
    """Land and wait until on ground."""
    print("[agent] Land")
    await drone.action.land()
    async def _on_ground():
        in_air = await _get_in_air_once(drone)
        return in_air is False
    ok = await wait_until(_on_ground, timeout_s=60.0, interval_s=0.2)
    if not ok:
        print("[agent] Land: timeout waiting for disarm/on-ground.")

async def do_rtl(drone):
    print("[agent] Return-To-Launch (RTL)")
    await drone.action.return_to_launch()

async def rotate_yaw(deg: float, drone, yaw_rate_deg_s: float = 30.0):
    """deg>0 clockwise, deg<0 CCW."""
    from mavsdk.offboard import VelocityBodyYawspeed
    await ensure_armed(drone)
    if not await ensure_offboard(drone):
        print("[agent] cannot enter offboard; aborting rotate")
        return
    rate = max(5.0, float(yaw_rate_deg_s))
    yawspeed = rate if deg >= 0 else -rate
    duration = abs(float(deg)) / rate
    print(f"[agent] Rotate: {deg:.1f}° at {rate:.1f}°/s (~{duration:.1f}s)")
    t0 = asyncio.get_event_loop().time()
    try:
        while asyncio.get_event_loop().time() - t0 < duration:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, yawspeed))
            await asyncio.sleep(0.1)
    finally:
        await OFFBOARD.stop()
    print("[agent] Rotate: done")

async def move_body(drone, fwd: float, right: float, dur: float, down: float = 0.0):
    """Body-frame move for duration."""
    from mavsdk.offboard import VelocityBodyYawspeed
    await ensure_armed(drone)
    if not await ensure_offboard(drone):
        print("[agent] cannot enter offboard; aborting move")
        return
    print(f"[agent] Move: fwd={fwd:.2f} m/s, right={right:.2f} m/s, down={down:.2f} m/s for {dur:.2f}s")
    t0 = asyncio.get_event_loop().time()
    try:
        while asyncio.get_event_loop().time() - t0 < dur:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(fwd, right, down, 0.0))
            await asyncio.sleep(0.1)
    finally:
        await OFFBOARD.stop()
    print("[agent] Move: done")

async def move_distance(drone, direction: str, meters: float, speed_m_s: float = 1.0):
    meters = abs(float(meters))
    speed  = max(0.1, float(speed_m_s))
    dur    = meters / speed
    if direction in ("front","forward"): fwd, right = speed, 0.0
    elif direction in ("back","backward"): fwd, right = -speed, 0.0
    elif direction == "right": fwd, right = 0.0, speed
    elif direction == "left":  fwd, right = 0.0, -speed
    else:
        print(f"[agent] unknown direction: {direction}"); return
    await move_body(drone, fwd, right, dur)

async def fly_at_altitude(drone, target_rel_alt_m: float, climb_speed_m_s: float = 1.0):
    from mavsdk.offboard import VelocityBodyYawspeed
    tol = 0.25
    await ensure_armed(drone)
    if not await ensure_offboard(drone):
        print("[agent] cannot enter offboard; aborting altitude change")
        return
    print(f"[agent] Fly-at-alt: target {target_rel_alt_m:.2f} m")
    try:
        while True:
            pos = await _get_position_once(drone)
            rel = getattr(pos, "relative_altitude_m", None)
            if rel is None:
                await asyncio.sleep(0.1); continue
            err = target_rel_alt_m - rel
            if abs(err) <= tol:
                print(f"[agent] Altitude reached ~{rel:.2f} m")
                break
            down = -climb_speed_m_s if err > 0 else climb_speed_m_s  # down>0 = descend
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, down, 0.0))
            await asyncio.sleep(0.1)
    finally:
        await OFFBOARD.stop()

# ---------- Survey helpers ----------
def meters_to_deg_lat(m: float) -> float: return m / 111320.0
def meters_to_deg_lon(m: float, lat_deg: float) -> float: return m / (111320.0 * max(0.1, math.cos(math.radians(lat_deg))))

@dataclass
class CircleSpec:
    diameter_m: float
    altitude_m: float
    speed_m_s: float
    points: int = 24

@dataclass
class RectSpec:
    width_m: float
    height_m: float
    spacing_m: float
    altitude_m: float
    speed_m_s: float

async def get_home(drone):
    async for pos in drone.telemetry.position():
        return pos.latitude_deg, pos.longitude_deg, pos.absolute_altitude_m, pos.relative_altitude_m

async def fly_mission(drone, items: List[Dict[str, float]]):
    from mavsdk import mission as M
    MissionItem, MissionPlan = M.MissionItem, M.MissionPlan
    mis = []
    for it in items:
        mis.append(
            MissionItem(
                it["lat"], it["lon"], it["rel_alt"], it["speed"],
                True,                         # is_fly_through
                float('nan'),                 # gimbal_pitch_deg
                float('nan'),                 # gimbal_yaw_deg
                M.MissionItem.CameraAction.NONE,
                0.0, 0.0,                     # loiter, photo_interval
                2.0,                          # acceptance_radius
                float('nan'),                 # yaw_deg
                0.0,                          # camera_photo_distance
                M.MissionItem.VehicleAction.NONE
            )
        )
    plan = MissionPlan(mis)
    try:
        await drone.mission.clear_mission()
    except Exception:
        pass
    await drone.mission.upload_mission(plan)
    print(f"[agent] Mission uploaded: {len(mis)} items")
    await ensure_armed(drone)
    await drone.mission.start_mission()
    print("[agent] Mission started")
    async for p in drone.mission.mission_progress():
        print(f"[agent] progress {p.current}/{p.total}", end="\r", flush=True)
        if p.total > 0 and p.current >= p.total:
            break
    print("\n[agent] Mission finished")
    # Exit mission mode cleanly so Offboard can start next
    try:
        await drone.action.hold()
        await asyncio.sleep(0.8)
    except Exception:
        pass

async def survey_circle(drone, spec: CircleSpec):
    lat0, lon0, *_ = await get_home(drone)
    r = spec.diameter_m / 2.0
    items = []
    for k in range(spec.points):
        ang = (2*math.pi) * (k/spec.points)
        dx, dy = r*math.cos(ang), r*math.sin(ang)
        lat = lat0 + meters_to_deg_lat(dy)
        lon = lon0 + meters_to_deg_lon(dx, lat0)
        items.append({"lat":lat, "lon":lon, "rel_alt":spec.altitude_m, "speed":spec.speed_m_s})
    items.append(items[0])
    await fly_mission(drone, items)

async def survey_rect(drone, spec: RectSpec):
    lat0, lon0, *_ = await get_home(drone)
    half_w, half_h = spec.width_m/2.0, spec.height_m/2.0
    y_lines, y = [], -half_h
    while y <= half_h + 1e-6:
        y_lines.append(y)
        y += max(1.0, spec.spacing_m)
    left  = lon0 + meters_to_deg_lon(-half_w, lat0)
    right = lon0 + meters_to_deg_lon( half_w, lat0)
    items = []
    for i, yy in enumerate(y_lines):
        lat = lat0 + meters_to_deg_lat(yy)
        if i % 2 == 0:
            items += [
                {"lat":lat,"lon":left,"rel_alt":spec.altitude_m,"speed":spec.speed_m_s},
                {"lat":lat,"lon":right,"rel_alt":spec.altitude_m,"speed":spec.speed_m_s}
            ]
        else:
            items += [
                {"lat":lat,"lon":right,"rel_alt":spec.altitude_m,"speed":spec.speed_m_s},
                {"lat":lat,"lon":left,"rel_alt":spec.altitude_m,"speed":spec.speed_m_s}
            ]
    await fly_mission(drone, items)

# ---------- Parsing (LLM-first + strong rules) ----------
KNOWN_TOKENS = [
    "arm","takeoff","take","off","land","rtl","return","home",
    "rotate","yaw","left","right","cw","ccw","clockwise","counter","counterclockwise",
    "go","front","forward","back","backward","move",
    "fly","at","to","alt","altitude","meters","meter","m","for","seconds","s","speed",
    "survey","circle","rect","rectangle","box",
    "tilt","up","down"
]
TOKEN_MAP = {
    # typos & variants
    "tehn":"then", "then":"then",
    "cirlce":"circle", "circel":"circle",
    "bakc":"back", "bac":"back", "bak":"back",
    "mov":"move", "frwd":"forward", "forwad":"forward", "fornt":"front", "frnot":"front",
    "rigth":"right", "lef":"left",
    "rtoate":"rotate", "rtoat":"rotate",
    "yawn":"yaw", "yawh":"yaw",
    "metres":"meters", "metre":"meter",
}

def fuzzy_normalize(text: str) -> List[str]:
    # Split by 'then' / ',' / 'and' as step separators, but also allow no separator patterns.
    raw = re.split(r"\bthen\b|;|,|\band\b", text, flags=re.I)
    parts = [t.strip().lower() for t in raw if t.strip()]
    fixed = []
    for s in parts:
        toks = re.findall(r"[a-zA-Z]+|\d+\.?\d*|[^\s]", s)
        out = []
        for t in toks:
            if t in TOKEN_MAP:
                out.append(TOKEN_MAP[t]); continue
            if re.fullmatch(r"[a-zA-Z]+", t) and t not in KNOWN_TOKENS:
                cand = difflib.get_close_matches(t, KNOWN_TOKENS, n=1, cutoff=0.78)
                out.append(cand[0] if cand else t)
            else:
                out.append(t)
        fixed.append(" ".join(out))
    return fixed

def parse_numbers(s: str):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", s)]

def _parse_go(s: str) -> Optional[Dict[str, Any]]:
    # accept "go/move <dir> <n>", "<dir> <n>", "back 3 rotate 30" (no comma)
    m_dir = re.search(r"\b(?:go|move)?\s*(front|forward|back|backward|left|right)\b", s)
    if not m_dir:
        # pattern like "forward 3"
        m_dir = re.search(r"\b(front|forward|back|backward|left|right)\s+\d", s)
    if not m_dir:
        return None
    direction = m_dir.group(1) if isinstance(m_dir.group, type(re.Match.group)) else m_dir[1]
    meters = None
    m_m = re.search(r"\b(\d+\.?\d*)\s*(?:m|meters?|meter)\b", s)
    if m_m:
        meters = float(m_m.group(1))
    else:
        m_bare = re.search(r"\b(\d+\.?\d*)\b(?!\s*s)", s)
        if m_bare:
            meters = float(m_bare.group(1))
    speed = None
    m_speed = re.search(r"(?:at|speed)\s+(\d+\.?\d*)\s*(?:m/s|mps)?", s)
    if m_speed:
        speed = float(m_speed.group(1))
    duration = None
    m_t = re.search(r"(?:for)\s+(\d+\.?\d*)\s*(?:s|sec|secs|seconds?)", s)
    if m_t:
        duration = float(m_t.group(1))
    return {"cmd":"go", "dir":direction, "meters":meters, "speed":speed, "duration":duration}

def rule_parse_one(seg: str) -> Optional[Dict[str, Any]]:
    s = seg.strip()
    if not s: return None

    if s in ("q","quit","exit"): return {"cmd":"quit"}
    if re.search(r"\barm\b", s): return {"cmd":"arm"}

    if re.search(r"\b(takeoff|take off)\b", s):
        nums = parse_numbers(s); alt = nums[0] if nums else 3.0
        return {"cmd":"takeoff", "alt":float(alt)}

    if re.search(r"\bland\b", s): return {"cmd":"land"}
    if re.search(r"\b(rtl|return( to)? (home|launch))\b", s):
        return {"cmd":"rtl"}

    # rotate / yaw (accept "rotate 45", "rotate right 45", "turn left 30", etc.)
    if any(k in s for k in ("rotate","yaw","turn")):
        nums = parse_numbers(s)
        deg = float(nums[0]) if nums else 0.0
        if "ccw" in s or "counter" in s or "left" in s:
            deg = -abs(deg) if deg != 0 else -30.0
        elif "cw" in s or "clockwise" in s or "right" in s:
            deg =  abs(deg) if deg != 0 else  30.0
        if "rotate" in s and not any(x in s for x in ("ccw","counter","left","right","cw","clockwise")):
            deg = deg if deg != 0 else 30.0
        rate = None
        m = re.search(r"(?:rate|at)\s+(\d+\.?\d*)\s*(?:deg/s|deg|dps)?", s)
        if m: rate = float(m.group(1))
        return {"cmd":"rotate", "deg":deg, "rate":rate}

    # fly at altitude (fly at 3 | alt 3 | fly to 3)
    if re.search(r"\bfly\s+at\s+\d", s) or re.search(r"\b(alt|altitude)\s+\d", s) or re.fullmatch(r"fly\s+(to\s+)?\d+\.?\d*", s):
        nums = parse_numbers(s); alt = nums[0] if nums else 3.0
        return {"cmd":"fly_at", "alt":float(alt), "speed":1.0}

    # "fly 3 meters" => go front 3
    if re.search(r"\bfly\s+(\d+\.?\d*)\s*(m|meter|meters)?\b", s):
        nums = parse_numbers(s)
        if nums:
            return {"cmd":"go", "dir":"front", "meters":float(nums[0]), "speed":1.0}

    go = _parse_go(s)
    if go: return go

    if "tilt" in s:
        direction = "forward" if "forward" in s else "back" if "back" in s else "forward"
        nums = parse_numbers(s)
        meters = nums[0] if nums else 1.0
        return {"cmd":"tilt", "dir":direction, "meters":meters}

    if "survey" in s and "circle" in s:
        nums = parse_numbers(s); out = {"cmd":"survey_circle"}
        if len(nums)>=1: out["diameter_m"]=float(nums[0])
        if len(nums)>=2: out["altitude_m"]=float(nums[1])
        if len(nums)>=3: out["speed_m_s"]=float(nums[2])
        return out

    if "survey" in s and any(k in s for k in ("rect","rectangle","box")):
        nums = parse_numbers(s)
        keys=["width_m","height_m","spacing_m","altitude_m","speed_m_s"]
        out={"cmd":"survey_rect"}
        for i,k in enumerate(keys):
            if i<len(nums): out[k]=float(nums[i])
        return out

    return None

def rule_parse_multi(text: str) -> List[Dict[str, Any]]:
    fixed_parts = fuzzy_normalize(text)
    cmds = []
    for seg in fixed_parts:
        # Also split again by implicit adjacency like "back 3 rotate 30"
        implicit = re.split(r"\b(?=front|forward|back|backward|left|right|rotate|turn|yaw|fly|takeoff|take off|land|rtl)\b", seg)
        for chunk in implicit:
            p = rule_parse_one(chunk.strip())
            if p: cmds.append(p)
    return cmds or [{"cmd":None}]

LLM_SYS = (
 "You are a command planner for a MAVSDK drone agent.\n"
 "Output ONLY JSON (no text). When multiple actions are requested, output a JSON array of command objects in order.\n"
 "Supported commands and fields:\n"
 "  {\"cmd\":\"arm\"}\n"
 "  {\"cmd\":\"takeoff\",\"altitude_m\":<number>}\n"
 "  {\"cmd\":\"land\"}\n"
 "  {\"cmd\":\"rtl\"}\n"
 "  {\"cmd\":\"rotate\",\"deg\":<number>,\"rate\":<number optional>}\n"
 "  {\"cmd\":\"fly_at\",\"altitude_m\":<number>,\"speed\":<number optional>}\n"
 "  {\"cmd\":\"go\",\"dir\":\"front|back|left|right\",\"meters\":<number optional>,\"duration\":<number optional>,\"speed\":<number optional>}\n"
 "  {\"cmd\":\"tilt\",\"dir\":\"forward|back\",\"meters\":<number optional>}\n"
 "  {\"cmd\":\"survey_circle\",\"diameter_m\":<n>,\"altitude_m\":<n>,\"speed_m_s\":<n>}\n"
 "  {\"cmd\":\"survey_rect\",\"width_m\":<n>,\"height_m\":<n>,\"spacing_m\":<n>,\"altitude_m\":<n>,\"speed_m_s\":<n>}\n"
 "Defaults: takeoff.altitude_m=3; rotate.rate=30; fly_at.speed=1; go.speed=1; tilt.meters=1.\n"
 "Normalize variants like 'fly 3 meters' => {cmd:'go',dir:'front',meters:3} and 'turn right 45' => {cmd:'rotate',deg:45}.\n"
)

def llm_parse_multi(text: str) -> List[Dict[str, Any]]:
    if not USE_LLM:
        return rule_parse_multi(text)
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":LLM_SYS},
                  {"role":"user","content":f"{text}\nReturn JSON or JSON array ONLY."}],
        temperature=0.0, max_tokens=600,
    )
    raw = resp.choices[0].message.content.strip()
    m = re.search(r"\[.*\]|\{.*\}", raw, flags=re.S)
    if not m:
        return rule_parse_multi(text)
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict): return [obj]
        if isinstance(obj, list): return [x for x in obj if isinstance(x, dict)]
        return rule_parse_multi(text)
    except Exception:
        return rule_parse_multi(text)

def ask_f(prompt: str, default: float) -> float:
    s = input(f"{prompt} [{default}]: ").strip()
    try:
        return float(s) if s else default
    except:
        return default

# ---------- Tool execution ----------
async def exec_one(drone, p: Dict[str, Any]):
    cmd = p.get("cmd")

    if cmd == "arm":
        await ensure_armed(drone)

    elif cmd == "takeoff":
        alt = float(p.get("alt") or p.get("altitude_m") or 3.0)
        await do_takeoff(drone, alt)

    elif cmd == "land":
        await do_land(drone)

    elif cmd == "rtl":
        await do_rtl(drone)

    elif cmd == "rotate":
        deg  = float(p.get("deg", 0.0))
        rate = float(p.get("rate", 30.0)) if p.get("rate") is not None else 30.0
        await rotate_yaw(deg, drone, yaw_rate_deg_s=rate)

    elif cmd == "go":
        direction = p.get("dir")
        meters    = p.get("meters")
        speed     = p.get("speed")
        duration  = p.get("duration")
        if meters is not None:
            spd = float(speed) if speed is not None else 1.0
            await move_distance(drone, direction, float(meters), spd)
        elif duration is not None:
            spd = float(speed) if speed is not None else 1.0
            if direction in ("front","forward"): fwd, right = spd, 0.0
            elif direction in ("back","backward"): fwd, right = -spd, 0.0
            elif direction == "right": fwd, right = 0.0, spd
            elif direction == "left":  fwd, right = 0.0, -spd
            else:
                print("[agent] unknown direction for go"); return
            await move_body(drone, fwd, right, float(duration))
        else:
            spd = float(speed) if speed is not None else 1.0
            await move_distance(drone, direction, 2.0, spd)

    elif cmd == "fly_at":
        alt = float(p.get("alt") or p.get("altitude_m") or 3.0)
        spd = float(p.get("speed", 1.0))
        await fly_at_altitude(drone, alt, spd)

    elif cmd == "tilt":
        direction = p.get("dir","forward")
        meters = float(p.get("meters", 1.0))
        meters = max(0.2, meters)
        spd = 0.5
        if direction.startswith("back"):
            await move_distance(drone, "back", meters, spd)
        else:
            await move_distance(drone, "front", meters, spd)

    elif cmd == "survey_circle":
        diameter = float(p.get("diameter_m") or ask_f("Diameter (m)", 20.0))
        alt      = float(p.get("altitude_m") or ask_f("Altitude (m)", 10.0))
        speed    = float(p.get("speed_m_s")  or ask_f("Speed (m/s)", 3.0))
        await survey_circle(drone, CircleSpec(diameter, alt, speed))

    elif cmd == "survey_rect":
        width   = float(p.get("width_m")    or ask_f("Width (m)",   40.0))
        height  = float(p.get("height_m")   or ask_f("Height (m)",  30.0))
        spacing = float(p.get("spacing_m")  or ask_f("Spacing (m)",  6.0))
        alt     = float(p.get("altitude_m") or ask_f("Altitude (m)",12.0))
        speed   = float(p.get("speed_m_s")  or ask_f("Speed (m/s)",  3.0))
        await survey_rect(drone, RectSpec(width, height, spacing, alt, speed))

    elif cmd == "quit":
        print("[agent] bye")
        raise SystemExit

    else:
        print("[agent] Command not implemented:", cmd)

# ---------- Main loop ----------
DEFAULT_MAVSDK_URL = "udpin://0.0.0.0:14541"


async def main():
    url = os.environ.get("MAVSDK_URL", DEFAULT_MAVSDK_URL)
    try:
        from mavsdk import System  # noqa
    except ModuleNotFoundError:
        print("Please: python3 -m pip install mavsdk"); sys.exit(1)

    drone = await connect(url)
    print("\nExamples:\n"
          "  take off 3 then back 3, rotate right 45\n"
          "  rotate 45 ccw, front 2\n"
          "  fly at 2, rotate 180\n"
          "  survey circle\n"
          "  rtl\n"
          "q to quit.\n")

    while True:
        try:
            line = input("[you] > ")
        except (KeyboardInterrupt, EOFError):
            line = "q"

        if not line.strip():
            continue

        try:
            plans = llm_parse_multi(line)
        except Exception:
            plans = rule_parse_multi(line)

        # nothing parsed?
        if not plans or (len(plans) == 1 and plans[0].get("cmd") in (None,)):
            print("[agent] Couldn’t parse. Try: 'take off 3 then back 3 then rotate right 45'")
            continue

        # Execute strictly in order
        for p in plans:
            try:
                await exec_one(drone, p)
            except SystemExit:
                return
            except Exception as e:
                print("[agent] error executing", p.get("cmd"), ":", e)

if __name__ == "__main__":
    asyncio.run(main())
