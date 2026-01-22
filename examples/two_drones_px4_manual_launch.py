#!/usr/bin/env python3

import os
import subprocess
import time
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.world import World
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS
from scipy.spatial.transform import Rotation

PX4_PATH = os.path.expanduser("~/PX4-Autopilot/build/px4_sitl_default/bin/px4")

# PX4 launch helper ---------------------------------------------------------
def launch_px4_instance(vehicle_id, home_pos, out_port, in_port):
    env = os.environ.copy()
    env["PX4_SIM_HOST_ADDR"] = "127.0.0.1"
    env["PX4_SIM_MODEL"] = "iris"
    env["HOME_POS_LAT"] = str(home_pos[0])
    env["HOME_POS_LON"] = str(home_pos[1])
    env["HOME_POS_ALT"] = str(home_pos[2])

    print(f"[PX4-{vehicle_id}] Launching PX4 SITL on ports {out_port}/{in_port}...")

    return subprocess.Popen([
        PX4_PATH,
        f"-i{vehicle_id}",
        f"-d{out_port}",
        f"-u{in_port}"
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


class TwoDroneApp:
    def __init__(self):
        self.world = World(stage_units_in_meters=True)
        self.world.scene.add_default_ground_plane()

        self.px4_procs = []

        # DRONE 0 PX4 --------------------------------------------------------
        p0 = launch_px4_instance(
            vehicle_id=0,
            home_pos=[47.397742, 8.545594, 488.0],
            out_port=14560,
            in_port=14540  # MAVSDK default
        )
        self.px4_procs.append(p0)

        # DRONE 1 PX4 --------------------------------------------------------
        p1 = launch_px4_instance(
            vehicle_id=1,
            home_pos=[47.397742, 8.545594, 488.0],
            out_port=14570,
            in_port=14541
        )
        self.px4_procs.append(p1)

        time.sleep(5)  # Let PX4 boot and show messages

        print("\n=== PX4 BOOT LOGS ===")
        for idx, proc in enumerate(self.px4_procs):
            try:
                for _ in range(15):
                    line = proc.stdout.readline().strip()
                    if line:
                        print(f"[PX4-{idx}] {line}")
            except:
                pass

        # DRONE MODELS IN SIM ------------------------------------------------
        cfg = MultirotorConfig()

        quat0 = Rotation.from_euler("XYZ", [0,0,0], degrees=True).as_quat()
        self.drone0 = Multirotor("/World/drone00", ROBOTS["Iris"], 0,
                                 [0, 0, 0.2], quat0, config=cfg)

        quat1 = Rotation.from_euler("XYZ", [0,0,0], degrees=True).as_quat()
        self.drone1 = Multirotor("/World/drone01", ROBOTS["Iris"], 1,
                                 [2, 0, 0.2], quat1, config=cfg)

        self.world.reset()

        print("\n=== READY ===")
        print("Drone 0 = MAVSDK on udpin://0.0.0.0:14540")
        print("Drone 1 = MAVSDK on udpin://0.0.0.0:14541")

    def run(self):
        while simulation_app.is_running():
            self.world.step(render=True)


if __name__ == "__main__":
    app = TwoDroneApp()
    app.run()

