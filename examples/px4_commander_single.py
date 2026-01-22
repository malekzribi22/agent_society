#!/usr/bin/env python3
"""
Minimal Pegasus scene with one PX4-controlled drone.
PX4 is *not* autolaunched: start SITL yourself in another terminal, e.g.

    PX4_SYS_AUTOSTART=10016 PX4_SIM_MODEL=none \\
        ./build/px4_sitl_default/bin/px4 -s etc/init.d-posix/rcS \\
        -i 0 -w /tmp/pegasus_px4_0 -d

Pegasus connects over TCP to PX4's default 4560 port.
Start PX4 manually, then inside the pxh> prompt:
    commander arm
    commander takeoff
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig


class PX4CommanderSingle:
    """One PX4 drone with nothing but PX4 shell control."""

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()

        # Spawn the standard Curved Gridroom so there is plenty of space for a takeoff
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        config = MultirotorConfig()
        px4_cfg = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": False,
                "connection_type": "tcp",
                "connection_ip": "127.0.0.1",
                "connection_baseport": 4560,
            }
        )
        config.backends = [PX4MavlinkBackend(px4_cfg)]

        Multirotor(
            "/World/px4_commander_drone",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config,
        )

        self.pg.set_viewport_camera([5.0, 5.0, 3.0], [0.0, 0.0, 0.0])
        self.world.reset()

        self._print_usage()

    def _print_usage(self):
        print("\n" + "=" * 72)
        print("PX4 COMMANDER SINGLE DRONE READY")
        print("Pegasus waits for an external PX4 SITL instance.")
        print("Start PX4 manually (new terminal, inside PX4-Autopilot):")
        print("  ROMFS=$(pwd)/ROMFS/px4fmu_common")
        print("  PX4_SYS_AUTOSTART=10016 PX4_SIM_MODEL=none \\")
        print("    ./build/px4_sitl_default/bin/px4 \"$ROMFS\" \\")
        print("    -s \"$ROMFS/init.d-posix/rcS\" -i 0 -w /tmp/pegasus_px4_0 -d")
        print("PX4 TCP server : localhost:4560 (default px4 instance 0)")
        print("Once pxh> shows 'Ready for takeoff':")
        print("  commander arm")
        print("  commander takeoff")
        print("No automatic PX4 launch or mission control is performed here.")
        print("=" * 72 + "\n")

    def run(self):
        self.timeline.play()
        try:
            while simulation_app.is_running():
                self.world.step(render=True)
        finally:
            self.timeline.stop()
            simulation_app.close()


def main():
    PX4CommanderSingle().run()


if __name__ == "__main__":
    main()
