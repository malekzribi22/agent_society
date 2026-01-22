# mavsdk_adapter.py
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from mavsdk.action import ActionError

class MavsdkAdapter:
    def __init__(self, system_url: str = "udp://:14540"):
        self.system_url = system_url
        self.drone = System()
        self._connected = False

    async def connect(self):
        if self._connected:
            return
        await self.drone.connect(system_address=self.system_url)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self._connected = True
                break

    async def arm(self):
        try:
            await self.drone.action.arm()
        except ActionError as e:
            raise RuntimeError(f"Arm failed: {e}")

    async def disarm(self):
        try:
            await self.drone.action.disarm()
        except ActionError:
            pass

    async def start_offboard(self):
        # send an initial setpoint before starting offboard
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        try:
            await self.drone.offboard.start()
        except OffboardError as e:
            raise RuntimeError(f"Offboard start failed: {e}")

    async def stop_offboard(self):
        try:
            await self.drone.offboard.stop()
        except OffboardError:
            pass

    async def takeoff_slow(self, climb_rate: float = -0.5, seconds: float = 3.0):
        await self.arm()
        await self.start_offboard()
        steps = int(max(1, seconds * 10))
        for _ in range(steps):
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, climb_rate, 0.0)
            )
            await asyncio.sleep(0.1)

    async def land_slow(self, descend_rate: float = 0.5, seconds: float = 3.0):
        steps = int(max(1, seconds * 10))
        for _ in range(steps):
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, descend_rate, 0.0)
            )
            await asyncio.sleep(0.1)
        await self.stop_offboard()
        try:
            await self.drone.action.land()
        except ActionError:
            pass

    async def move_vel(self, north_m_s: float, east_m_s: float, down_m_s: float, yaw_deg: float, duration_s: float = 1.0):
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(north_m_s, east_m_s, down_m_s, yaw_deg)
        )
        await asyncio.sleep(max(0.1, duration_s))

