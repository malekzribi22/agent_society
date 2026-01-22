#!/bin/bash
# Start two PX4 SITL instances with proper port separation
# Save as: start_dual_px4.sh
# Usage: ./start_dual_px4.sh

# Path to your PX4-Autopilot directory
PX4_DIR="$HOME/PX4-Autopilot"

echo "Starting PX4 SITL instances for dual drone setup..."
echo "=================================================="

# Start drone0 (index 0, UDP port 14540)
echo "Starting drone0 on UDP port 14540..."
gnome-terminal -- bash -c "
cd $PX4_DIR;
PX4_SYS_AUTOSTART=10015 \
PX4_SIM_MODEL=none \
PX4_SIM_HOSTNAME=localhost \
PX4_SITL_UDP_PORT=14540 \
./build/px4_sitl_default/bin/px4 ./ROMFS/px4fmu_common -i 0 -s etc/init.d-posix/rcS;
exec bash"

sleep 2

# Start drone1 (index 1, UDP port 14541)
echo "Starting drone1 on UDP port 14541..."
gnome-terminal -- bash -c "
cd $PX4_DIR;
PX4_SYS_AUTOSTART=10015 \
PX4_SIM_MODEL=none \
PX4_SIM_HOSTNAME=localhost \
PX4_SITL_UDP_PORT=14541 \
./build/px4_sitl_default/bin/px4 ./ROMFS/px4fmu_common -i 1 -s etc/init.d-posix/rcS;
exec bash"

echo ""
echo "Both PX4 instances started!"
echo "Drone0: UDP 14540 -> MAVSDK udpin://0.0.0.0:14540"
echo "Drone1: UDP 14541 -> MAVSDK udpin://0.0.0.0:14541"
echo ""
echo "Now run:"
echo "  python3 agent_for_drone.py 0"
echo "  python3 agent_for_drone.py 1"

