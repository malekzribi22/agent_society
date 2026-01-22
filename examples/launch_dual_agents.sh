#!/bin/bash
# Launch both agents in separate terminals for parallel control
# Usage: ./launch_dual_agents.sh
#
# This script will:
# 1. Launch Agent 0 (controls drone 0) in Terminal 1
# 2. Launch Agent 1 (controls drone 1) in Terminal 2
#
# Make sure two_drones_dual_agents.py is running first!

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Dual Agents for Parallel Control"
echo "=========================================="
echo ""
echo "Agent 0 will control drone 0 (MAVSDK port 14540)"
echo "Agent 1 will control drone 1 (MAVSDK port 14541)"
echo ""
echo "Make sure two_drones_dual_agents.py is running first!"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Launch Agent 0 in first terminal
echo "Launching Agent 0 in new terminal..."
gnome-terminal --title="Agent 0 - Drone 0" -- bash -c "
cd '$SCRIPT_DIR';
echo '========================================';
echo 'Agent 0 - Controlling Drone 0';
echo 'MAVSDK URL: udpin://0.0.0.0:14540';
echo '========================================';
echo '';
python3 agent_for_drone.py 0;
echo '';
echo 'Agent 0 terminated. Press Enter to close...';
read
" 2>/dev/null || xterm -title "Agent 0 - Drone 0" -e bash -c "
cd '$SCRIPT_DIR';
echo '========================================';
echo 'Agent 0 - Controlling Drone 0';
echo 'MAVSDK URL: udpin://0.0.0.0:14540';
echo '========================================';
echo '';
python3 agent_for_drone.py 0;
echo '';
echo 'Agent 0 terminated. Press Enter to close...';
read
" 2>/dev/null || konsole --title "Agent 0 - Drone 0" -e bash -c "
cd '$SCRIPT_DIR';
echo '========================================';
echo 'Agent 0 - Controlling Drone 0';
echo 'MAVSDK URL: udpin://0.0.0.0:14540';
echo '========================================';
echo '';
python3 agent_for_drone.py 0;
echo '';
echo 'Agent 0 terminated. Press Enter to close...';
read
" 2>/dev/null || echo "Could not launch terminal. Please run manually: python3 agent_for_drone.py 0"

sleep 1

# Launch Agent 1 in second terminal
echo "Launching Agent 1 in new terminal..."
gnome-terminal --title="Agent 1 - Drone 1" -- bash -c "
cd '$SCRIPT_DIR';
echo '========================================';
echo 'Agent 1 - Controlling Drone 1';
echo 'MAVSDK URL: udpin://0.0.0.0:14541';
echo '========================================';
echo '';
python3 agent_for_drone.py 1;
echo '';
echo 'Agent 1 terminated. Press Enter to close...';
read
" 2>/dev/null || xterm -title "Agent 1 - Drone 1" -e bash -c "
cd '$SCRIPT_DIR';
echo '========================================';
echo 'Agent 1 - Controlling Drone 1';
echo 'MAVSDK URL: udpin://0.0.0.0:14541';
echo '========================================';
echo '';
python3 agent_for_drone.py 1;
echo '';
echo 'Agent 1 terminated. Press Enter to close...';
read
" 2>/dev/null || konsole --title "Agent 1 - Drone 1" -e bash -c "
cd '$SCRIPT_DIR';
echo '========================================';
echo 'Agent 1 - Controlling Drone 1';
echo 'MAVSDK URL: udpin://0.0.0.0:14541';
echo '========================================';
echo '';
python3 agent_for_drone.py 1;
echo '';
echo 'Agent 1 terminated. Press Enter to close...';
read
" 2>/dev/null || echo "Could not launch terminal. Please run manually: python3 agent_for_drone.py 1"

echo ""
echo "=========================================="
echo "Both agents launched!"
echo "=========================================="
echo ""
echo "Agent 0 is controlling drone 0 (port 14540)"
echo "Agent 1 is controlling drone 1 (port 14541)"
echo ""
echo "You can now control each drone independently in their terminals."
echo ""

