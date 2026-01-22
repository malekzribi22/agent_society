# Running Dual Agents in Parallel

This guide explains how to run two agents simultaneously, each controlling a different drone.

## Overview

- **Agent 0** controls **Drone 0** (MAVSDK port 14540)
- **Agent 1** controls **Drone 1** (MAVSDK port 14541)
- Both agents run in **separate terminals** and operate **in parallel**

## Prerequisites

1. Make sure you have the required dependencies:
   ```bash
   pip install mavsdk
   ```

2. The simulation must be running with `two_drones_dual_agents.py`

## Method 1: Using the Launch Script (Recommended)

1. **Start the simulation** in one terminal:
   ```bash
   cd /isaac-sim-4.5/PegasusSimulator/examples
   isaacpy two_drones_dual_agents.py
   ```

2. **Wait for the simulation to fully start** (you'll see the mapping printed)

3. **Launch both agents** using the helper script:
   ```bash
   ./launch_dual_agents.sh
   ```

   This will automatically open two new terminals:
   - Terminal 1: Agent 0 controlling Drone 0
   - Terminal 2: Agent 1 controlling Drone 1

## Method 2: Manual Launch (Two Terminals)

If you prefer to launch manually:

### Terminal 1 - Agent 0:
```bash
cd /isaac-sim-4.5/PegasusSimulator/examples
python3 agent_for_drone.py 0
```

### Terminal 2 - Agent 1:
```bash
cd /isaac-sim-4.5/PegasusSimulator/examples
python3 agent_for_drone.py 1
```

## How It Works

1. **`two_drones_dual_agents.py`**:
   - Sets up the simulation with 2 drones
   - Each drone has its own PX4 backend
   - Drone 0: MAVSDK port 14540
   - Drone 1: MAVSDK port 14541

2. **`agent_for_drone.py`**:
   - Takes a drone index (0 or 1) as argument
   - Calculates the correct MAVSDK port
   - Sets environment variables (`MAVSDK_URL`, `AGENT_NAME`)
   - Launches `agent_mavsdk_pro.py` with the correct configuration

3. **`agent_mavsdk_pro.py`**:
   - Connects to the MAVSDK URL from environment
   - Uses per-drone state (no shared state between agents)
   - Provides interactive control via command line

## Example Commands

Once both agents are running, you can control each drone independently:

### In Agent 0 terminal (Drone 0):
```
[you] > take off 3
[you] > forward 2
[you] > rotate right 45
```

### In Agent 1 terminal (Drone 1):
```
[you] > take off 3
[you] > back 2
[you] > rotate left 45
```

Both drones will execute their commands **simultaneously**!

## Troubleshooting

### Issue: "Connection timeout" or "Could not connect"
- Make sure `two_drones_dual_agents.py` is running and fully initialized
- Wait a few seconds after starting the simulation before launching agents
- Check that PX4 instances are running (they auto-start with the simulation)

### Issue: Agents interfere with each other
- Each agent uses its own `DroneState` object (no shared state)
- Each agent connects to a different MAVSDK port
- This should not happen, but if it does, restart both agents

### Issue: Terminal doesn't open
- The script tries multiple terminal emulators (gnome-terminal, xterm, konsole)
- If none work, use Method 2 to launch manually

## Port Mapping

| Drone | Index | MAVSDK Port | PX4 TCP Port | ROS Namespace |
|-------|-------|-------------|--------------|---------------|
| drone0 | 0 | 14540 | 4560 | drone00 |
| drone1 | 1 | 14541 | 4561 | drone01 |

## Notes

- Both agents run **completely independently** - they don't share any state
- Commands in one terminal only affect that agent's drone
- You can run different commands in each terminal at the same time
- To quit an agent, type `q` or `quit` in that agent's terminal


