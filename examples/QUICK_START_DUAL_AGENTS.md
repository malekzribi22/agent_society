# Quick Start: Dual Agents

## Step-by-Step Instructions

### 1. Start the Simulation
```bash
cd /isaac-sim-4.5/PegasusSimulator/examples
isaacpy two_drones_dual_agents.py
```

Wait until you see:
```
TWO DRONES READY — PX4 + MAVSDK mapping
```

### 2. Launch Both Agents (Choose one method)

#### Option A: Automatic (Recommended)
```bash
./launch_dual_agents.sh
```

#### Option B: Manual - Terminal 1
```bash
python3 agent_for_drone.py 0
```

#### Option C: Manual - Terminal 2  
```bash
python3 agent_for_drone.py 1
```

### 3. Control Your Drones

Each agent runs independently. You can now:
- Type commands in Agent 0 terminal → controls Drone 0
- Type commands in Agent 1 terminal → controls Drone 1
- Both execute **at the same time**!

### Example Commands
```
take off 3
forward 2
rotate right 45
back 1
land
```

### To Quit
Type `q` or `quit` in any agent terminal.


