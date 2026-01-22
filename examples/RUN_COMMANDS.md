# Commands to Run Dual Agents

## Step 1: Start the Simulation

Open **Terminal 1** and run:
```bash
cd /isaac-sim-4.5/PegasusSimulator/examples
isaacpy two_drones_dual_agents.py
```

Wait until you see:
```
TWO DRONES READY — PX4 + MAVSDK mapping
```

## Step 2: Launch Agent 0 (Drone 0)

Open **Terminal 2** and run:
```bash
cd /isaac-sim-4.5/PegasusSimulator/examples
python3 agent_for_drone.py 0
```

You should see:
- `[launcher] Starting Agent0 for drone0 on udpin://0.0.0.0:14540`
- `[Agent0] MAVSDK connected. UUID: ..., System ID: 1`
- `[Agent0] Ready on udpin://0.0.0.0:14540`

## Step 3: Launch Agent 1 (Drone 1)

Open **Terminal 3** and run:
```bash
cd /isaac-sim-4.5/PegasusSimulator/examples
python3 agent_for_drone.py 1
```

You should see:
- `[launcher] Starting Agent1 for drone1 on udpin://0.0.0.0:14541`
- `[Agent1] MAVSDK connected. UUID: ..., System ID: 2`
- `[Agent1] Ready on udpin://0.0.0.0:14541`

## Step 4: Control the Drones

### In Terminal 2 (Agent 0 - Drone 0):
```
[you] > take off 3
[you] > forward 2
[you] > rotate right 45
```

### In Terminal 3 (Agent 1 - Drone 1):
```
[you] > take off 3
[you] > back 2
[you] > rotate left 45
```

## Quick Check: Verify System IDs

When each agent connects, check the System ID:
- **Agent 0** should show: `System ID: 1` ✅
- **Agent 1** should show: `System ID: 2` ✅

If Agent 1 shows `System ID: 1`, it's connected to the wrong drone!

## Alternative: Use the Launch Script

Instead of manually opening terminals, you can use:
```bash
./launch_dual_agents.sh
```

This will automatically open two new terminals with the agents.

## Troubleshooting

If you see "Address in use" errors:
- Make sure the simulation is fully started before launching agents
- Wait a few seconds after starting the simulation
- Check that no other processes are using ports 14540 or 14541

If Agent 1 controls Drone 0:
- Check the System ID in the connection message
- Verify Agent 1 shows `udpin://0.0.0.0:14541` (not 14540)
- Make sure you're running `agent_for_drone.py 1` (not 0)


