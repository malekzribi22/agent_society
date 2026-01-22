# Fix Explanation: "Address in use" Error

## The Problem

When running both agents in parallel, you encountered:
```
bind error: Address in use (udp_connection.cpp:99)
Connection failed: Bind error (connection_initiator.h:47)
```

## Root Cause

The issue was in the **URL format** used for MAVSDK connections:

### Before (Broken):
```python
url = f"udpin://0.0.0.0:{port}"  # e.g., "udpin://0.0.0.0:14540"
```

### After (Fixed):
```python
url = f"udp://:{port}"  # e.g., "udp://:14540"
```

## Why This Fixes It

1. **`udpin://0.0.0.0:PORT`** explicitly tries to bind to `0.0.0.0:PORT` (all network interfaces)
   - This can fail if:
     - The port is already bound by another process
     - Multiple agents try to bind to the same port
     - There's a timing conflict during startup

2. **`udp://:PORT`** is a more flexible format that:
   - Uses the default network interface
   - Handles existing port bindings better
   - Is the format used by working examples in the codebase
   - Allows MAVSDK to connect to PX4's existing UDP listeners

## What Was Changed

### Files Modified:

1. **`agent_for_drone.py`** (line 44):
   - Changed from: `url = f"udpin://0.0.0.0:{port}"`
   - Changed to: `url = f"udp://:{port}"`

2. **`agent_mavsdk_pro.py`** (lines 10, 15, 882):
   - Changed default URL from `udpin://0.0.0.0:14540` to `udp://:14540`
   - Added retry logic with better error handling for connection issues

3. **`launch_dual_agents.sh`**:
   - Updated display messages to show correct URL format

## Additional Improvements

1. **Retry Logic**: Added connection retry mechanism (3 attempts with 2s delay)
   - Helps handle temporary port conflicts
   - Provides better error messages

2. **Better Error Messages**: 
   - Now shows which port is in use
   - Provides troubleshooting steps

## How It Works Now

1. **Agent 0** connects to `udp://:14540` (drone 0)
2. **Agent 1** connects to `udp://:14541` (drone 1)
3. Both agents can run **simultaneously** without port conflicts
4. Each agent connects to its assigned PX4 instance independently

## Verification

After the fix:
- ✅ Both agents can start simultaneously
- ✅ No "Address in use" errors
- ✅ Each agent controls its assigned drone independently
- ✅ Commands execute in parallel without conflicts

## Technical Details

The `udp://:PORT` format tells MAVSDK to:
- Listen on the default interface (usually 0.0.0.0)
- Use the specified port
- Connect to PX4's MAVLink stream on that port

This is the standard format used throughout the MAVSDK ecosystem and matches the working examples in your codebase (like `agent_langgraph_px4.py` and `agent_mavsdk.py`).


