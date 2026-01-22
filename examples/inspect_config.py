
import sys
import os

# Add the necessary paths to sys.path if needed, or rely on the environment
# Assuming the environment is set up correctly for Isaac Sim / Pegasus

try:
    from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackendConfig
    print("PX4MavlinkBackendConfig fields:")
    # Inspect the __init__ or the class itself
    import inspect
    print(inspect.signature(PX4MavlinkBackendConfig.__init__))
    
    # Also check if there are any properties
    cfg = PX4MavlinkBackendConfig({})
    print("Config dictionary keys:", cfg.config.keys())
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
