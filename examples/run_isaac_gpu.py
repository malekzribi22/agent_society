#!/usr/bin/env python3
from isaacsim import SimulationApp
app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"})

# set sane defaults in case a scene overrides them
import omni.kit.app
from omni.kit.app import get_app_interface
appiface = get_app_interface()
appiface.set_setting("/app/framerateLimitEnabled", False)  # no vsync cap
appiface.set_setting("/rtx/meshlet/enabled", True)         # typical RTX defaults

# enable GPU PhysX on the default physics scene if present or created later
from pxr import Usd, UsdPhysics, PhysxSchema
import omni.usd
stage = omni.usd.get_context().get_stage()
if stage.GetDefaultPrim() is None:
    stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
physx.CreateGpuDynamicsEnabledAttr(True)
physx.CreateBroadphaseTypeAttr("GPU")
scene.CreateStepsPerSecondAttr(120.0)       # 120 Hz sim step
scene.CreateTimeCodesPerSecondAttr(120.0)
physx.CreateSolverTypeAttr("TGS")
physx.CreateSolverPositionIterationCountAttr(6)   # keep moderate
physx.CreateSolverVelocityIterationCountAttr(1)
stage.Save()

# open an empty sample scene to test pure renderer+physics
from omni.isaac.core.utils.stage import add_reference_to_stage
# simple ground plane USD provided with kit, adjust if path differs
add_reference_to_stage(usd_path="/Isaac/Environments/Simple_Warehouse/warehouse.usd", prim_path="/World/Env")

# drop a dynamic cube to exercise physics
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import create_prim
create_prim("/World/Cube", "Cube", translation=(0,0,3))
from omni.isaac.core.utils.physics import enable_physics
enable_physics()

# run for a short time so you can see FPS and motion
while app.is_running():
    app.update()

