from dataclasses import dataclass
from typing import List

@dataclass
class MissionTemplate:
    id: str
    prompt: str          # e.g. "fly at ALT m"
    description: str     # natural language description
    drones: int          # 1, 2, 3...
    altitude_m: float    # canonical altitude
    pattern: str         # e.g. "hover", "grid", "circle", "expanding_square"
    notes: str           # extra info if needed
    canonical_plan: str  # canonical mission plan text

def get_uav_mission_templates() -> List[MissionTemplate]:
    """
    Returns a list of canonical UAV mission templates for supervisor evaluation.
    """
    return [
        MissionTemplate(
            id="basic_takeoff_hover",
            prompt="take off and hover",
            description="Basic takeoff and hover mission",
            drones=1,
            altitude_m=10.0,
            pattern="hover",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 10 m.\n"
                "3. Hover for duration.\n"
                "4. Land."
            )
        ),
        MissionTemplate(
            id="fly_to_waypoint",
            prompt="fly to waypoint",
            description="Navigate to a specific waypoint",
            drones=1,
            altitude_m=15.0,
            pattern="waypoint",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 15 m.\n"
                "3. Fly to waypoint coordinates.\n"
                "4. Hover at destination.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="survey_area_A",
            prompt="survey area A",
            description="Survey Area A with grid pattern",
            drones=1,
            altitude_m=20.0,
            pattern="grid",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 20 m.\n"
                "3. Fly to Area A.\n"
                "4. Perform grid survey at 20 m (spacing 10 m).\n"
                "5. Return to launch and land."
            )
        ),
        MissionTemplate(
            id="inspect_building_B",
            prompt="inspect building B",
            description="Inspect Building B",
            drones=1,
            altitude_m=20.0,
            pattern="orbit",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 20 m.\n"
                "3. Fly to Building B.\n"
                "4. Orbit building structure.\n"
                "5. Capture imagery.\n"
                "6. Return to launch and land."
            )
        ),
        MissionTemplate(
            id="multi_drone_survey",
            prompt="use 2 drones to survey",
            description="Multi-drone survey mission",
            drones=2,
            altitude_m=15.0,
            pattern="grid",
            notes="Split area between agents",
            canonical_plan=(
                "Agent 1:\n"
                "1. Arm motors.\n"
                "2. Takeoff to 15 m.\n"
                "3. Survey Sector 1.\n"
                "4. Land.\n\n"
                "Agent 2:\n"
                "1. Arm motors.\n"
                "2. Takeoff to 15 m.\n"
                "3. Survey Sector 2.\n"
                "4. Land."
            )
        ),
        MissionTemplate(
            id="search_rescue_woods",
            prompt="search for missing person in woods",
            description="Search and rescue in woods",
            drones=1,
            altitude_m=30.0,
            pattern="search",
            notes="Thermal sensor implied",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 30 m.\n"
                "3. Fly to woods area.\n"
                "4. Execute search pattern (expanding square).\n"
                "5. Scan with thermal camera.\n"
                "6. Report findings.\n"
                "7. Return to launch."
            )
        ),
        MissionTemplate(
            id="orbit_tower",
            prompt="orbit the tower",
            description="Orbit a tower structure",
            drones=1,
            altitude_m=40.0,
            pattern="orbit",
            notes="Radius 50m implied from examples",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 40 m.\n"
                "3. Fly to tower center.\n"
                "4. Orbit at radius 50 m.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="follow_river",
            prompt="follow the river",
            description="River surveillance",
            drones=1,
            altitude_m=25.0,
            pattern="path_follow",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 25 m.\n"
                "3. Locate river start point.\n"
                "4. Follow river path downstream.\n"
                "5. Record video feed.\n"
                "6. Return to launch."
            )
        ),
        MissionTemplate(
            id="inspect_power_lines",
            prompt="inspect power lines",
            description="Power line inspection",
            drones=1,
            altitude_m=15.0,
            pattern="linear",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 15 m.\n"
                "3. Fly along power line corridor.\n"
                "4. Inspect insulators and cables.\n"
                "5. Return to launch."
            )
        ),
        MissionTemplate(
            id="multi_drone_perimeter",
            prompt="3 drones perimeter patrol",
            description="Perimeter patrol with 3 drones",
            drones=3,
            altitude_m=20.0,
            pattern="patrol",
            notes="",
            canonical_plan=(
                "Agent 1:\n"
                "1. Patrol North sector.\n\n"
                "Agent 2:\n"
                "1. Patrol East sector.\n\n"
                "Agent 3:\n"
                "1. Patrol South/West sector."
            )
        ),
        MissionTemplate(
            id="crop_monitoring",
            prompt="monitor crops in field",
            description="Agricultural crop monitoring",
            drones=1,
            altitude_m=10.0,
            pattern="grid",
            notes="Multispectral sensor",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 10 m.\n"
                "3. Fly grid over field.\n"
                "4. Capture multispectral data.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="traffic_monitoring",
            prompt="monitor traffic at intersection",
            description="Traffic monitoring",
            drones=1,
            altitude_m=30.0,
            pattern="hover",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 30 m.\n"
                "3. Hover over intersection.\n"
                "4. Count vehicles and monitor flow.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="bridge_inspection",
            prompt="inspect bridge",
            description="Bridge structure inspection",
            drones=1,
            altitude_m=10.0,
            pattern="inspection",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff.\n"
                "3. Fly under/over bridge structure.\n"
                "4. Inspect pillars and deck.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="solar_panel_inspection",
            prompt="inspect solar panels",
            description="Solar farm inspection",
            drones=1,
            altitude_m=15.0,
            pattern="grid",
            notes="Thermal",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 15 m.\n"
                "3. Fly grid over solar array.\n"
                "4. Scan for hotspots.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="delivery_mission",
            prompt="deliver package to point B",
            description="Package delivery",
            drones=1,
            altitude_m=20.0,
            pattern="delivery",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 20 m.\n"
                "3. Fly to Point B.\n"
                "4. Descend to 2m.\n"
                "5. Release package.\n"
                "6. Return to launch."
            )
        ),
        MissionTemplate(
            id="mapping_construction",
            prompt="map construction site",
            description="Construction site mapping",
            drones=1,
            altitude_m=40.0,
            pattern="grid",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 40 m.\n"
                "3. Fly grid pattern.\n"
                "4. Capture overlapping photos.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="wildfire_monitoring",
            prompt="monitor wildfire",
            description="Wildfire perimeter mapping",
            drones=1,
            altitude_m=100.0,
            pattern="perimeter",
            notes="Thermal",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 100 m.\n"
                "3. Fly to fire boundary.\n"
                "4. Map fire front progression.\n"
                "5. Return to launch."
            )
        ),
        MissionTemplate(
            id="flood_assessment",
            prompt="assess flood damage",
            description="Flood damage assessment",
            drones=1,
            altitude_m=50.0,
            pattern="survey",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 50 m.\n"
                "3. Fly over flooded area.\n"
                "4. Capture imagery of damage.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="pipeline_inspection",
            prompt="inspect pipeline",
            description="Pipeline inspection",
            drones=1,
            altitude_m=20.0,
            pattern="linear",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 20 m.\n"
                "3. Follow pipeline route.\n"
                "4. Check for leaks or damage.\n"
                "5. Return to launch."
            )
        ),
        MissionTemplate(
            id="night_patrol",
            prompt="night patrol of facility",
            description="Night security patrol",
            drones=1,
            altitude_m=15.0,
            pattern="patrol",
            notes="Thermal",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 15 m.\n"
                "3. Fly designated patrol route.\n"
                "4. Scan with thermal camera.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="roof_inspection",
            prompt="inspect roof",
            description="Roof inspection",
            drones=1,
            altitude_m=10.0,
            pattern="inspection",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 10 m.\n"
                "3. Fly over roof surface.\n"
                "4. Capture detailed photos.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="crowd_monitoring",
            prompt="monitor crowd at event",
            description="Event crowd monitoring",
            drones=1,
            altitude_m=40.0,
            pattern="hover",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 40 m.\n"
                "3. Hover over event area.\n"
                "4. Monitor crowd density.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="wind_turbine_inspection",
            prompt="inspect wind turbine",
            description="Wind turbine blade inspection",
            drones=1,
            altitude_m=80.0,
            pattern="vertical_scan",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 80 m.\n"
                "3. Fly vertical scan of each blade.\n"
                "4. Inspect nacelle.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="multi_drone_formation",
            prompt="formation flight",
            description="Multi-drone formation flight",
            drones=2,
            altitude_m=20.0,
            pattern="formation",
            notes="",
            canonical_plan=(
                "Agent 1:\n"
                "1. Lead formation.\n\n"
                "Agent 2:\n"
                "1. Follow leader at offset."
            )
        ),
        MissionTemplate(
            id="cinematic_shot",
            prompt="cinematic shot of subject",
            description="Cinematic filming",
            drones=1,
            altitude_m=10.0,
            pattern="tracking",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff.\n"
                "3. Track subject.\n"
                "4. Execute smooth camera move.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="calibration_flight",
            prompt="calibrate sensors",
            description="Sensor calibration",
            drones=1,
            altitude_m=5.0,
            pattern="calibration",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff.\n"
                "3. Rotate 360 degrees.\n"
                "4. Fly figure 8 pattern.\n"
                "5. Land."
            )
        ),
        MissionTemplate(
            id="emergency_landing",
            prompt="emergency land",
            description="Emergency landing procedure",
            drones=1,
            altitude_m=0.0,
            pattern="land",
            notes="",
            canonical_plan=(
                "1. Stop mission immediately.\n"
                "2. Descend at current location.\n"
                "3. Land and disarm."
            )
        ),
        MissionTemplate(
            id="return_to_home",
            prompt="return to home",
            description="Return to launch (RTL)",
            drones=1,
            altitude_m=20.0,
            pattern="rtl",
            notes="",
            canonical_plan=(
                "1. Abort current task.\n"
                "2. Climb to safe altitude.\n"
                "3. Return to launch point.\n"
                "4. Land."
            )
        ),
        MissionTemplate(
            id="hover_test",
            prompt="hover test",
            description="Stability hover test",
            drones=1,
            altitude_m=2.0,
            pattern="hover",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 2 m.\n"
                "3. Hover for specified duration.\n"
                "4. Land."
            )
        ),
        MissionTemplate(
            id="square_pattern",
            prompt="fly square pattern",
            description="Square flight pattern",
            drones=1,
            altitude_m=10.0,
            pattern="square",
            notes="",
            canonical_plan=(
                "1. Arm motors.\n"
                "2. Takeoff to 10 m.\n"
                "3. Fly leg 1.\n"
                "4. Turn 90 degrees.\n"
                "5. Fly leg 2.\n"
                "6. Turn 90 degrees.\n"
                "7. Fly leg 3.\n"
                "8. Turn 90 degrees.\n"
                "9. Fly leg 4.\n"
                "10. Land."
            )
        ),
    ]
