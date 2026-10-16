import os
import time
import mujoco
import mujoco.viewer

# Dimensions validated from the workspace analysis
TABLE_HEIGHT = 0.30
FOOTPRINT = (0.15, 0.35, -0.45, -0.05)  # x_min, x_max, y_min, y_max
TRAY_CENTER = (0.24, -0.16)
TRAY_OPENING = 0.16
TRAY_WALL_HEIGHT = 0.05
TRAY_WALL_THICKNESS = 0.01
TABLE_THICKNESS = 0.03
CUBE_SIZE = 0.04

# Spawn exactly on the table surface at the validated success pick location
# Table top surface is at Z = 0.30. Cube center should be Z = 0.30 + CUBE_SIZE/2 = 0.32.
INITIAL_CUBE_POS = [0.32, -0.38, 0.32]

def create_task_scene(model_path: str) -> mujoco.MjModel:
    x_min, x_max, y_min, y_max = FOOTPRINT
    tray_x, tray_y = TRAY_CENTER
    tray_half = TRAY_OPENING / 2
    wall_half = TRAY_WALL_THICKNESS / 2
    outer_half = tray_half + TRAY_WALL_THICKNESS
    
    spec = mujoco.MjSpec.from_file(model_path)
    
    # Critically damped contact solver parameters [time_constant=20ms, damping_ratio=1.0]
    solref = [0.02, 1.0]
    solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
    
    # 1. Table Top
    spec.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[(x_min + x_max) / 2, (y_min + y_max) / 2, TABLE_HEIGHT - TABLE_THICKNESS / 2],
        size=[(x_max - x_min) / 2, (y_max - y_min) / 2, TABLE_THICKNESS / 2],
        contype=1,
        conaffinity=1,
        rgba=[0.5, 0.35, 0.22, 1.0],
        friction=[1.0, 0.05, 0.0001],
        solref=solref,
        solimp=solimp,
    )
    
    # 2. Tray Floor
    spec.worldbody.add_geom(
        name="tray_floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[tray_x, tray_y, TABLE_HEIGHT + 0.002],
        size=[outer_half, outer_half, 0.002],
        contype=1,
        conaffinity=1,
        rgba=[0.2, 0.55, 0.85, 1.0],
        friction=[1.0, 0.05, 0.0001],
        solref=solref,
        solimp=solimp,
    )
    
    # 3. Tray Walls
    wall_z = TABLE_HEIGHT + TRAY_WALL_HEIGHT / 2
    wall_specs = [
        ("px", tray_x + tray_half + wall_half, tray_y, wall_half, outer_half),
        ("nx", tray_x - tray_half - wall_half, tray_y, wall_half, outer_half),
        ("py", tray_x, tray_y + tray_half + wall_half, tray_half, wall_half),
        ("ny", tray_x, tray_y - tray_half - wall_half, tray_half, wall_half),
    ]
    for label, x, y, half_x, half_y in wall_specs:
        spec.worldbody.add_geom(
            name=f"tray_wall_{label}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, wall_z],
            size=[half_x, half_y, TRAY_WALL_HEIGHT / 2],
            contype=1,
            conaffinity=1,
            rgba=[0.2, 0.55, 0.85, 1.0],
            friction=[1.0, 0.05, 0.0001],
            solref=solref,
            solimp=solimp,
        )
        
    # 4. Pick Cube
    cube_body = spec.worldbody.add_body(
        name="target_cube",
        pos=INITIAL_CUBE_POS,
    )
    cube_body.add_freejoint(name="cube_freejoint")
    
    cube_body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_SIZE / 2] * 3,
        contype=1,
        conaffinity=1,
        rgba=[0.86, 0.22, 0.18, 1.0],
        friction=[1.0, 0.05, 0.0001],
        mass=0.1,
        solref=solref,
        solimp=solimp,
    )
    
    return spec.compile()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "Lab3", "robot_descriptions")
    model_path = os.path.join(model_dir, "arm_gripper.xml")

    original_dir = os.getcwd()
    try:
        os.chdir(model_dir)
        model = create_task_scene(model_path)
    finally:
        os.chdir(original_dir)
        
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Task Scene spawned successfully. The scene is completely frozen.")
        
        while viewer.is_running():
            # Apply gravity compensation to arm DOFs only
            data.qfrc_applied[:] = 0.0
            for i in range(model.nv):
                jnt_id = model.dof_jntid[i]
                if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
                    data.qfrc_applied[i] = data.qfrc_bias[i]
            
            # Step the simulation cleanly
            mujoco.mj_step(model, data)
            
            # Force ALL velocities (arm and cube) to exactly 0 to freeze the scene
            data.qvel[:] = 0.0
            
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()