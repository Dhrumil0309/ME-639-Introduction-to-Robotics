"""
config.py - every number the project uses lives here, in one place.

Why one file?  When you tune something (table size, speeds, gains) you only
edit this file, and every script (workspace, pick_place, compare) picks it up.

Units: metres, radians, seconds.  World frame: robot base at the origin,
+x points forward (away from the robot), +z points up.
"""
import numpy as np

# ---------------------------------------------------------------- robot model
PANDA_XML = "assets/panda/panda.xml"     # Franka Panda from mujoco_menagerie
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]   # the 7 arm joints we control
TCP_OFFSET = 0.1034   # tool-centre-point: 10.34 cm below the hand flange,
                      # i.e. the middle of the two finger pads
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
# Panda joint velocity limits (rad/s) from the Franka datasheet
V_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0   # actuator8 command range 0..255

# --------------------------------------------- table (chosen by workspace.py)
# These values come from the workspace analysis (see results/table_design.md).
TABLE_H = 0.10                  # table top height above the robot base (H)
TABLE_X = (0.30, 0.65)          # table spans x_min..x_max -> length L = 0.35
TABLE_Y = (-0.45, 0.45)         # table spans y_min..y_max -> width  W = 0.90
TABLE_THICK = 0.03

# ------------------------------------------------------------- cube and tray
CUBE_HALF = 0.02                # 4 cm cube
CUBE_MASS = 0.05
TRAY_POS = (0.475, -0.32)       # tray centre (x, y) on the table
TRAY_INNER = 0.08               # inner half-size of the tray -> 16 x 16 cm
TRAY_WALL_H = 0.05              # wall height above the table
TRAY_WALL_T = 0.008             # wall thickness

# cube spawn region (inside the table, away from the tray and table edges)
SPAWN_X = (0.34, 0.61)
SPAWN_Y = (-0.14, 0.40)
SPAWN_YAW = (-np.pi, np.pi)

# ------------------------------------------------------------ motion planning
PREGRASP_DZ = 0.12              # pre-grasp is 12 cm above the cube
SAFE_MARGIN = 0.05              # extra clearance over the tray rim in transit
STEP_POS = 0.01                 # Cartesian waypoint spacing (1 cm)
STEP_ROT = 0.10                 # orientation waypoint spacing (rad)
EE_SPEED = 0.20                 # TCP speed used to time the trajectory (m/s)
ROT_SPEED = 1.2                 # wrist rotation speed (rad/s)
POS_TOL, ROT_TOL = 1e-3, 1e-2   # IK converged when error below these
MAX_IK_ITERS = 150              # per waypoint
PHYS_DT = 0.002                 # MuJoCo timestep

# ------------------------------------------------------- custom IK parameters
DLS_LAMBDA = 1e-2               # damping lambda in (J^T J + lambda I)^-1
IK_GAIN = 0.8                   # fraction of the error corrected per iteration
NULL_ALPHA = 0.05               # null-space joint-centring gain (DLS)
QP_DT = 0.05                    # virtual time step used in the QP velocity limit
JOINT_MARGIN = 0.03             # stay 0.03 rad inside the joint limits (QP)
