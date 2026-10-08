import mujoco
import mujoco.viewer
import numpy as np
import time

# =====================================================================
# 1. LOAD MODEL
# =====================================================================
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/skydio_x2/scene.xml")
data = mujoco.MjData(model)

try:
    CHASSIS_NAME = "x2"
    chassis_id = model.body(CHASSIS_NAME).id
except KeyError:
    CHASSIS_NAME = "base"
    chassis_id = model.body(CHASSIS_NAME).id

model.opt.gravity[:] = [0.0, 0.0, -9.81]
GRAVITY = 9.81

joint_id = model.body_jntadr[chassis_id]
assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE, "Chassis must have a free joint"
qpos_adr = model.jnt_qposadr[joint_id]
dof_adr = model.jnt_dofadr[joint_id]

START_POS = np.array([0.0, 0.0, 0.3])
data.qpos[qpos_adr:qpos_adr + 3] = START_POS
data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)

# =====================================================================
# 2. PHYSICAL PARAMETERS
# =====================================================================
MASS = model.body_subtreemass[chassis_id]
HOVER_THRUST_TOTAL = MASS * GRAVITY
print(f"[info] mass = {MASS:.3f} kg, hover thrust = {HOVER_THRUST_TOTAL:.3f} N, "
      f"{model.nu} actuators")

# =====================================================================
# 3. CONTROL GAINS
# =====================================================================
MAX_TILT_RAD = np.deg2rad(12.0)   # tilt limit used by the mission (gentle flight)

KP_ATT = 8.0
KD_ATT = 1.2
KP_YAWRATE = 2.5
KP_VZ = 18.0
LIN_DRAG_COEFF = 0.6

# Outer position loop
KP_POS = 1.5      # [m/s^2 per m]
KD_POS = 2.5      # [m/s^2 per m/s]
KP_Z = 1.5        # altitude error -> climb rate
VZ_LIMIT = 1.5    # [m/s]

# =====================================================================
# 4. MISSION DEFINITION
# =====================================================================
T_START = 1.0     # seconds of hover before the mission begins
V_REF = 1.2       # speed of the moving reference point [m/s]
HOLD_TIME = 1.0   # pause at each waypoint [s]
POS_TOL = 0.20    # how close the drone must get before the next waypoint [m]

WAYPOINTS = np.array([
    [0.0,  0.0, 1.5],   # take off
    [2.0,  0.0, 1.5],   # forward
    [2.0,  2.0, 2.0],   # right + climb
    [-2.0, 2.0, 1.5],   # back
    [-2.0, -2.0, 1.0],  # left + descend
    [0.0, -2.0, 1.5],
    [0.0,  0.0, 1.5],   # above the start point
    [0.0,  0.0, 0.3],   # land back at the initial position
])

mission = {
    "ref": START_POS.copy(),   # moving reference point the drone chases
    "vref": np.zeros(3),
    "idx": 0,
    "hold_until": None,
    "done": False,
}


def update_reference(t, dt, pos):
    """Move the reference point toward the current waypoint at V_REF."""
    m = mission
    m["vref"] = np.zeros(3)
    if t < T_START or m["done"]:
        return

    target = WAYPOINTS[m["idx"]]
    diff = target - m["ref"]
    dist = np.linalg.norm(diff)
    step = V_REF * dt

    if dist > step:
        direction = diff / dist
        m["ref"] = m["ref"] + direction * step
        m["vref"] = direction * V_REF
        return

    # reference has reached the waypoint; wait for the drone to arrive
    m["ref"] = target.copy()
    if np.linalg.norm(target - pos) < POS_TOL:
        if m["hold_until"] is None:
            m["hold_until"] = t + HOLD_TIME
        elif t >= m["hold_until"]:
            m["hold_until"] = None
            m["idx"] += 1
            if m["idx"] >= len(WAYPOINTS):
                m["done"] = True
                m["idx"] = len(WAYPOINTS) - 1
                print(f"\n[mission] complete at t = {t:.1f} s, drone is back at start.")


def rotmat_to_euler(R):
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def position_controller(pos, vel, yaw):
    """World position error -> desired roll, pitch and climb rate."""
    m = mission
    a_des = (KP_POS * (m["ref"] - pos)
             + KD_POS * (m["vref"] - vel)
             + (LIN_DRAG_COEFF / MASS) * m["vref"])   # drag feed-forward

    # limit horizontal acceleration to what the tilt limit allows
    a_max = GRAVITY * np.tan(MAX_TILT_RAD)
    a_xy = a_des[:2]
    n = np.linalg.norm(a_xy)
    if n > a_max:
        a_xy = a_xy * (a_max / n)

    # rotate the world-frame acceleration into the yaw-aligned frame
    c, s = np.cos(yaw), np.sin(yaw)
    ax_b = c * a_xy[0] + s * a_xy[1]
    ay_b = -s * a_xy[0] + c * a_xy[1]

    # R = Rz Ry Rx: thrust direction = [sin(pitch), -sin(roll), cos(...)]
    pitch_des = np.clip(ax_b / GRAVITY, -MAX_TILT_RAD, MAX_TILT_RAD)
    roll_des = np.clip(-ay_b / GRAVITY, -MAX_TILT_RAD, MAX_TILT_RAD)

    vz_des = np.clip(KP_Z * (m["ref"][2] - pos[2]) + m["vref"][2], -VZ_LIMIT, VZ_LIMIT)
    return roll_des, pitch_des, vz_des


def compute_ctrl(roll_des, pitch_des, vz_des):
    """PD attitude + altitude controller with actuator-Jacobian allocation."""
    R = data.xmat[chassis_id].reshape(3, 3)
    roll, pitch, yaw = rotmat_to_euler(R)

    p, q, r = data.qvel[dof_adr + 3: dof_adr + 6]
    vz_actual = data.qvel[dof_adr + 2]

    tau_x = KP_ATT * (roll_des - roll) - KD_ATT * p
    tau_y = KP_ATT * (pitch_des - pitch) - KD_ATT * q
    tau_z = KP_YAWRATE * (0.0 - r)   # hold heading

    tilt = min(max(abs(roll), abs(pitch)), np.deg2rad(60))
    Fz_world = HOVER_THRUST_TOTAL / np.cos(tilt) + KP_VZ * (vz_des - vz_actual)

    wrench_des = np.array([0.0, 0.0, Fz_world, tau_x, tau_y, tau_z])

    raw = np.asarray(data.actuator_moment)
    if raw.ndim == 2:
        moment_dense = raw
    elif hasattr(data, "actuator_moment_rownnz"):
        moment_dense = np.zeros((model.nu, model.nv))
        mujoco.mju_sparse2dense(
            moment_dense,
            data.actuator_moment,
            data.actuator_moment_rownnz,
            data.actuator_moment_rowadr,
            data.actuator_moment_colind,
        )
    else:
        moment_dense = raw.reshape(model.nu, model.nv)
    M = moment_dense[:, dof_adr: dof_adr + 6]
    ctrl = np.linalg.pinv(M.T) @ wrench_des

    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    return np.clip(ctrl, lo, hi)


# =====================================================================
# 5. DRAWING HELPERS
# =====================================================================
def draw_axis(scn, pos, mat, size=0.4, width=0.01):
    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for axis in range(3):
        end = pos + mat[:, axis] * size
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9),
            np.array(colors[axis], dtype=np.float32)
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, pos, end)
        scn.ngeom += 1


def draw_matrix_label(scn, pos, mat, row_gap=0.14):
    """3D floating matrix: title + one label per row, stacked above the drone."""
    lines = [
        "Rotation Matrix (Drone to World)",
        f"[ {mat[0,0]:+.2f}   {mat[0,1]:+.2f}   {mat[0,2]:+.2f} ]",
        f"[ {mat[1,0]:+.2f}   {mat[1,1]:+.2f}   {mat[1,2]:+.2f} ]",
        f"[ {mat[2,0]:+.2f}   {mat[2,1]:+.2f}   {mat[2,2]:+.2f} ]",
    ]
    for i, text in enumerate(lines):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.001, 0, 0]), pos + np.array([0, 0, -i * row_gap]),
            np.eye(3).flatten(),
            np.array([1, 1, 1, 0], dtype=np.float32)
        )
        g.label = text
        scn.ngeom += 1


def matrix_overlay_text(mat):
    return "\n".join(
        f"[ {mat[r,0]:+.2f}  {mat[r,1]:+.2f}  {mat[r,2]:+.2f} ]" for r in range(3)
    )


# =====================================================================
# 6. MAIN LOOP
# =====================================================================
FRAME_DT = 1.0 / 60.0
SUBSTEPS = max(1, int(round(FRAME_DT / model.opt.timestep)))
dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:

    with viewer.lock():
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 9.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

    print("\n=================================================")
    print(f"Autonomous mission: hover {T_START:.0f} s, then fly {len(WAYPOINTS)} waypoints,")
    print("ending back at the initial position.")
    print("=================================================\n")

    step_count = 0
    world_pos = np.zeros(3)
    world_mat = np.eye(3)

    while viewer.is_running():
        frame_start = time.time()

        for _ in range(SUBSTEPS):
            pos = data.qpos[qpos_adr:qpos_adr + 3].copy()
            vel = data.qvel[dof_adr:dof_adr + 3].copy()
            R = data.xmat[chassis_id].reshape(3, 3)
            _, _, yaw = rotmat_to_euler(R)

            update_reference(data.time, dt, pos)
            roll_des, pitch_des, vz_des = position_controller(pos, vel, yaw)

            # synthetic air drag
            data.xfrc_applied[chassis_id, 0:3] = -LIN_DRAG_COEFF * vel
            data.xfrc_applied[chassis_id, 3:6] = 0.0

            mujoco.mj_step1(model, data)
            data.ctrl[:] = compute_ctrl(roll_des, pitch_des, vz_des)
            mujoco.mj_step2(model, data)

        chassis_pos = data.xpos[chassis_id].copy()
        chassis_mat = data.xmat[chassis_id].reshape(3, 3).copy()

        viewer.user_scn.ngeom = 0
        draw_axis(viewer.user_scn, world_pos, world_mat, size=0.6, width=0.015)      # ground frame
        draw_axis(viewer.user_scn, chassis_pos, chassis_mat, size=0.35, width=0.01)  # drone frame
        draw_matrix_label(viewer.user_scn, chassis_pos + np.array([0, 0, 0.8]), chassis_mat)

        # fixed on-screen overlays: proper 3x3 matrix + mission status
        status = "LANDED / DONE" if mission["done"] else (
            "HOVER (waiting)" if data.time < T_START else f"WAYPOINT {mission['idx'] + 1}/{len(WAYPOINTS)}")
        viewer.set_texts([
            (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
             "Rotation Matrix (Drone to World)", matrix_overlay_text(chassis_mat)),
            (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
             f"t = {data.time:5.1f} s   {status}",
             f"pos = [{chassis_pos[0]:+.2f} {chassis_pos[1]:+.2f} {chassis_pos[2]:+.2f}]"),
        ])

        if step_count % 15 == 0:
            print(f"t={data.time:5.1f}s  pos={np.round(chassis_pos, 2)}")
            print(np.round(chassis_mat, 3))

        step_count += 1
        viewer.sync()

        # keep roughly real time
        sleep_left = FRAME_DT - (time.time() - frame_start)
        if sleep_left > 0:
            time.sleep(sleep_left)
            
