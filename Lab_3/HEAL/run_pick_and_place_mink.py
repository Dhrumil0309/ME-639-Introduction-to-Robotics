import os
import time
import random
import mujoco
import mujoco.viewer
import numpy as np
import mink

TABLE_HEIGHT = 0.30
TRAY_CENTER = (0.24, -0.16)
TRAY_OPENING = 0.16
TRAY_WALL_HEIGHT = 0.05
TRAY_WALL_THICKNESS = 0.01
TRAY_FLOOR_TOP = TABLE_HEIGHT + 0.004
CUBE_SIZE = 0.04
N_EPISODES = 3

GRASP_DEPTH = 0.010  # TCP 1cm below top face: pads grip center, follower meshes do not pinch
TCP_FALLBACK_LOCAL = [0.0, 0.0, 0.13]

GRASP_Z = TABLE_HEIGHT + CUBE_SIZE - GRASP_DEPTH
PLACE_Z = TRAY_FLOOR_TOP + CUBE_SIZE - GRASP_DEPTH

# --- Arm servo tuning (stiff PD tracking without lag or sag) ---
ARM_KP = 800.0
ARM_KV = 80.0
ARM_FORCE = 120.0       # N*m cap per joint
MAX_LAG = 0.20          # rad; anti-windup command clipping
FINGER_FORCE_CAP = None  # leave XML limits


def create_task_scene(model_path, tcp_local=None):
    spec = mujoco.MjSpec.from_file(model_path)
    solref, solimp = [0.02, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0]

    if tcp_local is not None:
        spec.body("gripper_base").add_site(
            name="tcp", pos=list(tcp_local), size=[0.004, 0.004, 0.004],
            rgba=[0, 1, 0, 1],
        )

    # These integrator settings vastly improve gripping physics stability
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0

    spec.worldbody.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.25, -0.25, TABLE_HEIGHT - 0.015], size=[0.10, 0.20, 0.015],
        contype=1, conaffinity=1, rgba=[0.5, 0.35, 0.22, 1.0],
        friction=[1.0, 0.05, 0.0001], solref=solref, solimp=solimp,
    )

    tray_half = TRAY_OPENING / 2
    wall_half = TRAY_WALL_THICKNESS / 2
    outer_half = tray_half + TRAY_WALL_THICKNESS
    tray_x, tray_y = TRAY_CENTER

    spec.worldbody.add_geom(
        name="tray_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[tray_x, tray_y, TABLE_HEIGHT + 0.002],
        size=[outer_half, outer_half, 0.002],
        contype=1, conaffinity=1, rgba=[0.2, 0.55, 0.85, 1.0],
        friction=[1.0, 0.05, 0.0001], solref=solref, solimp=solimp,
    )

    wall_z = TABLE_HEIGHT + TRAY_WALL_HEIGHT / 2
    wall_specs = [
        ("px", tray_x + tray_half + wall_half, tray_y, wall_half, outer_half),
        ("nx", tray_x - tray_half - wall_half, tray_y, wall_half, outer_half),
        ("py", tray_x, tray_y + tray_half + wall_half, tray_half, wall_half),
        ("ny", tray_x, tray_y - tray_half - wall_half, tray_half, wall_half),
    ]
    for label, x, y, half_x, half_y in wall_specs:
        spec.worldbody.add_geom(
            name=f"tray_wall_{label}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, wall_z], size=[half_x, half_y, TRAY_WALL_HEIGHT / 2],
            contype=1, conaffinity=1, rgba=[0.2, 0.55, 0.85, 1.0],
            friction=[1.0, 0.05, 0.0001], solref=solref, solimp=solimp,
        )

    cube_body = spec.worldbody.add_body(name="target_cube", pos=[0, 0, 1.0])
    cube_body.add_freejoint(name="cube_freejoint")
    cube_body.add_geom(
        name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_SIZE / 2] * 3, contype=1, conaffinity=1,
        rgba=[0.86, 0.22, 0.18, 1.0], friction=[1.5, 0.05, 0.0001],
        mass=0.1, solref=solref, solimp=solimp,
    )

    mocap_body = spec.worldbody.add_body(name="target_mocap")
    mocap_body.mocap = True
    return spec.compile()


def compute_tcp_local(model, data):
    pad_ids = [
        g for g in range(model.ngeom)
        if "pad" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
    ]
    if not pad_ids:
        return TCP_FALLBACK_LOCAL
    bid = model.body("gripper_base").id
    R = data.xmat[bid].reshape(3, 3)
    center = data.geom_xpos[pad_ids].mean(axis=0)
    return R.T @ (center - data.xpos[bid])


def build_model(model_path):
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(model_path))
    try:
        probe = create_task_scene(model_path)
        probe_data = mujoco.MjData(probe)
        mujoco.mj_forward(probe, probe_data)
        tcp_local = compute_tcp_local(probe, probe_data)

        arm_joints = []
        for j in range(probe.njnt):
            if probe.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_JOINT, j)
            assert name, f"joint {j} is unnamed; name it in the XML"
            arm_joints.append(name)
        arm_joints = arm_joints[:6]

        # Note: We deliberately removed the clash detection here. We will hijack
        # the existing actuators dynamically in memory instead!

        return create_task_scene(model_path, tcp_local=tcp_local), arm_joints
    finally:
        os.chdir(original_dir)


def check_collisions(model, data):
    # Gripper-mechanism bodies are allowed near the table during grasp approach.
    # Only hard arm-link bodies contacting the table/tray constitute a failure.
    GRIPPER_BODIES = {
        "gripper_base",
        "left_driver", "right_driver",
        "left_coupler", "right_coupler",
        "left_spring_link", "right_spring_link",
        "left_follower", "right_follower",
        "left_pad", "right_pad",
    }
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > 0:
            continue
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]) or "world"
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]) or "world"

        # Never flag cube contacts (it is supposed to touch everything)
        if "target_cube" in (b1, b2):
            continue

        # Check for contact with table/tray surfaces
        is_table_contact = (
            "table_top" in (g1, g2)
            or "tray_floor" in (g1, g2)
            or "tray_wall" in g1 or "tray_wall" in g2
        )
        if is_table_contact:
            # Allow gripper-mechanism bodies (fingers) near the table during grasp
            if b1 in GRIPPER_BODIES or b2 in GRIPPER_BODIES:
                continue
            return True, f"Collision: {b1} ({g1 or 'unnamed'}) hit {b2} ({g2 or 'unnamed'})"
    return False, ""



def randomize_cube(model, data):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    adr = model.jnt_qposadr[jid]
    rand_x = random.uniform(0.28, 0.33)
    rand_y = random.uniform(-0.40, -0.28)
    data.qpos[adr:adr+3] = [rand_x, rand_y, TABLE_HEIGHT + CUBE_SIZE / 2 + 0.001]
    data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    data.qvel[model.jnt_dofadr[jid]:model.jnt_dofadr[jid]+6] = 0.0
    return rand_x, rand_y


def placement_failure(cube_pos, cube_rotation, cube_velocity):
    half_extents = np.abs(cube_rotation) @ np.full(3, CUBE_SIZE / 2)
    tray_half = TRAY_OPENING / 2
    if (abs(cube_pos[0] - TRAY_CENTER[0]) + half_extents[0] > tray_half
            or abs(cube_pos[1] - TRAY_CENTER[1]) + half_extents[1] > tray_half):
        return "place_miss (box is outside the tray opening)"

    floor_gap = cube_pos[2] - half_extents[2] - TRAY_FLOOR_TOP
    if abs(floor_gap) > 0.01:
        return "place_miss (box is not resting on the tray floor)"
    if np.linalg.norm(cube_velocity[:3]) > 0.03 or np.linalg.norm(cube_velocity[3:]) > 0.2:
        return "place_unsettled (box is still moving)"
    return None


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "Lab3", "robot_descriptions", "arm_gripper.xml")

    model, arm_joints = build_model(model_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # Gravity-compensate everything except the cube so the servos don't sag
    model.body_gravcomp[:] = 1.0
    model.body_gravcomp[model.body("target_cube").id] = 0.0

    arm_jids = [model.joint(n).id for n in arm_joints]
    arm_qadr = model.jnt_qposadr[arm_jids]
    arm_dadr = model.jnt_dofadr[arm_jids]

    # THE FIX: Dynamically hijack the existing XML actuators mapped to the arm
    # and turn them into strictly tuned PD Servos for safe compliant gripping.
    arm_act = []
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            if model.actuator_trnid[a, 0] in arm_jids:
                arm_act.append(a)
                model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
                model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
                model.actuator_gainprm[a, :] = 0.0
                model.actuator_gainprm[a, 0] = ARM_KP
                model.actuator_biasprm[a, :] = 0.0
                model.actuator_biasprm[a, 1] = -ARM_KP
                model.actuator_biasprm[a, 2] = -ARM_KV
                model.actuator_forcelimited[a] = 1
                model.actuator_forcerange[a, :] = [-ARM_FORCE, ARM_FORCE]

    fingers_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    # ctrl=0   -> zero tendon force -> spring opens fingers -> OPEN
    # ctrl=160 -> sufficient tendon force to hold 100g cube compliant without pinching/repulsion
    open_cmd  = float(model.actuator_ctrlrange[fingers_actuator, 0])   # 0
    close_cmd = 160.0

    # Ensure only pad boxes contact the cube (follower meshes are non-convex and pinch)
    for g in [30, 33, 34, 42, 45, 46]:
        model.geom_contype[g] = 0
        model.geom_conaffinity[g] = 0

    print("\nfingers_actuator gain/bias/forcerange:",
          model.actuator_gainprm[fingers_actuator, :3],
          model.actuator_biasprm[fingers_actuator, :3],
          model.actuator_forcerange[fingers_actuator])

    if FINGER_FORCE_CAP is not None:
        model.actuator_forcelimited[fingers_actuator] = 1
        model.actuator_forcerange[fingers_actuator] = [-FINGER_FORCE_CAP, FINGER_FORCE_CAP]

    configuration = mink.Configuration(model)
    tcp_task = mink.FrameTask(frame_name="tcp", frame_type="site",
                              position_cost=1.0, orientation_cost=1.0)
    posture_task = mink.PostureTask(model, cost=1e-2)
    downward_quat = np.array([0.0, 1.0, 0.0, 0.0])
    tcp_site_id = model.site("tcp").id
    mocap_id = model.body("target_mocap").mocapid[0]
    cube_body_id = model.body("target_cube").id
    cube_joint_id = model.joint("cube_freejoint").id
    cube_dofadr = model.jnt_dofadr[cube_joint_id]

    def ik_step(q_cmd, target_pos, target_quat):
        data.mocap_pos[mocap_id] = target_pos
        data.mocap_quat[mocap_id] = target_quat
        mujoco.mj_kinematics(model, data)
        tcp_task.set_target(mink.SE3.from_mocap_name(model, data, "target_mocap"))
        q_full = data.qpos.copy()
        q_full[arm_qadr] = q_cmd
        configuration.update(q_full)
        vel = mink.solve_ik(configuration, [tcp_task, posture_task], dt,
                            solver="daqp", damping=1e-2)
        return q_cmd + vel[arm_dadr] * dt

    print("\n--- Warming up IK to find safe starting pose ---")
    safe_home_pos = np.array([0.15, -0.20, TABLE_HEIGHT + 0.35])
    mujoco.mj_forward(model, data)
    posture_task.set_target(data.qpos.copy())
    q_cmd = data.qpos[arm_qadr].copy()
    for _ in range(500):
        q_cmd = ik_step(q_cmd, safe_home_pos, downward_quat)
        data.qpos[arm_qadr] = q_cmd
    home_q = q_cmd.copy()
    posture_target = data.qpos.copy()

    def physics_step(grip, q_cmd):
        q_act = data.qpos[arm_qadr]
        q_cmd = q_act + np.clip(q_cmd - q_act, -MAX_LAG, MAX_LAG)  # anti-windup
        data.ctrl[arm_act] = q_cmd
        data.ctrl[fingers_actuator] = grip
        mujoco.mj_step(model, data)
        return q_cmd

    with mujoco.viewer.launch_passive(model, data) as viewer:
        episode_results = []
        for episode in range(1, N_EPISODES + 1):
            if not viewer.is_running():
                break
            mujoco.mj_resetData(model, data)
            data.qpos[arm_qadr] = home_q
            cube_x, cube_y = randomize_cube(model, data)
            mujoco.mj_forward(model, data)
            initial_cube_z = data.xpos[cube_body_id, 2]
            posture_task.set_target(posture_target)

            q_cmd = home_q.copy()
            data.ctrl[arm_act] = q_cmd
            data.ctrl[fingers_actuator] = open_cmd
            for _ in range(100):
                q_cmd = physics_step(open_cmd, q_cmd)

            data.mocap_pos[mocap_id] = data.site_xpos[tcp_site_id]
            data.mocap_quat[mocap_id] = downward_quat

            tx, ty = TRAY_CENTER
            # PLACE_Z_RELEASE places the cube so its bottom rests gently on the tray floor (0.304)
            # TCP is 0.010 below top face -> cube bottom is 0.030 below TCP -> TCP at 0.334
            PLACE_Z_RELEASE = 0.334
            waypoints = [
                {"name": "hover_cube",     "pos": [cube_x, cube_y, TABLE_HEIGHT + 0.25], "grip": open_cmd,  "speed": 0.15},
                {"name": "pre_grasp",      "pos": [cube_x, cube_y, TABLE_HEIGHT + 0.15], "grip": open_cmd,  "speed": 0.10},
                {"name": "grasp_approach", "pos": [cube_x, cube_y, GRASP_Z],             "grip": open_cmd,  "speed": 0.02},
                {"name": "grasp",          "pos": [cube_x, cube_y, GRASP_Z],             "grip": close_cmd, "wait": 300, "ramp": True},
                {"name": "lift",           "pos": [cube_x, cube_y, TABLE_HEIGHT + 0.20], "grip": close_cmd, "speed": 0.03},
                {"name": "transit",        "pos": [tx, ty, TABLE_HEIGHT + 0.20],         "grip": close_cmd, "speed": 0.03},
                {"name": "place",          "pos": [tx, ty, PLACE_Z_RELEASE],             "grip": close_cmd, "speed": 0.02},
                {"name": "release",        "pos": [tx, ty, PLACE_Z_RELEASE],             "grip": open_cmd,  "wait": 400, "ramp": True},
                {"name": "retreat",        "pos": [tx, ty, TABLE_HEIGHT + 0.25],         "grip": open_cmd,  "speed": 0.05},
            ]

            success, reason = True, "Completed Successfully"
            print(f"\n[Episode {episode}] Cube: X={cube_x:.3f}, Y={cube_y:.3f}")

            prev_grip = open_cmd  # tracks grip at start of each waypoint's wait phase
            for wp in waypoints:
                if not success:
                    break
                if not viewer.is_running():
                    success, reason = False, "Viewer closed during episode"
                    break

                if wp["name"] == "retreat":
                    curr_tcp = data.site_xpos[tcp_site_id].copy()
                    start = curr_tcp.copy()
                    data.mocap_pos[mocap_id] = start
                    target_pos = np.array([curr_tcp[0], curr_tcp[1], TABLE_HEIGHT + 0.25])
                else:
                    target_pos = np.array(wp["pos"])
                    start = data.mocap_pos[mocap_id].copy()

                dist = np.linalg.norm(target_pos - start)
                n_steps = max(int((dist / wp.get("speed", 0.10)) / dt), 1)

                for step in range(n_steps):
                    if not viewer.is_running():
                        success, reason = False, "Viewer closed during episode"
                        break

                    alpha = (step + 1) / n_steps
                    q_cmd = ik_step(q_cmd, start + alpha * (target_pos - start), downward_quat)
                    q_cmd = physics_step(wp["grip"], q_cmd)
                    is_col, col_reason = check_collisions(model, data)
                    if is_col:
                        success, reason = False, f"{col_reason} during '{wp['name']}'"
                        break
                    if not (np.isfinite(data.qpos).all()
                            and np.isfinite(data.qvel).all()
                            and np.isfinite(data.qacc).all()):
                        success, reason = False, f"Non-finite simulation state during '{wp['name']}'"
                        break
                    viewer.sync()

                if not success:
                    break

                n_wait = wp.get("wait", 0)
                ramp_start = prev_grip   # grip value at the beginning of this wait
                for w in range(n_wait):
                    if not viewer.is_running():
                        success, reason = False, "Viewer closed during episode"
                        break
                    grip = wp["grip"]
                    if wp.get("ramp"):
                        p = min(1.0, w / (0.4 * n_wait))
                        grip = ramp_start + p * (wp["grip"] - ramp_start)
                    q_cmd = physics_step(grip, q_cmd)
                    is_col, col_reason = check_collisions(model, data)
                    if is_col:
                        success, reason = False, f"{col_reason} during '{wp['name']}'"
                        break
                    if not (np.isfinite(data.qpos).all()
                            and np.isfinite(data.qvel).all()
                            and np.isfinite(data.qacc).all()):
                        success, reason = False, f"Non-finite simulation state during '{wp['name']}'"
                        break
                    viewer.sync()

                # Update prev_grip so the next waypoint knows where we left off
                prev_grip = wp["grip"]

                if not success:
                    break

                cube_p = data.xpos[cube_body_id].copy()
                tcp_p = data.site_xpos[tcp_site_id].copy()
                print(f"    [{wp['name']}] TCP: {tcp_p.round(3)} | Cube: {cube_p.round(3)}")

                if wp["name"] == "lift":
                    cube_pos = data.xpos[cube_body_id]
                    if cube_pos[2] < initial_cube_z + CUBE_SIZE / 2:
                        success, reason = False, "grasp_miss (box did not lift with the gripper)"
                    elif np.linalg.norm(cube_pos - data.site_xpos[tcp_site_id]) > CUBE_SIZE:
                        success, reason = False, "grasp_slip (box is not held by the gripper)"
                elif wp["name"] in ("transit", "place"):
                    if np.linalg.norm(
                            data.xpos[cube_body_id] - data.site_xpos[tcp_site_id]) > CUBE_SIZE:
                        success, reason = False, f"grasp_slip (box lost during '{wp['name']}')"

            if success:
                # Allow cube 3.0s to settle completely under active arm holding
                for _ in range(max(1, int(3.0 / dt))):
                    if not viewer.is_running():
                        success, reason = False, "Viewer closed while settling"
                        break
                    q_cmd = physics_step(open_cmd, q_cmd)
                    is_col, col_reason = check_collisions(model, data)
                    if is_col:
                        success, reason = False, f"{col_reason} while settling"
                        break
                    viewer.sync()
                    if not (np.isfinite(data.qpos).all()
                            and np.isfinite(data.qvel).all()
                            and np.isfinite(data.qacc).all()):
                        success, reason = False, "Non-finite simulation state while settling"
                        break
                if success:
                    print(f"    [settled] Cube: {data.xpos[cube_body_id].round(3)} | Vel: {np.linalg.norm(data.qvel[cube_dofadr:cube_dofadr+3]):.3f}")
                    reason = placement_failure(
                        data.xpos[cube_body_id],
                        data.xmat[cube_body_id].reshape(3, 3),
                        data.qvel[cube_dofadr:cube_dofadr + 6],
                    )
                    if reason is None:
                        reason = "Completed Successfully"
                    else:
                        success = False

            print(f"  -> Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"  -> Reason: {reason}")
            episode_results.append((success, reason))
            if not viewer.is_running():
                break

        successes = sum(ok for ok, _ in episode_results)
        attempted = len(episode_results)
        print(f"\nSummary: {successes}/{attempted} successful "
              f"({100 * successes / attempted:.1f}%)" if attempted
              else "\nSummary: no episodes completed")
        failures = {}
        for ok, failure_reason in episode_results:
            if not ok:
                failures[failure_reason] = failures.get(failure_reason, 0) + 1
        for failure_reason, count in failures.items():
            print(f"  {count}x {failure_reason}")

    time.sleep(0.5)

if __name__ == "__main__":
    main()