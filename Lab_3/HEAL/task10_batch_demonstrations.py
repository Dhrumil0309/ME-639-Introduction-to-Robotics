import os
import sys
import time
import random
import csv
import mujoco
import numpy as np
import mink
import matplotlib.pyplot as plt

from run_pick_and_place_mink import (
    TABLE_HEIGHT, TRAY_CENTER, TRAY_OPENING, TRAY_WALL_HEIGHT,
    TRAY_WALL_THICKNESS, TRAY_FLOOR_TOP, CUBE_SIZE, GRASP_DEPTH,
    GRASP_Z, PLACE_Z, ARM_KP, ARM_KV, ARM_FORCE, MAX_LAG, FINGER_FORCE_CAP,
    build_model, placement_failure, check_collisions
)

N_EPISODES = 25
OUTPUT_DIR = "task10_batch"
CSV_PATH = os.path.join(OUTPUT_DIR, "batch_logs.csv")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "screenshots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def min_distance_to_obstacles(model, data):
    min_dist = float("inf")
    for c in range(data.ncon):
        con = data.contact[c]
        min_dist = min(min_dist, con.dist)
    return min_dist if min_dist != float("inf") else 0.05

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "Lab3", "robot_descriptions", "arm_gripper.xml")

    model, arm_joints = build_model(model_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    renderer = mujoco.Renderer(model, height=480, width=640)

    model.body_gravcomp[:] = 1.0
    model.body_gravcomp[model.body("target_cube").id] = 0.0

    arm_jids = [model.joint(n).id for n in arm_joints]
    arm_qadr = model.jnt_qposadr[arm_jids]
    arm_dadr = model.jnt_dofadr[arm_jids]

    arm_act = []
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] in arm_jids:
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
    open_cmd = float(model.actuator_ctrlrange[fingers_actuator, 0])
    close_cmd = 160.0

    for g in [30, 33, 34, 42, 45, 46]:
        model.geom_contype[g] = 0
        model.geom_conaffinity[g] = 0

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

    def physics_step(grip, q_cmd):
        q_act = data.qpos[arm_qadr]
        q_cmd = q_act + np.clip(q_cmd - q_act, -MAX_LAG, MAX_LAG)
        data.ctrl[arm_act] = q_cmd
        data.ctrl[fingers_actuator] = grip
        mujoco.mj_step(model, data)
        return q_cmd

    def capture_image(filename):
        renderer.update_scene(data)
        img = renderer.render()
        plt.imsave(filename, img)

    safe_home_pos = np.array([0.15, -0.20, TABLE_HEIGHT + 0.35])
    mujoco.mj_forward(model, data)
    posture_task.set_target(data.qpos.copy())
    q_cmd = data.qpos[arm_qadr].copy()
    for _ in range(500):
        q_cmd = ik_step(q_cmd, safe_home_pos, downward_quat)
        data.qpos[arm_qadr] = q_cmd
    home_q = q_cmd.copy()
    posture_target = data.qpos.copy()

    records = []
    print("==================================================")
    print(f"Starting Task 10: Running {N_EPISODES} Batch Demonstrations")
    print("==================================================")

    for episode in range(1, N_EPISODES + 1):
        random.seed(1000 + episode)
        np.random.seed(1000 + episode)

        mujoco.mj_resetData(model, data)
        data.qpos[arm_qadr] = home_q
        
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        adr = model.jnt_qposadr[jid]
        rand_x = random.uniform(0.28, 0.33)
        rand_y = random.uniform(-0.40, -0.28)
        data.qpos[adr:adr+3] = [rand_x, rand_y, TABLE_HEIGHT + CUBE_SIZE / 2 + 0.001]
        data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
        data.qvel[model.jnt_dofadr[jid]:model.jnt_dofadr[jid]+6] = 0.0
        mujoco.mj_forward(model, data)
        posture_task.set_target(posture_target)

        q_cmd = home_q.copy()
        data.ctrl[arm_act] = q_cmd
        data.ctrl[fingers_actuator] = open_cmd
        for _ in range(100):
            q_cmd = physics_step(open_cmd, q_cmd)

        data.mocap_pos[mocap_id] = data.site_xpos[tcp_site_id]
        data.mocap_quat[mocap_id] = downward_quat

        tx, ty = TRAY_CENTER
        PLACE_Z_RELEASE = 0.334
        waypoints = [
            {"name": "hover_cube",     "pos": [rand_x, rand_y, TABLE_HEIGHT + 0.25], "grip": open_cmd,  "speed": 0.15},
            {"name": "pre_grasp",      "pos": [rand_x, rand_y, TABLE_HEIGHT + 0.15], "grip": open_cmd,  "speed": 0.10},
            {"name": "grasp_approach", "pos": [rand_x, rand_y, GRASP_Z],             "grip": open_cmd,  "speed": 0.02},
            {"name": "grasp",          "pos": [rand_x, rand_y, GRASP_Z],             "grip": close_cmd, "wait": 300, "ramp": True},
            {"name": "lift",           "pos": [rand_x, rand_y, TABLE_HEIGHT + 0.20], "grip": close_cmd, "speed": 0.03},
            {"name": "transit",        "pos": [tx, ty, TABLE_HEIGHT + 0.20],         "grip": close_cmd, "speed": 0.03},
            {"name": "place",          "pos": [tx, ty, PLACE_Z_RELEASE],             "grip": close_cmd, "speed": 0.02},
            {"name": "release",        "pos": [tx, ty, PLACE_Z_RELEASE],             "grip": open_cmd,  "wait": 400, "ramp": True},
            {"name": "retreat",        "pos": [tx, ty, TABLE_HEIGHT + 0.25],         "grip": open_cmd,  "speed": 0.05},
        ]

        t0_solve = time.perf_counter()
        ik_iterations = 0
        success = True
        reason = "Completed Successfully"
        prev_grip = open_cmd

        for wp in waypoints:
            if not success:
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
                alpha = (step + 1) / n_steps
                q_cmd = ik_step(q_cmd, start + alpha * (target_pos - start), downward_quat)
                ik_iterations += 1
                q_cmd = physics_step(wp["grip"], q_cmd)

                is_col, col_reason = check_collisions(model, data)
                if is_col:
                    success, reason = False, f"{col_reason} during '{wp['name']}'"
                    break
                if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                    success, reason = False, f"Numerical instability during '{wp['name']}'"
                    break

            if not success:
                break

            n_wait = wp.get("wait", 0)
            ramp_start = prev_grip
            for w in range(n_wait):
                grip = wp["grip"]
                if wp.get("ramp"):
                    p = min(1.0, w / (0.4 * n_wait))
                    grip = ramp_start + p * (wp["grip"] - ramp_start)
                q_cmd = physics_step(grip, q_cmd)
                is_col, col_reason = check_collisions(model, data)
                if is_col:
                    success, reason = False, f"{col_reason} during '{wp['name']}' wait"
                    break
            prev_grip = wp["grip"]

        if success:
            for _ in range(max(1, int(3.0 / dt))):
                q_cmd = physics_step(open_cmd, q_cmd)
                is_col, col_reason = check_collisions(model, data)
                if is_col:
                    success, reason = False, f"{col_reason} while settling"
                    break
            if success:
                reason = placement_failure(
                    data.xpos[cube_body_id],
                    data.xmat[cube_body_id].reshape(3, 3),
                    data.qvel[cube_dofadr:cube_dofadr + 6],
                )
                if reason is None:
                    reason = "Completed Successfully"
                else:
                    success = False

        time_to_solve = time.perf_counter() - t0_solve
        min_clearance = min_distance_to_obstacles(model, data)

        status_tag = "SUCCESS" if success else "FAILURE"
        img_filename = os.path.join(IMAGES_DIR, f"episode_{episode:02d}_{status_tag}.png")
        capture_image(img_filename)

        print(f"[Episode {episode:02d}/25] Init: ({rand_x:.3f}, {rand_y:.3f}) | Result: {status_tag} | Reason: {reason} | Time: {time_to_solve:.2f}s | Image: {os.path.basename(img_filename)}")

        records.append({
            "episode": episode,
            "cube_x": round(rand_x, 4),
            "cube_y": round(rand_y, 4),
            "success": success,
            "failure_reason": reason if not success else "",
            "min_distance_margin_m": round(min_clearance, 4),
            "time_to_solve_s": round(time_to_solve, 3),
            "ik_iterations": ik_iterations,
            "screenshot": img_filename
        })

    keys = records[0].keys()
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    successes = sum(r["success"] for r in records)
    mean_time = np.mean([r["time_to_solve_s"] for r in records])
    mean_iters = np.mean([r["ik_iterations"] for r in records])
    print(f"\n================ Task 10 Summary ================")
    print(f"Total Episodes: {N_EPISODES}")
    print(f"Success Rate: {successes}/{N_EPISODES} ({100 * successes / N_EPISODES:.1f}%)")
    print(f"Mean Time to Solve: {mean_time:.2f} s")
    print(f"Mean IK Iterations: {mean_iters:.1f}")
    print(f"Saved CSV to: {CSV_PATH}")
    print(f"Saved Screenshots to: {IMAGES_DIR}")

if __name__ == "__main__":
    main()
