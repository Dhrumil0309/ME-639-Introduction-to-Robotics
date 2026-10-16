import os
import sys
import time
import random
import csv
from collections import defaultdict
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_pick_and_place_mink import (
    TABLE_HEIGHT, TRAY_CENTER, TRAY_OPENING, TRAY_WALL_HEIGHT,
    TRAY_WALL_THICKNESS, TRAY_FLOOR_TOP, CUBE_SIZE, GRASP_DEPTH,
    GRASP_Z, PLACE_Z, ARM_KP, ARM_KV, ARM_FORCE, MAX_LAG, FINGER_FORCE_CAP,
    build_model, placement_failure, check_collisions
)

from task12_custom_ik.custom_ik_solvers import DLSIK, QPIK, MinkIKSolver

METHODS = ["mink", "dls", "qp"]
METHOD_NAMES = {
    "mink": "Mink (Off-the-shelf)",
    "dls": "DLS-IK (Closed-loop Damped)",
    "qp": "QP-IK (Joint & Velocity Constrained)"
}
METHOD_COLORS = {"mink": "#1f77b4", "dls": "#ff7f0e", "qp": "#2ca02c"}

OUTPUT_DIR = "task13_comparison"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

N_EPISODES = 25

def run_experiment_for_method(method_name, seeds):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(current_dir, "Lab3", "robot_descriptions", "arm_gripper.xml")
    model, arm_joints = build_model(model_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

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

    if method_name == "mink":
        solver = MinkIKSolver(model, dt=dt)
    elif method_name == "dls":
        solver = DLSIK(model, damping=1e-3, null_alpha=0.1)
    elif method_name == "qp":
        solver = QPIK(model, damping=1e-3, dt=dt, margin=0.05)

    downward_quat = np.array([0.0, 1.0, 0.0, 0.0])
    tcp_site_id = model.site("tcp").id
    mocap_id = model.body("target_mocap").mocapid[0]
    cube_body_id = model.body("target_cube").id
    cube_joint_id = model.joint("cube_freejoint").id
    cube_dofadr = model.jnt_dofadr[cube_joint_id]

    def physics_step(grip, q_cmd):
        q_act = data.qpos[arm_qadr]
        q_cmd = q_act + np.clip(q_cmd - q_act, -MAX_LAG, MAX_LAG)
        data.ctrl[arm_act] = q_cmd
        data.ctrl[fingers_actuator] = grip
        mujoco.mj_step(model, data)
        return q_cmd

    safe_home_pos = np.array([0.15, -0.20, TABLE_HEIGHT + 0.35])
    mujoco.mj_forward(model, data)
    res_home = solver.solve(data.qpos[arm_qadr].copy(), safe_home_pos, downward_quat)
    home_q = res_home.q.copy()
    data.qpos[arm_qadr] = home_q
    mujoco.mj_forward(model, data)

    records = []
    print(f"\n---> Running Method: {METHOD_NAMES[method_name]} across {len(seeds)} episodes...")

    for ep_idx, seed in enumerate(seeds, 1):
        random.seed(seed)
        np.random.seed(seed)

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
        joint_limit_viols = 0
        vel_viols = 0
        max_vel = 0.0
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
                des_pos = start + alpha * (target_pos - start)
                res = solver.solve(q_cmd, des_pos, downward_quat)
                ik_iterations += res.iters

                dq = (res.q - q_cmd) / dt
                max_v = np.max(np.abs(dq))
                max_vel = max(max_vel, max_v)
                if max_v > np.pi + 0.05:
                    vel_viols += 1
                for j_idx in range(6):
                    j_range = model.joint(arm_joints[j_idx]).range
                    if res.q[j_idx] < j_range[0] - 1e-4 or res.q[j_idx] > j_range[1] + 1e-4:
                        joint_limit_viols += 1

                q_cmd = res.q
                q_cmd = physics_step(wp["grip"], q_cmd)

                is_col, col_reason = check_collisions(model, data)
                if is_col:
                    success, reason = False, f"{col_reason} during '{wp['name']}'"
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
            reason = placement_failure(
                data.xpos[cube_body_id],
                data.xmat[cube_body_id].reshape(3, 3),
                data.qvel[cube_dofadr:cube_dofadr + 6],
            )
            if reason is None:
                reason = "Completed Successfully"
            else:
                success = False

        total_time = time.perf_counter() - t0_solve
        records.append({
            "method": method_name,
            "episode": ep_idx,
            "seed": seed,
            "cube_x": round(rand_x, 4),
            "cube_y": round(rand_y, 4),
            "success": success,
            "failure_reason": reason if not success else "",
            "time_to_solve_s": round(total_time, 3),
            "ik_iterations": ik_iterations,
            "joint_limit_violations": joint_limit_viols,
            "velocity_violations": vel_viols,
            "max_joint_vel": round(max_vel, 3)
        })

    return records

def generate_comparison_plots(records):
    print("\nGenerating Task 13 Comparison Plots...")

    methods_order = [m for m in METHODS if any(r["method"] == m for r in records)]
    labels = [METHOD_NAMES[m] for m in methods_order]
    colors = [METHOD_COLORS[m] for m in methods_order]

    # 1. Success Rate Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = []
    for m in methods_order:
        m_rec = [r for r in records if r["method"] == m]
        succ = sum(1 for r in m_rec if r["success"]) / len(m_rec) * 100
        vals.append(succ)

    bars = ax.bar(labels, vals, color=colors, width=0.45)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Success Rate Comparison Across IK Methods (25 Identical Seeds)", fontsize=12, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "cmp_success_rate.png"), dpi=150)
    plt.close(fig)

    # 2. Planning Time & IK Iterations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    time_data = [[r["time_to_solve_s"] for r in records if r["method"] == m] for m in methods_order]
    bp1 = ax1.boxplot(time_data, patch_artist=True, tick_labels=labels)
    for patch, m in zip(bp1['boxes'], methods_order):
        patch.set_facecolor(METHOD_COLORS[m])
    ax1.set_ylabel("Planning Time (s)", fontsize=11)
    ax1.set_title("Mean Time to Solve", fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)

    iter_data = [[r["ik_iterations"] for r in records if r["method"] == m] for m in methods_order]
    bp2 = ax2.boxplot(iter_data, patch_artist=True, tick_labels=labels)
    for patch, m in zip(bp2['boxes'], methods_order):
        patch.set_facecolor(METHOD_COLORS[m])
    ax2.set_ylabel("IK Iterations per Episode", fontsize=11)
    ax2.set_title("IK Convergence Iterations", fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "cmp_time_and_iters.png"), dpi=150)
    plt.close(fig)

    # 3. Constraint Violations
    fig, ax = plt.subplots(figsize=(7, 4.5))
    joint_viols = [sum(r["joint_limit_violations"] for r in records if r["method"] == m) for m in methods_order]
    vel_viols = [sum(r["velocity_violations"] for r in records if r["method"] == m) for m in methods_order]
    x = np.arange(len(methods_order))
    width = 0.35
    ax.bar(x - width/2, joint_viols, width, label="Joint Limit Violations", color="#d62728")
    ax.bar(x + width/2, vel_viols, width, label="Velocity Violations", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Violations (25 Episodes)", fontsize=11)
    ax.set_title("Joint Limit & Velocity Constraint Violations", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "cmp_constraint_violations.png"), dpi=150)
    plt.close(fig)
    print(f"All comparison plots saved to: {PLOTS_DIR}")

def main():
    seeds = [1000 + i for i in range(1, N_EPISODES + 1)]
    all_records = []

    for method in METHODS:
        records = run_experiment_for_method(method, seeds)
        all_records.extend(records)

    csv_file = os.path.join(OUTPUT_DIR, "ik_comparison_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nSaved all comparison logs to: {csv_file}")

    # Generate Statistical Summary Table
    summary = []
    for m in METHODS:
        m_rec = [r for r in all_records if r["method"] == m]
        summary.append({
            "Method": METHOD_NAMES[m],
            "Episodes": len(m_rec),
            "Success Rate (%)": round(sum(1 for r in m_rec if r["success"]) / len(m_rec) * 100, 1),
            "Mean Time (s)": round(np.mean([r["time_to_solve_s"] for r in m_rec]), 2),
            "Mean IK Iters": round(np.mean([r["ik_iterations"] for r in m_rec]), 1),
            "Joint Violations": int(sum(r["joint_limit_violations"] for r in m_rec)),
            "Velocity Violations": int(sum(r["velocity_violations"] for r in m_rec)),
            "Max Joint Vel (rad/s)": round(max(r["max_joint_vel"] for r in m_rec), 2)
        })

    sum_csv = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    with open(sum_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\n================ Task 13 Comparison Summary ================")
    for row in summary:
        print(f"{row['Method']}: Success={row['Success Rate (%)']}%, Time={row['Mean Time (s)']}s, Iters={row['Mean IK Iters']}, JointViol={row['Joint Violations']}, VelViol={row['Velocity Violations']}")

    generate_comparison_plots(all_records)

if __name__ == "__main__":
    main()
