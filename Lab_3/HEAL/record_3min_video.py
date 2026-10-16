"""
record_3min_video.py for HEAL - Records 1-minute video per IK method (Mink, DLS, QP)
with on-screen TCP trails, concatenating into HEAL_3min_IK_comparison.mp4.
"""
import os
import sys
import time
import random
import subprocess
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_pick_and_place_mink import (
    TABLE_HEIGHT, TRAY_CENTER, TRAY_OPENING, TRAY_WALL_HEIGHT,
    TRAY_WALL_THICKNESS, TRAY_FLOOR_TOP, CUBE_SIZE, GRASP_DEPTH,
    GRASP_Z, PLACE_Z, ARM_KP, ARM_KV, ARM_FORCE, MAX_LAG, FINGER_FORCE_CAP,
    build_model, placement_failure, check_collisions
)
from task12_custom_ik.custom_ik_solvers import DLSIK, QPIK, MinkIKSolver

METHODS = ["mink", "dls", "qp"]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
COLORS = {"mink": (25, 200, 255), "dls": (255, 140, 20), "qp": (80, 220, 50)}

def record_heal_method(method, out_mp4, duration_s=60, fps=25):
    current_dir = os.path.dirname(os.path.abspath(__file__))
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

    if method == "mink":
        solver = MinkIKSolver(model, dt=dt)
    elif method == "dls":
        solver = DLSIK(model, damping=1e-3, null_alpha=0.1)
    elif method == "qp":
        solver = QPIK(model, damping=1e-3, dt=dt, margin=0.05)

    downward_quat = np.array([0.0, 1.0, 0.0, 0.0])
    tcp_site_id = model.site("tcp").id
    mocap_id = model.body("target_mocap").mocapid[0]
    cube_body_id = model.body("target_cube").id
    cube_joint_id = model.joint("cube_freejoint").id
    cube_dofadr = model.jnt_dofadr[cube_joint_id]

    renderer = mujoco.Renderer(model, 480, 640, max_geom=10000)

    proc = subprocess.Popen([
        FFMPEG, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", "640x480", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", out_mp4
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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

    total_frames = int(duration_s * fps)
    frames_written = 0
    ep = 0
    trails = []
    old_trails = []

    print(f"\nRecording {duration_s}s video for HEAL ({method.upper()})...")

    while frames_written < total_frames:
        ep += 1
        random.seed(2000 + ep)
        np.random.seed(2000 + ep)

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
        for _ in range(50):
            q_cmd = physics_step(open_cmd, q_cmd)

        data.mocap_pos[mocap_id] = data.site_xpos[tcp_site_id]
        data.mocap_quat[mocap_id] = downward_quat

        if trails:
            old_trails.append(trails[::5])
            old_trails = old_trails[-4:]
        trails = []

        tx, ty = TRAY_CENTER
        PLACE_Z_RELEASE = 0.334
        waypoints = [
            {"name": "hover_cube",     "pos": [rand_x, rand_y, TABLE_HEIGHT + 0.25], "grip": open_cmd,  "speed": 0.20},
            {"name": "pre_grasp",      "pos": [rand_x, rand_y, TABLE_HEIGHT + 0.15], "grip": open_cmd,  "speed": 0.15},
            {"name": "grasp_approach", "pos": [rand_x, rand_y, GRASP_Z],             "grip": open_cmd,  "speed": 0.04},
            {"name": "grasp",          "pos": [rand_x, rand_y, GRASP_Z],             "grip": close_cmd, "wait": 150, "ramp": True},
            {"name": "lift",           "pos": [rand_x, rand_y, TABLE_HEIGHT + 0.20], "grip": close_cmd, "speed": 0.06},
            {"name": "transit",        "pos": [tx, ty, TABLE_HEIGHT + 0.20],         "grip": close_cmd, "speed": 0.06},
            {"name": "place",          "pos": [tx, ty, PLACE_Z_RELEASE],             "grip": close_cmd, "speed": 0.04},
            {"name": "release",        "pos": [tx, ty, PLACE_Z_RELEASE],             "grip": open_cmd,  "wait": 200, "ramp": True},
            {"name": "retreat",        "pos": [tx, ty, TABLE_HEIGHT + 0.25],         "grip": open_cmd,  "speed": 0.10},
        ]

        prev_grip = open_cmd
        step_counter = 0

        for wp in waypoints:
            if frames_written >= total_frames:
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
                q_cmd = res.q
                q_cmd = physics_step(wp["grip"], q_cmd)

                step_counter += 1
                if step_counter % 20 == 0 and frames_written < total_frames:
                    tcp_cur = data.site_xpos[tcp_site_id].copy()
                    trails.append(tcp_cur)
                    renderer.update_scene(data)
                    col_rgb = COLORS.get(method, (255, 255, 255))
                    for tr, alpha_tr in [(t, 0.3) for t in old_trails] + [(trails, 1.0)]:
                        for p in tr:
                            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                                break
                            g = renderer.scene.geoms[renderer.scene.ngeom]
                            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0],
                                                np.asarray(p, float), np.eye(3).flatten(),
                                                np.array([col_rgb[0]/255, col_rgb[1]/255, col_rgb[2]/255, alpha_tr], np.float32))
                            renderer.scene.ngeom += 1

                    img_arr = renderer.render().copy()
                    img = Image.fromarray(img_arr)
                    draw = ImageDraw.Draw(img)
                    draw.text((15, 15), f"Robot: 6-DOF HEAL | Method: {method.upper()} | Episode: {ep}", fill=(255, 255, 255))
                    draw.text((15, 35), f"Cube: ({rand_x:.2f}, {rand_y:.2f}) | Waypoint: {wp['name']}", fill=(220, 220, 220))
                    proc.stdin.write(np.array(img).tobytes())
                    frames_written += 1

            n_wait = wp.get("wait", 0)
            ramp_start = prev_grip
            for w in range(n_wait):
                grip = wp["grip"]
                if wp.get("ramp"):
                    p = min(1.0, w / (0.4 * n_wait))
                    grip = ramp_start + p * (wp["grip"] - ramp_start)
                q_cmd = physics_step(grip, q_cmd)
                step_counter += 1
                if step_counter % 20 == 0 and frames_written < total_frames:
                    renderer.update_scene(data)
                    img_arr = renderer.render().copy()
                    img = Image.fromarray(img_arr)
                    draw = ImageDraw.Draw(img)
                    draw.text((15, 15), f"Robot: 6-DOF HEAL | Method: {method.upper()} | Episode: {ep}", fill=(255, 255, 255))
                    draw.text((15, 35), f"Cube: ({rand_x:.2f}, {rand_y:.2f}) | Waypoint: {wp['name']} (wait)", fill=(220, 220, 220))
                    proc.stdin.write(np.array(img).tobytes())
                    frames_written += 1
            prev_grip = wp["grip"]

    proc.stdin.close()
    proc.wait()
    renderer.close()
    print(f"Finished {out_mp4} ({frames_written} frames)!")

def concat_heal_videos(v_files, final_output):
    list_file = "HEAL_concat_list.txt"
    with open(list_file, "w") as f:
        for vf in v_files:
            f.write(f"file '{os.path.basename(vf)}'\n")
    print(f"\nConcatenating HEAL videos into {final_output}...")
    cmd = [
        FFMPEG, "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", final_output
    ]
    subprocess.run(cmd, cwd=os.path.dirname(final_output) or ".", check=True)
    if os.path.exists(list_file):
        os.remove(list_file)
    print(f"Successfully generated final 3-minute video: {final_output}!")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    v_files = []
    for m in METHODS:
        out_v = os.path.join(current_dir, f"video_{m}.mp4")
        record_heal_method(m, out_v, duration_s=60, fps=25)
        v_files.append(out_v)
    final_output = os.path.join(current_dir, "HEAL_3min_IK_comparison.mp4")
    concat_heal_videos(v_files, final_output)

if __name__ == "__main__":
    main()
