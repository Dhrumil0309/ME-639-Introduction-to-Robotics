import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import tkinter as tk

ROBOTS = {
    "1": dict(name="Franka Emika Panda (7-DOF)",
              path="mujoco_menagerie/franka_emika_panda/scene.xml", ee="hand"),
    "2": dict(name="Universal Robots UR5e (6-DOF)",
              path="mujoco_menagerie/universal_robots_ur5e/scene.xml", ee="wrist_3_link"),
}

print("Which official robot do you want to simulate?")
for k, r in ROBOTS.items():
    print(f"{k}: {r['name']}")
choice = input("Enter 1 or 2: ").strip()
if choice not in ROBOTS:
    print("Invalid choice.")
    exit()

cfg = ROBOTS[choice]
if not os.path.exists(cfg["path"]):
    print(f"\nERROR: Cannot find '{cfg['path']}'.")
    exit()

# ---------------------------------------------------------------- model
model = mujoco.MjModel.from_xml_path(cfg["path"])
data = mujoco.MjData(model)

ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["ee"])
if ee_id < 0:
    print(f"ERROR: body '{cfg['ee']}' not found in the model.")
    exit()

# start from the model's home pose instead of all-zero targets
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
home_ctrl = data.ctrl.copy()


# ---------------------------------------------------------------- helpers
def draw_axis(scn, pos, mat, size=0.15, width=0.006):
    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]  # X, Y, Z
    for axis in range(3):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                            np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(colors[axis], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                             pos, pos + mat[:, axis] * size)
        scn.ngeom += 1


# ---------------------------------------------------------------- slider window
root = tk.Tk()
root.title(f"Joint control: {cfg['name']}")
alive = [True]
root.protocol("WM_DELETE_WINDOW", lambda: alive.__setitem__(0, False))

vars_, scales = [], []
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i + 1}"

    is_angle = (model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and
                model.jnt_type[model.actuator_trnid[i, 0]] == mujoco.mjtJoint.mjJNT_HINGE)
    scale = 180.0 / np.pi if is_angle else 1.0
    unit = "deg" if is_angle else "raw"

    if model.actuator_ctrllimited[i]:
        lo, hi = model.actuator_ctrlrange[i]
    else:
        lo, hi = -np.pi, np.pi

    var = tk.DoubleVar(value=home_ctrl[i] * scale)
    row = tk.Frame(root)
    row.pack(fill="x", padx=8, pady=2)
    tk.Label(row, text=f"{name} [{unit}]", width=22, anchor="w").pack(side="left")
    tk.Scale(row, from_=lo * scale, to=hi * scale, orient="horizontal",
             resolution=0.5 if is_angle else 1.0, length=380,
             variable=var).pack(side="left")
    vars_.append(var)
    scales.append(scale)


def set_all(values):
    for v, s, val in zip(vars_, scales, values):
        v.set(val * s)


btns = tk.Frame(root)
btns.pack(pady=6)
tk.Button(btns, text="Reset to home", width=16,
          command=lambda: set_all(home_ctrl)).pack(side="left", padx=4)
tk.Button(btns, text="Zero all", width=16,
          command=lambda: set_all(np.zeros(model.nu))).pack(side="left", padx=4)

ee_label = tk.Label(root, text="", font=("Courier", 12), pady=8)
ee_label.pack()

# ---------------------------------------------------------------- main loop
SUBSTEPS = max(1, int(round(0.02 / model.opt.timestep)))   # ~50 Hz refresh, real time

with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE   # we draw our own frames

    print("Use the slider window to move the robot (click a slider, then use the arrow keys for fine steps).")
    print("The viewer's own Control panel is overridden by the sliders.")

    while viewer.is_running() and alive[0]:
        t0 = time.time()

        data.ctrl[:] = [v.get() / s for v, s in zip(vars_, scales)]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        pos = data.xpos[ee_id].copy()
        mat = data.xmat[ee_id].reshape(3, 3).copy()

        viewer.user_scn.ngeom = 0
        draw_axis(viewer.user_scn, np.zeros(3), np.eye(3), size=0.3, width=0.008)  # world frame
        draw_axis(viewer.user_scn, pos, mat)                                       # end-effector frame

        try:
            ee_label.config(text=f"End-effector (m)\nX: {pos[0]:+.4f}\nY: {pos[1]:+.4f}\nZ: {pos[2]:+.4f}")
            root.update()
        except tk.TclError:
            break

        viewer.sync()
        left = 0.02 - (time.time() - t0)
        if left > 0:
            time.sleep(left)

try:
    root.destroy()
except tk.TclError:
    pass
print("\nSimulation closed.")
