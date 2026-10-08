import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("robotis_mujoco_menagerie/robotis_tb3/scene_turtlebot3_waffle_pi.xml")
data = mujoco.MjData(model)

CHASSIS_NAME = "base"
chassis_id = model.body(CHASSIS_NAME).id

SHOW_OVERLAY = True  # fixed on-screen matrix in the top-left corner


def keyboard_control(keycode):
    # W (87) or UP Arrow (265)
    if keycode == 87 or keycode == 265:
        data.ctrl[0] = 8.0
        data.ctrl[1] = 8.0
    # S (83) or DOWN Arrow (264)
    elif keycode == 83 or keycode == 264:
        data.ctrl[0] = -8.0
        data.ctrl[1] = -8.0
    # A (65) or LEFT Arrow (263)
    elif keycode == 65 or keycode == 263:
        data.ctrl[0] = -8.0
        data.ctrl[1] = 8.0
    # D (68) or RIGHT Arrow (262)
    elif keycode == 68 or keycode == 262:
        data.ctrl[0] = 8.0
        data.ctrl[1] = -8.0
    # Spacebar (32): Stop
    elif keycode == 32:
        data.ctrl[0] = 0.0
        data.ctrl[1] = 0.0


def draw_axis(scn, pos, mat, size=0.4, width=0.01):
    """Draw one RGB coordinate triad at `pos` with rotation `mat` (3x3)."""
    colors = [
        [1, 0, 0, 1],  # X = red
        [0, 1, 0, 1],  # Y = green
        [0, 0, 1, 1],  # Z = blue
    ]
    for axis in range(3):
        direction = mat[:, axis]
        end = pos + direction * size

        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9),
            np.array(colors[axis], dtype=np.float32)
        )
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_ARROW, width,
            pos, end
        )
        scn.ngeom += 1


def draw_matrix_label(scn, pos, mat, row_gap=0.12):
    """Render the rotation matrix as stacked labels (title + one per row) in the 3D view."""
    lines = [
        "Rotation Matrix (Body to Ground)",
        f"[ {mat[0,0]:+.2f}   {mat[0,1]:+.2f}   {mat[0,2]:+.2f} ]",
        f"[ {mat[1,0]:+.2f}   {mat[1,1]:+.2f}   {mat[1,2]:+.2f} ]",
        f"[ {mat[2,0]:+.2f}   {mat[2,1]:+.2f}   {mat[2,2]:+.2f} ]",
    ]
    for i, text in enumerate(lines):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        # first line sits highest, each following line drops by row_gap
        label_pos = pos + np.array([0, 0, -i * row_gap])
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.001, 0, 0]), label_pos, np.eye(3).flatten(),
            np.array([1, 1, 1, 0], dtype=np.float32)  # invisible anchor for the text
        )
        g.label = text
        scn.ngeom += 1


def matrix_overlay_text(mat):
    """Multi-line string of the matrix for the fixed on-screen overlay."""
    return "\n".join(
        f"[ {mat[r,0]:+.2f}  {mat[r,1]:+.2f}  {mat[r,2]:+.2f} ]" for r in range(3)
    )


with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_control) as viewer:

    # Turn OFF the built-in per-body frames, we draw our own instead
    with viewer.lock():
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

    print("\n=============================================")
    print("CONTROLS: WASD or Arrow Keys to move. SPACE to stop.")
    print("IMPORTANT: Click the 3D grid window first so it captures your keys!")
    print("=============================================\n")

    step_count = 0
    world_pos = np.zeros(3)
    world_mat = np.eye(3)

    while viewer.is_running():
        mujoco.mj_step(model, data)

        chassis_pos = data.xpos[chassis_id].copy()
        chassis_mat = data.xmat[chassis_id].reshape(3, 3).copy()

        # Reset custom scene geoms each frame
        viewer.user_scn.ngeom = 0

        draw_axis(viewer.user_scn, world_pos, world_mat, size=0.8, width=0.015)     # fixed ground frame
        draw_axis(viewer.user_scn, chassis_pos, chassis_mat, size=0.8, width=0.01)  # moving chassis frame
        draw_matrix_label(viewer.user_scn, chassis_pos + np.array([0, 0, 0.9]), chassis_mat)

        if SHOW_OVERLAY:
            viewer.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "Rotation Matrix (Body to Ground)",
                matrix_overlay_text(chassis_mat),
            ))

        # Throttled terminal print to keep the terminal clean
        if step_count % 10 == 0:
            print("\n--- Chassis-to-Ground Rotation Matrix ---")
            print(np.round(chassis_mat, 3))

        step_count += 1
        viewer.sync()
        time.sleep(0.01)
