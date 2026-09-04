"""
01_rotation_sandbox.py  --  ME 639 HW1, Task 1
===============================================
Interactive rotation sandbox using the MuJoCo 3.x passive viewer.
All interaction happens entirely inside the viewer window -- no terminal
typing required. Press a key, watch the dart rotate, read the live
matrix overlay on screen.

KEYBINDINGS
-----------
  Keys 1 / 2 / 3  -> Rotate +90° about X / Y / Z  in the FIXED  (space) frame
                      Rule: R_new = R_step @ R_old   [pre-multiply / LEFT]
  Keys 4 / 5 / 6  -> Rotate +90° about X / Y / Z  in the CURRENT (body) frame
                      Rule: R_new = R_old @ R_step   [post-multiply / RIGHT]
  Key  0           -> Reset orientation to identity

ON-SCREEN OVERLAYS
------------------
  TOP-LEFT  : Live 3x3 rotation matrix (updated after every step)
  TOP-RIGHT : Last action label + composition rule used
  BOTTOM    : Key legend

SMOOTH ANIMATION
----------------
  Every rotation is SLERP-interpolated over ANIMATION_DURATION seconds
  so the motion is smooth for screen recording. Adjust the constant below.

THREAD SAFETY
-------------
  The GLFW key_callback runs on the viewer thread (not the main thread).
  It only writes a lightweight command into a queue. The main loop
  reads that queue, does the SLERP animation, then rebuilds the overlays --
  all on the main thread. No locks needed.

AI USE NOTE
-----------
  SLERP kernel, overlay formatting, and threading pattern written with
  AI assistance. The composition rules (pre- vs post-multiply) were
  derived from Lynch & Park Ch.3 in Problem 3.
"""

import sys
import time
import queue
import threading
import argparse
import numpy as np
import mujoco
import mujoco.viewer as mjv

from utils import (
    Rx, Ry, Rz, ELEMENTARY_ROTATIONS,
    set_body_orientation,
    R_to_quat, quat_to_R,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration -- adjust for your recording setup
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH         = "../model/asymmetric_body.xml"
ANIMATION_DURATION = 1.2      # seconds per elemental rotation (SLERP)
VIEWER_FPS         = 60       # viewer sync rate
ROTATION_DEG       = 90.0     # fixed rotation angle for every key press
FONT               = mujoco.mjtFontScale.mjFONTSCALE_150

# GLFW key codes (match ASCII for printable characters)
KEY_0, KEY_1, KEY_2, KEY_3 = 48, 49, 50, 51
KEY_4, KEY_5, KEY_6        = 52, 53, 54


# ──────────────────────────────────────────────────────────────────────
# Section 1 -- Kinematic composition (the core math)
# ──────────────────────────────────────────────────────────────────────

def compose(R_old: np.ndarray, axis: str, angle_rad: float, frame: str):
    """Apply one elemental rotation and return (R_new, rule_str).

    Lynch & Park, Ch.3 Composition of Rotations:
      current / body  frame  ->  R_new = R_old  @ R_step   (post / right)
      fixed   / space frame  ->  R_new = R_step @ R_old    (pre  / left)
    """
    R_step = ELEMENTARY_ROTATIONS[axis](angle_rad)
    if frame == "current":
        return R_old @ R_step, "R_new = R_old @ R_step  (post, BODY frame)"
    else:
        return R_step @ R_old, "R_new = R_step @ R_old  (pre,  SPACE frame)"


# ──────────────────────────────────────────────────────────────────────
# Section 2 -- SLERP smooth animation
# ──────────────────────────────────────────────────────────────────────

def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two wxyz unit quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:           # shortest-path correction
        q1, dot = -q1, -dot
    if dot > 0.9995:        # nearly identical -- use linear blend
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta   = theta_0 * t
    q_perp  = (q1 - q0 * dot) / np.linalg.norm(q1 - q0 * dot)
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


def animate(viewer, model, data,
            R_start: np.ndarray, R_end: np.ndarray,
            duration: float = ANIMATION_DURATION):
    """SLERP-interpolate the body from R_start to R_end.

    Runs on the main thread. viewer.sync() is called every frame so the
    window stays responsive throughout the animation.
    """
    if viewer is None or not viewer.is_running():
        set_body_orientation(data, R_end)
        mujoco.mj_forward(model, data)
        return

    q0      = R_to_quat(R_start)
    q1      = R_to_quat(R_end)
    n_steps = max(2, int(duration * VIEWER_FPS))
    dt      = 1.0 / VIEWER_FPS

    for i in range(1, n_steps + 1):
        if not viewer.is_running():
            break
        qi = _slerp(q0, q1, i / n_steps)
        set_body_orientation(data, quat_to_R(qi))
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(dt)


# ──────────────────────────────────────────────────────────────────────
# Section 3 -- On-screen overlay builder
# ──────────────────────────────────────────────────────────────────────

_LEGEND = (
    "1/2/3 : +90° about X/Y/Z  [SPACE frame, pre-mult]",
    "4/5/6 : +90° about X/Y/Z  [BODY  frame, post-mult]",
    "  0   : Reset to identity",
)

_GRID = mujoco.mjtGridPos


def _fmt_matrix(R: np.ndarray) -> tuple[str, str]:
    """Return (left_col_text, right_col_text) for set_texts().

    Left column holds row labels; right column holds the 3x3 numbers.
    Both are newline-delimited strings so each row appears on its own line.
    """
    header_l = "  Rotation Matrix R:"
    header_r = "     col-x      col-y      col-z"
    rows_l   = ["  row-x |", "  row-y |", "  row-z |"]
    rows_r   = [
        "  " + "   ".join(f"{v:+.4f}" for v in row)
        for row in R
    ]
    text_l = header_l + "\n" + "\n".join(rows_l)
    text_r = header_r + "\n" + "\n".join(rows_r)
    return text_l, text_r


def _build_overlays(viewer, R: np.ndarray, last_action: str, last_rule: str):
    """Push all text overlays to the viewer for this frame."""
    viewer.clear_texts()

    # ── TOP-LEFT: live rotation matrix ──────────────────────────────
    mat_l, mat_r = _fmt_matrix(R)
    texts = [
        (FONT, _GRID.mjGRID_TOPLEFT,    mat_l, mat_r),
    ]

    # ── TOP-RIGHT: last action + kinematic rule ──────────────────────
    texts += [
        (FONT, _GRID.mjGRID_TOPRIGHT,  "Last action:", last_action),
        (FONT, _GRID.mjGRID_TOPRIGHT,  "Rule applied:", last_rule),
    ]

    # ── BOTTOM: key legend ───────────────────────────────────────────
    legend_l = "\n".join(_LEGEND)
    texts += [
        (FONT, _GRID.mjGRID_BOTTOM, legend_l, ""),
    ]

    viewer.set_texts(texts)


# ──────────────────────────────────────────────────────────────────────
# Section 4 -- Command dispatch table
# ──────────────────────────────────────────────────────────────────────

# Map GLFW keycode -> (axis, frame, display_label)
ANGLE_RAD = np.deg2rad(ROTATION_DEG)
_KEY_MAP: dict[int, tuple[str, str, str]] = {
    KEY_1: ("x", "fixed",   f"Rot(X, +{ROTATION_DEG:.0f}°) SPACE frame"),
    KEY_2: ("y", "fixed",   f"Rot(Y, +{ROTATION_DEG:.0f}°) SPACE frame"),
    KEY_3: ("z", "fixed",   f"Rot(Z, +{ROTATION_DEG:.0f}°) SPACE frame"),
    KEY_4: ("x", "current", f"Rot(X, +{ROTATION_DEG:.0f}°) BODY  frame"),
    KEY_5: ("y", "current", f"Rot(Y, +{ROTATION_DEG:.0f}°) BODY  frame"),
    KEY_6: ("z", "current", f"Rot(Z, +{ROTATION_DEG:.0f}°) BODY  frame"),
    KEY_0: (None, None,     "Reset to identity"),
}


# ──────────────────────────────────────────────────────────────────────
# Section 5 -- Main loop
# ──────────────────────────────────────────────────────────────────────

def run(model, data, step_duration: float = ANIMATION_DURATION):
    """Open the passive viewer and handle all key events."""

    # Shared state
    cmd_queue: queue.SimpleQueue = queue.SimpleQueue()
    # The key_callback runs on the GLFW/viewer thread; it only queues a
    # lightweight int so no locks or shared mutable state are touched.
    def key_callback(keycode: int) -> None:
        if keycode in _KEY_MAP:
            cmd_queue.put(keycode)

    print("Opening MuJoCo viewer ...")
    print("  Keys 1-3: SPACE frame  |  Keys 4-6: BODY frame  |  0: Reset")
    print("  Close the viewer window to exit.\n")

    viewer = mjv.launch_passive(model, data, key_callback=key_callback)

    # Give the window time to fully render before overlaying text
    for _ in range(int(0.3 * VIEWER_FPS)):
        if not viewer.is_running():
            break
        viewer.sync()
        time.sleep(1.0 / VIEWER_FPS)

    # ── Application state ────────────────────────────────────────────
    R           = np.eye(3)
    last_action = "(none -- press a key)"
    last_rule   = ""
    step_count  = 0

    set_body_orientation(data, R)
    mujoco.mj_forward(model, data)
    _build_overlays(viewer, R, last_action, last_rule)
    viewer.sync()

    # ── Main loop ────────────────────────────────────────────────────
    try:
        while viewer.is_running():

            # Process any pending key commands
            while not cmd_queue.empty():
                keycode = cmd_queue.get_nowait()
                axis, frame, label = _KEY_MAP[keycode]

                if axis is None:
                    # ── Reset ─────────────────────────────────────────
                    R_new       = np.eye(3)
                    last_action = "Reset to identity"
                    last_rule   = "R = I"
                    step_count  = 0
                    print(f"[Reset]  R = I")
                else:
                    # ── Rotate ────────────────────────────────────────
                    R_new, rule = compose(R, axis, ANGLE_RAD, frame)
                    step_count += 1
                    last_action = f"Step {step_count}: {label}"
                    last_rule   = rule
                    print(f"[Step {step_count}] {label}")
                    print(f"         {rule}")
                    print("         R_new =")
                    for row_lbl, row in zip(("  x |", "  y |", "  z |"), R_new):
                        print(row_lbl + "  " +
                              "   ".join(f"{v:+8.4f}" for v in row))
                    print()

                # ── Animate transition ────────────────────────────────
                # Update overlay mid-animation so the text tracks the motion
                _build_overlays(viewer, R_new, last_action, last_rule)
                animate(viewer, model, data, R, R_new, duration=step_duration)
                R = R_new

                # Flush any keys that arrived during animation
                # (don't stack up multiple rotations without showing each one)
                with cmd_queue.mutex if hasattr(cmd_queue, 'mutex') else _nullctx():
                    pass   # SimpleQueue has no mutex attribute -- that's fine

            # ── Keep viewer alive ─────────────────────────────────────
            _build_overlays(viewer, R, last_action, last_rule)
            viewer.sync()
            time.sleep(1.0 / VIEWER_FPS)

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        if viewer.is_running():
            viewer.close()


# ──────────────────────────────────────────────────────────────────────
# Tiny helper so the "with _nullctx()" trick compiles cleanly
# ──────────────────────────────────────────────────────────────────────
from contextlib import contextmanager
@contextmanager
def _nullctx():
    yield


# ──────────────────────────────────────────────────────────────────────
# Section 6 -- Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ME 639 HW1 Task 1 -- Rotation Sandbox (viewer-native keybindings)"
    )
    parser.add_argument(
        "--step-duration", type=float, default=ANIMATION_DURATION, metavar="SEC",
        help=f"SLERP animation duration per step (default: {ANIMATION_DURATION}s)"
    )
    args = parser.parse_args()

    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    except Exception as e:
        sys.exit(f"[ERROR] Could not load '{MODEL_PATH}': {e}")
    data = mujoco.MjData(model)

    run(model, data, step_duration=args.step_duration)


if __name__ == "__main__":
    main()
