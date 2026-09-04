"""
01_rotation_sandbox.py  --  ME 639 HW1, Task 1
===============================================
Interactive rotation sandbox using the MuJoCo 3.x passive viewer.
Everything happens inside the 3D viewer window: press a key, watch
the dart SLERP-animate smoothly, read the live 3×3 matrix on screen.

KEYBINDINGS
-----------
  Digits 0-9  : Build a custom rotation angle (type digits then press 1-6)
  '-'         : Negate the angle currently in the buffer
  Backspace   : Delete the last digit from the buffer
  Enter       : Lock in the typed angle (does not rotate by itself)

  Key 1 / 2 / 3  -> Rotate ±angle° about X / Y / Z  in FIXED  (space) frame
                     Rule: R_new = R_step @ R_old   [pre-multiply / LEFT]
  Key 4 / 5 / 6  -> Rotate ±angle° about X / Y / Z  in CURRENT (body) frame
                     Rule: R_new = R_old @ R_step   [post-multiply / RIGHT]
  Key R           -> Reset orientation to identity  (clears angle buffer too)

ON-SCREEN OVERLAYS
------------------
  TOP-LEFT  : Live 3×3 rotation matrix R
  TOP-RIGHT : Last action + composition rule
  BOTTOM    : Key legend + angle buffer display

CUSTOM ANGLE FLOW
-----------------
  1. Type digits (e.g. "4", "5" -> "45") while watching the buffer update.
  2. Optionally press '-' to negate.
  3. Press Backspace to delete the last digit.
  4. Press 1-6 to immediately apply the buffered angle (default: 90°).
  The buffer resets automatically after each rotation.

BUGS FIXED vs. PREVIOUS VERSION
---------------------------------
  - Animation stacking eliminated: each rotation command is processed ONE
    AT A TIME; keys pressed during animation are queued and played after the
    current animation finishes (not mid-animation with a wrong R_start).
  - Overlapping-rotation / absurd-motion bug eliminated by the above.
  - Key '0' no longer conflicts: digits build the angle buffer; 'R' resets.

THREAD SAFETY
-------------
  GLFW key_callback runs on the viewer thread -> only puts a lightweight
  dict into a thread-safe queue.SimpleQueue. The main thread does all
  animation, matrix math, and overlay rebuilding.

AI USE NOTE
-----------
  SLERP kernel, overlay formatting, and threading pattern written with AI
  assistance. Composition rules (pre- vs. post-multiply) were derived
  independently from Lynch & Park Ch.3 in Problem 3.
"""

import sys
import time
import queue
import argparse
import numpy as np
import mujoco
import mujoco.viewer as mjv
from contextlib import contextmanager

from utils import (
    Rx, Ry, Rz, ELEMENTARY_ROTATIONS,
    set_body_orientation,
    R_to_quat, quat_to_R,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH         = "../model/asymmetric_body.xml"
ANIMATION_DURATION = 1.2      # seconds per rotation (SLERP)
VIEWER_FPS         = 60       # viewer sync rate
DEFAULT_ANGLE_DEG  = 90.0     # angle used when buffer is empty
FONT               = mujoco.mjtFontScale.mjFONTSCALE_150
_GRID              = mujoco.mjtGridPos

# GLFW key codes (printable chars match ASCII)
_DIGIT_KEYS   = set(range(48, 58))      # '0'=48 .. '9'=57
_KEY_MINUS    = 45                       # '-'
_KEY_BACKSPACE = 259                     # GLFW_KEY_BACKSPACE
_KEY_ENTER    = 257                      # GLFW_KEY_ENTER (also 335 = numpad Enter)
_KEY_ENTER_NP = 335
_KEY_R        = 82                       # 'R' -- reset

# Rotation keys -> (axis, frame)
_ROT_KEYS: dict[int, tuple[str, str]] = {
    49: ("x", "fixed"),    # '1'
    50: ("y", "fixed"),    # '2'
    51: ("z", "fixed"),    # '3'
    52: ("x", "current"),  # '4'
    53: ("y", "current"),  # '5'
    54: ("z", "current"),  # '6'
}

_LEGEND_LINES = [
    "1/2/3  :  +angle about X/Y/Z  [SPACE frame,  pre-mult  R_new = R_step @ R_old]",
    "4/5/6  :  +angle about X/Y/Z  [BODY  frame,  post-mult R_new = R_old  @ R_step]",
    "R      :  Reset to identity",
    "0-9    :  Type custom angle (deg) -- apply with 1-6",
    "-      :  Negate angle  |  Bksp : Delete digit  |  Enter : Confirm angle",
]


# ══════════════════════════════════════════════════════════════════════
# Section 1 -- Kinematic composition
# ══════════════════════════════════════════════════════════════════════

def compose(R_old: np.ndarray, axis: str, angle_rad: float, frame: str):
    """Return (R_new, rule_description).

    body  / current  ->  R_new = R_old  @ R_step   (post-multiply, RIGHT)
    space / fixed    ->  R_new = R_step @ R_old    (pre-multiply,  LEFT)
    """
    R_step = ELEMENTARY_ROTATIONS[axis](angle_rad)
    if frame == "current":
        return R_old @ R_step, "R_new = R_old @ R_step   [post / BODY  frame]"
    else:
        return R_step @ R_old, "R_new = R_step @ R_old   [pre  / SPACE frame]"


# ══════════════════════════════════════════════════════════════════════
# Section 2 -- SLERP smooth animation
# ══════════════════════════════════════════════════════════════════════

def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-path SLERP between two wxyz unit quaternions.

    BUG-FIX NOTE: We normalise both inputs first. If dot(q0, q1) < 0
    we negate q1 to force the shortest geodesic arc on SO(3).
    Without this flip, a 90-degree rotation about a body axis can appear
    to sweep 270 degrees in the other direction (the 'absurd rotation').
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))

    # Shortest-path: flip q1 if it is on the opposite hemisphere
    if dot < 0.0:
        q1  = -q1
        dot = -dot

    # Near-identical orientations: use normalised linear blend (NLERP)
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))   # total arc angle
    theta   = theta_0 * t                            # arc angle at time t
    q_perp  = q1 - q0 * dot
    q_perp  = q_perp / np.linalg.norm(q_perp)       # orthogonal component
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


def animate(viewer, model, data,
            R_start: np.ndarray, R_end: np.ndarray,
            duration: float = ANIMATION_DURATION):
    """SLERP the body from R_start to R_end over `duration` seconds.

    KEY FIX: q0 is read from data.qpos (MuJoCo's live quaternion) rather
    than recomputed via R_to_quat(R_start). This guarantees q0 is on the
    same hemisphere as the state MuJoCo has already rendered, preventing
    a one-frame sign-flip artefact at the very start of each animation.

    Runs entirely on the main thread; viewer.sync() is called every frame.
    """
    if viewer is None or not viewer.is_running():
        set_body_orientation(data, R_end)
        mujoco.mj_forward(model, data)
        return

    # Read q0 directly from MuJoCo's current state (not from R matrix)
    q0 = data.qpos[3:7].copy()   # wxyz
    q1 = R_to_quat(R_end)        # target quaternion

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


# ══════════════════════════════════════════════════════════════════════
# Section 3 -- Overlay builder
# ══════════════════════════════════════════════════════════════════════

def _fmt_matrix(R: np.ndarray) -> tuple[str, str]:
    """(left_col, right_col) for the 3×3 matrix overlay."""
    left  = "  Rotation Matrix R:\n  row-x |\n  row-y |\n  row-z |"
    right = (
        "     col-x      col-y      col-z\n"
        + "\n".join(
            "  " + "   ".join(f"{v:+.4f}" for v in row)
            for row in R
        )
    )
    return left, right


def _build_overlays(viewer, R, last_action, last_rule, angle_buf):
    """Rebuild all on-screen text overlays."""
    viewer.clear_texts()
    texts = []

    # TOP-LEFT: live 3×3 matrix
    ml, mr = _fmt_matrix(R)
    texts.append((FONT, _GRID.mjGRID_TOPLEFT, ml, mr))

    # TOP-RIGHT: last action + rule
    texts.append((FONT, _GRID.mjGRID_TOPRIGHT, "Last action:", last_action))
    texts.append((FONT, _GRID.mjGRID_TOPRIGHT, "Rule applied:", last_rule))

    # BOTTOM: legend + angle buffer
    legend = "\n".join(_LEGEND_LINES)
    # Show the typed angle buffer (or default)
    if angle_buf:
        buf_display = f"Next angle: {angle_buf}°  (press 1-6 to apply)"
    else:
        buf_display = f"Next angle: {DEFAULT_ANGLE_DEG:.0f}° (default)  -- type digits to change"
    texts.append((FONT, _GRID.mjGRID_BOTTOM, legend, buf_display))

    viewer.set_texts(texts)


# ══════════════════════════════════════════════════════════════════════
# Section 4 -- Angle buffer helper
# ══════════════════════════════════════════════════════════════════════

def _buf_to_angle(buf: str) -> float:
    """Parse the digit buffer to a float angle (degrees).
    Returns DEFAULT_ANGLE_DEG if the buffer is empty or invalid.
    """
    if not buf or buf in ("-", ""):
        return DEFAULT_ANGLE_DEG
    try:
        val = float(buf)
        return val if val != 0.0 else DEFAULT_ANGLE_DEG
    except ValueError:
        return DEFAULT_ANGLE_DEG


# ══════════════════════════════════════════════════════════════════════
# Section 5 -- Main loop
# ══════════════════════════════════════════════════════════════════════

def run(model, data, step_duration: float = ANIMATION_DURATION):
    """Open the passive viewer and handle all key events."""

    # ── Command queue (GLFW thread -> main thread) ───────────────────
    cmd_queue: queue.SimpleQueue = queue.SimpleQueue()

    def key_callback(keycode: int) -> None:
        """Called by the GLFW/viewer thread on every key-press.

        Only enqueues a lightweight dict; no shared mutable state touched.
        """
        if keycode in _ROT_KEYS:
            cmd_queue.put({"type": "rotate", "key": keycode})
        elif keycode in _DIGIT_KEYS:
            cmd_queue.put({"type": "digit", "char": chr(keycode)})
        elif keycode == _KEY_MINUS:
            cmd_queue.put({"type": "minus"})
        elif keycode in (_KEY_ENTER, _KEY_ENTER_NP):
            cmd_queue.put({"type": "confirm"})
        elif keycode == _KEY_BACKSPACE:
            cmd_queue.put({"type": "backspace"})
        elif keycode == _KEY_R:
            cmd_queue.put({"type": "reset"})

    print("Opening MuJoCo viewer ...")
    print("  1-3: SPACE frame  |  4-6: BODY frame  |  R: Reset")
    print("  Type digits for a custom angle, then press 1-6 to apply.")
    print("  Close the viewer window to exit.\n")

    viewer = mjv.launch_passive(model, data, key_callback=key_callback)

    # Initial sync burst so the window appears rendered before we overlay text
    for _ in range(int(0.3 * VIEWER_FPS)):
        if not viewer.is_running():
            break
        viewer.sync()
        time.sleep(1.0 / VIEWER_FPS)

    # ── Application state ────────────────────────────────────────────
    R           = np.eye(3)
    last_action = "(none)"
    last_rule   = "Press 1-6 to start"
    angle_buf   = ""       # typed digit buffer
    step_count  = 0
    animating   = False    # True while a SLERP is in progress

    set_body_orientation(data, R)
    mujoco.mj_forward(model, data)
    _build_overlays(viewer, R, last_action, last_rule, angle_buf)
    viewer.sync()

    # ── Main loop ────────────────────────────────────────────────────
    try:
        while viewer.is_running():

            # Process ONE pending rotation (never stack mid-animation)
            # ── Buffer/control events: drain all ─────────────────────
            # ── Rotation events: process exactly one, then animate ───
            processed_rotation = False

            while not cmd_queue.empty():
                event = cmd_queue.get_nowait()
                etype = event["type"]

                if etype == "digit":
                    # Build up the angle string (max 5 digits)
                    raw = angle_buf.lstrip("-")
                    if len(raw) < 5:
                        angle_buf = (angle_buf[0] if angle_buf.startswith("-") else "") \
                                    + raw + event["char"]

                elif etype == "minus":
                    # Toggle sign
                    if angle_buf.startswith("-"):
                        angle_buf = angle_buf[1:]
                    else:
                        angle_buf = "-" + angle_buf

                elif etype == "backspace":
                    angle_buf = angle_buf[:-1]

                elif etype == "confirm":
                    # Enter pressed: just print the locked-in angle
                    deg = _buf_to_angle(angle_buf)
                    print(f"[Angle locked] {deg:.1f}° -- press 1-6 to apply")

                elif etype == "reset":
                    R_new       = np.eye(3)
                    last_action = "Reset to identity"
                    last_rule   = "R = I"
                    angle_buf   = ""
                    step_count  = 0
                    print("[Reset]  R = I")
                    _build_overlays(viewer, R_new, last_action, last_rule, angle_buf)
                    animate(viewer, model, data, R, R_new, duration=0.4)
                    R = R_new
                    processed_rotation = True
                    break   # stop draining; one action per loop iteration

                elif etype == "rotate" and not processed_rotation:
                    # ── Consume the buffered angle ────────────────────
                    angle_deg = _buf_to_angle(angle_buf)
                    angle_rad = np.deg2rad(angle_deg)
                    angle_buf = ""   # clear buffer immediately

                    key  = event["key"]
                    axis, frame = _ROT_KEYS[key]
                    frame_label = "SPACE" if frame == "fixed" else "BODY"

                    R_new, rule = compose(R, axis, angle_rad, frame)
                    step_count += 1
                    last_action = (f"Step {step_count}: "
                                   f"Rot({axis.upper()}, {angle_deg:+.1f}°) "
                                   f"{frame_label} frame")
                    last_rule   = rule

                    print(f"[Step {step_count}] Rot({axis.upper()}, {angle_deg:+.1f}°) "
                          f"{frame_label} frame")
                    print(f"  Rule: {rule}")
                    print("  R_new =")
                    for row_lbl, row in zip(("  x |", "  y |", "  z |"), R_new):
                        print(row_lbl + "  " +
                              "   ".join(f"{v:+8.4f}" for v in row))
                    print()

                    # Update overlay BEFORE animating so text shows target R
                    _build_overlays(viewer, R_new, last_action, last_rule, angle_buf)

                    # ── SLERP animate (blocks until done) ────────────
                    animate(viewer, model, data, R, R_new, duration=step_duration)
                    R = R_new
                    processed_rotation = True

                    # After animation, drain any digits/control keys that
                    # arrived while we were animating, but do NOT process
                    # another rotate command this iteration -- that will
                    # happen on the NEXT loop iteration so R_start is correct.
                    break

            # ── Keep viewer alive (idle frame) ────────────────────────
            _build_overlays(viewer, R, last_action, last_rule, angle_buf)
            viewer.sync()
            time.sleep(1.0 / VIEWER_FPS)

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        if viewer.is_running():
            viewer.close()


# ══════════════════════════════════════════════════════════════════════
# Section 6 -- Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ME 639 HW1 Task 1 -- Rotation Sandbox (viewer-native)"
    )
    parser.add_argument(
        "--step-duration", type=float, default=ANIMATION_DURATION, metavar="SEC",
        help=f"SLERP animation duration per step in seconds (default: {ANIMATION_DURATION})"
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
