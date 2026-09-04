"""
01_rotation_sandbox.py -- HW1 Part 2, Task 1: does rotation order matter?

STARTER CODE. The model loading, viewer, and simulation loop below
are complete and working -- run this file as-is and you should see
the asymmetric dart sitting in the viewer. Your job is to fill in
the TODOs so that:

  1. The user can queue up a sequence of elemental rotations
     (about x, y, or z), each one EITHER about the current body
     frame OR about the fixed space frame (their choice).
  2. The dart's orientation updates to reflect that sequence.
  3. You can run the SAME sequence of angles twice -- once
     "current frame" and once "fixed frame" -- and see (and
     screen-record) that the final orientation is visibly
     different, exactly as you proved symbolically in HW1
     Problem 3 (Lynch & Park Ch.3 Ex.3.4-style reasoning) and
     the "Composition of Rotations" lecture derivation.

This is intentionally a plain script, not a GUI app -- editing the
`rotation_sequence` list below and re-running is a perfectly good
"sandbox." A slider UI is a nice-to-have, not a requirement. Use AI
freely here; document what you asked it for in your AI Use Note.
"""

import sys
import time
import argparse
import numpy as np
import mujoco
try:
    import mujoco.viewer
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False

from utils import Rx, Ry, Rz, ELEMENTARY_ROTATIONS, set_body_orientation, get_body_orientation, R_to_quat, quat_to_R

MODEL_PATH = "../model/asymmetric_body.xml"


# ---------------------------------------------------------------
# Default demonstration sequence:
# Each entry is (axis, angle_radians, frame) where frame is either:
# "current" (compose on the RIGHT: R_new = R_old @ R_step) or
# "fixed"   (compose on the LEFT:  R_new = R_step @ R_old).
# ---------------------------------------------------------------
default_rotation_sequence = [
    ("z", np.deg2rad(90), "current"),
    ("x", np.deg2rad(90), "current"),
]


def compose_step(R_current, axis, angle, frame):
    """Compose a single elemental rotation with the current orientation R_current.

    Rules from Composition of Rotations (Lynch & Park Ch. 3):
      - Current (body) frame: POST-multiply on the RIGHT
            R_new = R_current @ R_step
      - Fixed (space) frame:   PRE-multiply on the LEFT
            R_new = R_step @ R_current
    """
    axis = axis.lower().strip()
    if axis not in ELEMENTARY_ROTATIONS:
        raise ValueError(f"Invalid axis '{axis}'. Must be one of 'x', 'y', 'z'.")

    R_step = ELEMENTARY_ROTATIONS[axis](angle)
    frame_norm = frame.lower().strip()

    if frame_norm in ("current", "body", "c", "b"):
        return R_current @ R_step
    elif frame_norm in ("fixed", "space", "f", "s"):
        return R_step @ R_current
    else:
        raise ValueError(f"Unknown frame '{frame}'. Must be 'current' or 'fixed'.")


def compose_sequence(sequence):
    """Given a list of (axis, angle, frame) tuples, return the final
    3x3 rotation matrix R obtained by applying them in order,
    starting from R = identity.
    """
    R = np.eye(3)
    for axis, angle, frame in sequence:
        R = compose_step(R, axis, angle, frame)
    return R


def slerp_quat(q0, q1, t):
    """Spherical linear interpolation between two quaternions q0 and q1."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    q_perp = q1 - q0 * dot
    q_perp = q_perp / np.linalg.norm(q_perp)
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


def animate_transition(viewer, model, data, R_start, R_target, duration=1.0, fps=60):
    """Smoothly interpolate the body orientation from R_start to R_target."""
    if viewer is None or not viewer.is_running():
        set_body_orientation(data, R_target)
        mujoco.mj_forward(model, data)
        return

    q_start = R_to_quat(R_start)
    q_target = R_to_quat(R_target)
    n_frames = max(1, int(duration * fps))

    for i in range(1, n_frames + 1):
        if not viewer.is_running():
            break
        t = i / float(n_frames)
        q_interp = slerp_quat(q_start, q_target, t)
        R_interp = quat_to_R(q_interp)
        set_body_orientation(data, R_interp)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0 / fps)


def parse_user_step(step_str):
    """Parse a string like 'z 90 current' or 'x -45 fixed'."""
    parts = step_str.strip().split()
    if len(parts) != 3:
        raise ValueError("Expected 3 arguments: <axis> <angle_deg> <frame>")
    axis = parts[0].lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"Axis must be x, y, or z; got '{axis}'")
    angle_deg = float(parts[1])
    angle_rad = np.deg2rad(angle_deg)
    frame = parts[2].lower()
    if frame not in ("current", "fixed", "body", "space", "c", "f"):
        raise ValueError(f"Frame must be 'current' or 'fixed'; got '{frame}'")
    canonical_frame = "current" if frame in ("current", "body", "c") else "fixed"
    return axis, angle_rad, canonical_frame, angle_deg


def interactive_input_sequence():
    """Prompt the user step-by-step to define a rotation sequence."""
    print("\n--- Interactive Rotation Sequence Builder ---")
    print("Enter each elemental rotation step.")
    print("Format: <axis> <angle_deg> <frame>")
    print("  axis:   x, y, or z")
    print("  angle:  degrees (e.g., 90, -45, 180)")
    print("  frame:  'current' (body frame) or 'fixed' (space frame)")
    print("Example:  'z 90 current'  or  'x 45 fixed'")
    print("Type 'done' or press Enter on empty line to finish.")
    print("---------------------------------------------\n")

    sequence = []
    step_num = 1
    while True:
        try:
            val = input(f"Step {step_num} [axis angle frame | 'done']: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting input.")
            break

        if not val or val.lower() in ("done", "exit", "q"):
            break

        try:
            axis, angle_rad, frame, angle_deg = parse_user_step(val)
            sequence.append((axis, angle_rad, frame))
            print(f"  -> Added step {step_num}: Rot({axis}, {angle_deg}°) about {frame.upper()} frame.")
            step_num += 1
        except Exception as e:
            print(f"  [Error] {e}. Please try again.")

    if not sequence:
        print("No steps entered. Using default sequence: [('z', 90°, 'current'), ('x', 90°, 'current')].")
        return default_rotation_sequence

    return sequence


def run_sequence_on_model(model, data, sequence, animate=True, step_duration=1.0):
    """Execute a rotation sequence on the model, animating each step."""
    print("\nExecuting rotation sequence:")
    R_current = np.eye(3)
    set_body_orientation(data, R_current)
    mujoco.mj_forward(model, data)

    viewer = None
    if HAS_VIEWER:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:
            print(f"Note: Viewer could not be opened ({e}). Running in headless mode.")

    try:
        if viewer is not None:
            viewer.sync()
            time.sleep(0.5)

        for i, (axis, angle, frame) in enumerate(sequence):
            angle_deg = np.rad2deg(angle)
            R_next = compose_step(R_current, axis, angle, frame)
            print(f"Step {i + 1}: Rot({axis}, {angle_deg:.1f}°) about {frame.upper()} frame")
            print(f"  Composition rule: {'R_new = R_old @ R_step' if frame == 'current' else 'R_new = R_step @ R_old'}")

            if viewer is not None and animate:
                animate_transition(viewer, model, data, R_current, R_next, duration=step_duration)
                time.sleep(0.3)
            else:
                set_body_orientation(data, R_next)
                mujoco.mj_forward(model, data)

            R_current = R_next

        print("\nFinal Rotation Matrix R:")
        print(np.array2string(R_current, precision=4, suppress_small=True))

        if viewer is not None:
            print("\nSequence complete. Viewer running -- close viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
    finally:
        if viewer is not None:
            viewer.close()

    return R_current


def run_comparison(model, data):
    """Compare the exact same sequence of elemental rotations in 'current' vs 'fixed' frame."""
    print("\n" + "=" * 60)
    print("Task 1 Demonstration: Current Frame vs. Fixed Frame Comparison")
    print("=" * 60)
    test_angles = [("z", np.deg2rad(90)), ("x", np.deg2rad(90))]

    seq_current = [(axis, ang, "current") for axis, ang in test_angles]
    seq_fixed = [(axis, ang, "fixed") for axis, ang in test_angles]

    R_curr = compose_sequence(seq_current)
    R_fix = compose_sequence(seq_fixed)

    print("\nSequence: Rotate 90° about z, then rotate 90° about x.")
    print("\n1. All rotations about CURRENT (body) frame (Post-multiplication: R = Rz @ Rx):")
    print(np.array2string(R_curr, precision=4, suppress_small=True))

    print("\n2. All rotations about FIXED (space) frame (Pre-multiplication: R = Rx @ Rz):")
    print(np.array2string(R_fix, precision=4, suppress_small=True))

    diff = np.linalg.norm(R_curr - R_fix)
    print(f"\nMatrix Difference ||R_current - R_fixed||_F = {diff:.4f}")
    if diff > 1e-4:
        print(">> PROVED VISUALLY & NUMERICALLY: Rotations do NOT commute!")
        print("   Current-frame composition Rz @ Rx differs from fixed-frame composition Rx @ Rz.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ME 639 HW1 Task 1: Rotation Sandbox")
    parser.add_argument("--interactive", "-i", action="store_true", help="Prompt for steps interactively")
    parser.add_argument("--compare", "-c", action="store_true", help="Run current vs. fixed comparison demonstration")
    parser.add_argument("--no-animate", action="store_true", help="Skip smooth animation transitions")
    parser.add_argument("--step-duration", type=float, default=1.0, help="Duration (s) per animated rotation step")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if args.compare:
        run_comparison(model, data)
        return

    if args.interactive or (sys.stdin.isatty() and len(sys.argv) == 1):
        sequence = interactive_input_sequence()
    else:
        sequence = default_rotation_sequence

    run_sequence_on_model(model, data, sequence, animate=not args.no_animate, step_duration=args.step_duration)


if __name__ == "__main__":
    main()
