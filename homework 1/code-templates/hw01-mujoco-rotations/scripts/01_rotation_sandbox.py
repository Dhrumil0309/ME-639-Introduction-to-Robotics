"""
01_rotation_sandbox.py  --  ME 639 HW1, Task 1
===============================================
Rotation Sandbox: Apply elemental rotations about the CURRENT (body)
frame or the FIXED (space) frame and see the asymmetric dart animate
smoothly in the MuJoCo viewer.

KEY FEATURES
------------
1. Interactive CLI  : Background thread reads stdin; main thread drives
                      the viewer -- no blocking, no frozen window.
2. Rigorous Kinematics:
      Current (body)  frame  ->  R_new = R_old @ R_step   (post-multiply)
      Fixed   (space) frame  ->  R_new = R_step @ R_old   (pre-multiply)
   The 3x3 rotation matrix is printed after EVERY step.
3. Smooth Animation : Rotations are SLERP-interpolated over many sub-steps
                      inside the viewer sync loop -- great for recording.
4. Non-commutativity demo (--compare):
                      Plays the same two-rotation sequence twice (once with
                      all-current, once with all-fixed frame) so the visibly
                      different final orientations prove non-commutativity.

USAGE
-----
  # Fully interactive -- type steps at runtime, viewer stays live:
  python 01_rotation_sandbox.py

  # Play the built-in default sequence and watch:
  python 01_rotation_sandbox.py --demo

  # Non-commutativity demo (ideal for screen recording):
  python 01_rotation_sandbox.py --compare

  # Slow things down for recording:
  python 01_rotation_sandbox.py --compare --step-duration 2.5

INPUT FORMAT (interactive mode)
--------------------------------
  z 90 current   <- 90 deg about current z-axis  (body frame)
  x 45 fixed     <- 45 deg about fixed  x-axis   (space frame)
  y -30 c        <- shortcuts: c=current, f=fixed, b=body, s=space
  go / run       <- play accumulated sequence in viewer
  reset          <- reset dart to identity, clear sequence
  demo           <- load built-in default sequence
  help           <- print this help again
  quit           <- close viewer and exit

AI USE NOTE
-----------
  Threading pattern, SLERP kernel, and matrix printer written with AI
  assistance. Composition rules (pre- vs post-multiply) and the
  non-commutativity proof were derived independently in Problem 3.
"""

import sys
import time
import queue
import threading
import argparse
import numpy as np
import mujoco

try:
    import mujoco.viewer as _mj_viewer
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False

from utils import (
    Rx, Ry, Rz, ELEMENTARY_ROTATIONS,
    set_body_orientation, get_body_orientation,
    R_to_quat, quat_to_R,
)

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
MODEL_PATH = "../model/asymmetric_body.xml"

DEFAULT_SEQUENCE = [
    ("z", np.deg2rad(90),  "current"),
    ("x", np.deg2rad(90),  "current"),
]

COMPARE_ANGLES = [
    ("z", np.deg2rad(90)),
    ("x", np.deg2rad(90)),
]

FPS       = 60      # Viewer refresh rate (Hz)
SLERP_FPS = 60      # Sub-step rate for smooth animation
PAUSE     = 0.6     # Seconds of hold after each step completes

HELP_TEXT = """
  COMMANDS
  --------
  <axis> <angle_deg> <frame>   Queue one rotation step.
     axis  : x, y, or z
     angle : degrees (e.g. 90, -45, 180)
     frame : current | body | c    (body  frame -> post-multiply R_old @ R_step)
             fixed   | space | f   (space frame -> pre-multiply  R_step @ R_old)

  Examples:   "z 90 current"   "x 45 fixed"   "y -30 c"

  go / run         Play the queued sequence in the viewer.
  reset / clear    Clear queue and reset dart to identity.
  demo             Load the built-in default sequence.
  help / h / ?     Show this help text.
  quit / q / exit  Close viewer and exit.
"""


# ================================================================== #
# Section 1 -- Kinematic composition (the core math)
# ================================================================== #

def compose_step(R_old, axis, angle_rad, frame):
    """Apply one elemental rotation and return the new SO(3) matrix.

    Lynch & Park, Ch.3 Composition of Rotations:
      body  / current  =>  R_new = R_old @ R_step    (right / post-multiply)
      space / fixed    =>  R_new = R_step @ R_old    (left  / pre-multiply)

    Returns
    -------
    R_new : ndarray shape (3,3)
    rule  : str   human-readable description of the multiplication used
    """
    if axis not in ELEMENTARY_ROTATIONS:
        raise ValueError(f"Axis must be x/y/z, got '{axis}'")
    R_step = ELEMENTARY_ROTATIONS[axis](angle_rad)

    f = frame.lower().strip()
    if f in ("current", "body", "c", "b"):
        return R_old @ R_step, "R_new = R_old @ R_step  [POST-multiply, body  frame]"
    elif f in ("fixed", "space", "f", "s"):
        return R_step @ R_old, "R_new = R_step @ R_old  [PRE-multiply,  space frame]"
    else:
        raise ValueError(f"Frame must be 'current' or 'fixed', got '{frame}'")


def print_R(R, label="R"):
    """Pretty-print a 3x3 rotation matrix."""
    bar = "-" * 42
    print(f"\n  {bar}")
    print(f"  {label}")
    print(f"  {bar}")
    header = "          {:>9s}   {:>9s}   {:>9s}".format("x-col", "y-col", "z-col")
    print(header)
    for row_lbl, row in zip(("  x-row |", "  y-row |", "  z-row |"), R):
        print(row_lbl + "  " + "   ".join(f"{v:+9.5f}" for v in row))
    print(f"  {bar}\n")


# ================================================================== #
# Section 2 -- SLERP smooth animation
# ================================================================== #

def _slerp(q0, q1, t):
    """Spherical linear interpolation (wxyz quaternion convention)."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    th0 = np.arccos(np.clip(dot, -1.0, 1.0))
    th  = th0 * t
    qp  = q1 - q0 * dot
    qp  = qp / np.linalg.norm(qp)
    return q0 * np.cos(th) + qp * np.sin(th)


def animate_transition(viewer, model, data, R_start, R_end, duration=1.5):
    """SLERP-interpolate orientation from R_start to R_end over `duration` s.

    Runs on the main thread, calls viewer.sync() every frame.
    """
    if viewer is None or not viewer.is_running():
        set_body_orientation(data, R_end)
        mujoco.mj_forward(model, data)
        return

    q0 = R_to_quat(R_start)
    q1 = R_to_quat(R_end)
    n  = max(2, int(duration * SLERP_FPS))
    dt = 1.0 / SLERP_FPS

    for i in range(1, n + 1):
        if not viewer.is_running():
            break
        qi = _slerp(q0, q1, i / float(n))
        set_body_orientation(data, quat_to_R(qi))
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(dt)


def _hold(viewer, seconds):
    """Keep viewer alive (sync) for a fixed number of seconds."""
    if viewer is None:
        time.sleep(seconds)
        return
    t0 = time.time()
    while time.time() - t0 < seconds:
        if not viewer.is_running():
            return
        viewer.sync()
        time.sleep(1.0 / FPS)


# ================================================================== #
# Section 3 -- Thread-safe input reader
# ================================================================== #

class _InputThread(threading.Thread):
    """Daemon thread that reads stdin and enqueues lines.

    The main viewer loop polls get() without ever calling input()
    itself, so the MuJoCo window is never frozen while waiting for
    the user to type.
    """

    def __init__(self):
        super().__init__(daemon=True, name="InputReader")
        self._q     = queue.Queue()
        self._alive = threading.Event()
        self._alive.set()

    def run(self):
        while self._alive.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                line = ""
            if line == "":          # EOF
                self._q.put(None)
                return
            self._q.put(line.rstrip("\n"))

    def poll(self):
        """Return the next pending line or None (non-blocking)."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._alive.clear()


# ================================================================== #
# Section 4 -- Input parsing
# ================================================================== #

def _parse(line):
    """Parse one user input line into a command dict."""
    line = line.strip()
    if not line:
        return {"cmd": "empty"}

    lo = line.lower()
    if lo in ("quit", "q", "exit"):
        return {"cmd": "quit"}
    if lo in ("help", "h", "?"):
        return {"cmd": "help"}
    if lo in ("go", "run", "play"):
        return {"cmd": "go"}
    if lo in ("reset", "clear", "r"):
        return {"cmd": "reset"}
    if lo in ("demo", "default"):
        return {"cmd": "demo"}

    parts = lo.split()
    if len(parts) != 3:
        return {"cmd": "error",
                "msg": f"Expected '<axis> <angle_deg> <frame>', got: '{line}'"}

    axis, ang_str, frame = parts
    if axis not in ("x", "y", "z"):
        return {"cmd": "error", "msg": f"Axis must be x/y/z, got '{axis}'"}
    try:
        angle_deg = float(ang_str)
    except ValueError:
        return {"cmd": "error", "msg": f"Angle must be a number, got '{ang_str}'"}
    if frame not in ("current", "body", "c", "b", "fixed", "space", "f", "s"):
        return {"cmd": "error",
                "msg": f"Frame must be 'current' or 'fixed', got '{frame}'"}

    canonical = "current" if frame in ("current", "body", "c", "b") else "fixed"
    return {
        "cmd":       "step",
        "axis":      axis,
        "angle_deg": angle_deg,
        "angle_rad": np.deg2rad(angle_deg),
        "frame":     canonical,
    }


# ================================================================== #
# Section 5 -- Sequence runner
# ================================================================== #

def run_sequence(viewer, model, data, sequence, step_duration=1.5, label=""):
    """Execute a list of (axis, angle_rad, frame) steps with animation.

    After every step the updated 3x3 R matrix is printed to the terminal.
    """
    R = np.eye(3)
    set_body_orientation(data, R)
    mujoco.mj_forward(model, data)
    if viewer is not None and viewer.is_running():
        viewer.sync()
    time.sleep(PAUSE)

    print("\n" + "=" * 60)
    if label:
        print(f"  {label}")
    print(f"  {len(sequence)} step(s)  |  step duration: {step_duration:.1f}s")
    print("=" * 60)
    print_R(R, label="R_0 = I  (identity -- dart at rest)")

    for i, (axis, angle_rad, frame) in enumerate(sequence):
        if viewer is not None and not viewer.is_running():
            print("[Viewer closed -- stopping]")
            return R

        deg = np.rad2deg(angle_rad)
        R_new, rule = compose_step(R, axis, angle_rad, frame)

        print(f"  STEP {i+1}/{len(sequence)}")
        print(f"    Rotation : {deg:+.1f}° about {axis.upper()}-axis")
        print(f"    Frame    : {frame.upper()} "
              f"({'body' if frame=='current' else 'space'} frame)")
        print(f"    Rule     : {rule}")
        print_R(R_new, label=f"R after step {i+1}")

        animate_transition(viewer, model, data, R, R_new, duration=step_duration)
        R = R_new
        _hold(viewer, PAUSE)

    print("=" * 60)
    print("  SEQUENCE COMPLETE")
    print_R(R, label="R_final")
    print("=" * 60 + "\n")
    return R


# ================================================================== #
# Section 6 -- Non-commutativity comparison
# ================================================================== #

def run_comparison(viewer, model, data, step_duration=2.0):
    """Show non-commutativity visually: same angles, two frame choices,
    two visibly different final orientations.

    Perfect for a 10-20s screen recording:
      Part A: z90 then x90 about CURRENT frame  -> one final pose
      (3s hold to observe)
      Part B: z90 then x90 about FIXED   frame  -> different final pose
      (3s hold to compare)
      Numerical printout confirms ||R_A - R_B||_F >> 0
    """
    seq_a = [(ax, ang, "current") for ax, ang in COMPARE_ANGLES]
    seq_b = [(ax, ang, "fixed")   for ax, ang in COMPARE_ANGLES]

    print("\n" + "=" * 60)
    print("  NON-COMMUTATIVITY DEMONSTRATION  (Problem 3 visual proof)")
    print("=" * 60)
    print("  Sequence: Rz(90°) followed by Rx(90°)")
    print("  Part A -- CURRENT (body)  frame: post-multiply")
    print("  Part B -- FIXED   (space) frame: pre-multiply")
    print("  If rotation is non-commutative, A and B end at DIFFERENT")
    print("  orientations. Watch the dart to see this live.\n")

    input("  Press ENTER to start Part A ...")
    print("\n  >>> PART A: all rotations in CURRENT (body) FRAME <<<")
    R_a = run_sequence(viewer, model, data, seq_a,
                       step_duration=step_duration,
                       label="PART A -- current (body) frame")
    print("  Observe the dart orientation. Holding for 3 seconds ...")
    _hold(viewer, 3.0)

    input("\n  Press ENTER to start Part B (dart will reset first) ...")
    print("\n  >>> PART B: all rotations in FIXED (space) FRAME <<<")
    R_b = run_sequence(viewer, model, data, seq_b,
                       step_duration=step_duration,
                       label="PART B -- fixed (space) frame")
    print("  Observe the DIFFERENT final orientation. Holding for 3 seconds ...")
    _hold(viewer, 3.0)

    # Numerical summary
    diff = np.linalg.norm(R_a - R_b, ord="fro")
    print("\n" + "=" * 60)
    print("  NUMERICAL RESULT")
    print("=" * 60)
    print_R(R_a, label="R_A  (current-frame composition)")
    print_R(R_b, label="R_B  (fixed-frame  composition)")
    print(f"  ||R_A - R_B||_F  =  {diff:.6f}")
    if diff > 1e-6:
        print("\n  >> The two matrices ARE different (norm >> 0).")
        print("  >> CONCLUSION: Rotation is NOT commutative.")
        print("  >> This is the visual counterpart of the symbolic proof")
        print("     in Problem 3:  Rz @ Rx  !=  Rx @ Rz")
    else:
        print("\n  [Note: matrices appear equal -- check COMPARE_ANGLES]")
    print("=" * 60 + "\n")


# ================================================================== #
# Section 7 -- Main interactive loop
# ================================================================== #

def interactive_loop(viewer, model, data, step_duration=1.5):
    """CLI loop that accumulates steps and plays them on 'go'.

    Uses BackgroundInputThread so typing never freezes the viewer.
    """
    print(HELP_TEXT)
    print("  Viewer is live. Type commands below and press Enter.\n")

    pending: list = []
    reader = _InputThread()
    reader.start()

    try:
        while True:
            # Keep viewer alive
            if viewer is not None:
                if viewer.is_running():
                    viewer.sync()
                else:
                    print("  Viewer closed. Exiting.")
                    break
            time.sleep(1.0 / FPS)

            # Non-blocking poll for user input
            line = reader.poll()
            if line is None:
                continue

            cmd = _parse(line)
            name = cmd["cmd"]

            if name == "empty":
                continue

            elif name == "quit":
                print("  Exiting.")
                break

            elif name == "help":
                print(HELP_TEXT)

            elif name == "reset":
                pending.clear()
                set_body_orientation(data, np.eye(3))
                mujoco.mj_forward(model, data)
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                print("  Reset: dart is at identity. Sequence cleared.\n")

            elif name == "demo":
                pending = list(DEFAULT_SEQUENCE)
                print(f"  Default sequence loaded ({len(pending)} steps). "
                      f"Type 'go' to play.\n")

            elif name == "go":
                if not pending:
                    print("  Nothing queued. Type steps first, or 'demo'.\n")
                    continue
                print(f"  Playing {len(pending)} step(s) ...\n")
                run_sequence(viewer, model, data, pending,
                             step_duration=step_duration,
                             label="User sequence")
                print("  Done. Add more steps or 'reset' to start fresh.\n")

            elif name == "step":
                ax  = cmd["axis"]
                ang = cmd["angle_rad"]
                fr  = cmd["frame"]
                deg = cmd["angle_deg"]
                pending.append((ax, ang, fr))
                n = len(pending)
                print(f"  + Step {n}: Rot({ax.upper()}, {deg:+.1f}°)  "
                      f"frame={fr}   [{n} step{'s' if n>1 else ''} queued]\n")

            elif name == "error":
                print(f"  [!] {cmd['msg']}")
                print("  Type 'help' for the command format.\n")

    finally:
        reader.stop()


# ================================================================== #
# Section 8 -- Entry point
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="ME 639 HW1 Task 1 -- Rotation Sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--compare", action="store_true",
                        help="Non-commutativity demo (run both frames, show difference)")
    parser.add_argument("--demo", action="store_true",
                        help="Play default built-in sequence then keep viewer open")
    parser.add_argument("--step-duration", type=float, default=1.5, metavar="SEC",
                        help="Seconds per animated rotation step (default: 1.5)")
    parser.add_argument("--headless", action="store_true",
                        help="No viewer; print matrices only (for CI / testing)")
    args = parser.parse_args()

    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    except Exception as e:
        sys.exit(f"[ERROR] Could not load '{MODEL_PATH}': {e}")
    data = mujoco.MjData(model)

    viewer = None
    if HAS_VIEWER and not args.headless:
        try:
            viewer = _mj_viewer.launch_passive(model, data)
        except Exception as e:
            print(f"[Warning] Viewer unavailable ({e}). Running headless.")

    # Initial sync so the viewer window appears fully rendered
    if viewer is not None:
        for _ in range(int(0.4 * FPS)):
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(1.0 / FPS)

    try:
        if args.compare:
            run_comparison(viewer, model, data, step_duration=args.step_duration)
            print("  [Close the viewer window to exit]\n")
            _hold(viewer, seconds=99999)

        elif args.demo:
            run_sequence(viewer, model, data, DEFAULT_SEQUENCE,
                         step_duration=args.step_duration,
                         label="Default demo")
            print("  [Close the viewer window to exit]\n")
            _hold(viewer, seconds=99999)

        else:
            interactive_loop(viewer, model, data,
                             step_duration=args.step_duration)

    except KeyboardInterrupt:
        print("\n  [Interrupted]")
    finally:
        if viewer is not None and viewer.is_running():
            viewer.close()


if __name__ == "__main__":
    main()
