"""
02_verify_skew_properties.py  --  ME 639 HW1, Task 2 (Problem 8)
=================================================================
Interactive live verification of the two skew-symmetric / rotation identities
proved analytically in Problem 5:

  Identity 1 (cross-product equivariance):
      R (v × w)  =  (Rv) × (Rw)      for all v, w ∈ ℝ³

  Identity 2 (adjoint / skew similarity):
      R hat(ω) Rᵀ  =  hat(Rω)        for all ω ∈ ℝ³

KEYBINDINGS  (inside the MuJoCo viewer window)
-------------------
  Space   → toggle pause / resume physics
  Enter   → print full side-by-side matrix equality proof at current R(t)
  T       → run silent stats check and print a table row (same as auto-check)
  Q       → quit cleanly
  Escape  → quit cleanly

TERMINAL OUTPUT (auto, every CHECK_INTERVAL simulated seconds)
--------------------------------------------------------------
  A compact table row: time | residual Id1 | residual Id2 | PASS?

TERMINAL OUTPUT (on Enter keypress)
------------------------------------
  Full side-by-side printout of every element of both LHS and RHS so you
  can verify by eye that every number is identical.

HOW IT WORKS
------------
  - The passive MuJoCo viewer runs continuously.
  - A time-varying ω(t) is injected every physics step (no gravity).
  - A queue.SimpleQueue bridges the viewer's GLFW key_callback thread and
    the main physics/rendering thread (zero shared mutable state, no locks).
  - Pausing stops mj_step() but keeps viewer.sync() running so the window
    remains responsive.

EXPECTED RESIDUALS
------------------
  Both identities are exact algebraic identities in SO(3), so residuals
  should be ≈ 1e-14 to 1e-16 (within ~10× double-precision machine epsilon
  ε ≈ 2.22e-16). A small residual does NOT prove the identity (we sample
  finitely many R and vectors) but strongly supports it numerically.

USAGE
-----
  cd scripts/
  python 02_verify_skew_properties.py
  python 02_verify_skew_properties.py --headless --duration 5  # CI mode

AI USE NOTE
-----------
  Threading pattern, side-by-side formatter, and CSV writer written with AI
  assistance. Mathematical identities (hat operator, composition rules) were
  derived independently in Problem 5.
"""

import sys
import time
import queue
import argparse
import csv

import numpy as np
import mujoco
import mujoco.viewer as mjv

from utils import get_body_orientation, is_close_to_identity

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH      = "../model/asymmetric_body.xml"
CHECK_INTERVAL  = 0.5     # simulated seconds between automatic stat rows
N_RANDOM_CHECKS = 8       # random triples per automatic check
VIEWER_FPS      = 60      # viewer sync rate
CSV_PATH        = "skew_verification_residuals.csv"

# GLFW key codes
KEY_SPACE  = 32
KEY_ENTER  = 257
KEY_ENTER2 = 335    # numpad Enter
KEY_ESCAPE = 256
KEY_Q      = 81
KEY_T      = 84

try:
    _FONT = mujoco.mjtFontScale.mjFONTSCALE_150
    _GRID = mujoco.mjtGridPos
    _HAS_OVERLAY = True
except Exception:
    _HAS_OVERLAY = False


# ══════════════════════════════════════════════════════════════════════
# Section 1 -- Mathematics
# ══════════════════════════════════════════════════════════════════════

def hat(omega: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of ω ∈ ℝ³.

    Defined so that  hat(ω) · v = ω × v  for every v ∈ ℝ³.
    This is the standard ω̂ (hat) operator from Lynch & Park §3.2.

         ⎡  0   -ω₃   ω₂ ⎤
    ω̂ = ⎢  ω₃    0  -ω₁ ⎥
         ⎣ -ω₂   ω₁    0 ⎦
    """
    wx, wy, wz = float(omega[0]), float(omega[1]), float(omega[2])
    return np.array([
        [ 0.0, -wz,  wy],
        [ wz,   0.0, -wx],
        [-wy,   wx,   0.0],
    ])


def time_varying_omega(t: float) -> np.ndarray:
    """Continuously varying angular velocity ω(t) ∈ ℝ³ (rad/s).

    Non-periodic, excites all three body axes simultaneously.
    """
    return np.array([
        1.5 * np.sin(2.5 * t) + 0.5 * np.cos(5.0 * t),
        1.2 * np.cos(3.0 * t) - 0.4 * np.sin(1.5 * t),
        2.0 * np.sin(t) * np.cos(2.0 * t) + 0.8 * np.cos(0.5 * t),
    ])


def check_identities(R: np.ndarray, rng: np.random.Generator) -> dict:
    """Evaluate both identities for N_RANDOM_CHECKS random vector triples.

    Returns dict with max/mean residuals for each identity.
    """
    resids_cross, resids_skew = [], []

    for _ in range(N_RANDOM_CHECKS):
        v     = rng.standard_normal(3)
        w     = rng.standard_normal(3)
        omega = rng.standard_normal(3)

        # Identity 1: R(v×w) = (Rv)×(Rw)
        lhs1 = R @ np.cross(v, w)
        rhs1 = np.cross(R @ v, R @ w)
        resids_cross.append(np.linalg.norm(lhs1 - rhs1))

        # Identity 2: R hat(ω) Rᵀ = hat(Rω)
        lhs2 = R @ hat(omega) @ R.T
        rhs2 = hat(R @ omega)
        resids_skew.append(np.linalg.norm(lhs2 - rhs2, ord="fro"))

    return {
        "max_resid_cross":  max(resids_cross),
        "max_resid_skew":   max(resids_skew),
        "mean_resid_cross": float(np.mean(resids_cross)),
        "mean_resid_skew":  float(np.mean(resids_skew)),
    }


# ══════════════════════════════════════════════════════════════════════
# Section 2 -- Explicit side-by-side proof printer (Enter key)
# ══════════════════════════════════════════════════════════════════════

_W = 74    # total terminal width for boxes

def _hline(char="─"):
    return "  " + char * (_W - 2)

def _box(title):
    return f"\n  ╔{'═' * (_W - 4)}╗\n  ║  {title:<{_W - 6}}║\n  ╚{'═' * (_W - 4)}╝"


def print_proof(R: np.ndarray, t: float, rng: np.random.Generator):
    """Generate one random triple and print every element of LHS and RHS
    for both identities, side by side, so the equality is visually obvious.
    """
    v     = rng.standard_normal(3)
    w     = rng.standard_normal(3)
    omega = rng.standard_normal(3)

    # ── Pre-compute ──────────────────────────────────────────────────
    cross_vw  = np.cross(v, w)
    lhs1      = R @ cross_vw          # R(v×w)
    Rv, Rw    = R @ v, R @ w
    rhs1      = np.cross(Rv, Rw)     # (Rv)×(Rw)
    resid1    = np.linalg.norm(lhs1 - rhs1)

    lhs2      = R @ hat(omega) @ R.T  # R hat(ω) Rᵀ
    Romega    = R @ omega
    rhs2      = hat(Romega)           # hat(Rω)
    resid2    = np.linalg.norm(lhs2 - rhs2, ord="fro")

    eps = np.finfo(float).eps
    check = lambda r: "✓  machine ε" if r < 1e-10 else "✗  LARGE!"

    # ── Header ───────────────────────────────────────────────────────
    print()
    print("  " + "═" * (_W - 2))
    print(f"  PROVING EQUALITIES   at  t = {t:.4f} s")
    print("  " + "═" * (_W - 2))

    # ── Current R ────────────────────────────────────────────────────
    print()
    print("  Current rotation matrix  R(t):")
    print(f"              col-x          col-y          col-z")
    for lbl, row in zip(("  row-x │", "  row-y │", "  row-z │"), R):
        vals = "   ".join(f"{v:+13.9f}" for v in row)
        print(f"  {lbl}  {vals}")

    # ── Random vectors ───────────────────────────────────────────────
    print()
    print("  Random test vectors  (drawn from N(0, I)):")
    print(f"    v = [ {v[0]:+.6f},  {v[1]:+.6f},  {v[2]:+.6f} ]")
    print(f"    w = [ {w[0]:+.6f},  {w[1]:+.6f},  {w[2]:+.6f} ]")
    print(f"    ω = [ {omega[0]:+.6f},  {omega[1]:+.6f},  {omega[2]:+.6f} ]")

    # ════════════════════════════════════════════════════════════════
    # Identity 1 -- 3×1 vector comparison
    # ════════════════════════════════════════════════════════════════
    print()
    print(_hline("─"))
    print("  Identity 1:   R(v × w)  =  (Rv) × (Rw)")
    print(_hline("─"))
    print(f"    {'':5}  {'LHS  =  R @ (v × w)':>22}    {'RHS  =  (R@v) × (R@w)':>24}    {'diff':>14}")
    print(_hline())
    for i, lbl in enumerate(("  x │", "  y │", "  z │")):
        l, r, d = lhs1[i], rhs1[i], lhs1[i] - rhs1[i]
        print(f"  {lbl}   {l:+22.15f}    {r:+24.15f}    {d:+14.3e}")
    print()
    print(f"  Residual  ||LHS − RHS||₂  =  {resid1:.4e}    {check(resid1)}")

    # ════════════════════════════════════════════════════════════════
    # Identity 2 -- 3×3 matrix comparison
    # ════════════════════════════════════════════════════════════════
    print()
    print(_hline("─"))
    print("  Identity 2:   R · hat(ω) · Rᵀ  =  hat(R ω)")
    print(_hline("─"))
    print(f"    {'Rω':}\n"
          f"      = [{Romega[0]:+.6f}, {Romega[1]:+.6f}, {Romega[2]:+.6f}]")
    print()

    col_lhs = 26    # column width for LHS matrix
    print(f"  {'':3}  {'LHS  =  R @ hat(ω) @ Rᵀ':^{col_lhs}}      {'RHS  =  hat(Rω)':^{col_lhs}}      {'diff (LHS-RHS)':^18}")
    print(_hline())
    row_labels = ("  [0,·] │", "  [1,·] │", "  [2,·] │")
    for i, rlbl in enumerate(row_labels):
        lhs_row = "  ".join(f"{lhs2[i,j]:+9.5f}" for j in range(3))
        rhs_row = "  ".join(f"{rhs2[i,j]:+9.5f}" for j in range(3))
        dif_row = "  ".join(f"{lhs2[i,j]-rhs2[i,j]:+9.2e}" for j in range(3))
        print(f"  {rlbl}  [ {lhs_row} ]   [ {rhs_row} ]   [ {dif_row} ]")
    print()
    print(f"  Residual  ||LHS − RHS||_F  =  {resid2:.4e}    {check(resid2)}")

    # ── Footer ───────────────────────────────────────────────────────
    print()
    print("  " + "═" * (_W - 2))
    print(f"  Machine epsilon ε ≈ {eps:.3e}     "
          f"Id1: {resid1/eps:.1f}·ε     Id2: {resid2/eps:.1f}·ε")
    print("  " + "═" * (_W - 2))
    print()

    return {"max_resid_cross": resid1, "max_resid_skew": resid2}


# ══════════════════════════════════════════════════════════════════════
# Section 3 -- Compact stat table (auto-check)
# ══════════════════════════════════════════════════════════════════════

_TABLE_W = 72

def _print_header():
    print()
    print("=" * _TABLE_W)
    print("  Task 2: Numerical Verification of Skew-Symmetric Identities")
    print("=" * _TABLE_W)
    print("  Identity 1: R(v×w) = (Rv)×(Rw)   resid = ||LHS-RHS||₂")
    print("  Identity 2: Rhat(ω)Rᵀ = hat(Rω)   resid = ||LHS-RHS||_F")
    print(f"  Auto-check every {CHECK_INTERVAL}s  |  Press ENTER for full proof  |  SPACE=pause  Q=quit")
    print("=" * _TABLE_W)
    hdr = (f"  {'#':>4}  {'t (s)':>6}  "
           f"{'max|Id1|':>14}  {'max|Id2|':>14}  "
           f"{'Id1/ε':>8}  {'Id2/ε':>8}  {'ok?':>4}")
    print(hdr)
    print("  " + "─" * (_TABLE_W - 2))


def _print_row(idx: int, t: float, results: dict):
    eps    = np.finfo(float).eps
    rc, rs = results["max_resid_cross"], results["max_resid_skew"]
    ok     = "✓" if rc < 1e-10 and rs < 1e-10 else "✗"
    print(
        f"  {idx:>4}  {t:>6.3f}  "
        f"{rc:>14.3e}  {rs:>14.3e}  "
        f"{rc/eps:>8.1f}  {rs/eps:>8.1f}  {ok:>4}",
        flush=True,
    )


def _print_footer(all_results: list):
    eps = np.finfo(float).eps
    rc_all = [r["max_resid_cross"] for r in all_results]
    rs_all = [r["max_resid_skew"]  for r in all_results]
    print("  " + "─" * (_TABLE_W - 2))
    print(f"  Overall max Id1: {max(rc_all):.3e}   "
          f"Overall max Id2: {max(rs_all):.3e}   "
          f"ε = {eps:.3e}")
    if max(rc_all) < 1e-10 and max(rs_all) < 1e-10:
        print("  ✓  VERIFIED: both identities hold to floating-point precision.")
    else:
        print("  ✗  WARNING: residuals exceeded 1e-10 -- check implementation!")
    print("=" * _TABLE_W)


# ══════════════════════════════════════════════════════════════════════
# Section 4 -- Viewer overlay
# ══════════════════════════════════════════════════════════════════════

def _overlay(viewer, t: float, step_idx: int, results,
             omega: np.ndarray, R: np.ndarray, paused: bool):
    if not _HAS_OVERLAY:
        return
    viewer.clear_texts()
    texts = []

    # TOP-LEFT: live R matrix
    left  = "  R(t) -- live rotation matrix:\n  row-x |\n  row-y |\n  row-z |"
    right = (
        "     col-x      col-y      col-z\n"
        + "\n".join(
            "  " + "   ".join(f"{v:+.4f}" for v in row)
            for row in R
        )
    )
    texts.append((_FONT, _GRID.mjGRID_TOPLEFT, left, right))

    # TOP-RIGHT: status + last residuals
    status = "[ PAUSED ]" if paused else "[ LIVE   ]"
    omega_s = f"ω = [{omega[0]:+.2f}, {omega[1]:+.2f}, {omega[2]:+.2f}] rad/s"
    if results is not None:
        rc, rs = results["max_resid_cross"], results["max_resid_skew"]
        res_s = (f"t = {t:.3f}s  step {step_idx}\n"
                 f"Id1 resid: {rc:.2e}\n"
                 f"Id2 resid: {rs:.2e}")
    else:
        res_s = f"t = {t:.3f}s  (warming up ...)"
    texts.append((_FONT, _GRID.mjGRID_TOPRIGHT,
                  f"{status}  {omega_s}", res_s))

    # BOTTOM: legend
    legend = (
        "SPACE: pause/resume  |  ENTER: print full proof  |  T: stat row  |  Q/ESC: quit\n"
        "Id1:  R(v×w) = (Rv)×(Rw)            residual → 0 near machine ε\n"
        "Id2:  R hat(ω) Rᵀ = hat(Rω)          residual → 0 near machine ε"
    )
    texts.append((_FONT, _GRID.mjGRID_BOTTOM, legend, ""))
    viewer.set_texts(texts)


# ══════════════════════════════════════════════════════════════════════
# Section 5 -- CSV
# ══════════════════════════════════════════════════════════════════════

def _open_csv(path: str):
    fh = open(path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["step", "sim_time_s",
                 "max_resid_cross", "mean_resid_cross",
                 "max_resid_skew",  "mean_resid_skew",
                 "omega_x", "omega_y", "omega_z"])
    fh.flush()
    return fh, wr


def _write_csv(wr, fh, idx, t, results, omega):
    wr.writerow([idx, f"{t:.4f}",
                 f"{results['max_resid_cross']:.6e}",
                 f"{results['mean_resid_cross']:.6e}",
                 f"{results['max_resid_skew']:.6e}",
                 f"{results['mean_resid_skew']:.6e}",
                 f"{omega[0]:.4f}", f"{omega[1]:.4f}", f"{omega[2]:.4f}"])
    fh.flush()


# ══════════════════════════════════════════════════════════════════════
# Section 6 -- Main interactive loop
# ══════════════════════════════════════════════════════════════════════

def run(model, data, duration_s: float | None, headless: bool):
    """Physics + rendering loop.

    key_callback (GLFW thread) -> cmd_queue -> main thread.
    All mutable state (paused, R, step counters) lives on the main thread.
    """
    rng = np.random.default_rng(seed=42)
    cmd_queue: queue.SimpleQueue = queue.SimpleQueue()

    def key_callback(keycode: int) -> None:
        if   keycode == KEY_SPACE:               cmd_queue.put("pause")
        elif keycode in (KEY_ENTER, KEY_ENTER2): cmd_queue.put("proof")
        elif keycode == KEY_T:                   cmd_queue.put("stat")
        elif keycode in (KEY_Q, KEY_ESCAPE):     cmd_queue.put("quit")

    # ── Open viewer ─────────────────────────────────────────────────
    viewer = None
    if not headless:
        try:
            viewer = mjv.launch_passive(model, data, key_callback=key_callback)
            for _ in range(int(0.3 * VIEWER_FPS)):
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(1.0 / VIEWER_FPS)
        except Exception as e:
            print(f"[Warning] Viewer unavailable ({e}). Running headless.")

    # ── Open CSV ────────────────────────────────────────────────────
    try:
        csv_fh, csv_wr = _open_csv(CSV_PATH)
        print(f"  Logging residuals to '{CSV_PATH}'")
    except Exception as e:
        print(f"  [Note] Could not open CSV: {e}")
        csv_fh, csv_wr = None, None

    _print_header()

    # ── State ───────────────────────────────────────────────────────
    paused         = False
    step_idx       = 0
    all_results    = []
    latest_results = None
    next_check_t   = CHECK_INTERVAL
    should_quit    = False

    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qvel[:]   = 0.0
    mujoco.mj_forward(model, data)

    # ── Main loop ───────────────────────────────────────────────────
    while True:
        sim_t = data.time

        # ── Stop conditions ──────────────────────────────────────────
        if should_quit:
            break
        if viewer is not None and not viewer.is_running():
            print("\n  [Viewer closed -- stopping]")
            break
        if duration_s is not None and sim_t >= duration_s:
            break

        # ── Drain key events (non-blocking) ──────────────────────────
        while not cmd_queue.empty():
            ev = cmd_queue.get_nowait()

            if ev == "quit":
                should_quit = True
                break

            elif ev == "pause":
                paused = not paused
                state  = "PAUSED" if paused else "RESUMED"
                print(f"\n  [ {state} at t = {sim_t:.3f} s ]\n")

            elif ev in ("proof", "stat"):
                # Always operate on current live R and time
                R_snap     = get_body_orientation(data)
                t_snap     = sim_t
                omega_snap = data.qvel[3:6].copy()

                if ev == "proof":
                    # Full side-by-side printout
                    r = print_proof(R_snap, t_snap, rng)
                    all_results.append(r)
                    latest_results = r
                else:
                    # Silent stat row
                    r = check_identities(R_snap, rng)
                    step_idx += 1
                    all_results.append(r)
                    latest_results = r
                    _print_row(step_idx, t_snap, r)
                    if csv_wr:
                        _write_csv(csv_wr, csv_fh, step_idx, t_snap, r, omega_snap)

        if should_quit:
            break

        # ── Physics step (only when not paused) ──────────────────────
        if not paused:
            data.qvel[3:6] = time_varying_omega(sim_t)
            mujoco.mj_step(model, data)

            # Auto-check every CHECK_INTERVAL simulated seconds
            if data.time >= next_check_t:
                R     = get_body_orientation(data)
                omega = data.qvel[3:6].copy()
                r     = check_identities(R, rng)
                step_idx  += 1
                all_results.append(r)
                latest_results = r
                _print_row(step_idx, data.time, r)
                if csv_wr:
                    _write_csv(csv_wr, csv_fh, step_idx, data.time, r, omega)
                next_check_t += CHECK_INTERVAL

        # ── Viewer render ─────────────────────────────────────────────
        if viewer is not None and viewer.is_running():
            R_live     = get_body_orientation(data)
            omega_live = data.qvel[3:6].copy()
            _overlay(viewer, data.time, step_idx, latest_results,
                     omega_live, R_live, paused)
            viewer.sync()
            time.sleep(1.0 / VIEWER_FPS)
        # Headless: no sleep (run as fast as possible)

    # ── Cleanup ──────────────────────────────────────────────────────
    if viewer is not None and viewer.is_running():
        viewer.close()
    if csv_fh:
        csv_fh.close()
        print(f"\n  Results saved → '{CSV_PATH}'")
    if all_results:
        _print_footer(all_results)


# ══════════════════════════════════════════════════════════════════════
# Section 7 -- Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ME 639 HW1 Task 2 -- interactive skew-identity verifier"
    )
    parser.add_argument(
        "--duration", type=float, default=None, metavar="SEC",
        help="Optional simulated-time limit in seconds. Default: run until Q/ESC/close."
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="No viewer window. Runs until --duration is reached."
    )
    args = parser.parse_args()

    if args.headless and args.duration is None:
        args.duration = 5.0   # sensible default for CI

    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    except Exception as e:
        sys.exit(f"[ERROR] Could not load '{MODEL_PATH}': {e}")
    data = mujoco.MjData(model)

    print()
    print("  ME 639 HW1 Task 2 -- Skew-Symmetric Identity Verifier")
    print("  Physics: tumbling dart, no gravity, time-varying ω(t)")
    print("  Keys (in viewer): SPACE=pause  ENTER=full proof  T=stat row  Q=quit")
    if args.duration:
        print(f"  Running for {args.duration:.1f} simulated seconds.")
    else:
        print("  Running indefinitely -- press Q in the viewer to quit.")

    run(model, data, duration_s=args.duration, headless=args.headless)


if __name__ == "__main__":
    main()
