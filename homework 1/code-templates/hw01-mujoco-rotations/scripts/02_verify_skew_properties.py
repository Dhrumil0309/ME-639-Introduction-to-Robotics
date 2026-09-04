"""
02_verify_skew_properties.py  --  ME 639 HW1, Task 2 (Problem 8)
=================================================================
Live numerical verification of the two skew-symmetric / rotation identities
that were proved analytically on paper in Problem 5:

  Identity 1 (cross-product equivariance):
      R (v × w)  =  (R v) × (R w)               for all v, w ∈ ℝ³

  Identity 2 (adjoint / similarity of skew matrices):
      R ω̂ Rᵀ  =  (R ω)^                         for all ω ∈ ℝ³
  equivalently written:  R hat(ω) Rᵀ = hat(R ω)

HOW IT WORKS
------------
1. The passive MuJoCo viewer is launched so the dart tumbles live on screen.
2. A time-varying angular velocity ω(t) is injected into the free-joint
   velocity every physics step. No gravity, no contacts -- pure rotation.
3. Every CHECK_INTERVAL seconds of simulated time the script:
     a. Reads the live 3×3 rotation matrix R(t) directly from MuJoCo's
        quaternion state (qpos[3:7]).
     b. Generates N_RANDOM_CHECKS random vectors v, w, ω ~ N(0, I).
     c. Computes the Frobenius-norm residuals of both identities.
     d. Prints a clean table row to the terminal.
     e. Writes the results to a running CSV file.
4. The viewer stays responsive throughout -- physics and terminal logging
   happen on the main thread; viewer.sync() is called every rendered frame.

EXPECTED RESIDUALS
------------------
Both identities hold to floating-point precision because they are exact
algebraic identities in SO(3) (no approximations). Residuals should be
   ~1e-14 to 1e-16   (near double-precision machine epsilon ε ≈ 2.2e-16)

A residual this small does NOT constitute a proof (we only sample finitely
many orientations and vectors). It is a numerical sanity check that
complements the paper proof from Problem 5.

USAGE
-----
  cd scripts/
  python 02_verify_skew_properties.py
  python 02_verify_skew_properties.py --duration 15  # run for 15 sim-seconds
  python 02_verify_skew_properties.py --headless      # no viewer, fast CI mode

AI USE NOTE
-----------
  Thread architecture, overlay text, and CSV writer written with AI
  assistance. The mathematical identities and the hat() operator are
  from Lynch & Park Ch.3 / HW1 Problem 5.
"""

import sys
import time
import argparse
import csv
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer as mjv

from utils import get_body_orientation, is_close_to_identity

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH      = "../model/asymmetric_body.xml"
CHECK_INTERVAL  = 0.5          # simulated seconds between identity checks
N_RANDOM_CHECKS = 8            # random (v, w, ω) triples per logged step
VIEWER_FPS      = 60           # viewer sync rate (Hz)
CSV_PATH        = "skew_verification_residuals.csv"

# Fonts / grid positions for the viewer overlay
try:
    _FONT = mujoco.mjtFontScale.mjFONTSCALE_150
    _GRID = mujoco.mjtGridPos
    _HAS_OVERLAY = True
except Exception:
    _HAS_OVERLAY = False


# ══════════════════════════════════════════════════════════════════════
# Section 1 -- Mathematical helpers
# ══════════════════════════════════════════════════════════════════════

def hat(omega: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of ω ∈ ℝ³.

    Defined so that  hat(ω) v = ω × v  for any v ∈ ℝ³.
    This is the standard ω̂ (hat) operator from Lynch & Park §3.2.

        ⎡  0   -ω₃   ω₂ ⎤
    ω̂ = ⎢  ω₃   0   -ω₁ ⎥
        ⎣ -ω₂   ω₁    0 ⎦
    """
    wx, wy, wz = float(omega[0]), float(omega[1]), float(omega[2])
    return np.array([
        [ 0.0, -wz,  wy],
        [ wz,   0.0, -wx],
        [-wy,   wx,   0.0],
    ])


def check_identities(R: np.ndarray, rng: np.random.Generator) -> dict:
    """Evaluate both identities for N_RANDOM_CHECKS random vector triples.

    Returns a dict with:
        'max_resid_cross'  : max Frobenius residual of Identity 1
        'max_resid_skew'   : max Frobenius residual of Identity 2
        'mean_resid_cross' : mean residual of Identity 1
        'mean_resid_skew'  : mean residual of Identity 2
    """
    resids_cross = []
    resids_skew  = []

    for _ in range(N_RANDOM_CHECKS):
        v     = rng.standard_normal(3)
        w     = rng.standard_normal(3)
        omega = rng.standard_normal(3)

        # ── Identity 1: R(v × w) = (Rv) × (Rw) ──────────────────────
        lhs1 = R @ np.cross(v, w)
        rhs1 = np.cross(R @ v, R @ w)
        resids_cross.append(np.linalg.norm(lhs1 - rhs1))   # Frobenius = L2 for vectors

        # ── Identity 2: R hat(ω) Rᵀ = hat(Rω) ───────────────────────
        lhs2 = R @ hat(omega) @ R.T
        rhs2 = hat(R @ omega)
        resids_skew.append(np.linalg.norm(lhs2 - rhs2, ord='fro'))

    return {
        "max_resid_cross":  max(resids_cross),
        "max_resid_skew":   max(resids_skew),
        "mean_resid_cross": float(np.mean(resids_cross)),
        "mean_resid_skew":  float(np.mean(resids_skew)),
    }


def time_varying_omega(t: float) -> np.ndarray:
    """Continuously varying angular velocity ω(t) ∈ ℝ³ (rad/s).

    Chosen to be non-periodic and to excite all three body axes so the
    dart tumbles in a visually interesting way across the full duration.

      ωₓ(t) = 1.5 sin(2.5t) + 0.5 cos(5.0t)
      ωᵧ(t) = 1.2 cos(3.0t) − 0.4 sin(1.5t)
      ω_z(t) = 2.0 sin(t)cos(2t) + 0.8 cos(0.5t)
    """
    return np.array([
        1.5 * np.sin(2.5 * t) + 0.5 * np.cos(5.0 * t),
        1.2 * np.cos(3.0 * t) - 0.4 * np.sin(1.5 * t),
        2.0 * np.sin(t) * np.cos(2.0 * t) + 0.8 * np.cos(0.5 * t),
    ])


# ══════════════════════════════════════════════════════════════════════
# Section 2 -- Terminal printing
# ══════════════════════════════════════════════════════════════════════

_COL_W = 72

def _print_header():
    print()
    print("=" * _COL_W)
    print("  Task 2: Numerical Verification of Skew-Symmetric Identities")
    print("=" * _COL_W)
    print("  Identity 1 (cross-product equivariance):")
    print("      R(v × w) = (Rv) × (Rw)       residual = ||LHS - RHS||_2")
    print()
    print("  Identity 2 (adjoint / skew similarity):")
    print("      R hat(ω) Rᵀ = hat(Rω)         residual = ||LHS - RHS||_F")
    print()
    print(f"  Checking {N_RANDOM_CHECKS} random vector triples every {CHECK_INTERVAL}s of sim time.")
    print(f"  Expected residuals: ~1e-14 to 1e-16  (near machine epsilon ε ≈ 2.2e-16)")
    print("=" * _COL_W)
    hdr = (f"  {'#':>3}  {'t (s)':>6}  "
           f"{'max |R(vxw)-(Rv)x(Rw)|':>26}  "
           f"{'max |Rhw^RT - h(Rw)|_F':>24}  "
           f"{'PASS?':>5}")
    print(hdr)
    print("  " + "-" * (_COL_W - 2))


def _print_row(idx: int, t: float, results: dict):
    r_c = results["max_resid_cross"]
    r_s = results["max_resid_skew"]
    eps = np.finfo(float).eps                # 2.22e-16
    passed = r_c < 1e-10 and r_s < 1e-10    # well below any rounding concern
    flag = "✓" if passed else "✗ FAIL"
    row = (f"  {idx:>3}  {t:>6.3f}  "
           f"{r_c:>26.3e}  "
           f"{r_s:>24.3e}  "
           f"  {flag}")
    print(row, flush=True)


def _print_footer(results_list: list):
    print("  " + "-" * (_COL_W - 2))
    all_c = [r["max_resid_cross"] for r in results_list]
    all_s = [r["max_resid_skew"]  for r in results_list]
    print(f"  Overall max residual Identity 1: {max(all_c):.3e}")
    print(f"  Overall max residual Identity 2: {max(all_s):.3e}")
    eps = np.finfo(float).eps
    print(f"  Machine epsilon (double):        {eps:.3e}")
    print()
    if max(all_c) < 1e-10 and max(all_s) < 1e-10:
        print("  ✓  VERIFIED: Both identities hold at floating-point precision.")
        print("     Residuals are < 1e-10 across all sampled orientations,")
        print("     consistent with exact algebraic identities in SO(3).")
    else:
        print("  ✗  WARNING: Some residuals exceeded 1e-10 -- check implementation.")
    print("=" * _COL_W)


# ══════════════════════════════════════════════════════════════════════
# Section 3 -- Viewer overlay
# ══════════════════════════════════════════════════════════════════════

def _overlay(viewer, t: float, step_idx: int, results: dict | None,
             omega: np.ndarray, R: np.ndarray):
    """Push live overlay text to the viewer window."""
    if not _HAS_OVERLAY:
        return

    viewer.clear_texts()
    texts = []

    # TOP-LEFT: live rotation matrix
    left_mat = "  R(t) -- live rotation matrix:\n  row-x |\n  row-y |\n  row-z |"
    right_mat = (
        "     col-x      col-y      col-z\n"
        + "\n".join(
            "  " + "   ".join(f"{v:+.4f}" for v in row)
            for row in R
        )
    )
    texts.append((_FONT, _GRID.mjGRID_TOPLEFT, left_mat, right_mat))

    # TOP-RIGHT: live ω(t) and latest residuals
    omega_str = (f"ω = [{omega[0]:+.2f}, {omega[1]:+.2f}, {omega[2]:+.2f}] rad/s")
    if results is not None:
        rc = results["max_resid_cross"]
        rs = results["max_resid_skew"]
        res_str = (f"t = {t:.2f}s  (step {step_idx})\n"
                   f"Resid Id1: {rc:.2e}\n"
                   f"Resid Id2: {rs:.2e}")
    else:
        res_str = f"t = {t:.2f}s  (warming up ...)"
    texts.append((_FONT, _GRID.mjGRID_TOPRIGHT, omega_str, res_str))

    # BOTTOM: instructions
    legend = (
        "Identity 1:  R(v×w) = (Rv)×(Rw)   residual → 0 at machine ε\n"
        "Identity 2:  R hat(ω) Rᵀ = hat(Rω)  residual → 0 at machine ε\n"
        "Checking every 0.5 sim-seconds  |  Close window to stop"
    )
    texts.append((_FONT, _GRID.mjGRID_BOTTOM, legend, ""))

    viewer.set_texts(texts)


# ══════════════════════════════════════════════════════════════════════
# Section 4 -- CSV writer
# ══════════════════════════════════════════════════════════════════════

def _open_csv(path: str) -> tuple[object, object]:
    """Open the CSV file and write the header row. Returns (file_handle, writer)."""
    fh = open(path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow([
        "step", "sim_time_s",
        "max_resid_cross", "mean_resid_cross",
        "max_resid_skew",  "mean_resid_skew",
        "omega_x", "omega_y", "omega_z",
    ])
    fh.flush()
    return fh, wr


def _write_csv_row(wr, fh, idx: int, t: float,
                   results: dict, omega: np.ndarray):
    wr.writerow([
        idx, f"{t:.4f}",
        f"{results['max_resid_cross']:.6e}", f"{results['mean_resid_cross']:.6e}",
        f"{results['max_resid_skew']:.6e}",  f"{results['mean_resid_skew']:.6e}",
        f"{omega[0]:.4f}", f"{omega[1]:.4f}", f"{omega[2]:.4f}",
    ])
    fh.flush()


# ══════════════════════════════════════════════════════════════════════
# Section 5 -- Main simulation + verification loop
# ══════════════════════════════════════════════════════════════════════

def run(model, data, duration_s: float, headless: bool):
    """Main loop: physics + identity verification + viewer rendering."""

    rng = np.random.default_rng(seed=42)

    # Open viewer (or headless)
    viewer = None
    if not headless:
        try:
            viewer = mjv.launch_passive(model, data)
            # Initial sync burst so window appears before we start logging
            for _ in range(int(0.3 * VIEWER_FPS)):
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(1.0 / VIEWER_FPS)
        except Exception as e:
            print(f"[Warning] Viewer unavailable ({e}). Running headless.")

    # Open CSV output
    csv_fh, csv_wr = None, None
    try:
        csv_fh, csv_wr = _open_csv(CSV_PATH)
        print(f"  Logging results to '{CSV_PATH}'")
    except Exception as e:
        print(f"  [Note] Could not open CSV ({e}) -- terminal only.")

    _print_header()

    # ── Simulation bookkeeping ─────────────────────────────────────────
    dt              = model.opt.timestep   # 0.005 s
    next_check_t    = CHECK_INTERVAL       # first identity check at t=0.5s
    step_idx        = 0
    all_results     = []
    latest_results  = None
    wall_t0         = time.time()

    # Initial physics state
    data.qpos[0:3] = [0.0, 0.0, 1.0]      # keep body at z=1 (clear of floor)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # ── Main loop ─────────────────────────────────────────────────────
    while True:
        sim_t = data.time

        # ── Stop condition ────────────────────────────────────────────
        if sim_t >= duration_s:
            break
        if viewer is not None and not viewer.is_running():
            print("\n  [Viewer closed by user -- stopping simulation]")
            break

        # ── Drive angular velocity: re-inject ω(t) every physics step ──
        # This continuously changes the angular velocity so the body tumbles
        # in a non-periodic, dynamically interesting way.
        data.qvel[3:6] = time_varying_omega(sim_t)

        # ── Advance one physics step ──────────────────────────────────
        mujoco.mj_step(model, data)

        # ── Identity check (every CHECK_INTERVAL simulated seconds) ───
        if data.time >= next_check_t:
            # Read R(t) from the live quaternion in qpos
            R = get_body_orientation(data)
            omega = data.qvel[3:6].copy()

            # Sanity check: R must be a valid rotation matrix
            R_err  = np.linalg.norm(R @ R.T - np.eye(3), ord='fro')
            det_R  = np.linalg.det(R)

            results = check_identities(R, rng)
            all_results.append(results)
            latest_results = results
            step_idx += 1

            _print_row(step_idx, data.time, results)

            if csv_wr is not None:
                _write_csv_row(csv_wr, csv_fh, step_idx, data.time, results, omega)

            # Annotate the terminal with a sanity line if R drifted
            if R_err > 1e-6 or abs(det_R - 1.0) > 1e-6:
                print(f"    [!] R drift: ||RRᵀ-I||_F={R_err:.1e}  det(R)={det_R:.6f}")

            next_check_t += CHECK_INTERVAL

        # ── Viewer render (every frame) ───────────────────────────────
        if viewer is not None and viewer.is_running():
            R_live    = get_body_orientation(data)
            omega_live = data.qvel[3:6].copy()
            _overlay(viewer, data.time, step_idx, latest_results, omega_live, R_live)
            viewer.sync()
            # Throttle: prevent the loop from spinning faster than VIEWER_FPS
            time.sleep(1.0 / VIEWER_FPS)
        # In headless mode: advance as fast as possible (no sleep)

    # ── Cleanup ───────────────────────────────────────────────────────
    if viewer is not None and viewer.is_running():
        viewer.close()

    if csv_fh is not None:
        csv_fh.close()
        print(f"\n  Results saved to '{CSV_PATH}'.")

    if all_results:
        _print_footer(all_results)
    else:
        print("  [No identity checks were completed -- increase duration or decrease CHECK_INTERVAL]")


# ══════════════════════════════════════════════════════════════════════
# Section 6 -- Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ME 639 HW1 Task 2 -- Numerical verification of skew-symmetric identities"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, metavar="SEC",
        help="Total simulated time in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without opening the viewer (fast mode for CI/testing)"
    )
    args = parser.parse_args()

    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    except Exception as e:
        sys.exit(f"[ERROR] Could not load '{MODEL_PATH}': {e}")
    data = mujoco.MjData(model)

    print(f"\n  Simulating {args.duration:.1f} s of tumbling rotation.")
    print(f"  Checking identities every {CHECK_INTERVAL} s  ({int(args.duration/CHECK_INTERVAL)} checks total).")
    if args.headless:
        print("  Headless mode: no viewer window.\n")

    run(model, data, duration_s=args.duration, headless=args.headless)


if __name__ == "__main__":
    main()
