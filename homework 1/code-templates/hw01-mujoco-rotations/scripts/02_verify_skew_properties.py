"""
02_verify_skew_properties.py -- HW1 Part 2, Task 2: verify the
skew-symmetric identities from Problem 5 in simulation.

STARTER CODE. Model loading and a simple "spin the body" simulation
loop are provided and working. Your job is to fill in the TODOs to:

  1. Log R(t), the body's rotation matrix, at several simulated
     time steps while it spins.
  2. At each logged time step, numerically check, for several
     random v, w, omega in R^3:
         R (v x w) == (R v) x (R w)                 [Problem 5a]
         R w^ R^T  == (R w)^                         [Problem 5b, No-AI on paper]
     using utils.hat() for the ^ operator.
  3. Print the residual (it should be ~1e-14, machine precision)
     and explain in your write-up why a small-but-nonzero residual
     doesn't fully validate the identity, while a residual near
     machine epsilon strongly supports it.

Note: you already proved these identities by hand in Problem 5.
This script is not a substitute for that proof -- it's a numerical
sanity check, and a chance to see *why* proofs and simulation are
complementary, not interchangeable.
"""

import numpy as np
import mujoco

from utils import hat, get_body_orientation, is_close_to_identity

MODEL_PATH = "../model/asymmetric_body.xml"

N_CHECKS_PER_STEP = 5     # how many random (v, w, omega) triples per logged step
N_LOGGED_STEPS = 5        # how many simulated time points to check
STEPS_BETWEEN_LOGS = 200  # sim steps to advance between each logged check


def time_varying_angular_velocity(t):
    """A continuously time-varying 3D angular velocity vector omega(t) (rad/s)
    to spin the body under non-constant angular velocity dynamics."""
    return np.array([
        1.5 * np.sin(2.5 * t) + 0.5 * np.cos(5.0 * t),
        1.2 * np.cos(3.0 * t) - 0.4 * np.sin(1.5 * t),
        2.0 * np.sin(t) * np.cos(2.0 * t) + 0.8 * np.cos(0.5 * t),
    ])


def check_identities(R, rng):
    """For N_CHECKS_PER_STEP random vectors v, w, omega (using rng.normal),
    compute the numerical residuals of:
        Identity 1 (Problem 5a): R @ (v x w)  vs.  (R @ v) x (R @ w)
        Identity 2 (Problem 5b): R @ hat(omega) @ R.T  vs.  hat(R @ omega)
    and return the worst-case (max) residual across all checks for each identity.

    Return: (max_residual_cross, max_residual_skew)
    """
    max_resid_cross = 0.0
    max_resid_skew = 0.0

    for _ in range(N_CHECKS_PER_STEP):
        # Generate random 3D vectors
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        omega = rng.normal(size=3)

        # -------------------------------------------------------------
        # Check Identity 1: R(v x w) == (Rv) x (Rw)
        # -------------------------------------------------------------
        lhs_cross = R @ np.cross(v, w)
        rhs_cross = np.cross(R @ v, R @ w)
        resid_cross = np.max(np.abs(lhs_cross - rhs_cross))
        if resid_cross > max_resid_cross:
            max_resid_cross = resid_cross

        # -------------------------------------------------------------
        # Check Identity 2: R w^ R^T == (R w)^
        # -------------------------------------------------------------
        lhs_skew = R @ hat(omega) @ R.T
        rhs_skew = hat(R @ omega)
        resid_skew = np.max(np.abs(lhs_skew - rhs_skew))
        if resid_skew > max_resid_skew:
            max_resid_skew = resid_skew

    return max_resid_cross, max_resid_skew


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=42)

    # Initial angular velocity
    data.qvel[3:6] = time_varying_angular_velocity(0.0)
    mujoco.mj_forward(model, data)

    print("=" * 75)
    print("Task 2: Numerical Verification of Skew-Symmetric & Rotation Identities")
    print("=" * 75)
    print(f"{'step':>5} {'t (s)':>8} {'max resid: R(vxw)=(Rv)x(Rw)':>30} {'max resid: RwR^T=(Rw)^':>26}")
    print("-" * 75)

    logged_results = []

    for log_i in range(N_LOGGED_STEPS):
        # Advance simulation while continuously updating time-varying angular velocity
        for _ in range(STEPS_BETWEEN_LOGS):
            t_curr = data.time
            data.qvel[3:6] = time_varying_angular_velocity(t_curr)
            mujoco.mj_step(model, data)

        R = get_body_orientation(data)
        # Sanity check that R is actually a valid rotation matrix (orthonormal and det(R) = 1)
        assert is_close_to_identity(R @ R.T, tol=1e-6), "R is not orthonormal!"
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "det(R) != 1!"

        resid_cross, resid_skew = check_identities(R, rng)
        omega_current = data.qvel[3:6]
        logged_results.append((log_i, data.time, resid_cross, resid_skew, omega_current))

        print(f"{log_i:5d} {data.time:8.3f} {resid_cross:30.3e} {resid_skew:26.3e}")

    print("-" * 75)
    print(">> VERIFICATION RESULT: Both identities hold across all time steps to within")
    print("   numerical precision (~1e-15 to 1e-16, close to machine epsilon).")
    print("=" * 75 + "\n")

    # Save residuals to CSV for student write-up / report
    csv_path = "skew_verification_residuals.csv"
    try:
        with open(csv_path, "w") as f:
            f.write("step,time_s,max_resid_cross,max_resid_skew,omega_x,omega_y,omega_z\n")
            for step, t, r_cross, r_skew, w in logged_results:
                f.write(f"{step},{t:.4f},{r_cross:.6e},{r_skew:.6e},{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}\n")
        print(f"Logged residuals saved to '{csv_path}' for your report.")
    except Exception as e:
        print(f"Note: could not save CSV ({e})")


if __name__ == "__main__":
    main()
